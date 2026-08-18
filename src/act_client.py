#!/usr/bin/env python3
"""
Cliente de inferencia para o ACT (3 cameras RGB, SEM depth),
conectado ao servidor serve_act.py (porta 8001).

DIFERENCAS em relacao ao cliente multimodal anterior:
  - SEM profundidade -- so' 3 subscribers de imagem (nao 6).
  - chunk_size=50, na grade das CAMERAS (nao mais 100Hz nativo). O
    espacamento real entre passos do chunk e' 1/fps_camera, nao
    ACT_STEP_DT=0.01 fixo -- ajuste CAMERA_FPS abaixo para o valor
    real das suas ZEDs.
  - Normalizacao: o servidor cuida disso internamente (NORM_INDICES
    salvos no checkpoint); o cliente so' manda qpos CRU, igual antes.
  - Mesma logica de histerese assimetrica no gripper e trajetoria
    suave via scaled_joint_trajectory_controller, reaproveitadas do
    cliente do SmolVLA/ACT multimodal.
"""

import asyncio
import base64
import json
import threading
import time
import cv2
import numpy as np
import websockets

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO
import message_filters
from cv_bridge import CvBridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO
# ══════════════════════════════════════════════════════════════════════════════

SERVER_URI = "ws://localhost:8001/infer"  # porta do serve_act.py novo

CHUNK_SIZE = 50  # FIXO pelo treino do ACT (grade da camera, nao 100Hz)

# Quantos passos do chunk previsto usar como alvo -- ajuste conforme
# testes no robo. Comeca conservador (menos passos = reage mais
# rapido a observacoes novas, mas trajetoria menos suave).
EXECUTION_STEPS = 25

# FPS real das cameras ZED usadas no treino (~28fps, conforme
# comentario no script de treino). Define o espacamento entre passos
# do chunk previsto.
CAMERA_FPS = 25.0
ACT_STEP_DT = 1.0 / CAMERA_FPS

IMG_H, IMG_W = 320, 320  # mesmo tamanho usado no treino (train_act.py)

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ══════════════════════════════════════════════════════════════════════════════
# NO ROS2 -- controle do robo real + captura de camera RGB (3 streams)
# ══════════════════════════════════════════════════════════════════════════════

class UR5RosBridge(Node):
    def __init__(self):
        super().__init__("ur5_act_inference_bridge")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self._current_joint_positions = None
        self._current_gripper_state = 0.0
        self._latest_images = None  # dict com 3 chaves: left_rgb, robot_rgb, right_rgb

        self.gripper_pin = 16
        self._last_sent_gripper_state = None
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        self.get_logger().info("Aguardando servico de I/O do robo...")
        self.io_client.wait_for_service()
        self.get_logger().info("Servico de I/O conectado!")

        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10
        )

        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb,
            qos_profile=qos_profile_sensor_data
        )

        qos_camera = QoSProfile(depth=10)
        qos_camera.reliability = ReliabilityPolicy.BEST_EFFORT

        # ── 3 subscribers RGB (sem depth) -- AJUSTE os topicos reais ──
        left_rgb_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_left/rgb/color/rect/image", qos_profile=qos_camera)
        robot_rgb_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_robot/rgb/color/rect/image", qos_profile=qos_camera)
        right_rgb_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_right/rgb/color/rect/image", qos_profile=qos_camera)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_rgb_sub, robot_rgb_sub, right_rgb_sub],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._camera_cb)

        self.get_logger().info("UR5RosBridge (3 RGB) pronto -- aguardando dados.")

    def _joint_state_cb(self, msg: JointState):
        try:
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            joint_pos = np.array([pos_map[j] for j in UR5_JOINT_NAMES], dtype=np.float32)
            with self._lock:
                self._current_joint_positions = joint_pos
        except KeyError:
            pass

    def _camera_cb(self, l_rgb, rob_rgb, r_rgb):
        try:
            def to_rgb(msg):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return cv_img[:, :, ::-1].copy()

            images = {
                "left_rgb": to_rgb(l_rgb),
                "robot_rgb": to_rgb(rob_rgb),
                "right_rgb": to_rgb(r_rgb),
            }
            with self._lock:
                self._latest_images = images
        except Exception as e:
            self.get_logger().error(f"Erro no callback de camera: {e}")

    def get_observation(self):
        with self._lock:
            if self._current_joint_positions is None or self._latest_images is None:
                return None
            return (
                self._current_joint_positions.copy(),
                self._current_gripper_state,
                dict(self._latest_images),
            )

    def get_current_joint_positions(self):
        with self._lock:
            if self._current_joint_positions is None:
                return None
            return self._current_joint_positions.copy()

    def publish_smooth_trajectory(self, actions_6d_list, duration_per_step: float):
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES

        for i, action_6d in enumerate(actions_6d_list):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in action_6d]
            t = (i + 1) * duration_per_step
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9)
            )
            msg.points.append(point)

        msg.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(msg)

    def publish_gripper_command(self, action_gripper_value: float,
                                 threshold_close: float = 0.2,
                                 threshold_open: float = 0.3):
        """Histerese assimetrica -- mesma logica validada no cliente do SmolVLA/ACT multimodal."""
        current_state = self._last_sent_gripper_state

        if current_state is None:
            new_state = 1 if action_gripper_value > threshold_close else 0
        elif current_state == 0:
            new_state = 1 if action_gripper_value > threshold_close else 0
        else:
            new_state = 0 if action_gripper_value < threshold_open else 1

        if new_state == current_state:
            return

        req = SetIO.Request()
        req.fun = 1
        req.pin = self.gripper_pin
        req.state = float(new_state)
        self.io_client.call_async(req)

        self._last_sent_gripper_state = new_state
        with self._lock:
            self._current_gripper_state = float(new_state)

        action_verb = "FECHANDO" if new_state == 1 else "ABRINDO"
        self.get_logger().info(
            f"[GRIPPER] {action_verb} (valor previsto: {action_gripper_value:.3f})"
        )

        if new_state == 1:
            self.get_logger().info("[GRIPPER] Pausando 2s apos fechar...")
            time.sleep(2.0)


def ros_spin_thread(node):
    rclpy.spin(node)


def encode_rgb_b64(img: np.ndarray) -> str:
    """Comprime RGB como JPEG (com perdas, ok para cor)."""
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Falha ao codificar RGB como JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


# ══════════════════════════════════════════════════════════════════════════════
# Cliente
# ══════════════════════════════════════════════════════════════════════════════

class ACTInferenceClient:
    def __init__(self, ros_bridge: UR5RosBridge):
        self.ros_bridge = ros_bridge
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(SERVER_URI, max_size=None)
        print(f"[CLIENTE] Conectado ao servidor: {SERVER_URI}")

    def _build_payload(self, qpos_7d, images):
        return {
            "qpos": qpos_7d,
            "img_left_rgb": encode_rgb_b64(images["left_rgb"]),
            "img_robot_rgb": encode_rgb_b64(images["robot_rgb"]),
            "img_right_rgb": encode_rgb_b64(images["right_rgb"]),
        }

    async def _request_inference(self, joint_pos, gripper_state, images):
        """Envia UMA observacao ao servidor e aguarda o chunk completo de volta."""
        t_a = time.perf_counter()
        qpos_7d = np.concatenate([joint_pos, [gripper_state]]).tolist()

        payload = await asyncio.to_thread(self._build_payload, qpos_7d, images)
        json_str = json.dumps(payload)

        await self.websocket.send(json_str)
        response_text = await self.websocket.recv()
        response = json.loads(response_text)

        t_g = time.perf_counter()

        actions = np.array(response["actions"], dtype=np.float32)  # (CHUNK_SIZE, 7)
        infer_ms = response.get("server_infer_time_ms", None)
        print(f"[INFERENCIA] Chunk recebido ({actions.shape[0]} acoes) | "
              f"round-trip total: {(t_g - t_a)*1000:.1f}ms"
              + (f" | calculo puro no servidor: {infer_ms:.1f}ms" if infer_ms else ""))
        return actions

    async def run(self, horizon_steps: int):
        step_counter = 0
        ARRIVAL_TOLERANCE = 0.001
        ARRIVAL_TIMEOUT_S = 0.3

        while step_counter < horizon_steps:
            obs = None
            while obs is None:
                obs = self.ros_bridge.get_observation()
                if obs is None:
                    await asyncio.sleep(0.05)

            joint_pos, gripper_state, images = obs

            print(f"[CLIENTE] step={step_counter} | pedindo novo chunk...")
            full_chunk = await self._request_inference(joint_pos, gripper_state, images)
            print(f"[GRIPPER] valores previstos ({EXECUTION_STEPS} steps usados): "
                  f"{np.mean(full_chunk[:EXECUTION_STEPS, 6]):.3f}")

            chunk_to_use = full_chunk[:EXECUTION_STEPS]
            final_target = chunk_to_use[-1][:6]
            gripper_value_final = np.mean(chunk_to_use[:, 6])

            total_duration = EXECUTION_STEPS * ACT_STEP_DT

            # NOTA: publicacao real do movimento fica comentada por
            # seguranca ate' voce validar os valores previstos --
            # descomente quando estiver pronto para mover o robo.
            self.ros_bridge.publish_smooth_trajectory([final_target], total_duration)
            self.ros_bridge.publish_gripper_command(float(gripper_value_final))

            print(f"[TRAJETORIA] step={step_counter} | alvo (duracao="
                  f"{total_duration*1000:.0f}ms): {np.round(final_target, 4)}")

            target = final_target
            t_wait_start = time.perf_counter()
            arrived = False
            while (time.perf_counter() - t_wait_start) < ARRIVAL_TIMEOUT_S:
                current = self.ros_bridge.get_current_joint_positions()
                if current is not None:
                    error = np.max(np.abs(current - target))
                    if error < ARRIVAL_TOLERANCE:
                        arrived = True
                        break
                await asyncio.sleep(0.02)

            if not arrived:
                print(f"[AVISO] step={step_counter} | timeout esperando chegada "
                      f"(erro final={error:.4f} rad) -- prosseguindo mesmo assim.")

            step_counter += EXECUTION_STEPS

        print("[CLIENTE] Horizon concluido.")


async def main_async():
    ros_bridge_holder = {}

    def start_ros():
        rclpy.init()
        node = UR5RosBridge()
        ros_bridge_holder["node"] = node
        ros_spin_thread(node)

    ros_thread = threading.Thread(target=start_ros, daemon=True)
    ros_thread.start()

    while "node" not in ros_bridge_holder:
        await asyncio.sleep(0.1)

    ros_bridge = ros_bridge_holder["node"]

    client = ACTInferenceClient(ros_bridge)
    await client.connect()
    await client.run(horizon_steps=20000)


if __name__ == "__main__":
    asyncio.run(main_async())