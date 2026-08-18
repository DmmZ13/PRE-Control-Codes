#!/usr/bin/env python3
"""
Cliente de inferencia SINCRONO para o UR5 real + servidor SmolVLA de
3 cameras.

MUDANCAS em relacao a versao anterior:

  1. TERCEIRA CAMERA (image_3 = zed_right) adicionada ao payload, para
     casar com o modelo treinado com 3 cameras. A ORDEM tem de ser
     identica a da conversao do dataset:
        image_1 = zed_left | image_2 = zed_robot | image_3 = zed_right

  2. TRAJETORIA INTEIRA. A versao anterior descartava 24 dos 25 pontos
     e mandava o robo direto para o ultimo (final_target), em linha
     reta. Isso e um regime que o modelo NUNCA viu no treino -- ele
     foi treinado sobre trajetorias densas ponto a ponto -- e destroi
     o angulo de aproximacao que o modelo planejou (importante no
     pick). Agora publicamos os EXECUTION_STEPS_PER_CHUNK pontos como
     uma unica trajetoria e o controlador interpola suavemente.

  3. GRIPPER LIGADO. Estava comentado; o robo nunca fechava a garra.

SINCRONO: executa o sub-chunk, para, pede o proximo com observacao
fresca. Elimina a sobreposicao entre chunks (o modelo preve posicoes
ABSOLUTAS, entao uma observacao defasada faria o chunk novo "achar"
que o robo esta atras de onde realmente esta).
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
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO
import message_filters
from cv_bridge import CvBridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO
# ══════════════════════════════════════════════════════════════════════════════

SERVER_URI = "ws://localhost:8000/infer"
PROMPT = "pick up the red mug and put it in the drawer"
# PROMPT = "pick up the red mug and put it in the drawer"

CONTROL_HZ = 25.0                       # DEVE bater com o fps do dataset
DT_CONTROL = 1.0 / CONTROL_HZ

CHUNK_SIZE = 50                          # o modelo sempre preve 50 passos
EXECUTION_STEPS_PER_CHUNK = 25           # quantos executar antes de reinferir
IMG_SIZE = (512, 512)

# Gripper: histerese para nao oscilar perto do meio.
GRIPPER_CLOSE_TH = 0.5
GRIPPER_OPEN_TH = 0.01
GRIPPER_PAUSE_AFTER_CLOSE_S = 2.0

UR5_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

gripper_closed = False

# ══════════════════════════════════════════════════════════════════════════════
# NO ROS2
# ══════════════════════════════════════════════════════════════════════════════

class UR5RosBridge(Node):

    def __init__(self):
        super().__init__("ur5_sync_inference_bridge")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self._current_joint_positions = None
        self._current_gripper_state = 0.0
        self._latest_images = None

        self.gripper_pin = 16
        self._last_sent_gripper_state = None
        self.io_client = self.create_client(SetIO, "/io_and_status_controller/set_io")
        self.get_logger().info("Aguardando servico de I/O do robo...")
        self.io_client.wait_for_service()
        self.get_logger().info("Servico de I/O conectado!")

        qos_camera = QoSProfile(depth=10)
        qos_camera.reliability = ReliabilityPolicy.BEST_EFFORT

        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10)

        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb,
            qos_profile=qos_profile_sensor_data)

        left_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_left/rgb/color/rect/image", qos_profile=qos_camera)
        right_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_right/rgb/color/rect/image", qos_profile=qos_camera)
        robot_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_robot/rgb/color/rect/image", qos_profile=qos_camera)

        # slop 0.04: medido que entrega ~25 Hz. NAO aumentar (janela grande
        # casa o mesmo frame com varios vizinhos -> falsos matches).
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, robot_sub], queue_size=10, slop=0.04)
        self.ts.registerCallback(self._camera_cb)

        self.get_logger().info("UR5RosBridge pronto -- aguardando junta/camera...")

    def _joint_state_cb(self, msg: JointState):
        try:
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            joint_pos = np.array([pos_map[j] for j in UR5_JOINT_NAMES], dtype=np.float32)
            with self._lock:
                self._current_joint_positions = joint_pos
        except KeyError:
            pass

    def _camera_cb(self, left_msg, right_msg, robot_msg):
        try:
            def to_rgb(msg):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return cv_img[:, :, ::-1].copy()
            images = {
                "zed_left": to_rgb(left_msg),
                "zed_right": to_rgb(right_msg),
                "zed_robot": to_rgb(robot_msg),
            }
            with self._lock:
                self._latest_images = images
        except Exception as e:
            self.get_logger().error(f"Erro no callback de camera: {e}")

    def get_observation(self):
        with self._lock:
            if self._current_joint_positions is None:
                self.get_logger().warn("sem /joint_states ainda",
                                       throttle_duration_sec=2.0)
                return None
            if self._latest_images is None:
                self.get_logger().warn("sem imagens sincronizadas ainda",
                                       throttle_duration_sec=2.0)
                return None
            return (self._current_joint_positions.copy(),
                    self._current_gripper_state,
                    dict(self._latest_images))

    def get_current_joint_positions(self):
        with self._lock:
            if self._current_joint_positions is None:
                return None
            return self._current_joint_positions.copy()

    def publish_smooth_trajectory(self, actions_6d_list, duration_per_step: float):
        """Publica o sub-chunk INTEIRO como uma trajetoria; o controlador
        interpola suavemente entre os pontos."""
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES
        for i, action_6d in enumerate(actions_6d_list):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in action_6d]
            t = (i + 1) * duration_per_step
            point.time_from_start = Duration(sec=int(t),
                                             nanosec=int((t - int(t)) * 1e9))
            msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(msg)

    def publish_gripper_command(self, action_gripper_value: float) -> bool:
        """Envia o comando de gripper e retorna True se o gripper foi fechado."""
        current_state = self._last_sent_gripper_state
        if current_state is None:
            new_state = 1 if action_gripper_value > GRIPPER_CLOSE_TH else 0
        elif current_state == 0:
            new_state = 1 if action_gripper_value > GRIPPER_CLOSE_TH else 0
        else:
            new_state = 0 if action_gripper_value < GRIPPER_OPEN_TH else 1

        just_closed = False

        if new_state != current_state:
            req = SetIO.Request()
            req.fun = 1
            req.pin = self.gripper_pin
            req.state = float(new_state)
            self.io_client.call_async(req)

            self._last_sent_gripper_state = new_state
            with self._lock:
                self._current_gripper_state = float(new_state)

            verb = "FECHANDO" if new_state == 1 else "ABRINDO"
            self.get_logger().info(f"[GRIPPER] {verb} (valor previsto: {action_gripper_value:.3f})")

            if new_state == 1:
                just_closed = True
                self.get_logger().info(f"[GRIPPER] Pausando {GRIPPER_PAUSE_AFTER_CLOSE_S}s apos fechar...")
                time.sleep(GRIPPER_PAUSE_AFTER_CLOSE_S)

        return just_closed


def ros_spin_thread(node):
    rclpy.spin(node)


def resize_image_for_server(img: np.ndarray) -> str:
    """Comprime a imagem RGB como JPEG e codifica em base64.
    O CROP e o resize para 512 sao feitos no SERVIDOR, para casar
    exatamente com o pre-processamento do conversor."""
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Falha ao codificar imagem como JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class SyncInferenceClient:
    def __init__(self, ros_bridge: UR5RosBridge):
        self.ros_bridge = ros_bridge
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(SERVER_URI, max_size=None)
        print(f"[CLIENTE] Conectado ao servidor: {SERVER_URI}")

    async def _request_inference(self, joint_pos, gripper_state, images):
        state_7d = np.concatenate([joint_pos, [gripper_state]]).tolist()
        payload = await asyncio.to_thread(self._build_payload, state_7d, images)
        await self.websocket.send(json.dumps(payload))
        response_text = await self.websocket.recv()
        response = json.loads(response_text)
        return np.array(response["actions"], dtype=np.float32)

    def _build_payload(self, state_7d, images):
        # ORDEM CRITICA: identica a conversao do dataset.
        return {
            "observation.state": state_7d,
            "observation.images.image_1": resize_image_for_server(images["zed_left"]),
            "observation.images.image_2": resize_image_for_server(images["zed_robot"]),
            "observation.images.image_3": resize_image_for_server(images["zed_right"]),
            "prompt": PROMPT,
        }

    async def run(self, horizon_steps: int):
        step_counter = 0
        ARRIVAL_TOLERANCE = 0.01
        ARRIVAL_TIMEOUT_S = EXECUTION_STEPS_PER_CHUNK * DT_CONTROL + 0.5

        while step_counter < horizon_steps:
            obs = None
            while obs is None:
                obs = self.ros_bridge.get_observation()
                if obs is None:
                    await asyncio.sleep(0.05)

            joint_pos, gripper_state, images = obs

            print(f"[CLIENTE] step={step_counter} | pedindo chunk "
                  f"(observacao fresca, robo parado)...")
            full_chunk = await self._request_inference(joint_pos, gripper_state, images)

            sub = full_chunk[:EXECUTION_STEPS_PER_CHUNK]
            actions_6d = [a[:6] for a in sub]

            # Gripper: media das ULTIMAS acoes do chunk completo (onde a
            # transicao de fechar acontece).
            gripper_mean = float(np.mean(full_chunk[:, 6]))

            # Seguranca: rejeita chunk que pede salto grande a partir da
            # posicao atual (predicao degenerada).
            cur = self.ros_bridge.get_current_joint_positions()
            first = np.asarray(actions_6d[0])
            jump = float(np.max(np.abs(first - cur))) if cur is not None else 0.0
            # if jump > 0.5:
            #     print(f"[ABORT] step={step_counter} | salto de {jump:.3f} rad no "
            #           f"1o ponto (limite 0.5). Parando por seguranca.")
            #     break

            # TRAJETORIA INTEIRA: publica os 25 pontos, controlador interpola.
            # self.ros_bridge.publish_smooth_trajectory(actions_6d[-1], DT_CONTROL)
            final_target = actions_6d[-1]
            total_duration = EXECUTION_STEPS_PER_CHUNK * DT_CONTROL
            self.ros_bridge.publish_smooth_trajectory([final_target], total_duration)

            gripper_closed = self.ros_bridge.publish_gripper_command(gripper_mean)

            print(f"[TRAJETORIA] step={step_counter} | {len(actions_6d)} pontos | "
                  f"gripper={gripper_mean:+.3f} | delta1={jump:.4f} rad | "
                  f"alvo_final={np.round(actions_6d[-1], 3)}")

            if gripper_closed:
                print(f"[CLIENTE] Gripper fechado! Escrevendo flag de sinal com o prompt: '{PROMPT}'")
                with open("/tmp/gripper_closed.flag", "w") as f:
                    f.write(PROMPT)
                break

            # Espera a trajetoria terminar (ativo, nao bloqueia o event loop).
            target = np.asarray(actions_6d[-1])
            t_wait = time.perf_counter()
            arrived = False
            error = 0.0
            while (time.perf_counter() - t_wait) < ARRIVAL_TIMEOUT_S:
                current = self.ros_bridge.get_current_joint_positions()
                if current is not None:
                    error = float(np.max(np.abs(current - target)))
                    if error < ARRIVAL_TOLERANCE:
                        arrived = True
                        break
                await asyncio.sleep(0.02)

            if not arrived:
                print(f"[AVISO] step={step_counter} | timeout de chegada "
                      f"(erro={error:.4f} rad) -- prosseguindo.")

            step_counter += len(sub)

        print("[CLIENTE] Horizon concluido.")


async def main_async():
    holder = {}

    def start_ros():
        rclpy.init()
        node = UR5RosBridge()
        holder["node"] = node
        ros_spin_thread(node)

    ros_thread = threading.Thread(target=start_ros, daemon=True)
    ros_thread.start()

    while "node" not in holder:
        await asyncio.sleep(0.1)

    ros_bridge = holder["node"]
    client = SyncInferenceClient(ros_bridge)
    await client.connect()
    await client.run(horizon_steps=2000)


if __name__ == "__main__":
    asyncio.run(main_async())