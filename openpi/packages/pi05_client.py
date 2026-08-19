#!/usr/bin/env python3
"""
Cliente de inferencia UR5 <-> servidor openpi (pi0.5) -- 2 cameras.

Correcoes em relacao ao cliente anterior:
  1. CROP por camera antes do resize (identico ao conversor do dataset).
  2. STRETCH para quadrado, nao 224 direto: a imagem chega quadrada e o
     resize_with_pad do servidor vira quase no-op (sem barras pretas).
  3. PROMPT adicionado a observacao -- o modelo treinou com
     prompt_from_task=True e 2 tasks, entao o prompt IMPORTA.

CHAVES DA OBSERVACAO
  base_rgb / wrist_rgb / state / prompt -- os nomes que UR5Inputs le.
  NAO use observation.images.* : so existem no repack do data loader.

TAXA DE CONTROLE
  action_horizon=50 passos na TAXA DO DATASET (~25 Hz). Execute a 25 Hz.
"""

import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO
import message_filters
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy

# ── Configuracao ────────────────────────────────────────────────────────
SERVER_HOST = "localhost"
SERVER_PORT = 8000

PROMPT = "pick up the red mug and put it in the microwave"
# troque para "...and put it in the drawer" para testar o outro destino

CONTROL_HZ = 25.0
IMG_SIZE = 224                # ja quadrado -> servidor nao pada
ACTIONS_PER_CHUNK = 25
GRIPPER_CLOSE_TH = 0.5
GRIPPER_OPEN_TH = 0.3
MAX_JOINT_STEP = 0.15         # rad; seguranca contra saltos absurdos

# CROP por camera -- IDENTICO ao conversor. Formato [x, y, w, h].
CROPS = {
    "zed_left":  [120, 0, 660, 660],   # image_1 = zed_left
    "zed_robot": [350, 0, 720, 720],   # image_2 = zed_robot
    "zed_right": [430, 100, 600, 400],   # image_3 = zed_right
}

UR5_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class UR5Bridge(Node):

    def __init__(self):
        super().__init__("ur5_pi0_client")
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._joints = None
        self._images = None
        self._gripper = 0.0
        self._last_gripper_sent = None

        self.gripper_pin = 16
        self.io_client = self.create_client(SetIO, "/io_and_status_controller/set_io")
        self.get_logger().info("Aguardando servico de I/O...")
        self.io_client.wait_for_service()
        self.get_logger().info("I/O conectado.")

        self.traj_pub = self.create_publisher(
            JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10)

        self.create_subscription(JointState, "/joint_states",
                                 self._joint_cb, qos_profile_sensor_data)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        zed_left = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_left/rgb/color/rect/image", qos_profile=qos)
        zed_robot = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_robot/rgb/color/rect/image", qos_profile=qos)
        zed_right = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_right/rgb/color/rect/image", qos_profile=qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [zed_left, zed_robot, zed_right], queue_size=30, slop=0.04)
        self.ts.registerCallback(self._camera_cb)

        self.get_logger().info("Bridge pronto.")

    def _joint_cb(self, msg):
        try:
            m = dict(zip(msg.name, msg.position))
            q = np.array([m[j] for j in UR5_JOINT_NAMES], dtype=np.float32)
        except KeyError:
            return
        with self._lock:
            self._joints = q

    def _camera_cb(self, zed_left_msg, zed_robot_msg, zed_right_msg):
        try:
            def prep(msg, crop):
                bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                x, y, w, h = crop
                rgb = rgb[y:y + h, x:x + w]                 # CROP igual ao treino
                return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), # STRETCH p/ quadrado
                                  interpolation=cv2.INTER_AREA)
            imgs = {
                "zed_left_rgb": prep(zed_left_msg, CROPS["zed_left"]),
                "zed_robot_rgb": prep(zed_robot_msg, CROPS["zed_robot"]),
                "zed_right_rgb": prep(zed_right_msg, CROPS["zed_right"]),
            }
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return
        with self._lock:
            self._images = imgs

    def get_observation(self):
        with self._lock:
            if self._joints is None:
                self.get_logger().warn("sem /joint_states ainda",
                                       throttle_duration_sec=2.0)
                return None
            if self._images is None:
                self.get_logger().warn("sem imagens sincronizadas ainda",
                                       throttle_duration_sec=2.0)
                return None
            state = np.concatenate([self._joints, [self._gripper]]).astype(np.float32)
            return {
                "zed_left_rgb": self._images["zed_left_rgb"],
                "zed_robot_rgb": self._images["zed_robot_rgb"],
                "zed_right_rgb": self._images["zed_right_rgb"],
                "state": state,
                "prompt": PROMPT,
            }

    def get_joints(self):
        with self._lock:
            return None if self._joints is None else self._joints.copy()

    def send_trajectory(self, actions_6d, dt):
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES
        for i, a in enumerate(actions_6d):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in a]
            t = (i + 1) * dt
            pt.time_from_start = Duration(sec=int(t),
                                          nanosec=int((t - int(t)) * 1e9))
            msg.points.append(pt)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(msg)

    def set_gripper(self, value):
        cur = self._last_gripper_sent
        if cur is None:
            new = 1 if value > GRIPPER_CLOSE_TH else 0
        elif cur == 0:
            new = 1 if value > GRIPPER_CLOSE_TH else 0
        else:
            new = 0 if value < GRIPPER_OPEN_TH else 1
        if new == cur:
            return
        req = SetIO.Request()
        req.fun, req.pin, req.state = 1, self.gripper_pin, float(new)
        self.io_client.call_async(req)
        self._last_gripper_sent = new
        with self._lock:
            self._gripper = float(new)
        self.get_logger().info(
            f"[GARRA] {'FECHANDO' if new else 'ABRINDO'} (previsto {value:.3f})")


def main():
    rclpy.init()
    node = UR5Bridge()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(f"Conectando ao servidor {SERVER_HOST}:{SERVER_PORT} ...")
    client = websocket_client_policy.WebsocketClientPolicy(
        host=SERVER_HOST, port=SERVER_PORT)
    print(f"Conectado. Prompt: {PROMPT!r}\n")

    dt = 1.0 / CONTROL_HZ
    step = 0

    try:
        while rclpy.ok():
            obs = None
            while obs is None:
                obs = node.get_observation()
                if obs is None:
                    time.sleep(0.05)

            t0 = time.perf_counter()
            result = client.infer(obs)
            infer_ms = (time.perf_counter() - t0) * 1000
            chunk = np.asarray(result["actions"], dtype=np.float32)

            sub = chunk[:ACTIONS_PER_CHUNK]
            actions_6d = sub[:, :6]

            cur = node.get_joints()
            jump = float(np.max(np.abs(actions_6d[0] - cur))) if cur is not None else 0.0
            if jump > MAX_JOINT_STEP:
                print(f"[ABORT] passo {step}: salto de {jump:.3f} rad no 1o ponto "
                      f"(limite {MAX_JOINT_STEP}). Nao executando.")
                break

            grip = float(np.mean(chunk[:, 6]))
            print(f"[{step:4d}] infer {infer_ms:6.1f} ms | garra {grip:+.3f} | "
                  f"delta1 {jump:.4f} rad | alvo final {np.round(actions_6d[-1], 3)}")

            node.set_gripper(grip)
            node.send_trajectory(actions_6d, dt)

            time.sleep(len(actions_6d) * dt + 0.05)
            step += 1

    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()