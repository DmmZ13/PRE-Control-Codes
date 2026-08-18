#!/usr/bin/env python3
"""
UR5 Multi-Modal Asynchronous High-Frequency Recorder (3-Camera ZED Version - EVENT-DRIVEN MP4)
ROS2 Humble | ur_robot_driver | 3x ZED Cameras

Grava imagens de 3 câmeras RGB sincronizadas por evento (sem timer fake),
resolvendo o problema de vídeo travado / frame drop.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from ur_msgs.msg import IOStates
import message_filters
import sys
import threading
import json
import os
import time
import numpy as np
from cv_bridge import CvBridge
import cv2

from evdev import InputDevice, list_devices, ecodes

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR5HighFreqRecorderNode(Node):
    def __init__(self):
        super().__init__('ur5_high_freq_recorder_node')

        self.gripper_pin = 16
        self.current_gripper_state = 0

        qos_profile_camera = QoSProfile(depth=10)
        qos_profile_camera.reliability = ReliabilityPolicy.BEST_EFFORT

        self.is_running = True
        self.is_recording = False
        self.is_saving = False

        self.last_pose_msg = None
        self.current_cartesian_pose = [0.0] * 7
        self.current_cartesian_velocity = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6

        self.camera_database = []
        self.robot_trajectory_database = []

        self.base_dataset_dir = '/home/ziqi/pre_ws/dataset'
        os.makedirs(self.base_dataset_dir, exist_ok=True)

        self.setup_new_dataset_folder()
        self.bridge = CvBridge()

        # Inscrições das câmeras ZED
        self.left_rgb_sub = message_filters.Subscriber(self, Image, '/zed_multi/zed_left/rgb/color/rect/image', qos_profile=qos_profile_camera)
        self.right_rgb_sub = message_filters.Subscriber(self, Image, '/zed_multi/zed_right/rgb/color/rect/image', qos_profile=qos_profile_camera)
        self.robot_rgb_sub = message_filters.Subscriber(self, Image, '/zed_multi/zed_robot/rgb/color/rect/image', qos_profile=qos_profile_camera)

        # Tolerância de sincronização ajustada para 40ms (permite flutuações sem perder frames)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_rgb_sub, self.right_rgb_sub, self.robot_rgb_sub],
            queue_size=100,
            slop=0.05
        )
        self.ts.registerCallback(self.camera_perception_callback)

        self.pose_subscription = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.high_frequency_pose_callback, 10)
        self.joint_subscription = self.create_subscription(JointState, '/joint_states', self.high_frequency_joints_callback, 10)
        self.io_subscription = self.create_subscription(IOStates, '/io_and_status_controller/io_states', self.io_states_callback, 10)

        # Variáveis para cálculo do FPS REAL de gravação
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0.0

        print("\n" + "*"*60)
        print(" --> CLIQUE LATERAL 1: INICIA GRAVAÇÃO | CLIQUE LATERAL 2: SALVA MP4 <-- ")
        print("*"*60 + "\n")
        self.mouse_thread = threading.Thread(target=self.mouse_listener_thread, daemon=True)
        self.mouse_thread.start()

    def determine_next_dataset_folder(self):
        index = 1
        while True:
            folder_name = f"synchronized_dataset_{index}"
            full_path = os.path.join(self.base_dataset_dir, folder_name)
            if not os.path.exists(full_path):
                return full_path
            index += 1

    def setup_new_dataset_folder(self):
        self.output_dir = self.determine_next_dataset_folder()
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"🚀 Próximo dataset será gravado em: {self.output_dir}")

        self.video_paths = {
            'left': os.path.join(self.output_dir, 'zed_left.mp4'),
            'right': os.path.join(self.output_dir, 'zed_right.mp4'),
            'robot': os.path.join(self.output_dir, 'zed_robot.mp4')
        }
        self.manifest_path = os.path.join(self.output_dir, 'synchronized_metadata.json')

    def find_logitech_mouse(self):
        for path in list_devices():
            try:
                dev = InputDevice(path)
                if "logitech" not in dev.name.lower():
                    continue
                caps = dev.capabilities()
                if ecodes.EV_KEY not in caps:
                    continue
                keys = caps[ecodes.EV_KEY]
                if ecodes.BTN_SIDE in keys or ecodes.BTN_EXTRA in keys:
                    self.get_logger().info(f"Mouse encontrado: {dev.name} ({path})")
                    return path
            except Exception:
                continue
        return None

    def mouse_listener_thread(self):
        mouse_path = self.find_logitech_mouse()
        if mouse_path is None:
            self.get_logger().error("Nenhum mouse Logitech com botões laterais encontrado.")
            return
        try:
            device = InputDevice(mouse_path)
            self.get_logger().info(f"Monitorando mouse: {device.name}")
            for event in device.read_loop():
                if not self.is_running:
                    break
                if event.type != ecodes.EV_KEY or event.value != 1:
                    continue
                if event.code == ecodes.BTN_EXTRA:
                    if self.is_saving:
                        print("\n⚠️ Aguarde o salvamento do dataset anterior terminar!\n")
                        continue

                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                        print("\n\n>>> [MOUSE TRIGGER] GRAVAÇÃO INICIADA! <<<\n")
                    else:
                        print("\n\n||| [MOUSE TRIGGER] Salvando vídeos MP4 em background... |||\n")
                        save_thread = threading.Thread(target=self.async_save_and_reset_dataset, daemon=True)
                        save_thread.start()

                elif event.code == ecodes.BTN_SIDE:
                    if self.is_recording or len(self.camera_database) > 0:
                        print("\n\n||| [MOUSE TRIGGER] Gravação CANCELADA! Descartando dados... |||\n")
                        self.cancel_recording()

        except Exception as e:
            self.get_logger().error(f"Falha ao monitorar mouse: {e}")

    def high_frequency_joints_callback(self, msg: JointState):
        if not self.is_running:
            return
        try:
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            vel_map = {name: vel for name, vel in zip(msg.name, msg.velocity)}

            self.current_joint_positions = [float(pos_map[joint]) for joint in UR5_JOINT_NAMES]
            self.current_joint_velocities = [float(vel_map[joint]) for joint in UR5_JOINT_NAMES]

            if self.is_recording:
                timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
                self.robot_trajectory_database.append({
                    "timestamp": timestamp,
                    "joint_positions": list(self.current_joint_positions),
                    "joint_velocities": list(self.current_joint_velocities),
                    "cartesian_pose": list(self.current_cartesian_pose),
                    "cartesian_velocity": list(self.current_cartesian_velocity),
                    "gripper_io": int(self.current_gripper_state)
                })
        except KeyError:
            pass

    def high_frequency_pose_callback(self, msg: PoseStamped):
        if not self.is_running:
            return
        self.current_cartesian_pose = [
            float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z),
            float(msg.pose.orientation.x), float(msg.pose.orientation.y), float(msg.pose.orientation.z), float(msg.pose.orientation.w)
        ]
        if self.last_pose_msg is not None:
            t_current = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            t_last = self.last_pose_msg.header.stamp.sec + (self.last_pose_msg.header.stamp.nanosec * 1e-9)
            dt = t_current - t_last
            if dt > 0.001:
                vx = (msg.pose.position.x - self.last_pose_msg.pose.position.x) / dt
                vy = (msg.pose.position.y - self.last_pose_msg.pose.position.y) / dt
                vz = (msg.pose.position.z - self.last_pose_msg.pose.position.z) / dt
                q1 = np.array([self.last_pose_msg.pose.orientation.w, self.last_pose_msg.pose.orientation.x, self.last_pose_msg.pose.orientation.y, self.last_pose_msg.pose.orientation.z])
                q2 = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
                if np.dot(q1, q2) < 0.0:
                    q2 = -q2
                q_diff = (q2 - q1) / dt
                q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
                w_vector = 2.0 * np.array([
                    q_diff[0]*q1_conj[1] + q_diff[1]*q1_conj[0] + q_diff[2]*q1_conj[3] - q_diff[3]*q1_conj[2],
                    q_diff[0]*q1_conj[2] - q_diff[1]*q1_conj[3] + q_diff[2]*q1_conj[0] + q_diff[3]*q1_conj[1],
                    q_diff[0]*q1_conj[3] + q_diff[1]*q1_conj[2] - q_diff[2]*q1_conj[1] + q_diff[3]*q1_conj[0]
                ])
                self.current_cartesian_velocity = [float(vx), float(vy), float(vz), float(w_vector[0]), float(w_vector[1]), float(w_vector[2])]
        self.last_pose_msg = msg

    # ── CALLBACK DIRETO DE CÂMERA (EVENT-DRIVEN) ─────────────────────────
    def camera_perception_callback(self, l_rgb, r_rgb, rob_rgb):
        if not self.is_running or not self.is_recording:
            return

        try:
            cv_l = self.bridge.imgmsg_to_cv2(l_rgb, desired_encoding='bgr8')
            cv_r = self.bridge.imgmsg_to_cv2(r_rgb, desired_encoding='bgr8')
            cv_rob = self.bridge.imgmsg_to_cv2(rob_rgb, desired_encoding='bgr8')
        except Exception:
            return

        timestamp = l_rgb.header.stamp.sec + (l_rgb.header.stamp.nanosec * 1e-9)

        self.camera_database.append({
            'frame_id': len(self.camera_database),
            'timestamp': timestamp,
            'left_img': cv_l,
            'right_img': cv_r,
            'robot_img': cv_rob
        })

        # Calcula o FPS REAL em tempo real
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 0.5:
            self.actual_fps = self.fps_counter / elapsed

        sys.stdout.write(
            f"\r[GRAVANDO 3 CAMS MP4] FPS Real: {self.actual_fps:.1f} Hz | {len(self.camera_database)} frames | Robô: {len(self.robot_trajectory_database)} pts"
        )
        sys.stdout.flush()

    def io_states_callback(self, msg):
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = 1 if pin_state.state else 0
                    break
        except Exception:
            pass

    def cancel_recording(self):
        self.is_recording = False
        self.camera_database = []
        self.robot_trajectory_database = []
        self.last_pose_msg = None
        print(f"🗑️ Dados descartados. Pronto para gravar novamente em: {self.output_dir}/")

    # ── RENDERIZAÇÃO INTELIGENTE DE VÍDEO CONFORME O FPS REAL ────────────
    def async_save_and_reset_dataset(self):
        self.is_saving = True
        self.is_recording = False

        cam_db = list(self.camera_database)
        robot_db = list(self.robot_trajectory_database)
        output_dir = self.output_dir
        video_paths = dict(self.video_paths)
        manifest_path = self.manifest_path

        self.camera_database = []
        self.robot_trajectory_database = []
        self.last_pose_msg = None

        self.setup_new_dataset_folder()

        if len(cam_db) > 1:
            # Calcula a taxa de FPS exata baseada no tempo dos timestamps gravados
            total_duration = cam_db[-1]['timestamp'] - cam_db[0]['timestamp']
            calculated_fps = (len(cam_db) - 1) / total_duration if total_duration > 0 else 25.0
            
            # Limita a uma faixa aceitável
            output_fps = max(1.0, min(calculated_fps, 60.0))

            print(f"\n[BACKGROUND] Renderizando {len(cam_db)} frames a {output_fps:.2f} FPS REAL...")

            h, w = cam_db[0]['left_img'].shape[:2]

            # Sincronização Temporal Perfeita
            total_duration = cam_db[-1]['timestamp'] - cam_db[0]['timestamp']
            exact_fps = (len(cam_db) - 1) / total_duration if total_duration > 0 else 25.0

            # O OpenCV salva o arquivo MP4 na velocidade REAL em que os dados foram capturados
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer_l = cv2.VideoWriter(video_paths['left'], fourcc, exact_fps, (w, h))
            writer_r = cv2.VideoWriter(video_paths['right'], fourcc, exact_fps, (w, h))
            writer_rob = cv2.VideoWriter(video_paths['robot'], fourcc, exact_fps, (w, h))

            for item in cam_db:
                writer_l.write(item['left_img'])
                writer_r.write(item['right_img'])
                writer_rob.write(item['robot_img'])

            writer_l.release()
            writer_r.release()
            writer_rob.release()

            manifesto_final = {
                "recorded_fps": float(output_fps),
                "total_frames": len(cam_db),
                "video_files": {
                    "left_rgb": "zed_left.mp4",
                    "right_rgb": "zed_right.mp4",
                    "robot_rgb": "zed_robot.mp4"
                },
                "camera_samples": [
                    {"frame_id": c['frame_id'], "timestamp": c['timestamp']} for c in cam_db
                ],
                "robot_trajectory": robot_db
            }

            with open(manifest_path, 'w') as json_file:
                json.dump(manifesto_final, json_file, indent=4)

            print(f"\n✅ [BACKGROUND] Dataset salvo perfeitamente a {output_fps:.1f} FPS em: {output_dir}/")
        else:
            print("\nNenhum dado capturado neste dataset -- nada foi salvo.")

        self.is_saving = False
        print("\n" + "*"*60)
        print(" Pronto! Pode clicar para iniciar a PRÓXIMA gravação. ")
        print("*"*60 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = UR5HighFreqRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Ctrl+C detectado -- encerrando...")
        node.is_running = False
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()