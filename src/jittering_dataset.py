#!/usr/bin/env python3
"""
UR5 Minimal Joint Position & IO Recorder + First-Frame Image Capture
ROS2 Humble | ur_robot_driver | evdev | cv_bridge

MUDANÇA: adicionado a inscrição na câmera zed_right no tópico RGB.
Ao iniciar a gravação (Clique 1), ele captura APENAS o primeiro frame da zed_right
e salva uma imagem .png com o nome equivalente ao dataset.json 
(ex: dataset_1.png para dataset_1.json) no diretório de saída.
"""

import os
import sys
import threading
import json
import cv2
import rclpy
from rclpy.node import Node
from ur_msgs.msg import IOStates
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from evdev import InputDevice, list_devices, ecodes

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class UR5MinimalRecorderNode(Node):
    def __init__(self):
        super().__init__('ur5_minimal_recorder_node')

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()

        self.gripper_pin = 16
        self.current_gripper_state = 0

        self.is_running = True
        self.is_recording = False

        self.current_joint_positions = [0.0] * 6
        self.last_joint_timestamp = 0.0

        self.history_database = []

        self.output_dir = '/home/ziqi/pre_ws/jittering_dataset'
        os.makedirs(self.output_dir, exist_ok=True)

        self.io_subscription = self.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_states_callback,
            10
        )

        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        # Inscrição na câmera zed_right (tópico RGB)
        self.image_subscription = self.create_subscription(
            Image,
            '/zed_multi/zed_left/rgb/color/rect/image',  # Ajuste o tópico se na sua rede o namespace for diferente
            self.zed_right_callback,
            10
        )

        self.get_logger().info("Minimal Joint Position, IO & ZED Right Camera monitoring ACTIVE.")

        print("\n" + "*"*60)
        print(" --> CLIQUE 1: INICIA GRAVAÇÃO | CLIQUE 2: PAUSA, SALVA E ENTRA EM IDLE <-- ")
        print("*"*60 + "\n")
        self.mouse_thread = threading.Thread(target=self.mouse_listener_thread, daemon=True)
        self.mouse_thread.start()

    def zed_right_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.image_lock:
                self.latest_image = cv_img
        except Exception as e:
            self.get_logger().error(f"Erro ao converter imagem da zed_right: {e}")

    def find_logitech_mouse(self):
        for path in list_devices():
            try:
                dev = InputDevice(path)
                if "logitech" not in dev.name.lower(): continue
                caps = dev.capabilities()
                if ecodes.EV_KEY not in caps: continue
                keys = caps[ecodes.EV_KEY]
                if ecodes.BTN_SIDE in keys or ecodes.BTN_EXTRA in keys:
                    self.get_logger().info(f"Mouse encontrado: {dev.name} ({path})")
                    return path
            except Exception: continue
        return None

    def determine_next_dataset_index(self):
        index = 1
        while True:
            file_name = f"dataset_{index}.json"
            full_path = os.path.join(self.output_dir, file_name)
            if not os.path.exists(full_path):
                return index
            index += 1

    def save_first_frame(self, index):
        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().warn("⚠️ Nenhuma imagem da zed_right capturada até o momento do clique!")
                return False
            img_to_save = self.latest_image.copy()

        img_filename = f"dataset_{index}.png"
        img_path = os.path.join(self.output_dir, img_filename)

        try:
            cv2.imwrite(img_path, img_to_save)
            print(f"📸 Primeiro frame da zed_right salvo em: {img_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Falha ao salvar imagem da câmera: {e}")
            return False

    def save_and_reset_dataset(self, next_index):
        if len(self.history_database) == 0:
            print("\n[AVISO] Gravação interrompida, mas nenhum dado foi capturado. Retornando ao estado IDLE...")
            return

        target_path = os.path.join(self.output_dir, f"dataset_{next_index}.json")
        total_frames = len(self.history_database)

        print(f"\n\n[SALVANDO] Exportando {total_frames} frames para: {target_path}...")

        try:
            data_to_save = list(self.history_database)
            self.history_database = []

            with open(target_path, 'w') as file:
                json.dump(data_to_save, file, indent=4)

            print(f"✅ Concluído! Arquivo '{os.path.basename(target_path)}' gravado com sucesso.")
            print("\n" + "="*60)
            print(" 🔄 NÓ EM IDLE: Pronto para a próxima demonstração! (Clique lateral para iniciar) ")
            print("="*60 + "\n")
        except Exception as e:
            self.get_logger().error(f"Falha crítica de I/O ao salvar dataset: {e}")

    def mouse_listener_thread(self):
        mouse_path = self.find_logitech_mouse()
        if mouse_path is None:
            self.get_logger().error("Nenhum mouse Logitech com botões laterais encontrado.")
            return
        
        current_index = 1
        try:
            device = InputDevice(mouse_path)
            self.get_logger().info(f"Monitorando mouse: {device.name}")

            for event in device.read_loop():
                if not self.is_running: break
                if event.type != ecodes.EV_KEY: continue
                if event.value != 1: continue

                if event.code == ecodes.BTN_EXTRA:
                    self.is_recording = not self.is_recording

                    if self.is_recording:
                        current_index = self.determine_next_dataset_index()
                        print(f"\n\n>>> 🔴 [RECORDING STARTED - DATASET {current_index}] Capturando juntas e imagem inicial... <<<")
                        self.save_first_frame(current_index)
                    else:
                        print("\n\n||| ⏸️ [RECORDING PAUSED] Processando encerramento...")
                        self.save_and_reset_dataset(current_index)

                elif event.code == ecodes.BTN_SIDE:
                    if self.is_recording:
                        print("\n\n||| ⏹️ [RECORDING STOPPED] Gravação interrompida pelo botão lateral. Limpando memória...")
                        self.is_recording = False
                        self.history_database = []
                        print("\n" + "="*60)
                        print(" 🔄 NÓ EM IDLE: Pronto para a próxima demonstração! (Clique lateral para iniciar) ")
                        print("="*60 + "\n")

        except Exception as e:
            self.get_logger().error(f"Falha ao monitorar mouse: {e}")

    def io_states_callback(self, msg):
        if not self.is_running: return
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = 1 if pin_state.state else 0
                    break
        except Exception: pass

    def joint_states_callback(self, msg):
        if not self.is_running: return
        try:
            self.last_joint_timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)

            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self.current_joint_positions = [float(pos_map[joint]) for joint in UR5_JOINT_NAMES]
            self.current_joint_positions = [round(pos, 5) for pos in self.current_joint_positions]
        except KeyError:
            return

        frame_packet = {
            "timestamp": self.last_joint_timestamp,
            "joint_positions": list(self.current_joint_positions),
            "gripper_io": int(self.current_gripper_state)
        }

        if self.is_recording:
            self.history_database.append(frame_packet)

        status_prefix = "[GRAVANDO]" if self.is_recording else "[IDLE - ESPERA]"
        j_print = ", ".join([f"{j:+.4f}" for j in self.current_joint_positions])

        sys.stdout.write(
            f"\r{status_prefix} Juntas (rad) -> [{j_print}] | Frames Acumulados: {len(self.history_database)}"
        )
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = UR5MinimalRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SIGNAL] Ctrl+C detectado no terminal. Encerrando nó minimalista de forma limpa...")
    finally:
        node.is_running = False
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()