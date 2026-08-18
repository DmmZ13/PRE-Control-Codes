#!/usr/bin/env python3
"""
UR5 Dataset Player v12 -- Mouse + Terminal Interativo
ROS2 Humble | ur_robot_driver | Forward Position Controller | evdev

Comportamento dos botoes do mouse (inalterado da v11):
  - CLIQUE DIREITO (BTN_RIGHT): toca o PROXIMO dataset (ou repete o
    atual, se foi cancelado via BTN_LEFT antes).
  - BOTAO LATERAL (BTN_LEFT): cancela o streaming atual, mantendo o
    mesmo dataset para o proximo clique direito.

NOVO -- Terminal interativo (thread separada, roda em paralelo):
  Digite um NUMERO + Enter no terminal para tocar aquele dataset
  especifico diretamente (ex: digitar "42" toca dataset_42.json).
  Depois disso, o CLIQUE DIREITO continua a sequencia normalmente
  A PARTIR desse numero escolhido.

  Comandos adicionais no terminal:
    list        - lista os indices de dataset disponiveis na pasta
    <numero>    - toca o dataset daquele numero especifico

O no permanece vivo entre os cliques/comandos (nao desliga ao
terminar/cancelar um dataset).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ur_msgs.srv import SetIO
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import json
import os
import re
import sys
import glob
import threading

from evdev import InputDevice, list_devices, ecodes

# ══════════════════════════════════════════════════════════════════════════════
# Configuracoes Padrao
# ══════════════════════════════════════════════════════════════════════════════

DATASET_DIR = '/home/ziqi/pre_ws/jittering_dataset'
DATASET_PREFIX = 'dataset_'
DATASET_START_INDEX = 1
DEFAULT_CONTROL_FREQ = 100.0  # Hz

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR5MouseTriggeredPlayer(Node):

    def __init__(self):
        super().__init__("ur5_mouse_triggered_player")

        self.declare_parameter('frequency', DEFAULT_CONTROL_FREQ)
        self.control_frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.dt_control = 1.0 / self.control_frequency

        self._q_current = np.zeros(6)
        self._js_ok = False

        self.frames = []
        self.dataset_index = DATASET_START_INDEX - 1
        self.start_playback_time = None
        self.target_idx = 0

        self.is_executing = False
        self._replay_current_dataset = False

        self.last_sent_gripper_state = None

        self.forward_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            10
        )

        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        self.get_logger().info("Aguardando servico de I/O do robo...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error("Interrompido enquanto aguardava pelo servico de I/O.")
                return
            self.get_logger().info("Servico de I/O nao disponivel, tentando novamente...")
        self.get_logger().info("Servico de I/O conectado com sucesso!")

        self.create_subscription(
            JointState,
            "/joint_states",
            self._js_cb,
            qos_profile=qos_profile_sensor_data
        )

        self.timer = self.create_timer(self.dt_control, self._control_loop)

        print("\n" + "*"*60)
        print(" 🖱️  CLIQUE DIREITO : toca o proximo dataset ")
        print(" 🖱️  BOTAO LATERAL  : cancela e mantem o dataset atual p/ o proximo clique ")
        print(" ⌨️   TERMINAL       : digite um numero + Enter para tocar aquele dataset ")
        print("                     ('list' para ver quais existem) ")
        print("*"*60 + "\n")

        self.mouse_thread = threading.Thread(target=self.mouse_listener_thread, daemon=True)
        self.mouse_thread.start()

        self.terminal_thread = threading.Thread(target=self.terminal_input_thread, daemon=True)
        self.terminal_thread.start()

    # ── Deteccao do mouse ─────────────────────────────────────────────────

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
                if ecodes.BTN_RIGHT in keys:
                    return path
            except Exception:
                continue
        return None

    def mouse_listener_thread(self):
        mouse_path = self.find_logitech_mouse()
        if mouse_path is None:
            self.get_logger().error("Nenhum mouse Logitech encontrado.")
            return

        try:
            device = InputDevice(mouse_path)
            self.get_logger().info(f"Monitor de cliques ativo no mouse: {device.name}")

            for event in device.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.value != 1:
                    continue

                if event.code == ecodes.BTN_RIGHT:
                    if self.is_executing:
                        print("\n[AVISO] Ja existe um dataset em execucao! Clique ignorado.")
                        continue

                    if self._replay_current_dataset:
                        print("\n🖱️  [CLIQUE DIREITO] Repetindo o MESMO dataset (cancelado antes)...")
                        self.replay_current_dataset()
                    else:
                        print("\n  [CLIQUE DIREITO] Carregando proximo dataset...")
                        self.load_next_dataset()

                elif event.code == ecodes.BTN_LEFT:
                    if not self.is_executing:
                        print("\n[AVISO] Nenhum dataset em execucao no momento -- BTN_LEFT ignorado.")
                        continue
                    print("\n⏹️  [BOTAO LATERAL] Cancelando streaming atual! "
                          "Permanecendo no MESMO dataset -- aguardando proximo clique direito...")
                    self.cancel_current_playback()

        except Exception as e:
            self.get_logger().error(f"Falha na thread do mouse: {e}")

    # ── NOVO: Terminal interativo ─────────────────────────────────────────

    def list_available_datasets(self):
        """Lista os indices de dataset_N.json disponiveis na pasta,
        ordenados numericamente."""
        pattern = os.path.join(DATASET_DIR, f"{DATASET_PREFIX}*.json")
        paths = glob.glob(pattern)

        indices = []
        for p in paths:
            fname = os.path.basename(p)
            match = re.match(rf"^{re.escape(DATASET_PREFIX)}(\d+)\.json$", fname)
            if match:
                indices.append(int(match.group(1)))

        indices.sort()
        if not indices:
            print(f"\n[LISTA] Nenhum dataset encontrado em {DATASET_DIR}")
            return

        print(f"\n[LISTA] {len(indices)} datasets disponiveis em {DATASET_DIR}:")
        print(f"  Menor indice: {indices[0]} | Maior indice: {indices[-1]}")
        # Mostra a lista completa se for curta, ou so' um resumo se for longa
        if len(indices) <= 40:
            print(f"  Indices: {indices}")
        else:
            print(f"  Primeiros 20: {indices[:20]}")
            print(f"  Ultimos 20:   {indices[-20:]}")

    def play_specific_dataset(self, index: int):
        """Toca o dataset de indice ESPECIFICO escolhido pelo
        terminal. Atualiza self.dataset_index para esse valor, para
        que o CLIQUE DIREITO subsequente continue a sequencia normal
        a partir daqui."""
        if self.is_executing:
            print("\n[AVISO] Ja existe um dataset em execucao! Comando ignorado.")
            return

        print(f"\n⌨️  [TERMINAL] Carregando dataset {index} especificamente...")
        if self._load_dataset_file(index):
            self.dataset_index = index
            self._replay_current_dataset = False

    def terminal_input_thread(self):
        """Le comandos do terminal em paralelo -- nao interfere com
        os cliques do mouse, ambos disparam a mesma logica de
        playback por baixo dos panos."""
        print("[TERMINAL] Pronto para receber comandos (numero ou 'list').")
        while True:
            try:
                raw = input()
            except (EOFError, KeyboardInterrupt):
                break

            cmd = raw.strip()
            if not cmd:
                continue

            if cmd.lower() == "list":
                self.list_available_datasets()
                continue

            if cmd.lstrip("-").isdigit():
                index = int(cmd)
                self.play_specific_dataset(index)
            else:
                print(f"[TERMINAL] Comando nao reconhecido: '{cmd}' "
                      f"(digite um numero ou 'list')")

    # ── Carregamento / cancelamento do dataset ───────────────────────────

    def _load_dataset_file(self, index: int) -> bool:
        path = os.path.join(DATASET_DIR, f"{DATASET_PREFIX}{index}.json")

        if not os.path.exists(path):
            self.get_logger().error(
                f"Dataset nao encontrado: {path} -- verifique o numero/pasta."
            )
            return False

        with open(path, 'r') as file:
            self.frames = json.load(file)

        self.target_idx = 0
        self.start_playback_time = None
        self.last_sent_gripper_state = None
        self.is_executing = True

        self.get_logger().info(
            f"Dataset '{path}' carregado! {len(self.frames)} frames -- iniciando streaming."
        )
        return True

    def load_next_dataset(self):
        candidate_index = self.dataset_index + 1
        if self._load_dataset_file(candidate_index):
            self.dataset_index = candidate_index
            self._replay_current_dataset = False

    def replay_current_dataset(self):
        if self._load_dataset_file(self.dataset_index):
            self._replay_current_dataset = False

    def cancel_current_playback(self):
        self.is_executing = False
        self.target_idx = 0
        self.start_playback_time = None
        self.last_sent_gripper_state = None
        self._replay_current_dataset = True

        print(f"\n[CANCELADO] Dataset {self.dataset_index} interrompido. "
              f"Proximo clique direito vai reproduzi-lo do inicio.")

    # ── Telemetria ────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q_current = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    # ── Loop de controle ─────────────────────────────────────────────────

    def _control_loop(self):
        if not self.is_executing:
            return

        if not self._js_ok:
            return

        current_ros_time = self.get_clock().now().nanoseconds * 1e-9

        if self.start_playback_time is None:
            self.start_playback_time = current_ros_time
            return

        elapsed_time = current_ros_time - self.start_playback_time

        if self.target_idx >= len(self.frames):
            print(f"\n[FIM] Dataset {self.dataset_index} concluido a "
                  f"{self.control_frequency}Hz. Aguardando proximo clique/comando...")
            self.is_executing = False
            self._replay_current_dataset = False
            return

        current_frame = self.frames[self.target_idx]
        q_ref = np.array(current_frame["joint_positions"])
        gripper_ref = current_frame["gripper_io"]

        self.target_idx += 1

        self._publish_forward_commands(q_ref)

        if gripper_ref != self.last_sent_gripper_state:
            self._publish_gripper_command(gripper_ref)
            self.last_sent_gripper_state = gripper_ref

        joint_error = np.linalg.norm(q_ref - self._q_current)
        gripper_status_str = "FECHANDO" if gripper_ref == 1 else "ABRINDO"

        sys.stdout.write(
            f"\r[DATASET {self.dataset_index}] {elapsed_time:.2f}s | "
            f"Frame: {self.target_idx}/{len(self.frames)} | "
            f"Garra: {gripper_status_str} | "
            f"Erro Juntas: {joint_error:.4f} rad"
        )
        sys.stdout.flush()

    def _publish_forward_commands(self, q_target: np.ndarray):
        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.forward_pub.publish(msg)

    def _publish_gripper_command(self, gripper_state: int):
        req = SetIO.Request()
        req.fun = 1
        req.pin = 16
        req.state = float(gripper_state)
        self.io_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = UR5MouseTriggeredPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuario. Parando.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()