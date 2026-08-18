#!/usr/bin/env python3
"""
Teleoperacao do UR5 REAL por teclado (evdev) -- APENAS controle,
sem gravacao/recorder. Mesmos binds do script de simulacao:

  Q/E         - Junta 0 (shoulder_pan) -/+
  W/S         - Junta 1 (shoulder_lift) ou Junta 3 (wrist_1) +/-
                (aperte Shift uma vez para alternar qual junta)
  Cima/Baixo  - Junta 2 (elbow) +/-
  Esq/Dir     - Junta 4 (wrist_2) -/+
  R/Shift+R   - Junta 5 (wrist_3) +/- (segure Shift p/ inverter)
  Espaco      - abre/fecha garra (toggle, via IO digital)
  ESC         - envia o robo para o HOME e fica pronto para a
                PROXIMA teleoperacao (NAO encerra o script)
  Ctrl+C (terminal) - encerra o script de verdade

ARQUITETURA:
- Controle de junta: URScript bruto por socket (porta 30002), enviando
  speedj(qd, a, t) repetidamente em alta frequencia.
- HOMING (novo): ao apertar ESC, dispara um movej() para a pose home
  e implementa uma pequena maquina de estados DENTRO do timer de
  controle (nao-bloqueante) que aguarda o robo chegar perto da pose
  alvo (via feedback de /joint_states) antes de voltar ao modo de jog
  normal -- assim o executor ROS2 nunca fica travado esperando.
- Garra: acionada via servico ROS2 SetIO, pino 16 (Tool Digital
  Output 0).
"""

import time
import socket
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO

from evdev import InputDevice, ecodes


class UR5GripperController:
    """Controla o gripper via servico oficial SetIO (nao trava o
    interpretador do robo como um URScript bruto travaria)."""

    def __init__(self, node: Node):
        self.node = node
        self.current_gripper_state = False
        self.gripper_pin = 16  # Tool Digital Output 0

        self.io_sub = self.node.create_subscription(
            IOStates, '/io_and_status_controller/io_states', self.io_states_callback, 10)

        self.io_client = self.node.create_client(SetIO, '/io_and_status_controller/set_io')

        self.node.get_logger().info("Aguardando servico de I/O do robo...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.node.get_logger().info("Servico de I/O nao disponivel, tentando novamente...")
        self.node.get_logger().info("Servico de I/O conectado com sucesso!")

    def io_states_callback(self, msg: IOStates):
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = bool(pin_state.state)
                    break
        except Exception:
            pass

    def set_gripper_state(self, close_gripper: bool):
        state_float = 1.0 if close_gripper else 0.0
        action_verb = "FECHANDO" if close_gripper else "ABRINDO"

        req = SetIO.Request()
        req.fun = 1
        req.pin = self.gripper_pin
        req.state = state_float

        self.io_client.call_async(req)
        self.node.get_logger().info(f"[GRIPPER] {action_verb} -> Solicitado via Servico (Pin {self.gripper_pin} = {state_float})")

    def toggle_gripper(self):
        novo_estado = not self.current_gripper_state
        self.set_gripper_state(close_gripper=novo_estado)

    def close(self):
        self.set_gripper_state(close_gripper=False)


# ── Configuracao ─────────────────────────────────────────────────────────

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

KEYBOARD_PATH = "/dev/input/event4"  # ajuste se necessario

ROBOT_IP = "147.250.35.40"
SCRIPT_PORT = 30002

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_KEY = ecodes.KEY_SPACE

# Ganho de velocidade de junta (rad/s) por tecla segurada.
JOINT_SPEED_RAD_S = 0.15
JOG_ACCEL = 0.5
JOG_CMD_PERIOD = DT * 2

JOINT_QE = 0
JOINT_WS_OFF = 1
JOINT_WS_ON = 3
JOINT_UP_DOWN = 2
JOINT_LEFT_RIGHT = 4
JOINT_R = 5

# ── HOME (novo) ───────────────────────────────────────────────────────────

HOME_POSITION = [0.0, -1.5708, 0.0, -1.5708, 1.5708, 0.0]
HOME_ACCEL = 1.0    # rad/s^2 -- movimento ponto-a-ponto, pode ser mais rapido que o jog
HOME_VEL = 0.5       # rad/s
HOME_TOLERANCE = 0.01  # rad -- considera "chegou" quando o erro maximo por junta for menor que isso
HOME_TIMEOUT_S = 15.0  # seguranca: desiste de esperar apos esse tempo (mas nao trava o no)


# ── Estado global compartilhado com a thread de teclado ─────────────────

pressed_keys = set()
keys_lock = threading.Lock()
running = True

shift_held = False
ws_toggle_state = False

gripper_toggle_requested = False

# NOVO: ESC agora dispara o homing, nao encerra mais o script
home_requested = False


# ── Thread de teclado (evdev) ────────────────────────────────────────────

def keyboard_thread():
    global running, shift_held, ws_toggle_state, gripper_toggle_requested, home_requested

    try:
        dev = InputDevice(KEYBOARD_PATH)
    except Exception as e:
        print(f"[ERRO] Nao foi possivel abrir {KEYBOARD_PATH}: {e}")
        running = False
        return

    print(f"[OK] Escutando teclado: {dev.name}")

    for event in dev.read_loop():
        if not running:
            break
        if event.type != ecodes.EV_KEY:
            continue

        code = event.code
        value = event.value

        if code in (ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT):
            with keys_lock:
                shift_held = (value != 0)
                if value == 1:
                    ws_toggle_state = not ws_toggle_state
                    print(f"[DEBUG] Toggle W/S: agora controla "
                          f"{'junta 3 (wrist_1)' if ws_toggle_state else 'junta 1 (shoulder_lift)'}")
            continue

        if code == ecodes.KEY_ESC and value == 1:
            # MUDANCA: ESC nao encerra mais o script -- so' sinaliza
            # o pedido de ir para o home. O loop de controle
            # (nao-bloqueante) cuida do resto e mantem o no vivo.
            print("\n[ESC] Enviando robo para o HOME...")
            home_requested = True
            continue

        if code == GRIPPER_KEY and value == 1:
            gripper_toggle_requested = True
            continue

        if value == 1:
            with keys_lock:
                pressed_keys.add(code)
        elif value == 0:
            with keys_lock:
                pressed_keys.discard(code)


def compute_joint_velocities():
    joint_vel = np.zeros(6)

    with keys_lock:
        keys_now = set(pressed_keys)
        shift_now = shift_held
        ws_toggle_now = ws_toggle_state

    sign = -1.0 if shift_now else 1.0

    if ecodes.KEY_R in keys_now:
        joint_vel[JOINT_R] += sign * JOINT_SPEED_RAD_S

    if ecodes.KEY_Q in keys_now:
        joint_vel[JOINT_QE] += JOINT_SPEED_RAD_S
    if ecodes.KEY_E in keys_now:
        joint_vel[JOINT_QE] -= JOINT_SPEED_RAD_S

    ws_target_joint = JOINT_WS_ON if ws_toggle_now else JOINT_WS_OFF
    if ecodes.KEY_W in keys_now:
        joint_vel[ws_target_joint] -= JOINT_SPEED_RAD_S
    if ecodes.KEY_S in keys_now:
        joint_vel[ws_target_joint] += JOINT_SPEED_RAD_S

    if ecodes.KEY_UP in keys_now:
        joint_vel[JOINT_UP_DOWN] -= JOINT_SPEED_RAD_S
    if ecodes.KEY_DOWN in keys_now:
        joint_vel[JOINT_UP_DOWN] += JOINT_SPEED_RAD_S

    if ecodes.KEY_LEFT in keys_now:
        joint_vel[JOINT_LEFT_RIGHT] += JOINT_SPEED_RAD_S
    if ecodes.KEY_RIGHT in keys_now:
        joint_vel[JOINT_LEFT_RIGHT] -= JOINT_SPEED_RAD_S

    return joint_vel


# ── No ROS2 ───────────────────────────────────────────────────────────────

class UR5RealTeleopNode(Node):
    def __init__(self):
        super().__init__("ur5_real_teleop_control_only_node")

        self.current_joint_positions = [0.0] * 6

        self._sock = None
        self._connect_script_socket()

        self.gripper = UR5GripperController(self)

        self.joint_subscription = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10)

        # ── Maquina de estados do homing (nao-bloqueante) ────────────────
        self.is_homing = False
        self.home_move_sent_at = None

        self.control_timer = self.create_timer(DT, self._control_step)

        print("\n" + "=" * 60)
        print("  Teleoperacao REAL (evdev) -- controle por JUNTA (sem gravacao)")
        print("=" * 60)
        print("  Q/E         - Junta 0 (shoulder_pan)")
        print("  W/S         - Junta 1 (shoulder_lift) ou Junta 3 (wrist_1)")
        print("                (Shift = toggle qual junta)")
        print("  Cima/Baixo  - Junta 2 (elbow)")
        print("  Esq/Dir     - Junta 4 (wrist_2)")
        print("  R/Shift+R   - Junta 5 (wrist_3)")
        print("  Espaco      - abre/fecha garra")
        print("  ESC         - vai para o HOME e fica pronto p/ a proxima teleop")
        print("  Ctrl+C (terminal) - encerra o script de verdade")
        print("=" * 60 + "\n")
        print("!!! ATENCAO: robo real. Mantenha o botao de emergencia por perto. !!!\n")

    def _connect_script_socket(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((ROBOT_IP, SCRIPT_PORT))
            self.get_logger().info("Conectado ao script socket do UR5.")
        except Exception as e:
            self.get_logger().error(f"Falha ao conectar no script socket: {e}")
            self._sock = None

    def _send_urscript(self, body: str):
        if self._sock is None:
            self._connect_script_socket()
            if self._sock is None:
                return
        try:
            self._sock.sendall(f"def prog():\n{body}\nend\n".encode("utf-8"))
        except Exception as e:
            self.get_logger().error(f"Erro ao enviar URScript: {e}")
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_speedj(self, qd):
        qd_str = "[" + ",".join(f"{v:.5f}" for v in qd) + "]"
        body = f"  speedj({qd_str}, a={JOG_ACCEL}, t={JOG_CMD_PERIOD:.3f})\n"
        self._send_urscript(body)

    def _send_stopj(self, decel=1.5):
        self._send_urscript(f"  stopj({decel})\n")

    def _send_movej_home(self):
        """Envia UM comando ponto-a-ponto (movej) para a pose home.
        Diferente do speedj (jog continuo), este e' um movimento
        discreto -- o proprio controlador do robo cuida da
        trajetoria/interpolacao ate o alvo."""
        q_str = "[" + ",".join(f"{v:.5f}" for v in HOME_POSITION) + "]"
        body = f"  movej({q_str}, a={HOME_ACCEL}, v={HOME_VEL})\n"
        self._send_urscript(body)

    def _joint_state_callback(self, msg: JointState):
        try:
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self.current_joint_positions = [float(pos_map[j]) for j in UR5_JOINT_NAMES]
        except KeyError:
            pass

    def _joint_error_to_home(self):
        return max(
            abs(cur - target)
            for cur, target in zip(self.current_joint_positions, HOME_POSITION)
        )

    def _control_step(self):
        global gripper_toggle_requested, home_requested, running

        if not running:
            self._send_stopj()
            rclpy.shutdown()
            return

        if gripper_toggle_requested:
            gripper_toggle_requested = False
            self.gripper.toggle_gripper()

        # ── Maquina de estados: inicia o homing ──────────────────────────
        if home_requested and not self.is_homing:
            home_requested = False
            self.is_homing = True
            self.home_move_sent_at = time.time()
            # Limpa quaisquer teclas de jog presas, para nao competir
            # com o movimento de homing
            with keys_lock:
                pressed_keys.clear()
            self._send_movej_home()
            self.get_logger().info(
                f"[HOME] Comando movej enviado. Alvo: {HOME_POSITION}"
            )
            return  # nao manda speedj neste mesmo ciclo

        # ── Maquina de estados: aguardando chegar no home (nao-bloqueante) ──
        if self.is_homing:
            error = self._joint_error_to_home()
            elapsed = time.time() - self.home_move_sent_at

            if error < HOME_TOLERANCE:
                self.is_homing = False
                self.get_logger().info(
                    f"[HOME] Robo chegou no home (erro={error:.4f} rad). "
                    f"Pronto para a proxima teleoperacao!"
                )
                print("\n" + "-" * 60)
                print(" ✅  Robo no HOME. Pode teleoperar novamente (jog livre). ")
                print("-" * 60 + "\n")
            elif elapsed > HOME_TIMEOUT_S:
                # Seguranca: nao trava o no esperando para sempre --
                # avisa e volta ao modo normal mesmo sem confirmar
                # a chegada exata (o proprio controlador do robo ja
                # deve ter concluido o movej de qualquer forma).
                self.is_homing = False
                self.get_logger().warn(
                    f"[HOME] Timeout aguardando confirmacao de chegada "
                    f"(erro atual={error:.4f} rad). Retomando modo normal "
                    f"mesmo assim -- verifique visualmente se o robo esta "
                    f"na pose esperada."
                )
            else:
                # Ainda se movendo -- nao manda speedj enquanto isso
                return

        # ── Modo normal: jog continuo via speedj ─────────────────────────
        qd = compute_joint_velocities()
        self._send_speedj(qd)


def main(args=None):
    global running

    rclpy.init(args=args)
    node = UR5RealTeleopNode()

    kb_thread = threading.Thread(target=keyboard_thread, daemon=True)
    kb_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        node._send_stopj()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Robo parado. Encerrado.")


if __name__ == "__main__":
    main()