#!/usr/bin/env python3
"""
UR5 -- Pick do objeto detectado + Place no microondas (workflow combinado)
ROS2 Humble | Dashboard Control | Scaled Joint Trajectory Controller

Combina:
  1. ur5_track_object_position.py -- calcula continuamente (em
     background, a cada PERIODIC_CALC_INTERVAL_S) os angulos de
     junta para pegar o objeto detectado via /object_position_yolo_world,
     imprimindo o resultado o tempo todo.
  2. ur5_go_to_microwave.py -- sequencia de poses fixas ate' o
     microondas, com controle de dashboard (play/stop).

Ao apertar ENTER, executa o workflow COMPLETO:
  [Vai ao objeto (10cm antes -> alvo)] -> [FECHA GRIPPER] -> [sleep 3s]
  -> [Poses intermediarias -> Microondas (Z ajustado por altura)]
  -> [ABRE GRIPPER] -> [Microondas OUT] -> [Dashboard STOP/PLAY]

AJUSTE DE ALTURA (suposicao explicita, CONFIRME/AJUSTE):
  A caneca AZUL e' o objeto de REFERENCIA:
    - Z de pick (robo) da caneca azul de referencia: 0.67m
    - Z de colocacao no microondas da caneca azul: 0.72m (pose
      microwave_pose_mug_by_name)
  Para OUTROS objetos, calculamos:
    diferenca_altura = Z_pick_objeto_atual - Z_PICK_REFERENCIA_CANECA_AZUL
    Z_microondas_ajustado = Z_PLACE_REFERENCIA_CANECA_AZUL + diferenca_altura
  Isso assume relacao DIRETA (soma simples) entre a diferenca medida
  no pick e o ajuste necessario no Z de colocacao -- se o sentido
  fisico estiver invertido na pratica, troque o sinal.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from ur_msgs.srv import SetIO
import numpy as np
import time
import threading

# ══════════════════════════════════════════════════════════════════════════════
# Configurações -- Pick via IK
# ══════════════════════════════════════════════════════════════════════════════

OBJECT_POSITION_TOPIC = "/object_position_yolo_world"

MOVE_DURATION_S = 5.0
MAX_IK_ITERATIONS = 200
IK_CONVERGENCE_THRESHOLD = 1e-5
PERIODIC_CALC_INTERVAL_S = 2.0
APPROACH_CLEARANCE_M = 0.10

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5_DH = np.array([
    [0.089201,  0.0,       np.pi/2],
    [0.0,      -0.425428,  0.0    ],
    [0.0,      -0.392387,  0.0    ],
    [0.110225,  0.0,       np.pi/2],
    [0.094859,  0.0,      -np.pi/2],
    [0.082384,  0.0,       0.0    ],
])

TCP_OFFSET_Z = 0.150
DAMPING = 0.05
JOINT_WEIGHTS = np.array([1.0, 5.0, 0.3, 1.0, 1.0, 1.0])

GRIPPER_PIN = 16
GRIPPER_SLEEP_AFTER_CLOSE_S = 3.0

# ══════════════════════════════════════════════════════════════════════════════
# Configuracoes -- Ajuste de altura (CONFIRME estes valores de referencia)
# ══════════════════════════════════════════════════════════════════════════════

Z_PICK_REFERENCE_BLUE_MUG = 0.67   # Z (robo) de onde a caneca azul de referencia foi pega
Z_PLACE_REFERENCE_BLUE_MUG = 0.72  # Z (robo) de onde a caneca azul e' colocada no microondas

# ══════════════════════════════════════════════════════════════════════════════
# Poses fixas do workflow do microondas -- mapeadas por NOME
# ══════════════════════════════════════════════════════════════════════════════

INTERMEDIATE_POSE_1 = {
    "shoulder_lift_joint": -0.9958761374102991,
    "elbow_joint":         -1.3366435209857386,
    "wrist_1_joint":       -0.8589761892901819,
    "wrist_2_joint":        1.6337573528289795,
    "wrist_3_joint":        0.00022770027862861753,
    "shoulder_pan_joint":  -0.045229736958638966,
}

INTERMEDIATE_POSE_2 = {
    "shoulder_lift_joint": -1.0318563620196741,
    "elbow_joint":         -1.4548152128802698,
    "wrist_1_joint":       -0.6828368345843714,
    "wrist_2_joint":        1.7203160524368286,
    "wrist_3_joint":        0.0,
    "shoulder_pan_joint":   0.8773535490036011,
}

INTERMEDIATE_POSE_3 = {
    "shoulder_lift_joint": -1.496312443410055,
    "elbow_joint":         -1.3220637480365198,
    "wrist_1_joint":       -0.29568463960756475,
    "wrist_2_joint":        0.8530086874961853,
    "wrist_3_joint":        0.002960478188470006,
    "shoulder_pan_joint":   1.0775395631790161,
}

# NOVO: microwave_mug e' a pose PRINCIPAL usada (nao mais microwave_cup)
MICROWAVE_POSE_MUG = {
    "shoulder_lift_joint": -1.8656962553607386,
    "elbow_joint":         -0.9146130720721644,
    "wrist_1_joint":       -0.36239463487734014,
    "wrist_2_joint":        0.8221152424812317,
    "wrist_3_joint":        0.009816204197704792,
    "shoulder_pan_joint":   0.8966574668884277,
}

MICROWAVE_OUT_POSE = {
    "shoulder_lift_joint": -1.2018223921405237,
    "elbow_joint":         -1.3745749632464808,
    "wrist_1_joint":       -0.41111928621401006,
    "wrist_2_joint":        0.8221871256828308,
    "wrist_3_joint":        0.007718590088188648,
    "shoulder_pan_joint":   1.121093511581421,
}

MOVE_DURATION_MICROWAVE_STEP_S = 4.0

# ══════════════════════════════════════════════════════════════════════════════
# Funções de Álgebra e Cinemática (reaproveitadas)
# ══════════════════════════════════════════════════════════════════════════════

def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ],
    ])

def ur5_fk_frames(q):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        d, a, alpha = UR5_DH[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
        frames.append(T.copy())
    return frames

def ur5_tcp_pose(q):
    frames = ur5_fk_frames(q)
    T_tool0 = frames[6]
    T_offset = np.eye(4)
    T_offset[2, 3] = TCP_OFFSET_Z
    return T_tool0 @ T_offset

def ur5_jacobian_tcp(q):
    frames = ur5_fk_frames(q)
    T_tcp = ur5_tcp_pose(q)
    p_tcp = T_tcp[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_tcp - p_i)
        J[3:, i] = z_i
    return J

# minimizar: ||J @ dq - erro||² + λ² · dqᵀ @ W @ dq
def weighted_damped_pinv(J, lam, joint_weights):
    W_inv = np.diag(1.0 / joint_weights)
    return W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + lam**2 * np.eye(6))

def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

def compute_pose_error(T_target, T_current):
    error = np.zeros(6)
    error[:3] = T_target[:3, 3] - T_current[:3, 3]
    R_target = T_target[:3, :3]
    R_current = T_current[:3, :3]
    R_error = R_target @ R_current.T
    acos_val = np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(acos_val)
    if theta > 1e-5:
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2.0 * np.sin(theta))
        error[3:] = axis * theta
    return error

def solve_ik(T_target, q_start):
    q_iter = q_start.copy()
    error_norm = None
    for iteration in range(MAX_IK_ITERATIONS):
        T_current = ur5_tcp_pose(q_iter)
        pose_error = compute_pose_error(T_target, T_current)
        error_norm = np.linalg.norm(pose_error)
        if error_norm < IK_CONVERGENCE_THRESHOLD:
            break
        J = ur5_jacobian_tcp(q_iter)
        Jp = weighted_damped_pinv(J, DAMPING, JOINT_WEIGHTS)
        dq = Jp @ pose_error
        q_iter = q_iter + dq
    return wrap_to_pi(q_iter), iteration + 1, error_norm

def compute_horizontal_approach_rotation(current_rotation):
    """Eixo Z da ferramenta sempre alinhado ao eixo X da base (fixo).
    Escolhe entre as 2 orientacoes validas (180 graus de diferenca em
    torno do eixo de abordagem) a que fica mais proxima da atual."""
    z_axis = np.array([1.0, 0.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_option_a = np.column_stack([x_axis, y_axis, z_axis])
    R_option_b = np.column_stack([-x_axis, -y_axis, z_axis])

    dist_a = np.linalg.norm(R_option_a - current_rotation)
    dist_b = np.linalg.norm(R_option_b - current_rotation)
    return R_option_a if dist_a <= dist_b else R_option_b

def pose_dict_to_ordered_array(pose_dict):
    return np.array([pose_dict[name] for name in UR5_JOINT_NAMES])

# ══════════════════════════════════════════════════════════════════════════════
# Nó combinado
# ══════════════════════════════════════════════════════════════════════════════

class UR5PickAndPlaceMicrowave(Node):

    def __init__(self):
        super().__init__("ur5_pick_and_place_microwave")

        self._q = np.zeros(6)
        self._js_ok = False
        self.target_position = None
        self.latest_computed_joints_pre = None
        self.latest_computed_joints_final = None
        self.is_executing = False

        # ── Publishers/Clients ──────────────────────────────────────────
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        self.play_client = self.create_client(Trigger, '/dashboard_client/play')
        self.stop_client = self.create_client(Trigger, '/dashboard_client/stop')
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')

        self.get_logger().info('Conectando aos serviços do robô...')
        self.play_client.wait_for_service()
        self.stop_client.wait_for_service()
        self.io_client.wait_for_service()
        self.get_logger().info('Serviços conectados!')

        self.create_subscription(
            JointState, "/joint_states", self._js_cb, qos_profile=qos_profile_sensor_data
        )
        self.create_subscription(
            Point, OBJECT_POSITION_TOPIC, self._object_position_cb, 10
        )

        # ── Recalculo periodico do PICK (background, so' informativo) ──
        self.calc_timer = self.create_timer(PERIODIC_CALC_INTERVAL_S, self._periodic_calculate)

        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            f"No pronto. Calculando pick automaticamente a cada "
            f"{PERIODIC_CALC_INTERVAL_S}s -- aperte ENTER para EXECUTAR o "
            f"workflow completo (pick -> microondas -> out)."
        )

    # ── Callbacks ────────────────────────────────────────────────────────

    def _js_cb(self, msg):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _object_position_cb(self, msg: Point):
        self.target_position = np.array([msg.x, msg.y, msg.z])

    def _keyboard_listener(self):
        print(f"\nPronto! Calculando pick automaticamente a cada "
              f"{PERIODIC_CALC_INTERVAL_S}s. Aperte ENTER para EXECUTAR o "
              f"workflow completo.\n")
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break
            if self.is_executing:
                print("[AVISO] Workflow já em execução! Enter ignorado.")
                continue
            threading.Thread(target=self.execute_full_workflow, daemon=True).start()

    # ── Calculo periodico do PICK (so' informativo, nao move) ───────────

    def _periodic_calculate(self):
        self.compute_pick_once()

    def compute_pick_once(self):
        if not self._js_ok or self.target_position is None:
            return

        current_rotation = ur5_tcp_pose(self._q)[:3, :3]
        R_target = compute_horizontal_approach_rotation(current_rotation)

        T_target = np.eye(4)
        T_target[:3, 3] = self.target_position
        T_target[:3, :3] = R_target

        q_target_joints, iterations_final, error_final = solve_ik(T_target, self._q)

        approach_z_axis = R_target[:, 2]
        pre_target_position = self.target_position - APPROACH_CLEARANCE_M * approach_z_axis

        T_pre_target = np.eye(4)
        T_pre_target[:3, 3] = pre_target_position
        T_pre_target[:3, :3] = R_target

        q_pre_target_joints, iterations_pre, error_pre = solve_ik(T_pre_target, q_target_joints)

        print(f"\n[PICK CALCULADO] Posicao alvo: {np.round(self.target_position, 4)} | "
              f"Z_pick={self.target_position[2]:.4f}m")
        print(f"[PICK CALCULADO] Convergencia: final={iterations_final}it "
              f"({error_final*1000:.3f}mm) | antes={iterations_pre}it ({error_pre*1000:.3f}mm)")
        print(f"[PICK CALCULADO] Angulos ALVO FINAL (rad): {np.round(q_target_joints, 4)}")
        print(f"[PICK CALCULADO] (aperte ENTER para EXECUTAR o workflow completo)")

        self.latest_computed_joints_pre = q_pre_target_joints
        self.latest_computed_joints_final = q_target_joints

    # ── Dashboard / Switch Controller / Gripper ─────────────────────────

    def call_dashboard_sync(self, client, command_name):
        self.get_logger().info(f"Enviando comando de {command_name}...")
        req = Trigger.Request()
        future = client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def set_gripper(self, close: bool):
        req = SetIO.Request()
        req.fun = 1
        req.pin = GRIPPER_PIN
        req.state = 1.0 if close else 0.0
        future = self.io_client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        action = "FECHANDO" if close else "ABRINDO"
        self.get_logger().info(f"[GRIPPER] {action}")
        return future.result()

    def send_trajectory_points(self, points_and_durations):
        """points_and_durations: lista de (q_array, duration_seconds),
        publica TODOS como uma unica trajetoria com time_from_start
        cumulativo."""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR5_JOINT_NAMES

        cumulative_time = 0.0
        for q_array, duration in points_and_durations:
            cumulative_time += duration
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q_array]
            point.time_from_start = Duration(seconds=cumulative_time).to_msg()
            traj.points.append(point)

        self.traj_pub.publish(traj)
        return cumulative_time

    # ── Workflow completo ────────────────────────────────────────────────

    def execute_full_workflow(self):
        self.is_executing = True

        if self.latest_computed_joints_pre is None or self.latest_computed_joints_final is None:
            print("[AVISO] Ainda não há cálculo de pick disponível -- aguarde.")
            self.is_executing = False
            return

        # ── AJUSTE DE ALTURA: calcula a diferenca em relacao a caneca
        # azul de referencia, ANTES de mover (usa o Z do alvo de pick
        # que acabou de ser usado) ─────────────────────────────────────
        z_pick_atual = self.target_position[2]
        height_diff = z_pick_atual - Z_PICK_REFERENCE_BLUE_MUG
        z_microondas_ajustado = Z_PLACE_REFERENCE_BLUE_MUG + height_diff

        print(f"\n[ALTURA] Z pick atual={z_pick_atual:.4f}m | Z pick referencia (caneca azul)="
              f"{Z_PICK_REFERENCE_BLUE_MUG:.4f}m | diferenca={height_diff:+.4f}m")
        print(f"[ALTURA] Z microondas ajustado: {Z_PLACE_REFERENCE_BLUE_MUG:.4f}m + "
              f"{height_diff:+.4f}m = {z_microondas_ajustado:.4f}m")

        q_pick_pre = self.latest_computed_joints_pre
        q_pick_final = self.latest_computed_joints_final

        # ── 1. Vai ao objeto (10cm antes -> alvo) ───────────────────────
        print("\n[EXECUTANDO] Indo ao objeto (10cm antes -> alvo)...")
        total_t = self.send_trajectory_points([
            (q_pick_pre, MOVE_DURATION_S * 0.8),
            (q_pick_final, MOVE_DURATION_S * 0.2),
        ])
        time.sleep(total_t + 0.3)

        # ── 2. Fecha o gripper ───────────────────────────────────────────
        self.set_gripper(close=True)
        time.sleep(GRIPPER_SLEEP_AFTER_CLOSE_S)

        # ── 3. Poses intermediarias -> microondas (Z ajustado) ──────────
        q_int1 = pose_dict_to_ordered_array(INTERMEDIATE_POSE_1)
        q_int2 = pose_dict_to_ordered_array(INTERMEDIATE_POSE_2)
        q_int3 = pose_dict_to_ordered_array(INTERMEDIATE_POSE_3)
        q_microwave_base = pose_dict_to_ordered_array(MICROWAVE_POSE_MUG)
        q_microwave_out = pose_dict_to_ordered_array(MICROWAVE_OUT_POSE)

        # Ajusta a pose do microondas SO' no Z, via refinamento rapido de
        # IK partindo da pose fixa original (mantem X, Y, orientacao)
        T_microwave_base = ur5_tcp_pose(q_microwave_base)
        T_microwave_adjusted = T_microwave_base.copy()
        T_microwave_adjusted[2, 3] = z_microondas_ajustado

        q_microwave_adjusted, iters_mw, error_mw = solve_ik(T_microwave_adjusted, q_microwave_base)
        print(f"[MICROONDAS] Pose ajustada em Z: convergiu em {iters_mw} iteracoes "
              f"(erro={error_mw*1000:.3f}mm)")

        print("\n[EXECUTANDO] Indo para poses intermediárias -> microondas -> Z corrigido...")
        total_t = self.send_trajectory_points([
            (q_int1, MOVE_DURATION_MICROWAVE_STEP_S),
            (q_int2, MOVE_DURATION_MICROWAVE_STEP_S),
            (q_int3, MOVE_DURATION_MICROWAVE_STEP_S),
            (q_microwave_base, MOVE_DURATION_MICROWAVE_STEP_S),      # pose original do microwave_mug
            (q_microwave_adjusted, MOVE_DURATION_MICROWAVE_STEP_S),  # mesma pose, so' com Z do TCP corrigido
        ])
        time.sleep(total_t + 0.3)

        # ── 4. Abre o gripper (entre microondas e microondas_out) ───────
        self.set_gripper(close=False)
        time.sleep(1.0)

        # ── 5. Vai para microondas_out ───────────────────────────────────
        print("\n[EXECUTANDO] Indo para microondas OUT...")
        total_t = self.send_trajectory_points([
            (q_int3, MOVE_DURATION_MICROWAVE_STEP_S),
            (q_microwave_out, MOVE_DURATION_MICROWAVE_STEP_S),
        ])
        time.sleep(total_t + 0.3)

        # ── 6. Dashboard STOP/PLAY final ─────────────────────────────────
        self.call_dashboard_sync(self.stop_client, "STOP")
        time.sleep(0.5)
        self.call_dashboard_sync(self.play_client, "PLAY")

        print("\n--- 🔄 Workflow completo! Pronto para o próximo ENTER ---")
        self.is_executing = False


def main(args=None):
    rclpy.init(args=args)
    node = UR5PickAndPlaceMicrowave()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuário.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()