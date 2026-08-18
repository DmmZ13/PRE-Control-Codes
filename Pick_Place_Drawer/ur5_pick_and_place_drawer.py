#!/usr/bin/env python3
"""
UR5 Object Position Tracker + Fechamento de Garra + Workflow Autonomo

Ao chegar no alvo (ENTER), o robo:
  1. Move ate' o ponto "10cm antes" e depois o alvo final (via IK)
  2. Fecha a garra (SetIO, pino 16)
  3. Executa o workflow autonomo (pose intermediaria -> reproduz
     dataset da gaveta -> home -> dashboard stop/play -> shutdown)
"""

import json
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
from ur_msgs.srv import SetIO
from std_srvs.srv import Trigger

# ══════════════════════════════════════════════════════════════════════════════
# Configurações
# ══════════════════════════════════════════════════════════════════════════════

OBJECT_POSITION_TOPIC = "/object_position_yolo_world"

MOVE_DURATION_S = 5.0
MAX_IK_ITERATIONS = 200
IK_CONVERGENCE_THRESHOLD = 1e-5

PERIODIC_CALC_INTERVAL_S = 2.0

APPROACH_CLEARANCE_M = 0.10
STAGE1_OFFSET_M = 0.10
STAGE1_DURATION_S = 4.0
STAGE2_DURATION_S = 2.0

# ── Garra via I/O digital do UR5 (mesmo padrao do cliente autonomo) ──
GRIPPER_IO_PIN = 16
GRIPPER_CLOSE_STATE = 1.0   # confirme a polaridade real da sua garra
GRIPPER_CLOSE_WAIT_S = 1.5  # tempo para a garra fechar fisicamente antes de seguir

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

TCP_OFFSET_Z   = 0.150
DAMPING        = 0.05

JOINT_WEIGHTS = np.array([1.0, 0.3, 5.0, 1.0, 1.0, 1.0])

GRASP_QUATERNION_XYZW = np.array([
    -0.6796924814975461,
     0.6299978960825222,
    -0.2772763062769206,
     0.25345341091555806,
])

# ── Workflow autonomo (dataset gravado da gaveta) ────────────────────
STEP_DECIMATION = 5
DATASET_PATHS = {"drawer": "/home/ziqi/pre_ws/jittering_dataset/drawer.json"}

INTERMEDIATE_POSE_DRAWER_BY_NAME = {
    "shoulder_lift_joint": -1.9332926909076136,
    "elbow_joint": -0.00021535554994756012,
    "wrist_1_joint": -1.6716049353228968,
    "wrist_2_joint": 1.5370337963104248,
    "wrist_3_joint": -3.5587941304981996e-05,
    "shoulder_pan_joint": 0.01216848287731409,
}

HOME_POSE_BY_NAME = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 0.0,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": 1.5708,
    "wrist_3_joint": 0.0,
}


def quaternion_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("quaternion de norma ~0")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


R_TARGET_FIXED = quaternion_to_rotation_matrix(GRASP_QUATERNION_XYZW)


def compute_grasp_rotation(current_rotation: np.ndarray) -> np.ndarray:
    return R_TARGET_FIXED


# ══════════════════════════════════════════════════════════════════════════════
# Álgebra e Cinemática (inalteradas)
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

def ur5_fk_frames(q: np.ndarray):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        d, a, alpha = UR5_DH[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
        frames.append(T.copy())
    return frames

def ur5_tcp_pose(q: np.ndarray) -> np.ndarray:
    frames = ur5_fk_frames(q)
    T_tool0 = frames[6]
    T_offset = np.eye(4)
    T_offset[2, 3] = TCP_OFFSET_Z
    return T_tool0 @ T_offset

def ur5_jacobian_tcp(q: np.ndarray) -> np.ndarray:
    frames = ur5_fk_frames(q)
    T_tcp  = ur5_tcp_pose(q)
    p_tcp  = T_tcp[:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_tcp - p_i)
        J[3:, i] = z_i
    return J

def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))

def weighted_damped_pinv(J: np.ndarray, lam: float, joint_weights: np.ndarray) -> np.ndarray:
    W_inv = np.diag(1.0 / joint_weights)
    return W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + lam**2 * np.eye(6))

def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
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

def solve_ik(T_target: np.ndarray, q_start: np.ndarray):
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

# ══════════════════════════════════════════════════════════════════════════════
# Nó de Rastreamento + Fechamento de Garra + Workflow
# ══════════════════════════════════════════════════════════════════════════════

class UR5ObjectPositionTracker(Node):

    def __init__(self):
        super().__init__("ur5_object_position_tracker")

        self._q = np.zeros(6)
        self._js_ok = False

        self.target_position = None
        self.latest_computed_joints_pre = None
        self.latest_computed_joints_final = None
        self.last_sent_gripper_state = None

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._js_cb,
            qos_profile=qos_profile_sensor_data
        )

        self.create_subscription(
            Point,
            OBJECT_POSITION_TOPIC,
            self._object_position_cb,
            10
        )

        # ── Cliente de IO (garra), igual ao cliente autonomo ──────────
        self.io_client = self.create_client(SetIO, "/io_and_status_controller/set_io")
        self.get_logger().info("Aguardando serviço de I/O do robô...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.get_logger().info("Serviço de I/O não disponível, tentando novamente...")

        # ── Clientes de dashboard (usados no workflow) ─────────────────
        self.play_client = self.create_client(Trigger, "/dashboard_client/play")
        self.stop_client = self.create_client(Trigger, "/dashboard_client/stop")

        self.calc_timer = self.create_timer(PERIODIC_CALC_INTERVAL_S, self._periodic_calculate)

        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            f"No pronto. Calculando automaticamente a cada "
            f"{PERIODIC_CALC_INTERVAL_S}s -- aperte ENTER no terminal para "
            f"EXECUTAR o ultimo calculo, mover ate' o objeto, fechar a "
            f"garra e iniciar o workflow autonomo."
        )

    # ── Callbacks ──────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _object_position_cb(self, msg: Point):
        self.target_position = np.array([msg.x, msg.y, msg.z])

    def _keyboard_listener(self):
        print(f"\nPronto! Calculando automaticamente a cada "
              f"{PERIODIC_CALC_INTERVAL_S}s. Aperte ENTER para EXECUTAR o "
              f"ultimo calculo (mover, fechar a garra, e rodar o workflow).\n")
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break
            self.execute_move()

    def _periodic_calculate(self):
        self.compute_once()

    # ── Garra (SetIO, pino 16 -- mesmo padrao do cliente autonomo) ─────

    def _publish_gripper_command(self, gripper_state: float):
        req = SetIO.Request()
        req.fun = 1
        req.pin = GRIPPER_IO_PIN
        req.state = float(gripper_state)
        future = self.io_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(f"[GARRA] Comando enviado (pino {GRIPPER_IO_PIN}, estado {gripper_state}).")
        )
        self.last_sent_gripper_state = gripper_state

    def close_gripper_and_wait(self):
        """Fecha a garra e aguarda GRIPPER_CLOSE_WAIT_S segundos
        (tempo fisico da garra fechar) antes de seguir para o
        workflow."""
        self.get_logger().info("[GARRA] Fechando garra...")
        self._publish_gripper_command(GRIPPER_CLOSE_STATE)
        time.sleep(GRIPPER_CLOSE_WAIT_S)

    # ── Cinematica: calculo e execucao do movimento ate' o objeto ──────

    def compute_once(self):
        if not self._js_ok:
            print("[AVISO] Ainda nao recebi /joint_states -- aguarde e tente de novo.")
            return

        if self.target_position is None:
            print(f"[AVISO] Ainda nao recebi nenhuma posicao em "
                  f"{OBJECT_POSITION_TOPIC} -- aguarde e tente de novo.")
            return

        current_rotation = ur5_tcp_pose(self._q)[:3, :3]
        R_target = compute_grasp_rotation(current_rotation)

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

        print(f"\n[CALCULADO] Posicao alvo do objeto usada: "
              f"{np.round(self.target_position, 4)}")
        print(f"[CALCULADO] Alvo final: convergiu em {iterations_final} iteracoes "
              f"(erro={error_final*1000:.4f}mm)")
        print(f"[CALCULADO] Ponto 10cm antes: convergiu em {iterations_pre} iteracoes "
              f"(erro={error_pre*1000:.4f}mm)")
        print(f"[CALCULADO] Angulos ATUAIS (rad):        {np.round(self._q, 4)}")
        print(f"[CALCULADO] Angulos PONTO ANTES (rad):   {np.round(q_pre_target_joints, 4)}")
        print(f"[CALCULADO] Angulos ALVO FINAL (rad):    {np.round(q_target_joints, 4)}")
        print(f"[CALCULADO] (aperte ENTER para EXECUTAR -- vai ate' o objeto, "
              f"fecha a garra, e roda o workflow)")

        self.latest_computed_joints_pre = q_pre_target_joints
        self.latest_computed_joints_final = q_target_joints

    def execute_move(self):
        """Move ate' o objeto (10cm antes -> final), fecha a garra, e
        entao dispara o workflow autonomo (pose intermediaria ->
        dataset -> home -> dashboard -> shutdown)."""
        if self.latest_computed_joints_pre is None or self.latest_computed_joints_final is None:
            print("[AVISO] Ainda nao ha' nenhum calculo disponivel -- aguarde.")
            return

        print(f"\n[EXECUTANDO] Indo primeiro ao ponto 10cm antes, depois ao "
              f"alvo final (duracao total={MOVE_DURATION_S}s)")
        self._publish_joint_trajectory(self.latest_computed_joints_pre, self.latest_computed_joints_final)

        # Agenda: espera o movimento terminar, fecha a garra, depois
        # roda o workflow. Isso bloqueia a callback do timer (rodando
        # dentro da thread do teclado, entao nao trava o spin do ROS).
        def _after_move():
            time.sleep(MOVE_DURATION_S + 0.2)  # garante que o robo chegou
            self.close_gripper_and_wait()
            self.execute_workflow()

        threading.Thread(target=_after_move, daemon=True).start()

    def _publish_joint_trajectory(self, q_pre: np.ndarray, q_final: np.ndarray):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR5_JOINT_NAMES

        t_pre = MOVE_DURATION_S * 0.8

        point_pre = JointTrajectoryPoint()
        point_pre.positions = q_pre.tolist()
        point_pre.time_from_start = Duration(seconds=t_pre).to_msg()

        point_final = JointTrajectoryPoint()
        point_final.positions = q_final.tolist()
        point_final.time_from_start = Duration(seconds=MOVE_DURATION_S).to_msg()

        traj.points = [point_pre, point_final]
        self.traj_pub.publish(traj)

    # ── Workflow autonomo (pose intermediaria -> dataset -> home) ──────

    def _pose_dict_to_ordered_array(self, pose_dict: dict) -> list:
        return [pose_dict[name] for name in UR5_JOINT_NAMES]

    def send_single_target_trajectory(self, target_joints: list, duration_seconds: float):
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(j) for j in target_joints]
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6

        point.time_from_start = Duration(
            seconds=duration_seconds
        ).to_msg()

        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(msg)

    def play_dataset_as_trajectory(self, json_file_path: str):
        if not os.path.exists(json_file_path):
            self.get_logger().error(f"Dataset não encontrado: {json_file_path}")
            return

        with open(json_file_path, "r") as f:
            frames = json.load(f)

        if not frames:
            return

        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES
        first_timestamp = frames[0]["timestamp"]

        decimated_frames = frames[::STEP_DECIMATION]
        if frames[-1] not in decimated_frames:
            decimated_frames.append(frames[-1])

        self.get_logger().info(f"Reduzindo {len(frames)} frames para {len(decimated_frames)} pontos para envio...")

        for item in decimated_frames:
            point = JointTrajectoryPoint()
            point.positions = [float(val) for val in item["joint_positions"]]
            point.velocities = [0.0] * 6

            t_rel = item["timestamp"] - first_timestamp
            point.time_from_start = Duration(seconds=t_rel).to_msg()
            msg.points.append(point)

        msg.header.stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(msg)

        start_time = time.time()
        for frame in frames:
            target_time = frame["timestamp"] - first_timestamp
            sleep_needed = target_time - (time.time() - start_time)
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            gripper_ref = int(frame.get("gripper_io", 0))
            if gripper_ref != self.last_sent_gripper_state:
                self._publish_gripper_command(gripper_ref)

        total_duration = frames[-1]["timestamp"] - first_timestamp
        time_left = total_duration - (time.time() - start_time)
        if time_left > 0:
            time.sleep(time_left)

        self.get_logger().info("Reprodução da trajetória concluída!")

    def call_dashboard_sync(self, client, command_name):
        self.get_logger().info(f"Enviando comando de {command_name} para o Teach Pendant...")
        req = Trigger.Request()
        future = client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def execute_workflow(self):
        """Roda apos o objeto ja ter sido pego (garra fechada):
        vai a pose intermediaria, reproduz o dataset da gaveta,
        volta pra home, e reativa o Polyscope."""
        target_key = "drawer"
        dataset_path = DATASET_PATHS[target_key]

        intermediate_point = self._pose_dict_to_ordered_array(INTERMEDIATE_POSE_DRAWER_BY_NAME)
        self.get_logger().info(f"Movendo para pose INTERMEDIÁRIA ({target_key})...")
        self.send_single_target_trajectory(intermediate_point, duration_seconds=4.0)
        time.sleep(4.2)

        self.get_logger().info(f"Reproduzindo dataset ({target_key})...")
        self.play_dataset_as_trajectory(dataset_path)

        home_point = self._pose_dict_to_ordered_array(HOME_POSE_BY_NAME)
        self.get_logger().info(f"Movendo para HOME ({target_key})...")
        self.send_single_target_trajectory(home_point, duration_seconds=4.0)
        time.sleep(4.2)

        if self.stop_client.wait_for_service(timeout_sec=1.0):
            self.call_dashboard_sync(self.stop_client, "STOP")
            time.sleep(0.5)

        if self.play_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Reativando Polyscope via comando PLAY...")
            self.call_dashboard_sync(self.play_client, "PLAY")

        self.get_logger().info("--- 🔄 Ciclo Concluído! ---")


def main():
    rclpy.init()
    node = UR5ObjectPositionTracker()

    print("\nJanela pronta -- aperte ENTER no terminal para executar o ciclo completo.\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()