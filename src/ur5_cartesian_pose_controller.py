#!/usr/bin/env python3
"""
UR5 Dataset Player v3 — Cartesian Pose Tracking via Joint Trajectory Controller
ROS2 Humble | ur_robot_driver | Scaled Joint Trajectory Controller

Este script lê as poses cartesianas salvas no arquivo JSON (posição e quatérnio),
interpola no tempo a 50Hz e calcula a trajetória de juntas correspondente através
de cinemática inversa diferencial em malha fechada, eliminando derivas cartesianas.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import json
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
# Configurações
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH   = '/home/ziqi/pre_ws/dataset.json'

ROBOT_FREQ     = 10.0  # Hz — frequência de comando enviada ao controlador
DT_ROBOT       = 1.0 / ROBOT_FREQ

DATASET_FREQ   = 10.0  # Hz — frequência nativa com que os dados foram gravados
DT_DATASET     = 1.0 / DATASET_FREQ

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# From cat ~/my_robot_calibration.yaml
UR5_DH = np.array([
    [0.089201,  0.0,       np.pi/2],  
    [0.0,      -0.425428,  0.0    ],  
    [0.0,      -0.392387,  0.0    ],
    [0.110225,  0.0,       np.pi/2], 
    [0.094859,  0.0,      -np.pi/2],  
    [0.082384,  0.0,       0.0    ],  
])

TCP_OFFSET_Z   = 0.150  # metros (offset da sua garra)
MAX_JOINT_VELOCITY = np.pi  # rad/s — limite máximo de velocidade por ciclo (segurança)
DAMPING        = 0.05   # Amortecimento para evitar singularidades

# ══════════════════════════════════════════════════════════════════════════════
# Funções de Álgebra e Cinemática Analítica
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

def quaternion_to_rotation_matrix(q):
    """ Converte vetor quaternion [x, y, z, w] para matriz de rotação 3x3 """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
    ])

def compute_pose_error(T_target, T_current):
    """ Calcula o erro cartesiano 6D (posição + orientação) entre duas matrizes homogêneas """
    error = np.zeros(6)
    # Erro de translação linear (XYZ)
    error[:3] = T_target[:3, 3] - T_current[:3, 3]
    
    # Erro de rotação (calculado no espaço de Lie através do vetor de rotação residual)
    R_target = T_target[:3, :3]
    R_current = T_current[:3, :3]
    R_error = R_target @ R_current.T
    
    # Eixo-ângulo a partir da matriz de rotação residual
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

# ══════════════════════════════════════════════════════════════════════════════
# Nó de Reprodução por Posição Cartesiana
# ══════════════════════════════════════════════════════════════════════════════

class UR5CartesianPosePlayer(Node):

    def __init__(self):
        super().__init__("ur5_cartesian_pose_player")

        self._q = np.zeros(6)
        self._js_ok = False
        self.frames = []
        self.start_playback_time = None

        self.load_dataset()

        self.target_idx = 0  # Índice do frame atual do dataset a ser rastreado

        # ── Mudança: Publisher configurado para o Scaled Joint Trajectory Controller ──
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

        # Loop estável a 50Hz para atualizar os alvos de trajetória dinamicamente
        self.timer = self.create_timer(DT_ROBOT, self._pose_tracking_loop)
        self.get_logger().info("Nó de Rastreamento de Pose pronto. Aguardando /joint_states...")

    def load_dataset(self):
        if not os.path.exists(DATASET_PATH):
            self.get_logger().error(f"Dataset não encontrado em {DATASET_PATH}")
            raise FileNotFoundError(DATASET_PATH)
        with open(DATASET_PATH, 'r') as file:
            self.frames = json.load(file)
        self.get_logger().info(f"Sucesso! {len(self.frames)} frames de pose carregados para o player.")

    def _js_cb(self, msg: JointState):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _pose_tracking_loop(self):
        if not self._js_ok:
            return

        current_ros_time = self.get_clock().now().nanoseconds * 1e-9

        if self.start_playback_time is None:
            self.start_playback_time = current_ros_time
            return

        # Time elapsed since the beginning of playback (seconds)
        elapsed_time = current_ros_time - self.start_playback_time

        # Stop condition: reached the end of the 125Hz recorded data
        if self.target_idx >= len(self.frames):
            self.get_logger().info("\n[COMPLETED] 125Hz Dataset played back successfully at 50Hz.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        # 1. EXTRACT THE EXACT TARGET POSE (No interpolation needed!)
        target_frame = self.frames[self.target_idx]["cartesian_pose"]
        self.target_idx += 1  # For logging purposes (1-based index)
        p_target = np.array(target_frame["position"])
        q_target = np.array(target_frame["orientation"])

        # Reconstruct the target SE(3) Homogeneous Matrix
        T_target = np.eye(4)
        T_target[:3, 3] = p_target
        T_target[:3, :3] = quaternion_to_rotation_matrix(q_target)

        # 2. CLOSED-LOOP DIFFERENTIAL INVERSE KINEMATICS
        T_current = ur5_tcp_pose(self._q)
        pose_error = compute_pose_error(T_target, T_current)

        J = ur5_jacobian_tcp(self._q)
        Jp = damped_pinv(J, DAMPING)
        
        # dq represents the joint displacement target to eliminate the Cartesian error
        dq = Jp @ pose_error

        # Convert velocity limit to position step limit per cycle
        dq = np.clip(dq, -MAX_JOINT_VELOCITY/DT_ROBOT, MAX_JOINT_VELOCITY/DT_ROBOT)  

        # Target joint positions for this control cycle
        q_target_joints = self._q + dq

        # Feedforward joint velocity profile derived from position step
        dq_vel = dq / DT_ROBOT  

        # 3. COMMAND THE ROBOT VIA TRAJECTORY PUBLISHER
        self._publish_joint_trajectory(q_target_joints, dq_vel)

        sys.stdout.write(
            f"\r[TRACKING] Time: {elapsed_time:.2f}s | "
            f"Dataset Frame: {self.target_idx}/{len(self.frames)} | "
            f"Linear Error: {np.linalg.norm(pose_error[:3])*1000:.3f}mm"
        )
        sys.stdout.flush()

    def _publish_joint_trajectory(self, q_target: np.ndarray, dq_target: np.ndarray):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR5_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = q_target.tolist()
        
        point.velocities = dq_target.tolist()  
        
        point.time_from_start = Duration(seconds=DT_ROBOT).to_msg()

        traj.points = [point]
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = UR5CartesianPosePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuário.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()