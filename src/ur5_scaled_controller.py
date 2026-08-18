#!/usr/bin/env python3
"""
UR5 Dataset Player v5 — Pure Position Reference with Dynamic Velocity Derivation
ROS2 Humble | ur_robot_driver | Scaled Joint Trajectory Controller

Este script lê apenas as posições das juntas salvas no arquivo JSON e aceita uma frequência
de controle customizada. A velocidade enviada ao controlador é calculada dinamicamente
com base no erro entre a posição alvo (q_ref) e a posição atual do robô (q_current).
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
# Configurações Padrão
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH     = '/home/ziqi/pre_ws/dataset.json'
DEFAULT_CONTROL_FREQ = 20.0  # Hz — Frequência padrão caso não seja passada via terminal

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ══════════════════════════════════════════════════════════════════════════════
# Nó de Reprodução por Posição com Feedback de Velocidade
# ══════════════════════════════════════════════════════════════════════════════

class UR5PositionTrajectoryPlayer(Node):

    def __init__(self):
        super().__init__("ur5_position_trajectory_player")

        # 🌟 Parâmetro dinâmico para escolher a frequência de controle do robô
        self.declare_parameter('frequency', DEFAULT_CONTROL_FREQ)
        self.control_frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.dt_control = 1.0 / self.control_frequency

        self._q_current = np.zeros(6)
        self._js_ok = False
        self.frames = []
        self.start_playback_time = None
        self.target_idx = 0  

        # Carrega o dataset contendo as posições
        self.load_dataset()

        # Publisher configurado para o Scaled Joint Trajectory Controller
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        # Subscreve no /joint_states para obter o q_current em tempo real (essencial para a velocidade)
        self.create_subscription(
            JointState, 
            "/joint_states", 
            self._js_cb, 
            qos_profile=qos_profile_sensor_data
        )

        # Loop de controle estável na frequência que você escolheu
        self.timer = self.create_timer(self.dt_control, self._control_loop)
        self.get_logger().info(f"Controle ativo a {self.control_frequency} Hz (dt: {self.dt_control:.4f}s). Aguardando hardware...")

    def load_dataset(self):
        if not os.path.exists(DATASET_PATH):
            self.get_logger().error(f"Dataset não encontrado em {DATASET_PATH}")
            raise FileNotFoundError(DATASET_PATH)
        with open(DATASET_PATH, 'r') as file:
            self.frames = json.load(file)
        self.get_logger().info(f"Sucesso! {len(self.frames)} frames carregados.")

    def _js_cb(self, msg: JointState):
        try:
            # Mapeia as juntas recebidas do robô (q_current) para a ordem padrão do UR5
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q_current = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _control_loop(self):
        if not self._js_ok:
            return

        current_ros_time = self.get_clock().now().nanoseconds * 1e-9

        if self.start_playback_time is None:
            self.start_playback_time = current_ros_time
            return

        elapsed_time = current_ros_time - self.start_playback_time

        # Condição de parada: fim do arquivo de posições
        if self.target_idx >= len(self.frames):
            self.get_logger().info(f"\n[FIM] Trajetória executada completamente a {self.control_frequency}Hz.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        # 1. PEGA APENAS A POSIÇÃO ALVO (q_ref) DO DATASET
        current_frame = self.frames[self.target_idx]
        q_ref = np.array(current_frame["joint_positions"])
        self.target_idx += 1 

        # 2. CALCULA A VELOCIDADE DA JUNTA: (q_ref - q_current) / dt
        # Isso simula o comportamento que a sua IA fará em malha fechada
        dq_derived = (q_ref - self._q_current) / self.dt_control

        # Limite de segurança física (Geralmente pi rad/s ou menos para evitar solavancos na bancada)
        dq_derived = np.clip(dq_derived, -np.pi, np.pi)

        # 3. ENVIA O COMANDO DE TRAJETÓRIA COMPLETO PARA O CONTROLADOR ESCALONADO
        self._publish_joint_trajectory(q_ref, dq_derived)

        # Print de monitoramento na tela
        joint_error = np.linalg.norm(q_ref - self._q_current)
        sys.stdout.write(
            f"\r[EXECUTANDO] {elapsed_time:.2f}s | "
            f"Frame: {self.target_idx}/{len(self.frames)} | "
            f"Erro Angular Total: {joint_error:.4f} rad"
        )
        sys.stdout.flush()

    def _publish_joint_trajectory(self, q_target: np.ndarray, dq_target: np.ndarray):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR5_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = q_target.tolist()      # Posição absoluta vinda da IA/Dataset
        # point.velocities = dq_target.tolist()    # Velocidade derivada da diferença real
        
        # O tempo para executar deve bater exatamente com o ciclo do seu timer
        point.time_from_start = Duration(seconds=self.dt_control).to_msg()

        traj.points = [point]
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = UR5PositionTrajectoryPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuário. Parando execução.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()