#!/usr/bin/env python3
"""
UR5 Dataset Player v9 — Position Streaming & Safe Gripper Service
ROS2 Humble | ur_robot_driver | Forward Position Controller

Este script lê as posições das juntas e o estado da garra do arquivo JSON, 
enviando as juntas para o Forward Position Controller e os estados do gripper
via Serviço oficial de I/O da UR para não interromper a trajetória.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ur_msgs.srv import SetIO  # 🌟 Serviço oficial de I/O da UR no ROS 2
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import json
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════
# Configurações Padrão
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH     = '/home/ziqi/pre_ws/jittering_dataset/dataset_6.json'
DEFAULT_CONTROL_FREQ = 125.0  # Hz — Frequência de controle escolhida por você

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ══════════════════════════════════════════════════════════════════════════════
# Nó de Envio de Posição e Garra (Service Client Base)
# ══════════════════════════════════════════════════════════════════════════════

class UR5PositionForwardPlayer(Node):

    def __init__(self):
        super().__init__("ur5_position_forward_player")

        # Parâmetro dinâmico de frequência de controle
        self.declare_parameter('frequency', DEFAULT_CONTROL_FREQ)
        self.control_frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.dt_control = 1.0 / self.control_frequency

        self._q_current = np.zeros(6)
        self._js_ok = False
        self.frames = []
        self.start_playback_time = None
        self.target_idx = 0  
        
        # Variável de controle para evitar mandar comando repetido para a garra a cada frame (Debounce)
        self.last_sent_gripper_state = None 

        # Carrega o dataset contendo as posições ordenadas e dados da garra
        self.load_dataset()

        # Publisher configurado para o canal de Forwarding de Posição do ros2_control
        self.forward_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            10
        )

        # 🌟 Cliente de serviço oficial para setar pinos digitais/ferramenta no driver da UR
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        
        self.get_logger().info("Aguardando serviço de I/O do robô...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error("Interrompido enquanto aguardava pelo serviço de I/O.")
                return
            self.get_logger().info("Serviço de I/O não disponível, tentando novamente...")
        self.get_logger().info("Serviço de I/O conectado com sucesso!")

        # Subscreve no /joint_states apenas para telemetria e cálculo do erro exibido na tela
        self.create_subscription(
            JointState, 
            "/joint_states", 
            self._js_cb, 
            qos_profile=qos_profile_sensor_data
        )

        # Loop estável na frequência parametrizada
        self.timer = self.create_timer(self.dt_control, self._control_loop)
        self.get_logger().info(f"Streaming de Juntas e Gripper ativo a {self.control_frequency} Hz.")

    def load_dataset(self):
        if not os.path.exists(DATASET_PATH):
            self.get_logger().error(f"Dataset não encontrado em {DATASET_PATH}")
            raise FileNotFoundError(DATASET_PATH)
        with open(DATASET_PATH, 'r') as file:
            self.frames = json.load(file)
        self.get_logger().info(f"Sucesso! {len(self.frames)} frames carregados para streaming.")

    def _js_cb(self, msg: JointState):
        try:
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

        # Condição de parada: Fim dos frames do arquivo
        if self.target_idx >= len(self.frames):
            self.get_logger().info(f"\n[FIM] Streaming concluído com sucesso a {self.control_frequency}Hz.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        # 1. Puxa os dados do frame atual do arquivo (juntas e garra)
        current_frame = self.frames[self.target_idx]
        q_ref = np.array(current_frame["joint_positions"])
        gripper_ref = current_frame["gripper_io"] # 1 para fechado, 0 para aberto
        
        self.target_idx += 1 

        # 2. Publica o comando de juntas puro
        self._publish_forward_commands(q_ref)

        # 3. 🌟 Altera a garra via chamada de serviço assíncrona se houver mudança de estado
        if gripper_ref != self.last_sent_gripper_state:
            self._publish_gripper_command(gripper_ref)
            self.last_sent_gripper_state = gripper_ref

        # Print de monitoramento do erro de rastreamento real
        joint_error = np.linalg.norm(q_ref - self._q_current)
        gripper_status_str = "FECHANDO" if gripper_ref == 1 else "ABRINDO"
        
        sys.stdout.write(
            f"\r[PLAYBACK] {elapsed_time:.2f}s | "
            f"Frame: {self.target_idx}/{len(self.frames)} | "
            f"Garra: {gripper_status_str} | "
            f"Erro Juntas: {joint_error:.4f} rad"
        )
        sys.stdout.flush()

    def _publish_forward_commands(self, q_target: np.ndarray):
        """ Cria e envia a mensagem compacta Float64MultiArray exigida pelo controlador """
        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.forward_pub.publish(msg)

    def _publish_gripper_command(self, gripper_state: int):
        """ 🌟 Usa requisição de serviço assíncrona nativa para chavear o pino físico da ferramenta """
        req = SetIO.Request()
        
        # 1 = FUNÇÃO PARA MUDAR DIGITAL OUT DA FERRAMENTA (TOOL DIGITAL OUT)
        # Se sua garra estivesse ligada na Controller Box normal, seria req.FUN_SET_DIGITAL_OUT (0)
        req.fun = 1 
        
        req.pin = 16         # Tool Output 0 (Pino correspondente ao set_tool_digital_out(0))
        req.state = float(gripper_state) # 1.0 para fechar, 0.0 para abrir
        
        # Chama o serviço em background de forma assíncrona para não engasgar o loop de 20Hz de juntas
        self.io_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = UR5PositionForwardPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuário. Parando streaming.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()