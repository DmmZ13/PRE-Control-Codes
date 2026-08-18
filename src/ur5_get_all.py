#!/usr/bin/env python3
"""
UR5 Multi-Modal Synchronized Recorder Node (Joints & Cartesian Space Only)
ROS2 Humble | ur_robot_driver | message_filters

Este nó escuta as poses cartesianas do TCP e os estados das juntas, ordena-as na 
sequência cinemática correta, calcula a velocidade cartesiana em tempo real 
e exporta tudo em um banco de dados JSON unificado baseado na frequência escolhida.

(Versão independente: Controle do robô e garras gerenciados externamente)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # Official UR Cartesian pose feedback type
from ur_msgs.msg import IOStates           # Official UR message type for gripper IO monitoring
from sensor_msgs.msg import JointState     # Official type for Joint States (Positions and Velocities)
import sys
import threading
import json

# 🌟 Definição estrita da ordem cinemática padrão do UR5
UR5_JOINT_NAMES = [
    "shoulder_pan_joint",  # Junta 0 (Base)
    "shoulder_lift_joint", # Junta 1 (Ombro)
    "elbow_joint",         # Junta 2 (Cotovelo)
    "wrist_1_joint",       # Junta 3 (Pulso 1 - geralmente Roll/Pitch)
    "wrist_2_joint",       # Junta 4 (Pulso 2 - Pitch/Yaw)
    "wrist_3_joint",       # Junta 5 (Pulso 3 - Roll terminal)
]

class UR5FreedriveNode(Node):
    def __init__(self):
        super().__init__('ur5_freedrive_node')
        
        # --- PARÂMETROS ---
        self.declare_parameter('frequency', 50.0)
        self.recording_frequency = self.get_parameter('frequency').get_parameter_value().double_value
        
        # Target digital input pin configuration for the gripper
        self.gripper_pin = 16
        self.current_gripper_state = 0  
        
        # Internal tracking state flags
        self.is_running = True
        self.is_recording = False  
        
        # Tracking variables for numerical velocity derivation
        self.last_pose_msg = None
        self.current_cartesian_pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        self.current_cartesian_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Tracking variables for Joint States (Garantindo inicialização limpa)
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        
        # --- DATASTRUCTURE TO STORE ALL ROBOT HISTORY ---
        self.history_database = []
        
        # Initialize ROS 2 Subscribers
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.pose_callback,
            10
        )

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
        
        # Timer dedicado para gravação baseada na frequência do usuário
        timer_period = 1.0 / self.recording_frequency
        self.recording_timer = self.create_timer(timer_period, self.recording_timer_callback)
        
        self.get_logger().info(f"Joint and Cartesian monitoring are ACTIVE. Recording frequency: {self.recording_frequency} Hz.")
        
        # Background thread para inputs do teclado
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener_loop, daemon=True)
        self.keyboard_thread.start()

    def keyboard_listener_loop(self):
        print("\n" + "*"*60)
        print(" --> PRESS [ENTER] TO START RECORDING DATASET <-- ")
        print("*"*60 + "\n")
        
        while self.is_running:
            input()  
            if not self.is_running:
                break
                
            self.is_recording = not self.is_recording  
            
            if self.is_recording:
                print("\n\n>>> RECORDING STARTED! Saving frames... (Press [ENTER] to pause) <<<\n")
            else:
                print("\n\n||| RECORDING PAUSED! (Press [ENTER] to resume, [Ctrl+C] to exit and save) |||\n")

    def io_states_callback(self, msg):
        if not self.is_running:
            return
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = 1 if pin_state.state else 0
                    break
        except Exception:
            pass

    def joint_states_callback(self, msg):
        """ 🌟 Callback para caçar e forçar a ordenação exata do UR5 """
        if not self.is_running:
            return
        try:
            # Cria mapas temporários relacionando Nome -> Posição e Nome -> Velocidade
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            vel_map = {name: vel for name, vel in zip(msg.name, msg.velocity)}
            
            # Reconstroi os vetores varrendo estritamente a lista UR5_JOINT_NAMES ordenada
            self.current_joint_positions = [float(pos_map[joint]) for joint in UR5_JOINT_NAMES]
            self.current_joint_velocities = [float(vel_map[joint]) for joint in UR5_JOINT_NAMES]
        except KeyError:
            # Evita quebra se o tópico cuspir mensagens parciais durante a inicialização
            pass

    def pose_callback(self, msg):
        if not self.is_running:
            return
            
        self.current_cartesian_pose = {
            "position": [float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z)],
            "orientation": [float(msg.pose.orientation.x), float(msg.pose.orientation.y), float(msg.pose.orientation.z), float(msg.pose.orientation.w)]
        }

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
                
                # Cálculo numérico limpo e corrigido do vetor de rotação
                w_vector = 2.0 * np.array([
                    q_diff[0]*q1_conj[1] + q_diff[1]*q1_conj[0] + q_diff[2]*q1_conj[3] - q_diff[3]*q1_conj[2],
                    q_diff[0]*q1_conj[2] - q_diff[1]*q1_conj[3] + q_diff[2]*q1_conj[0] + q_diff[3]*q1_conj[1],
                    q_diff[0]*q1_conj[3] + q_diff[1]*q1_conj[2] - q_diff[2]*q1_conj[1] + q_diff[3]*q1_conj[0]
                ])

                self.current_cartesian_velocity = [float(vx), float(vy), float(vz), float(w_vector[0]), float(w_vector[1]), float(w_vector[2])]

        self.last_pose_msg = msg

    def recording_timer_callback(self):
        if not self.is_running:
            return

        frame_packet = {
            "cartesian_pose": self.current_cartesian_pose,
            "cartesian_velocity": self.current_cartesian_velocity,
            "joint_positions": self.current_joint_positions,   
            "joint_velocities": self.current_joint_velocities, 
            "gripper_io": int(self.current_gripper_state)
        }
        
        if self.is_recording:
            self.history_database.append(frame_packet)
            
        status_prefix = "[RECORDING]" if self.is_recording else "[MONITORING]"
        j_print = ", ".join([f"{j:+.2f}" for j in self.current_joint_positions])
        
        sys.stdout.write(
            f"\r{status_prefix} "
            f"Juntas Ordenadas (rad) -> [{j_print}] | Salvos: {len(self.history_database)}"
        )
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = UR5FreedriveNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SIGNAL] Ctrl+C interceptado! Parando amostragem e consolidando dataset...")
        node.is_running = False
        node.is_recording = False
        
        if len(node.history_database) > 0:
            output_file_path = '/home/ziqi/pre_ws/dataset.json'
            print(f"\nDumping {len(node.history_database)} frames para '{output_file_path}'...")
            try:
                with open(output_file_path, 'w') as file:
                    json.dump(node.history_database, file, indent=4)
                print("Arquivo exportado com sucesso com as Juntas perfeitamente alinhadas para o Player!")
            except Exception as e:
                print(f"[ERROR] Falha ao salvar arquivo no disco: {e}")
        else:
            print("\nNenhum dado foi gravado. Pulando geração de arquivo.")
        
        print("\n" + "="*50)
        print("          REPORT COMPLETO DE TELEMETRIA ROBÓTICA          ")
        print("="*50)
        print(f" Total de frames gravados na frequência escolhida: {len(node.history_database)}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Exceção inesperada: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        print("Nó ROS 2 finalizado de forma limpa.")

if __name__ == '__main__':
    main()