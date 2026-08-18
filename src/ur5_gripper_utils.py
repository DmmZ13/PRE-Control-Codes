#!/usr/bin/env python3
"""
UR5 Gripper Utility Module with Interactive Keyboard Toggle (Safe Service Edition)
ROS2 Humble | ur_robot_driver | ur_msgs

This module provides an interactive interface to open/close the UR5 gripper
by pressing [ENTER] in the terminal, using the safe ROS 2 SetIO service.
"""

import rclpy
from rclpy.node import Node
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO  # 🌟 Serviço oficial de I/O da UR no ROS 2
import time
import threading
import sys

class UR5GripperController:
    def __init__(self, node: Node = None):
        """
        Initializes the gripper controller.
        """
        if node is not None:
            self.node = node
            self._own_node = False
        else:
            self.node = Node('ur5_gripper_utility_node')
            self._own_node = True

        self.current_gripper_state = False
        
        # 🌟 PIN 16 = Tool Digital Output 0 (Validado conforme o MATLAB)
        self.gripper_pin = 16

        # SUBSCRIBER para monitorar o estado dos pinos digitais de saída
        self.io_sub = self.node.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_states_callback,
            10
        )

        # 🌟 Cliente de serviço oficial para setar os pinos sem travar o interpretador do robô
        self.io_client = self.node.create_client(SetIO, '/io_and_status_controller/set_io')
        
        self.node.get_logger().info("Aguardando serviço de I/O do robô...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.node.get_logger().info("Serviço de I/O não disponível, tentando novamente...")
        self.node.get_logger().info("Serviço de I/O conectado com sucesso!")
        
        time.sleep(0.2)

    def io_states_callback(self, msg: IOStates):
        """ Callback assíncrono que monitora em tempo real o estado do pino 16 """
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = bool(pin_state.state)
                    break
        except Exception:
            pass

    def set_gripper_state(self, close_gripper: bool):
        """
        Changes the physical state of the gripper acting on Tool Output 0 via Service.
        """
        state_float = 1.0 if close_gripper else 0.0
        action_verb = "FECHANDO" if close_gripper else "ABRINDO"
        
        req = SetIO.Request()
        req.fun = 1          # 1 = Mudar Digital Out
        req.pin = 16         # Pin 16 corresponde à Tool DO 0
        req.state = state_float
        
        # Chama de forma assíncrona para não travar a execução
        self.io_client.call_async(req)
        self.node.get_logger().info(f"[GRIPPER] {action_verb} -> Solicitado via Serviço (Pin 16 = {state_float})")

    def toggle_gripper(self):
        """ Olha o estado atual registrado do pino 16 e inverte o sinal """
        novo_estado = not self.current_gripper_state
        self.set_gripper_state(close_gripper=novo_estado)

    def close(self):
        """ Opens the gripper and safely destroys the node if it was internally generated """
        self.set_gripper_state(close_gripper=False)
        time.sleep(0.2)
        if self._own_node:
            self.node.destroy_node()

# ══════════════════════════════════════════════════════════════════════════════
# LOOP INTERATIVO DE TECLADO
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    rclpy.init()
    
    gripper = UR5GripperController()
    
    # 🌟 Mudado para MultiThreadedExecutor para permitir chamadas de serviço em background
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gripper.node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    print("\n" + "="*60)
    print("   UR5 INTERACTIVE GRIPPER CONTROLLER (SAFE SERVICE MODE)   ")
    print("="*60)
    print(" -> Pressione [ENTER] para alternar o sinal da Garra (Tool 0 via Pin 16)")
    print(" -> Pressione [Ctrl+C] para sair com segurança")
    print("="*60 + "\n")
    
    try:
        time.sleep(0.5)
        
        while rclpy.ok():
            # Trava o loop esperando um ENTER do usuário
            input("Aperte [ENTER] para inverter a garra... ")
            
            # Executa a inversão baseada no feedback de telemetria do pino 16
            gripper.toggle_gripper()
            
            # Pequeno debounce para evitar cliques duplos acidentais
            time.sleep(0.1)
            
    except (KeyboardInterrupt, SystemExit):
        print("\n\n[SIGNAL] Finalizando utilitário de controle de garra...")
    finally:
        gripper.close()
        rclpy.shutdown()
        spin_thread.join()
        print("Módulo encerrado com sucesso.")