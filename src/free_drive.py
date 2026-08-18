#!/usr/bin/env python3
"""
UR5 Integrated Controller: FreeDrive + Safe Mouse Gripper Service (Pynput Version)
ROS2 Humble | ur_robot_driver | ur_msgs | pynput

Nó unificado para gerenciar o modo FreeDrive pelo terminal e o acionamento
do Gripper instantaneamente via clique GLOBAL do mouse (Botão Direito) ou ENTER,
utilizando chamadas de serviço e eliminando o seletor evdev de inicialização.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO  # 🌟 Serviço oficial de I/O da UR no ROS 2
import time
import threading
import socket
import os

# Monitoramento de hardware via evdev/pynput abstrato
from evdev import InputDevice, list_devices, ecodes

class UR5RobotController:
    def __init__(self, node: Node = None):
        if node is not None:
            self.node = node
            self._own_node = False
        else:
            self.node = Node('ur5_integrated_controller_node')
            self._own_node = True

        self.robot_ip = '147.250.35.40'
        self.dashboard_port = 29999

        self.current_gripper_state = False
        self.gripper_pin = 16

        # Publisher oficial do driver ROS2 para injetar comandos URScript
        self.script_pub = self.node.create_publisher(
            String,
            '/urscript_interface/script_command',
            10
        )
        
        # Subscriber para monitorar o estado dos pinos digitais
        self.io_sub = self.node.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_states_callback,
            10
        )

        # Cliente de serviço oficial para setar os pinos sem travar o robô
        self.io_client = self.node.create_client(SetIO, '/io_and_status_controller/set_io')
        
        self.node.get_logger().info("Aguardando serviço de I/O do robô...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.node.get_logger().info("Serviço de I/O não disponível, tentando novamente...")
        self.node.get_logger().info("Serviço de I/O conectado com sucesso!")
        
        time.sleep(0.2)

        self.mouse_thread = threading.Thread(target=self.mouse_listener_thread, daemon=True)
        self.mouse_thread.start()

    def io_states_callback(self, msg: IOStates):
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = bool(pin_state.state)
                    break
        except Exception:
            pass

    def send_urscript(self, script_string: str):
        msg = String()
        msg.data = script_string
        self.script_pub.publish(msg)

    def toggle_gripper(self):
        """ 
        Usa chamadas de serviço assíncronas para chavear a ferramenta. 
        O driver ROS altera os bits por baixo dos panos, impedindo 
        que a controladora da UR pause o programa atual do FreeDrive.
        """
        novo_estado = not self.current_gripper_state
        state_float = 1.0 if novo_estado else 0.0
        action_verb = "FECHANDO" if novo_estado else "ABRINDO"
        
        self.node.get_logger().info(f"[MOU_CLICK] {action_verb} -> Solicitando alteração via Serviço de I/O...")

        req = SetIO.Request()
        req.fun = 1         # 1 = Mudar Digital Out da Ferramenta (Tool Digital Out)
        req.pin = 16         # Tool Output 0
        req.state = state_float
        
        # Executa em background de forma assíncrona para não engasgar nenhuma thread
        self.io_client.call_async(req)

    def activate_freedrive(self):
        cmd = "def fd_on():\n  freedrive_mode()\n  while(True):\n    sync()\n  end\nend\n"
        self.send_urscript(cmd)
        self.node.get_logger().info("[ROBOT] ---> FREEDRIVE ATIVADO! Braço solto. <---")

    def force_dashboard_stop(self):
        try:
            self.node.get_logger().info("[EMERGÊNCIA] Conectando à porta 29999 para travar o robô...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.5)
            s.connect((self.robot_ip, self.dashboard_port))
            s.recv(512)
            s.sendall(b"stop\n")
            response = s.recv(512).decode().strip()
            s.close()
            self.node.get_logger().info(f"[Dashboard] Resposta: {response} -> FREIOS TRAVADOS!")
        except Exception as e:
            self.node.get_logger().error(f"Falha ao conectar no Dashboard: {e}")

    def close(self):
        self.force_dashboard_stop()
        time.sleep(0.1)
        if self._own_node:
            self.node.destroy_node()

    def find_logitech_mouse(self):
        for path in list_devices():
            try:
                dev = InputDevice(path)
                if "logitech" not in dev.name.lower(): continue
                caps = dev.capabilities()
                if ecodes.EV_KEY not in caps: continue
                keys = caps[ecodes.EV_KEY]
                if ecodes.BTN_RIGHT in keys:
                    self.node.get_logger().info(f"Mouse encontrado: {dev.name} ({path})")
                    return path
            except Exception: pass
        return None

    def mouse_listener_thread(self):
        mouse_path = self.find_logitech_mouse()
        if mouse_path is None:
            self.node.get_logger().error("Nenhum mouse Logitech encontrado.")
            return
        try:
            device = InputDevice(mouse_path)
            self.node.get_logger().info(f"Monitorando mouse: {device.name}")

            for event in device.read_loop():
                if event.type != ecodes.EV_KEY: continue
                if event.code != ecodes.BTN_RIGHT and event.code != ecodes.BTN_LEFT: continue
                if event.value != 1: continue # Apenas gatilho de descida (pressionado)

                if event.code == ecodes.BTN_RIGHT:
                    self.node.get_logger().info("[MOUSE] Clique direito detectado. Alternando estado do gripper...")
                    self.toggle_gripper()
                elif event.code == ecodes.BTN_LEFT:
                    self.node.get_logger().info("[MOUSE] Clique esquerdo detectado. Ativando FreeDrive...")
                    self.activate_freedrive()

        except Exception as e:
            self.node.get_logger().error(f"Falha ao monitorar mouse: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    rclpy.init()
    robot = UR5RobotController()
    
    # Executor MultiThreaded obrigatório para rodar serviços em paralelo com inputs
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot.node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    print("\n" + "="*60)
    print("   UR5 INTEGRATED CONTROLLER (COMBINED TRIGGER MODE)   ")
    print("="*60)
    print(" -> CONFIGURAÇÃO: Monitorando cliques de mouse globalmente")
    print(" -> CLIQUE DIREITO => Ativa GRIPPER + modo FREEDRIVE ao mesmo tempo!")
    print(" -> Digite [ENTER] (vazio)        => Força ativação manual do FreeDrive")
    print(" -> Pressione [Ctrl+C]            => Sair com segurança")
    print("="*60 + "\n")
    
    try:
        time.sleep(0.5)
        
        while rclpy.ok():
            user_input = input("Comando: ").strip().lower()
            
            # Mantém suporte para o ENTER ativar o freedrive caso o braço pese na bancada
            if user_input == '':
                robot.activate_freedrive()
            else:
                print("[AVISO] Use o Clique Direito do mouse para acionar o robô.")
            
            time.sleep(0.1)
            
    except (KeyboardInterrupt, SystemExit):
        print("\n\n[SIGNAL] Parando robô e limpando estados de pinos...")
    finally:
        robot.close()
        rclpy.shutdown()
        spin_thread.join()
        print("Módulo encerrado com sucesso.")