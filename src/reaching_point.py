#!/usr/bin/env python3
"""
UR5 Autonomous Trajectory Client (Mouse Triggered Version + ROS2 SetIO Gripper)
ROS2 Humble | Dashboard Control (Play/Stop) | evdev Interrupt | ROS2 SetIO Service

Aguardando clique no botão lateral do mouse (BTN_SIDE) para disparar:
[Checa garra via ROS2 IO -> Abre via SetIO Client se estiver fechada] -> 
[STOP TP] -> [PLAY] -> [Trajetória] -> [STOP TP] -> [PLAY]

NOVO: Aguardando pressionar a tecla ` (crase) no teclado -- dispara o
MESMO workflow (execute_workflow), levando o robô ao ponto final.
"""

import os
import sys
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates
from ur_msgs.srv import SetIO
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger
import time
import threading

from evdev import InputDevice, list_devices, ecodes

class UR5AutonomousClient(Node):
    def __init__(self):
        super().__init__('ur5_autonomous_client')
        self.get_logger().info('Inicializando Cliente Autónomo engatilhado por Mouse...')
        
        # Configuração da Garra (Pino Digital e estado)
        self.gripper_pin = 16
        self.current_gripper_state = None  # 1 = Fechada (ON / 24V), 0 = Aberta (OFF / 0V)

        # Ordem oficial das juntas para o mapeamento correto do qpos
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Flag de proteção contra disparos duplicados
        self.is_executing = False
        
        # ── CLIENTS DOS SERVIÇOS DO ROS2 (DASHBOARD E I/O) ──
        self.play_client = self.create_client(Trigger, '/dashboard_client/play')
        self.stop_client = self.create_client(Trigger, '/dashboard_client/stop')
        self.switch_ctrl_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        
        # Aguarda a ativação de todos os serviços
        self.get_logger().info('Conectando aos serviços do robô...')
        self.play_client.wait_for_service()
        self.stop_client.wait_for_service()
        self.switch_ctrl_client.wait_for_service()
        self.io_client.wait_for_service()
        self.get_logger().info('Todos os serviços ROS2 conectados com sucesso!')

        # Publisher de Trajetória
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        # Inscrição para estado das juntas
        self.current_joints_reading = None
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Inscrição para estado de I/O (garra)
        self.io_subscription = self.create_subscription(
            IOStates, '/io_and_status_controller/io_states', self.io_states_callback, 10
        )

        # Inicializa a thread de escuta do mouse em background
        print("\n" + "*"*60)
        print(" 🔘 AGUARDANDO CLIQUE NO BOTÃO LATERAL (BTN_SIDE) PARA DISPARAR ")
        print("*"*60 + "\n")
        self.mouse_thread = threading.Thread(target=self.mouse_listener_thread, daemon=True)
        self.mouse_thread.start()

        # NOVO: Inicializa a thread de escuta do teclado em background
        print("*"*60)
        print(" ⌨️  AGUARDANDO TECLA ` (CRASE) PARA DISPARAR O WORKFLOW ")
        print("*"*60 + "\n")
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener_thread, daemon=True)
        self.keyboard_thread.start()

    def joint_state_callback(self, msg):
        try:
            joints_dict = dict(zip(msg.name, msg.position))
            self.current_joints_reading = [joints_dict[name] for name in self.joint_names]
        except KeyError:
            pass 

    def io_states_callback(self, msg):
        try:
            for pin_state in msg.digital_out_states:
                if pin_state.pin == self.gripper_pin:
                    self.current_gripper_state = 1 if pin_state.state else 0
                    break
        except Exception:
            pass

    def set_gripper_state_sync(self, state: int):
        """
        Envia o comando de I/O via cliente do ROS2 (ur_msgs/srv/SetIO) de forma síncrona.
        state: 1 -> Fechada (1.0) | 0 -> Aberta (0.0)
        """
        req = SetIO.Request()
        req.fun = SetIO.Request.FUN_SET_DIGITAL_OUT
        req.pin = self.gripper_pin
        req.state = float(state)

        future = self.io_client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def open_gripper_if_closed(self):
        """ Checa se a garra está fechada e abre usando o cliente de I/O do ROS2 """
        self.get_logger().info("Verificando estado da garra...")
        
        # Aguarda receber o estado do I/O no tópico
        timeout = 2.0
        start = time.time()
        while self.current_gripper_state is None and (time.time() - start) < timeout:
            time.sleep(0.05)

        if self.current_gripper_state == 1:
            self.get_logger().info("⚠️ Garra detectada FECHADA. Enviando comando para ABRIR via Client ROS2...")
            self.set_gripper_state_sync(0) # 0 = Aberta
            time.sleep(0.5) # Tempo físico para o acionamento
            self.get_logger().info("✅ Garra aberta com sucesso.")
        elif self.current_gripper_state == 0:
            self.get_logger().info("✅ Garra já está ABERTA.")
        else:
            self.get_logger().warn("⚠️ Não foi possível ler o estado da garra. Enviando comando de abertura por segurança...")
            self.set_gripper_state_sync(0)
            time.sleep(0.5)

    def find_logitech_mouse(self):
        for path in list_devices():
            try:
                dev = InputDevice(path)
                if "logitech" not in dev.name.lower(): continue
                caps = dev.capabilities()
                if ecodes.EV_KEY not in caps: continue
                keys = caps[ecodes.EV_KEY]
                if ecodes.BTN_SIDE in keys or ecodes.BTN_EXTRA in keys:
                    return path
            except Exception: continue
        return None

    # NOVO: localiza um dispositivo de teclado (procura KEY_GRAVE nas capacidades)
    def find_keyboard_device(self):
        for path in list_devices():
            try:
                dev = InputDevice(path)
                caps = dev.capabilities()
                if ecodes.EV_KEY not in caps: continue
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_GRAVE in keys:
                    return path
            except Exception: continue
        return None

    def mouse_listener_thread(self):
        mouse_path = self.find_logitech_mouse()
        if mouse_path is None:
            self.get_logger().error("Nenhum mouse Logitech com botões laterais encontrado.")
            return
        
        try:
            device = InputDevice(mouse_path)
            self.get_logger().info(f"Monitor de interrupções ativo no mouse: {device.name}")
            
            for event in device.read_loop():
                if event.type != ecodes.EV_KEY: continue
                if event.value != 1: continue # Somente pressionamento (Pushed)
                
                if event.code == ecodes.BTN_SIDE:
                    if self.is_executing:
                        print("[AVISO] Workflow já está em execução! Clique ignorado por segurança.")
                        continue
                    
                    print("\n💥 [INTERRUPÇÃO] Botão lateral detectado! Iniciando automação...")
                    threading.Thread(target=self.execute_workflow, daemon=True).start()

        except Exception as e:
            self.get_logger().error(f"Falha na thread do mouse: {e}")

    # NOVO: thread de escuta do teclado -- tecla ` (crase) dispara o mesmo workflow
    def keyboard_listener_thread(self):
        keyboard_path = self.find_keyboard_device()
        if keyboard_path is None:
            self.get_logger().error("Nenhum teclado encontrado para escuta da tecla `.")
            return

        try:
            device = InputDevice(keyboard_path)
            self.get_logger().info(f"Monitor de interrupções ativo no teclado: {device.name}")

            for event in device.read_loop():
                if event.type != ecodes.EV_KEY: continue
                if event.value != 1: continue  # Somente pressionamento (Pushed)

                if event.code == ecodes.KEY_GRAVE:
                    if self.is_executing:
                        print("[AVISO] Workflow já está em execução! Tecla ` ignorada por segurança.")
                        continue

                    print("\n💥 [INTERRUPÇÃO] Tecla ` detectada! Iniciando automação...")
                    threading.Thread(target=self.execute_workflow, daemon=True).start()

        except Exception as e:
            self.get_logger().error(f"Falha na thread do teclado: {e}")

    def call_dashboard_sync(self, client, command_name):
        """ Envia comandos de Play/Stop de forma segura entre threads """
        self.get_logger().info(f"Enviando comando de {command_name} para o Teach Pendant...")
        req = Trigger.Request()
        
        future = client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def switch_controllers_sync(self, start_controllers, stop_controllers):
        req = SwitchController.Request()
        req.start_controllers = start_controllers
        req.stop_controllers = stop_controllers
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=1, nanosec=0)
        
        future = self.switch_ctrl_client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def send_trajectory_cmd(self, target_joints, duration_seconds):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(j) for j in target_joints]
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        point.time_from_start = Duration(
            sec=int(duration_seconds), 
            nanosec=int((duration_seconds - int(duration_seconds)) * 1e9)
        )
        
        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(msg)

    def execute_workflow(self):
        self.is_executing = True
        
        # 1. Aguarda telemetria inicial
        if self.current_joints_reading is None:
            self.get_logger().info("Aguardando telemetria inicial do UR5...")
            while self.current_joints_reading is None:
                time.sleep(0.1)

        # ── PASSO DE SEGURANÇA DA GARRA ──
        # Checa e abre a garra via client SetIO antes do movimento
        self.open_gripper_if_closed()
        time.sleep(0.5)
        
        ponto_2 = [0.0, -1.5708, 0.0, -1.5708, 1.5708, 0.0] # Ponto Alvo Final

        # ETAPA 1: Resetando Teach Pendant
        self.get_logger().info("Passo 1: Resetando Teach Pendant com comando STOP...")
        self.call_dashboard_sync(self.stop_client, "STOP")
        time.sleep(0.5)

        # ETAPA 2: Dá PLAY no Teach Pendant
        self.get_logger().info("Passo 2: Ativando Polyscope via comando PLAY...")
        self.call_dashboard_sync(self.play_client, "PLAY")
        time.sleep(1.0)

        # ETAPA 3: Ativa o controlador de trajetórias
        # self.switch_controllers_sync(
        #     start_controllers=['scaled_joint_trajectory_controller'],
        #     stop_controllers=['forward_position_controller']
        # )

        # ETAPA 4: Move até o ponto alvo
        self.get_logger().info(f"Passo 4: Movendo para Ponto Final {ponto_2}...")
        self.send_trajectory_cmd(ponto_2, 3.0)
        time.sleep(3.2)

        # ETAPA 5: Restaura o controlador padrão
        # self.switch_controllers_sync(
        #     start_controllers=['forward_position_controller'],
        #     stop_controllers=['scaled_joint_trajectory_controller']
        # )

        # ETAPA 6: Reseta Teach Pendant (STOP depois PLAY)
        self.call_dashboard_sync(self.stop_client, "STOP")
        time.sleep(0.5)

        self.get_logger().info("Reativando Polyscope via comando PLAY...")
        self.call_dashboard_sync(self.play_client, "PLAY")
        
        self.get_logger().info("--- 🔄 Ciclo Concluído! Pronto para o próximo clique lateral ---")
        
        self.is_executing = False


def main(args=None):
    rclpy.init(args=args)
    client = UR5AutonomousClient()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(client)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[SIGNAL] Finalizando nó autónomo de forma limpa...")
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()