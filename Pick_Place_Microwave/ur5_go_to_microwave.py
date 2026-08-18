#!/usr/bin/env python3
"""
UR5 Autonomous Trajectory Client (Enter Triggered Version)
ROS2 Humble | Dashboard Control (Play/Stop) | Enter Interrupt

Aguardando ENTER no terminal para disparar:
[STOP do Teach Pendant] -> [PLAY] -> [Pose Intermediaria] ->
[Pose Microondas] -> [STOP de novo]

NOTA sobre a pose cartesiana do microondas (MICROWAVE_CARTESIAN_REF):
guardada como referencia para uso FUTURO -- servira' de "ponto base"
para calcular a altura ajustada quando o objeto atual for de tamanho
diferente do copo usado para definir essa pose original. A logica de
AJUSTE de altura em si ainda NAO esta' implementada aqui -- so' as
poses fixas padrao estao sendo usadas por enquanto.
"""

import os
import sys
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger
import time
import threading


class UR5AutonomousClient(Node):
    def __init__(self):
        super().__init__('ur5_autonomous_client')
        self.get_logger().info('Inicializando Cliente Autónomo engatilhado por ENTER...')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # ── Poses fixas -- mapeadas por NOME (a ordem fornecida era
        # diferente do padrao acima, shoulder_pan por ultimo) ──────────
        self.intermediate_pose_by_name = {
            "shoulder_lift_joint": -0.9958761374102991,
            "elbow_joint":         -1.3366435209857386,
            "wrist_1_joint":       -0.8589761892901819,
            "wrist_2_joint":        1.6337573528289795,
            "wrist_3_joint":        0.00022770027862861753,
            "shoulder_pan_joint":  -0.045229736958638966,
        }

        self.second_intermediate_pose_by_name = {
            "shoulder_lift_joint": -1.0318563620196741,
            "elbow_joint":         -1.4548152128802698,
            "wrist_1_joint":       -0.6828368345843714,
            "wrist_2_joint":        1.7203160524368286,
            "wrist_3_joint":        0.0,
            "shoulder_pan_joint":   0.8773535490036011,
        }

        self.third_intermediate_pose_by_name = {
            "shoulder_lift_joint": -1.496312443410055,
            "elbow_joint":         -1.3220637480365198,
            "wrist_1_joint":       -0.29568463960756475,
            "wrist_2_joint":        0.8530086874961853,
            "wrist_3_joint":        0.002960478188470006,
            "shoulder_pan_joint":   1.0775395631790161,
        }


        self.microwave_pose_cup_by_name = {
            "shoulder_lift_joint": -1.766700569783346,
            "elbow_joint":         -1.0405243078814905,
            "wrist_1_joint":       -0.5025571028338831,
            "wrist_2_joint":        0.9978178143501282,
            "wrist_3_joint":        0.00016777915880084038,
            "shoulder_pan_joint":   0.8596329689025879,
        }

        self.microwave_pose_mug_by_name = {
            "shoulder_lift_joint": -1.8656962553607386,
            "elbow_joint":         -0.9146130720721644,
            "wrist_1_joint":       -0.36239463487734014,
            "wrist_2_joint":        0.8221152424812317,
            "wrist_3_joint":        0.009816204197704792,
            "shoulder_pan_joint":   0.8966574668884277,
        }

        self.microwave_out_pose_by_name = {
            "shoulder_lift_joint": -1.2018223921405237,
            "elbow_joint":         -1.3745749632464808,
            "wrist_1_joint":       -0.41111928621401006,
            "wrist_2_joint":        0.8221871256828308,
            "wrist_3_joint":        0.007718590088188648,
            "shoulder_pan_joint":   1.121093511581421,
        }



        # Pose cartesiana de referencia do microondas -- guardada para
        # uso FUTURO (ajuste de altura conforme tamanho do objeto),
        # ainda NAO implementado
        self.microwave_cartesian_ref = {
            "position": {
                "x": 0.6262977394302895,
                "y": 0.375412092172178,
                "z": 0.6844053859960908,
            },
            "orientation": {
                "x": -0.597365654524012,
                "y": 0.4173686129408671,
                "z": -0.40628463548964283,
                "w": 0.5512626512761077,
            },
        }

        self.microwave_cartesian_ref = {
            "position": {
                "x": 0.6233525927037412,
                "y": 0.34903031625013614,
                "z": 0.7272065926207144,
            },
            "orientation": {
                "x": -0.5342752507295292,
                "y": 0.46700217134338445,
                "z": -0.4551199279111491,
                "w": 0.5378891889939907,
            },
        }

        self.is_executing = False

        # ── CLIENTS DOS SERVIÇOS DO DASHBOARD (TEACH PENDANT) ──
        self.play_client = self.create_client(Trigger, '/dashboard_client/play')
        self.stop_client = self.create_client(Trigger, '/dashboard_client/stop')
        self.switch_ctrl_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        self.get_logger().info('Conectando aos serviços do robô...')
        self.play_client.wait_for_service()
        self.stop_client.wait_for_service()
        self.switch_ctrl_client.wait_for_service()

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.current_joints_reading = None
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        print("\n" + "*"*60)
        print(" 🔘 APERTE ENTER NO TERMINAL PARA DISPARAR ")
        print(" (intermediaria -> microondas) ")
        print("*"*60 + "\n")
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener_thread, daemon=True)
        self.keyboard_thread.start()

    def joint_state_callback(self, msg):
        try:
            joints_dict = dict(zip(msg.name, msg.position))
            self.current_joints_reading = [joints_dict[name] for name in self.joint_names]
        except KeyError:
            pass

    def keyboard_listener_thread(self):
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break

            if self.is_executing:
                print("[AVISO] Workflow já está em execução! Enter ignorado por segurança.")
                continue

            print("\n💥 [ENTER] Disparando automação (intermediaria -> microondas)...")
            threading.Thread(target=self.execute_workflow, daemon=True).start()

    def call_dashboard_sync(self, client, command_name):
        """ Envia comandos de Play/Stop de forma segura entre threads """
        self.get_logger().info(f"Enviando comando de {command_name} para o Teach Pendant...")
        req = Trigger.Request()

        future = client.call_async(req)
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

    def _pose_dict_to_ordered_array(self, pose_dict: dict) -> list:
        """Converte o dict {nome_junta: valor} para lista na ordem de
        self.joint_names -- evita depender da ordem originalmente
        fornecida (que tinha shoulder_pan por ultimo)."""
        return [pose_dict[name] for name in self.joint_names]

    def execute_workflow(self):
        self.is_executing = True

        if self.current_joints_reading is None:
            self.get_logger().info("Aguardando telemetria inicial do UR5...")
            while self.current_joints_reading is None:
                time.sleep(0.1)

        ponto_intermediario = self._pose_dict_to_ordered_array(self.intermediate_pose_by_name)
        ponto_segundo_intemediario = self._pose_dict_to_ordered_array(self.second_intermediate_pose_by_name)
        ponto_terceiro_intemediario = self._pose_dict_to_ordered_array(self.third_intermediate_pose_by_name)
        ponto_microondas_cup = self._pose_dict_to_ordered_array(self.microwave_pose_cup_by_name)
        ponto_microondas_mug = self._pose_dict_to_ordered_array(self.microwave_pose_mug_by_name)
        ponto_microondas_out = self._pose_dict_to_ordered_array(self.microwave_out_pose_by_name)

        self.get_logger().info(f"Passo 3: Movendo para pose INTERMEDIARIA {ponto_intermediario}...")
        self.send_trajectory_cmd(ponto_intermediario, 4.0)
        time.sleep(4.2)

        self.get_logger().info(f"Passo 4: Movendo para pose INTERMEDIARIA {ponto_segundo_intemediario}...")
        self.send_trajectory_cmd(ponto_segundo_intemediario, 4.0)
        time.sleep(4.2)

        self.get_logger().info(f"Passo 4: Movendo para pose INTERMEDIARIA {ponto_segundo_intemediario}...")
        self.send_trajectory_cmd(ponto_terceiro_intemediario, 4.0)
        time.sleep(4.2)

        self.get_logger().info(f"Passo 5: Movendo para pose do MICROONDAS {ponto_microondas_mug}...")
        self.send_trajectory_cmd(ponto_microondas_mug, 4.0)
        time.sleep(4.2)

        # self.get_logger().info(f"Passo 5: Movendo para pose do MICROONDAS {ponto_microondas_out}...")
        # self.send_trajectory_cmd(ponto_microondas_out, 4.0)
        # time.sleep(4.2)

        self.call_dashboard_sync(self.stop_client, "STOP")
        time.sleep(0.5)

        self.get_logger().info("Reativando Polyscope via comando PLAY...")
        self.call_dashboard_sync(self.play_client, "PLAY")

        self.get_logger().info("--- 🔄 Ciclo Concluído! Pronto para o próximo ENTER ---")

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