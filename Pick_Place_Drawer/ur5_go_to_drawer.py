#!/usr/bin/env python3
"""
UR5 Autonomous Client (Triggered by SmolVLA Gripper Close)
ROS2 Humble | Scaled Joint Trajectory Controller | SetIO (Pin 16)

Executa a pose intermediaria, reproduz o dataset sub-amostrado e
encerra o processo/no ROS2 automaticamente ao finalizar.
"""

import json
import os
import sys
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ur_msgs.srv import SetIO
from std_srvs.srv import Trigger
from rclpy.qos import qos_profile_sensor_data


UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Sub-amostragem: pega 1 ponto a cada STEP_DECIMATION para nao estourar o buffer (ex: 100Hz -> 20Hz)
STEP_DECIMATION = 5


class UR5AutonomousClient(Node):

    def __init__(self):
        super().__init__("ur5_autonomous_client")
        self.get_logger().info("Inicializando Cliente Autónomo via Scaled Joint Trajectory Controller...")

        self._q_current = None
        self._js_ok = False

        self.is_executing = False
        self.gripper_pin = 16
        self.last_sent_gripper_state = None


        self.intermediate_pose_drawer_by_name = {
            "shoulder_lift_joint": -1.9332926909076136,
            "elbow_joint": -0.00021535554994756012,
            "wrist_1_joint": -1.6716049353228968,
            "wrist_2_joint": 1.5370337963104248,
            "wrist_3_joint": -3.5587941304981996e-05,
            "shoulder_pan_joint": 0.01216848287731409,
        }

        self.home_pose_by_name = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 0.0,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": 1.5708,
            "wrist_3_joint": 0.0,
        }

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10
        )

        self.io_client = self.create_client(SetIO, "/io_and_status_controller/set_io")
        self.get_logger().info("Aguardando serviço de I/O do robô...")
        while not self.io_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.get_logger().info("Serviço de I/O não disponível, tentando novamente...")

        self.play_client = self.create_client(Trigger, "/dashboard_client/play")
        self.stop_client = self.create_client(Trigger, "/dashboard_client/stop")

        self.dataset_paths = {"drawer": "/home/ziqi/pre_ws/jittering_dataset/drawer.json"}
        self.create_subscription(
            JointState,
            "/joint_states",
            self._js_cb,
            qos_profile=qos_profile_sensor_data
        )

    def _js_cb(self, msg: JointState):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q_current = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _pose_dict_to_ordered_array(self, pose_dict: dict) -> list:
        return [pose_dict[name] for name in UR5_JOINT_NAMES]

    def _publish_gripper_command(self, gripper_state: int):
        req = SetIO.Request()
        req.fun = 1
        req.pin = self.gripper_pin
        req.state = float(gripper_state)
        self.io_client.call_async(req)

    def call_dashboard_sync(self, client, command_name):
        self.get_logger().info(f"Enviando comando de {command_name} para o Teach Pendant...")
        req = Trigger.Request()
        future = client.call_async(req)
        while not future.done():
            time.sleep(0.05)
        return future.result()

    def send_single_target_trajectory(self, target_joints: list, duration_seconds: float):
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(j) for j in target_joints]
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        
        point.time_from_start = Duration(
            sec=int(duration_seconds),
            nanosec=int((duration_seconds - int(duration_seconds)) * 1e9)
        )

        msg.points.append(point)
        msg.header.stamp = rclpy.time.Time().to_msg()
        self.trajectory_pub.publish(msg)

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
            point.time_from_start = Duration(
                sec=int(t_rel),
                nanosec=int((t_rel - int(t_rel)) * 1e9)
            )
            msg.points.append(point)

        msg.header.stamp = rclpy.time.Time().to_msg()
        self.trajectory_pub.publish(msg)

        start_time = time.time()
        for frame in frames:
            target_time = frame["timestamp"] - first_timestamp
            sleep_needed = target_time - (time.time() - start_time)
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            gripper_ref = int(frame.get("gripper_io", 0))
            if gripper_ref != self.last_sent_gripper_state:
                self._publish_gripper_command(gripper_ref)
                self.last_sent_gripper_state = gripper_ref

        total_duration = frames[-1]["timestamp"] - first_timestamp
        time_left = total_duration - (time.time() - start_time)
        if time_left > 0:
            time.sleep(time_left)

        self.get_logger().info("Reprodução da trajetória concluída!")

    def execute_workflow(self):
        self.is_executing = True
        
        intermediate_pose_dict = self.intermediate_pose_drawer_by_name
        target_key = "drawer"
        
        dataset_path = self.dataset_paths[target_key]

        # 1. Movimento para Pose Intermediária
        intermediate_point = self._pose_dict_to_ordered_array(intermediate_pose_dict)
        self.get_logger().info(f"Movendo para pose INTERMEDIÁRIA ({target_key})...")
        self.send_single_target_trajectory(intermediate_point, duration_seconds=4.0)
        time.sleep(4.2)

        # 2. Reprodução do Dataset Gravado
        self.get_logger().info(f"Reproduzindo dataset ({target_key})...")
        self.play_dataset_as_trajectory(dataset_path)

        home_point = self._pose_dict_to_ordered_array(self.home_pose_by_name)
        self.get_logger().info(f"Movendo para HOME ({target_key})...")
        self.send_single_target_trajectory(home_point, duration_seconds=4.0)
        time.sleep(4.2)

        # 3. Comandos de Dashboard (Stop / Play no Polyscope)
        if self.stop_client.wait_for_service(timeout_sec=1.0):
            self.call_dashboard_sync(self.stop_client, "STOP")
            time.sleep(0.5)

        if self.play_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Reativando Polyscope via comando PLAY...")
            self.call_dashboard_sync(self.play_client, "PLAY")

        self.get_logger().info("--- 🔄 Ciclo Concluído! Encerrando o nó... ---")
        
        # SINAL DE ENCERRAMENTO: Força o desligamento do nó ROS2 e da aplicação
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = UR5AutonomousClient()
    node.execute_workflow()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        print("\n[FIM] Código encerrado com sucesso.")


if __name__ == "__main__":
    main()