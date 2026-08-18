#!/usr/bin/env python3
"""
Versao AO VIVO (tempo real) do object_position_yolo_world_depth.py --
Publica e exibe SOMENTE a posicao no referencial MUNDO (World / ArUco)
para calibracao da camera.

Uso:
  python live_object_position_viewer.py --classes cup mug bowl
  (pressione 'q' na janela para sair)
"""

import argparse
import json
import sys

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import message_filters

from ultralytics import YOLOWorld

CAMERA_RGB_TOPIC = "/zed/zed_node/rgb/color/rect/image"
CAMERA_DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
CAMERA_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
CAMERA_POSE_PATH = "camera_pose.json"

YOLO_WORLD_MODEL = "yolov8s-world.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_RESIZE_WIDTH = 640
DETECTION_EVERY_N_FRAMES = 3


def sample_depth_robust(depth_image, bbox_xyxy, patch_size: int = 5):
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    half = patch_size // 2
    patch = depth_image[
        max(0, cy - half): cy + half + 1,
        max(0, cx - half): cx + half + 1,
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def backproject_to_camera_frame(pixel_uv, depth_m, camera_matrix):
    u, v = pixel_uv
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    return np.array([x_cam, y_cam, depth_m])


def camera_frame_to_world(point_camera, camera_rotation_world, camera_position_world):
    return camera_position_world + camera_rotation_world @ point_camera


class LiveObjectPositionViewer(Node):
    def __init__(self, class_names, camera_rotation_world, camera_position_world):
        super().__init__("live_object_position_viewer")

        self.class_names = class_names
        self.camera_rotation_world = camera_rotation_world
        self.camera_position_world = camera_position_world

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_matrix = None

        print(f"Carregando modelo YOLO-World ({YOLO_WORLD_MODEL}) em '{DEVICE}'...")
        self.model = YOLOWorld(YOLO_WORLD_MODEL)
        self.model.to(DEVICE)
        self.model.set_classes(class_names)

        self._frame_counter = 0
        self._last_overlay = None

        rgb_sub = message_filters.Subscriber(self, Image, CAMERA_RGB_TOPIC)
        depth_sub = message_filters.Subscriber(self, Image, CAMERA_DEPTH_TOPIC)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self._synced_callback)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._info_callback, 10)

        self.position_publisher = self.create_publisher(
            Point, '/object_position_yolo_world', 1
        )

        self.get_logger().info("Aguardando dados da camera...")

    def _synced_callback(self, rgb_msg, depth_msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    def _info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"camera_info capturado! Resolucao: {msg.width}x{msg.height}")

    def process_and_draw(self):
        if self.latest_rgb is None or self.latest_depth is None or self.camera_matrix is None:
            return None

        self._frame_counter += 1
        if self._frame_counter % DETECTION_EVERY_N_FRAMES != 0 and self._last_overlay is not None:
            return self._last_overlay

        rgb = self.latest_rgb
        depth = self.latest_depth
        overlay = rgb.copy()

        orig_h, orig_w = rgb.shape[:2]
        scale = DETECTION_RESIZE_WIDTH / orig_w
        resized = cv2.resize(rgb, (DETECTION_RESIZE_WIDTH, int(orig_h * scale)))

        results = self.model.predict(resized, verbose=False, device=DEVICE)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() / scale
            bbox_xyxy = (x1, y1, x2, y2)
            center_u = (x1 + x2) / 2.0
            center_v = (y1 + y2) / 2.0

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            depth_m = sample_depth_robust(depth, bbox_xyxy)

            if depth_m is None:
                text = f"{cls_name} {conf:.2f} | Sem profundidade"
            else:
                point_camera = backproject_to_camera_frame((center_u, center_v), depth_m, self.camera_matrix)
                point_world = camera_frame_to_world(
                    point_camera, self.camera_rotation_world, self.camera_position_world
                )

                text = (f"{cls_name} {conf:.2f} | {depth_m:.2f}m | "
                        f"world=({point_world[0]:.3f}, {point_world[1]:.3f}, {point_world[2]:.3f})")

                # Publica a posicao no referencial MUNDO (World / ArUco)
                point_msg = Point()
                point_msg.x = float(point_world[0])
                point_msg.y = float(point_world[1])
                point_msg.z = float(point_world[2])
                self.position_publisher.publish(point_msg)

                print(f"[{cls_name}] Posicao World: X={point_world[0]:.3f}, Y={point_world[1]:.3f}, Z={point_world[2]:.3f}")

            cv2.putText(overlay, text, (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        self._last_overlay = overlay
        return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                         help="Lista de classes a detectar (ex: --classes cup mug bowl)")
    parser.add_argument("--pose-path", type=str, default=CAMERA_POSE_PATH)
    args = parser.parse_args()

    try:
        with open(args.pose_path) as f:
            pose_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERRO] {args.pose_path} nao encontrado -- rode calibrate_camera_pose.py primeiro.")
        sys.exit(1)

    camera_rotation_world = np.array(pose_data["camera_rotation_matrix_world"])
    camera_position_world = np.array(pose_data["camera_position_world"])

    rclpy.init()
    node = LiveObjectPositionViewer(
        args.classes, camera_rotation_world, camera_position_world
    )

    print("\nJanela ao vivo iniciando -- pressione 'q' na janela para sair.\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            display_frame = node.process_and_draw()
            if display_frame is not None:
                cv2.imshow("Deteccao + Posicao World 3D", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()