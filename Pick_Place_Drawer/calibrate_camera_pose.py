#!/usr/bin/env python3
"""
Calibra a pose (posicao + orientacao) de uma camera em relacao ao
mundo, usando 4 marcadores ArUco em posicoes 3D conhecidas.

Captura a imagem DIRETO do topico ROS2 da camera -- conecta no
topico, espera o primeiro frame chegar, salva em disco (para
conferencia visual) e ja' usa esse frame para a calibracao.

Uso:
  1. Defina POSICOES_3D_MARCADORES abaixo com as posicoes REAIS
     medidas de cada marcador (em metros, no sistema de coordenadas
     que voce escolher como referencia).
  2. Garanta que os 4 marcadores estejam visiveis no campo de visao
     da camera (fisicamente, na hora de rodar o script).
  3. Rode este script -- ele captura um frame do topico ROS2, salva
     em disco, detecta os marcadores, resolve o PnP, e salva a pose
     resultante em camera_pose.json.
  4. Depois (dias/semanas depois), se os marcadores continuarem nos
     MESMOS lugares fisicos, rode de novo para recuperar a MESMA pose
     calibrada -- nao precisa remedir nada.

Alternativa (sem ROS2, usando uma foto ja salva):
  python calibrate_camera_pose.py --image caminho/para/foto.png
"""

import cv2
import numpy as np
import json
import argparse
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO -- AJUSTE AQUI
# ══════════════════════════════════════════════════════════════════════════════

# Tamanho real de cada marcador ArUco, em metros (lado do quadrado)
MARKER_SIZE_M = 0.08

# Dicionario ArUco usado (4x4 com 50 IDs possiveis -- comum e robusto)
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Topico ROS2 da camera a calibrar
CAMERA_TOPIC = "/zed/zed_node/rgb/color/rect/image"

# Onde salvar o frame capturado do topico, para conferencia visual
# posterior (confirmar que os marcadores realmente aparecem nitidos)
CAPTURED_FRAME_PATH = "captured_calibration_frame.png"

# Posicoes 3D REAIS medidas de cada marcador (em metros), no sistema
# de coordenadas do MUNDO que voce escolheu como referencia.
#
# IMPORTANTE: essas sao as coordenadas do CENTRO de cada marcador.
SIDE_SIZE = 0.04
POSICOES_3D_MARCADORES = {
    0: np.array([0.532 + SIDE_SIZE, 0.6695 - SIDE_SIZE, 0.0]),
    1: np.array([0.53 + SIDE_SIZE, 0.0955 - SIDE_SIZE, 0.0]),
    2: np.array([0.006 + SIDE_SIZE, 0.082 - SIDE_SIZE, 0.0]),
    3: np.array([0.006 + SIDE_SIZE, 0.668 - SIDE_SIZE, 0.0]),
}

# Matriz de intrinsecos REAL da camera ZED, extraida do topico
# /camera_info.
#
# ATENCAO: esses valores correspondem a RESOLUCAO NATIVA reportada
# pelo camera_info (1920x1080). O frame usado para calibracao
# PRECISA estar nessa MESMA resolucao para esses intrinsecos serem
# validos.
CAMERA_MATRIX = np.array([
    [1507.55419921875,               0.0, 954.78466796875],
    [              0.0, 1507.55419921875, 556.004638671875],
    [              0.0,               0.0,             1.0],
], dtype=np.float64)

NATIVE_IMAGE_WIDTH = 1920
NATIVE_IMAGE_HEIGHT = 1080

DIST_COEFFS = np.zeros(5, dtype=np.float64)

OUTPUT_PATH = "camera_pose.json"


# ══════════════════════════════════════════════════════════════════════════════
# Captura de frame via ROS2
# ══════════════════════════════════════════════════════════════════════════════

class SingleFrameCapture(Node):
    """No minimo que se inscreve no topico, guarda o PRIMEIRO frame
    que chegar, e sinaliza que terminou -- nao fica rodando continuamente."""

    def __init__(self, topic: str):
        super().__init__("calibration_frame_capture")
        self.bridge = CvBridge()
        self.captured_image = None

        self.subscription = self.create_subscription(
            Image, topic, self._callback, 10
        )
        self.get_logger().info(f"Aguardando frame em: {topic}")

    def _callback(self, msg: Image):
        if self.captured_image is not None:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.captured_image = cv_img
        self.get_logger().info("Frame capturado!")


def capture_frame_from_topic(topic: str, timeout_s: float = 10.0):
    rclpy.init()
    node = SingleFrameCapture(topic)

    import time
    t_start = time.time()
    while node.captured_image is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t_start > timeout_s:
            node.get_logger().error(
                f"Timeout de {timeout_s}s esperando frame em {topic} -- "
                f"verifique se o topico esta publicando (ros2 topic hz {topic})."
            )
            node.destroy_node()
            rclpy.shutdown()
            return None

    image = node.captured_image
    node.destroy_node()
    rclpy.shutdown()
    return image


# ══════════════════════════════════════════════════════════════════════════════
# Deteccao e PnP
# ══════════════════════════════════════════════════════════════════════════════

def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    detected = {}
    for i, marker_id in enumerate(ids.flatten()):
        detected[int(marker_id)] = corners[i][0]

    return detected


def solve_camera_pose(detected_markers, known_positions_3d):
    """
    Resolve a pose da camera via PnP.

    IMPORTANTE (corrigido): cv2.solvePnP retorna (rvec, tvec) que
    representam a transformacao MUNDO -> CAMERA, ou seja, tvec e' a
    posicao da ORIGEM DO MUNDO vista pela camera -- NAO e' a posicao
    da camera no mundo diretamente (confusao classica com solvePnP).
    Para obter a posicao REAL da camera no referencial do mundo, e'
    preciso inverter a transformacao:

        posicao_camera_no_mundo = -R^T @ tvec
    """
    object_points = []
    image_points = []

    for marker_id, corners_2d in detected_markers.items():
        if marker_id not in known_positions_3d:
            print(f"[AVISO] Marcador ID {marker_id} detectado mas sem posicao 3D conhecida -- ignorando.")
            continue

        center_2d = corners_2d.mean(axis=0)
        object_points.append(known_positions_3d[marker_id])
        image_points.append(center_2d)

    if len(object_points) < 4:
        raise ValueError(
            f"Apenas {len(object_points)} marcadores validos detectados -- "
            f"sao necessarios pelo menos 4 para resolver o PnP com robustez."
        )

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, CAMERA_MATRIX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP falhou -- verifique os pontos de entrada.")

    rotation_matrix, _ = cv2.Rodrigues(rvec)  # R (mundo -> camera)

    # Inverte a transformacao para obter a posicao REAL da camera no
    # referencial do mundo
    camera_position_world = -rotation_matrix.T @ tvec.flatten()
    camera_rotation_world = rotation_matrix.T

    return camera_rotation_world, camera_position_world, rvec.flatten()


def run_calibration(image):
    img_h, img_w = image.shape[:2]
    if (img_w, img_h) != (NATIVE_IMAGE_WIDTH, NATIVE_IMAGE_HEIGHT):
        print(f"[AVISO] O frame tem resolucao {img_w}x{img_h}, mas os intrinsecos "
              f"(CAMERA_MATRIX) foram calibrados para {NATIVE_IMAGE_WIDTH}x"
              f"{NATIVE_IMAGE_HEIGHT} -- os resultados do PnP podem ficar "
              f"incorretos.")

    print("Detectando marcadores ArUco...")
    detected = detect_markers(image)
    print(f"Marcadores detectados: {list(detected.keys())}")

    if len(detected) < 4:
        print(f"[AVISO] So' {len(detected)} marcadores detectados -- "
              f"idealmente precisa dos 4. Verifique iluminacao/angulo/foco.")

    rotation_matrix, camera_position, rvec = solve_camera_pose(detected, POSICOES_3D_MARCADORES)

    print("\n=== POSE DA CAMERA (posicao REAL no referencial do mundo, corrigido) ===")
    print(f"Posicao da camera (x, y, z): {camera_position}")
    print(f"Rotacao da camera (matriz, mundo->camera invertida):\n{rotation_matrix}")

    result = {
        "camera_position_world": camera_position.tolist(),
        "camera_rotation_matrix_world": rotation_matrix.tolist(),
        "rotation_vector_rodrigues_raw_pnp": rvec.tolist(),
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "dist_coeffs": DIST_COEFFS.tolist(),
        "marker_positions_used": {str(k): v.tolist() for k, v in POSICOES_3D_MARCADORES.items()},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nPose salva em: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="(Opcional) caminho para uma foto ja salva, em vez "
                              "de capturar do topico ROS2")
    parser.add_argument("--topic", type=str, default=CAMERA_TOPIC,
                         help=f"Topico ROS2 da camera (padrao: {CAMERA_TOPIC})")
    args = parser.parse_args()

    if args.image is not None:
        print(f"Usando imagem de arquivo: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERRO] Nao foi possivel abrir a imagem: {args.image}")
            sys.exit(1)
    else:
        print(f"Capturando frame do topico ROS2: {args.topic}")
        image = capture_frame_from_topic(args.topic)
        if image is None:
            print("[ERRO] Falha ao capturar frame do topico.")
            sys.exit(1)

        cv2.imwrite(CAPTURED_FRAME_PATH, image)
        print(f"Frame capturado salvo em: {CAPTURED_FRAME_PATH} "
              f"(confira visualmente se os marcadores aparecem nitidos)")

    run_calibration(image)

    print("\nPara replicar depois: basta rodar este script de novo (com o "
          "topico ativo), DESDE QUE os marcadores continuem nas MESMAS "
          "posicoes fisicas.")


if __name__ == "__main__":
    main()