#!/usr/bin/env python3
"""
Calibra a transformacao (rotacao + translacao) entre a base do robo
e a origem da mesa, usando MULTIPLOS pontos e o algoritmo de Kabsch.

Recebe a posicao conhecida de cada ponto automaticamente do topico
/object_position_yolo_world (publicado pelo
live_object_position_viewer.py, via deteccao YOLO-World + profundidade).

NOVO FLUXO por ponto:
  1. Aponte a camera para um objeto/marcador de referencia (a
     deteccao calcula a posicao dele no referencial da MESA
     automaticamente)
  2. Aperte ENTER -- salva essa posicao do MUNDO (a mais recente
     recebida do topico)
  3. Toque fisicamente a garra NESSE MESMO objeto
  4. Aperte ENTER de novo -- salva a posicao que o ROBO reporta
     (via /tcp_pose_broadcaster/pose) nesse instante
  5. Repete para 3+ pontos, depois resolve a transformacao via Kabsch

Uso:
  python calibrate_robot_to_table_multipoint.py
"""

import sys
import json
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point

from evdev import InputDevice, list_devices, ecodes

ROBOT_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
OBJECT_POSITION_TOPIC = "/object_position_yolo_world"
OUTPUT_PATH = "robot_to_table_transform.json"


def find_logitech_mouse():
    for path in list_devices():
        try:
            dev = InputDevice(path)
            if "logitech" not in dev.name.lower():
                continue
            caps = dev.capabilities()
            if ecodes.EV_KEY not in caps:
                continue
            keys = caps[ecodes.EV_KEY]
            if ecodes.BTN_SIDE in keys or ecodes.BTN_EXTRA in keys:
                return path
        except Exception:
            continue
    return None


def wait_for_mouse_button(device, target_button, other_button=None):
    """
    Bloqueia ate' o BOTAO ALVO ser pressionado no mouse. Se
    other_button for fornecido e for pressionado ANTES, retorna
    False (sinalizando "outro botao" em vez do alvo) -- usado para
    permitir cancelar/terminar via um botao diferente.
    """
    for event in device.read_loop():
        if event.type != ecodes.EV_KEY:
            continue
        if event.value != 1:  # so' no momento do press
            continue
        if event.code == target_button:
            return True
        if other_button is not None and event.code == other_button:
            return False


class MultiPointCalibrator(Node):
    def __init__(self):
        super().__init__("calibrate_robot_to_table_multipoint")
        self.latest_robot_pose = None
        self.latest_world_position = None

        self.robot_pose_sub = self.create_subscription(
            PoseStamped, ROBOT_POSE_TOPIC, self._robot_pose_callback, 10
        )
        self.object_position_sub = self.create_subscription(
            Point, OBJECT_POSITION_TOPIC, self._object_position_callback, 10
        )

        self.get_logger().info(
            f"Escutando {ROBOT_POSE_TOPIC} e {OBJECT_POSITION_TOPIC}..."
        )

    def _robot_pose_callback(self, msg: PoseStamped):
        # CORRIGIDO: PoseStamped tem .pose.position (nao
        # .transform.translation, que era da estrutura de
        # TransformStamped -- tipo de mensagem diferente)
        p = msg.pose.position
        self.latest_robot_pose = np.array([p.x, p.y, p.z])

    def _object_position_callback(self, msg: Point):
        self.latest_world_position = np.array([msg.x, msg.y, msg.z])


def kabsch_algorithm(P_table: np.ndarray, P_robot: np.ndarray):
    """
    Resolve a transformacao RIGIDA otima (rotacao R + translacao t)
    tal que: P_robot[i] ~= R @ P_table[i] + t, minimizando o erro
    quadratico total sobre TODOS os pontos simultaneamente.
    """
    assert P_table.shape == P_robot.shape
    assert P_table.shape[0] >= 3, "Precisa de pelo menos 3 pontos para o Kabsch."

    centroid_table = P_table.mean(axis=0)
    centroid_robot = P_robot.mean(axis=0)

    P_table_centered = P_table - centroid_table
    P_robot_centered = P_robot - centroid_robot

    H = P_table_centered.T @ P_robot_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])

    R = Vt.T @ D @ U.T
    t = centroid_robot - R @ centroid_table

    P_predicted = (R @ P_table.T).T + t
    residuals = np.linalg.norm(P_predicted - P_robot, axis=1)
    rms_error = np.sqrt(np.mean(residuals**2))

    return R, t, rms_error, residuals


def wait_for_click_then_capture(node, mouse_device, get_value_fn, clear_value_fn, prompt: str, timeout_s: float = 5.0):
    """Espera o clique do BOTAO EXTRA do mouse, depois captura o
    valor mais recente da funcao fornecida (com timeout de
    seguranca caso o topico ainda nao tenha publicado nada).

    CORRIGIDO: limpa o valor ANTES de esperar o clique -- sem isso,
    uma leitura ANTIGA (de antes do robo se mover fisicamente para o
    novo ponto) poderia ser retornada em vez de uma leitura FRESCA
    capturada de verdade no momento do clique, causando erros
    grandes por desalinhamento temporal entre a posicao do mundo e
    a posicao do robo.
    """
    clear_value_fn()
    print(prompt)
    wait_for_mouse_button(mouse_device, ecodes.BTN_EXTRA)

    t_start = time.time()
    while get_value_fn() is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t_start > timeout_s:
            print(f"[ERRO] Timeout esperando dado -- topico esta publicando?")
            return None

    rclpy.spin_once(node, timeout_sec=0.05)
    return get_value_fn()


def main():
    rclpy.init()
    node = MultiPointCalibrator()

    mouse_path = find_logitech_mouse()
    if mouse_path is None:
        print("[ERRO] Nenhum mouse Logitech com botoes laterais encontrado.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    mouse_device = InputDevice(mouse_path)
    print(f"Monitorando mouse: {mouse_device.name}")

    print("\n=== CALIBRACAO MULTI-PONTO: base do robo <-> mesa ===")
    print("Para cada ponto:")
    print("  1. Aponte a camera para um objeto de referencia, clique no BOTAO EXTRA")
    print("     (salva a posicao do MUNDO, vinda da deteccao)")
    print("  2. Toque a garra fisicamente NESSE MESMO objeto, clique no BOTAO EXTRA de novo")
    print("     (salva a posicao que o ROBO reporta)")
    print("Repita para 3+ pontos (recomendado 4+, bem espalhados).")
    print("Clique no BOTAO LATERAL (BTN_SIDE) no lugar do 1o clique de um novo ponto "
          "para terminar a coleta.\n")

    table_points = []
    robot_points = []

    while True:
        print(f"\n--- Ponto {len(table_points)+1} --- "
              f"Clique no BOTAO EXTRA para capturar a posicao do MUNDO "
              f"(ou BOTAO LATERAL para terminar): ")

        got_target = wait_for_mouse_button(mouse_device, ecodes.BTN_EXTRA, other_button=ecodes.BTN_SIDE)
        if not got_target:
            break  # BTN_SIDE pressionado -- terminar coleta

        node.latest_world_position = None
        t_start = time.time()
        while node.latest_world_position is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t_start > 5.0:
                print(f"[ERRO] Timeout esperando {OBJECT_POSITION_TOPIC} -- "
                      f"verifique se o live_object_position_viewer.py esta rodando.")
                break

        if node.latest_world_position is None:
            continue

        world_pos = node.latest_world_position.copy()
        print(f"✅ Posicao do MUNDO capturada: {np.round(world_pos, 4)}")

        robot_pos = wait_for_click_then_capture(
            node,
            mouse_device,
            lambda: node.latest_robot_pose,
            lambda: setattr(node, "latest_robot_pose", None),
            "Agora toque a garra fisicamente nesse MESMO objeto e clique no "
            "BOTAO EXTRA para capturar a posicao do ROBO...",
        )

        if robot_pos is None:
            print("[AVISO] Falha ao capturar posicao do robo -- descartando este ponto.")
            continue

        table_points.append(world_pos.tolist())
        robot_points.append(robot_pos.tolist())
        print(f"✅ Posicao do ROBO capturada: {np.round(robot_pos, 4)}")
        print(f"Ponto {len(table_points)} completo!")

    if len(table_points) < 3:
        print(f"\n[ERRO] So' {len(table_points)} pontos coletados -- "
              f"precisa de pelo menos 3 para o Kabsch.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    P_table = np.array(table_points)
    P_robot = np.array(robot_points)

    R, t, rms_error, residuals = kabsch_algorithm(P_table, P_robot)

    print("\n=== RESULTADO DA CALIBRACAO (Kabsch, {} pontos) ===".format(len(table_points)))
    print(f"Rotacao (R):\n{R}")
    print(f"Translacao (t): {t}")
    print(f"Erro RMS: {rms_error*1000:.3f}mm")
    print(f"Erros individuais por ponto (mm): {np.round(residuals*1000, 2)}")

    result = {
        "rotation_matrix": R.tolist(),
        "translation": t.tolist(),
        "rms_error_m": float(rms_error),
        "table_points_used": P_table.tolist(),
        "robot_points_captured": P_robot.tolist(),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResultado salvo em: {OUTPUT_PATH}")
    print("\nUse R e t para converter qualquer posicao da mesa para o "
          "referencial do robo: P_robo = R @ P_mesa + t")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()