#!/usr/bin/env python3
"""
Cliente de inferencia SINCRONO para o UR5 real, conectado ao
servidor SmolVLA (serve_smolvla_real_v2.py).

MUDANCA em relacao a versao assincrona anterior: a abordagem
assincrona (com gatilho por limiar g) causava sobreposicao entre
chunks consecutivos -- como o modelo prediz posicoes ABSOLUTAS de
junta, um chunk novo "pensa" que o robo ainda esta na posicao de
quando a observacao foi capturada, mas o robo ja avancou executando
o chunk anterior nesse meio tempo. Isso fazia o robo tentar refazer
um trecho de trajetoria que ja tinha percorrido.

A versao SINCRONA elimina esse problema por completo: sempre executa
o chunk atual por completo (EXECUTION_STEPS_PER_CHUNK acoes), depois
PARA e espera a proxima inferencia terminar -- usando uma observacao
capturada NA HORA, garantindo que reflita a posicao real do robo no
momento do pedido. O custo e' uma pausa a cada ciclo (~338ms
medidos), sem esconder a latencia de inferencia atras da execucao
como a versao assincrona tentava fazer.

NOVO: instrumentacao de debug comparando o ULTIMO COMANDO enviado
com a POSICAO REAL lida na proxima captura de observacao -- para
diagnosticar se o "salto para tras" observado entre chunks e'
causado por atraso de rastreamento do controlador (comando vs
execucao fisica real), nao por sobreposicao (que o modo sincrono
ja deveria ter eliminado).

ARQUITETURA:
- ROS2 roda numa THREAD SEPARADA (controle do robo real + captura
  de camera), mesmo padrao das threads de teclado/mouse ja usadas
  em outros scripts -- evita misturar o event loop do asyncio (usado
  para o WebSocket) com o spin() do rclpy no mesmo thread.
- WebSocket roda no thread PRINCIPAL, conectado ao servidor.
- Imagens sao comprimidas como JPEG+base64 antes do envio (nao mais
  como listas de pixels brutos) -- reduz drasticamente o overhead de
  serializacao (~360ms -> ~13ms medido).
"""

import asyncio
import base64
import json
import threading
import time
import cv2
import numpy as np
import websockets

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import SwitchController
from ur_msgs.srv import SetIO
import message_filters
from cv_bridge import CvBridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO
# ══════════════════════════════════════════════════════════════════════════════

SERVER_URI = "ws://localhost:8000/infer"  # AJUSTE para o IP/porta reais do servidor
PROMPT = "pick up the plastic cup and place it in the drawer"

CONTROL_HZ = 25.0
DT_CONTROL = 1.0 / CONTROL_HZ

CHUNK_SIZE = 50       # n -- FIXO pelo treino, o modelo sempre prediz 50 passos

# MODO SINCRONO: nao ha mais THRESHOLD_G/gatilho assincrono -- sempre
# executa o chunk atual por completo (EXECUTION_STEPS_PER_CHUNK acoes)
# e so' DEPOIS pede o proximo, com uma observacao fresca. Elimina a
# sobreposicao entre chunks consecutivos, ao custo de uma pausa a
# cada ciclo (~338ms medidos) enquanto aguarda a inferencia.

# Tabela 13 do artigo: 10 acoes por ciclo e' o ideal de QUALIDADE.
# Em modo SINCRONO nao precisamos mais do buffer de seguranca extra
# (20) que o modo assincrono exigia -- aqui sempre esperamos a
# inferencia terminar antes de trocar de chunk, entao nao ha risco
# de fila esvaziar.
EXECUTION_STEPS_PER_CHUNK = 25
IMG_SIZE = (512, 512)  # (H, W) -- mesmo tamanho usado no treino

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ══════════════════════════════════════════════════════════════════════════════
# NO ROS2 -- controle do robo real + captura de camera (thread separada)
# ══════════════════════════════════════════════════════════════════════════════

class UR5RosBridge(Node):
    """
    Roda em uma thread separada. Publica comandos de posicao no
    forward_position_controller e mantem a ultima observacao
    (estado + 3 imagens) disponivel de forma thread-safe para o
    cliente ler quando precisar.
    """

    def __init__(self):
        super().__init__("ur5_async_inference_bridge")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self._current_joint_positions = None  # (6,) ou None ate a 1a leitura
        self._current_gripper_state = 0.0
        self._latest_images = None  # dict {"zed_left":..., "zed_right":..., "zed_robot":...}

        # ── Gripper (mesmo padrao do gravador/player: servico SetIO,
        # so' manda comando quando o valor MUDA -- "debounce") ─────────
        self.gripper_pin = 16  # Tool Digital Output 0
        self._last_sent_gripper_state = None
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        self.get_logger().info("Aguardando servico de I/O do robo...")
        self.io_client.wait_for_service()
        self.get_logger().info("Servico de I/O conectado!")

        qos_camera = QoSProfile(depth=10)
        qos_camera.reliability = ReliabilityPolicy.BEST_EFFORT

        self.forward_pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )

        # ── NOVO: publisher de trajetoria suave (scaled_joint_trajectory_controller) ──
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10
        )

        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb,
            qos_profile=qos_profile_sensor_data
        )

        left_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_left/rgb/color/rect/image", qos_profile=qos_camera)
        right_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_right/rgb/color/rect/image", qos_profile=qos_camera)
        robot_sub = message_filters.Subscriber(
            self, Image, "/zed_multi/zed_robot/rgb/color/rect/image", qos_profile=qos_camera)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, robot_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._camera_cb)

        self.get_logger().info("UR5RosBridge pronto -- aguardando dados de junta/camera...")

    def _joint_state_cb(self, msg: JointState):
        try:
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            joint_pos = np.array([pos_map[j] for j in UR5_JOINT_NAMES], dtype=np.float32)
            with self._lock:
                self._current_joint_positions = joint_pos
        except KeyError:
            pass

    def _camera_cb(self, left_msg, right_msg, robot_msg):
        try:
            def to_rgb(msg):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return cv_img[:, :, ::-1].copy()

            images = {
                "zed_left": to_rgb(left_msg),
                "zed_right": to_rgb(right_msg),
                "zed_robot": to_rgb(robot_msg),
            }
            with self._lock:
                self._latest_images = images
        except Exception as e:
            self.get_logger().error(f"Erro no callback de camera: {e}")

    def get_observation(self):
        """Retorna (joint_pos, gripper_state, images) ou None se
        ainda nao houver dados suficientes."""
        with self._lock:
            if self._current_joint_positions is None or self._latest_images is None:
                return None
            return (
                self._current_joint_positions.copy(),
                self._current_gripper_state,
                dict(self._latest_images),
            )

    def get_current_joint_positions(self):
        """So' as posicoes de junta (sem imagens), para checagem
        rapida de chegada -- evita copiar imagens desnecessariamente
        quando so' precisamos monitorar convergencia."""
        with self._lock:
            if self._current_joint_positions is None:
                return None
            return self._current_joint_positions.copy()

    def publish_smooth_trajectory(self, actions_6d_list, duration_per_step: float):
        """
        Publica TODAS as acoes de um chunk como UMA UNICA trajetoria,
        deixando o controlador interpolar suavemente entre os pontos
        (em vez de rastrear comandos de posicao individuais a cada
        step, que pode causar movimento entrecortado/"bambeando").
        """
        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES

        for i, action_6d in enumerate(actions_6d_list):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in action_6d]
            t = (i + 1) * duration_per_step
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9)
            )
            msg.points.append(point)

        msg.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(msg)

    def publish_action(self, action_7d: np.ndarray):
        """Publica as 6 posicoes de junta. O gripper (7a dimensao)
        e' tratado separadamente via publish_gripper_command()."""
        msg = Float64MultiArray()
        msg.data = action_7d[:6].tolist()
        self.forward_pub.publish(msg)

    def publish_gripper_command(self, action_gripper_value: float,
                                 threshold_close: float = 0.99,
                                 threshold_open: float = 0.01):
        """
        Usa HISTERESE (limiares assimetricos) em vez de um unico
        threshold fixo, para evitar oscilacao/flickering perto de um
        ponto medio -- dado que os valores intermediarios do chunk
        (antes da transicao real) tem ruido pequeno (ex: -0.02 a
        0.04), um threshold unico tipo 0.5 poderia ser cruzado por
        acidente. Com limiares assimetricos, o gripper so' MUDA de
        estado quando ha' um sinal FORTE e confiante na direcao certa:
 
          - Se esta ABERTO e quer FECHAR: precisa ultrapassar 0.9
          - Se esta FECHADO e quer ABRIR: precisa cair abaixo de 0.02
 
        Fora dessas faixas de confianca alta, mantem o estado atual
        (nao muda).
        """
        current_state = self._last_sent_gripper_state
 
        if current_state is None:
            # Ainda nao definimos um estado -- usa o threshold de
            # fechar como criterio inicial (mais conservador, assume
            # aberto por padrao a menos que o sinal seja forte)
            new_state = 1 if action_gripper_value > threshold_close else 0
        elif current_state == 0:
            # Esta ABERTO -- so' fecha se o sinal for forte o suficiente
            new_state = 1 if action_gripper_value > threshold_close else 0
        else:
            # Esta FECHADO -- so' abre se o sinal for fraco o suficiente
            new_state = 0 if action_gripper_value < threshold_open else 1
 
        if new_state == current_state:
            return  # nao mudou, nao manda comando de novo
 
        req = SetIO.Request()
        req.fun = 1
        req.pin = self.gripper_pin
        req.state = float(new_state)
        self.io_client.call_async(req)
 
        self._last_sent_gripper_state = new_state
        with self._lock:
            self._current_gripper_state = float(new_state)
 
        action_verb = "FECHANDO" if new_state == 1 else "ABRINDO"
        self.get_logger().info(
            f"[GRIPPER] {action_verb} (valor previsto: {action_gripper_value:.3f}, "
            f"threshold_close={threshold_close}, threshold_open={threshold_open})"
        )

        if new_state == 1:
            self.get_logger().info("[GRIPPER] Pausando 2s apos fechar...")
            time.sleep(4.0)


def ros_spin_thread(node):
    rclpy.spin(node)


def resize_image_for_server(img: np.ndarray) -> str:
    """Comprime a imagem como JPEG e codifica em base64."""
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Falha ao codificar imagem como JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class SyncInferenceClient:
    def __init__(self, ros_bridge: UR5RosBridge):
        self.ros_bridge = ros_bridge
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(SERVER_URI, max_size=None)
        print(f"[CLIENTE] Conectado ao servidor: {SERVER_URI}")

    async def _request_inference(self, joint_pos, gripper_state, images):
        t_a = time.perf_counter()
        state_7d = np.concatenate([joint_pos, [gripper_state]]).tolist()

        t_b = time.perf_counter()
        payload = await asyncio.to_thread(self._build_payload, state_7d, images)
        t_c = time.perf_counter()

        json_str = json.dumps(payload)
        t_d = time.perf_counter()

        await self.websocket.send(json_str)
        t_e = time.perf_counter()

        response_text = await self.websocket.recv()
        t_f = time.perf_counter()

        response = json.loads(response_text)
        t_g = time.perf_counter()

        print(
            f"[DEBUG INFER] concat_state={  (t_b-t_a)*1000:.1f}ms | "
            f"to_thread_build_payload={(t_c-t_b)*1000:.1f}ms | "
            f"json_dumps={     (t_d-t_c)*1000:.1f}ms | "
            f"ws_send={         (t_e-t_d)*1000:.1f}ms | "
            f"ws_recv={         (t_f-t_e)*1000:.1f}ms | "
            f"json_loads={      (t_g-t_f)*1000:.1f}ms"
        )

        actions = np.array(response["actions"], dtype=np.float32)
        return actions

    def _build_payload(self, state_7d, images):
        return {
            "observation.state": state_7d,
            "observation.images.zed_left": resize_image_for_server(images["zed_left"]),
            "observation.images.zed_right": resize_image_for_server(images["zed_right"]),
            "observation.images.zed_robot": resize_image_for_server(images["zed_robot"]),
            "prompt": PROMPT,
        }

    async def run(self, horizon_steps: int):
        """
        Versao com TRAJETORIA SUAVE: em vez de mandar as 10 acoes
        como comandos de posicao individuais (que causavam movimento
        entrecortado, ja' que o forward_position_controller rastreia
        cada alvo novo sem interpolacao suave de verdade entre eles),
        publica TODAS as 10 acoes de uma vez como uma UNICA trajetoria
        no scaled_joint_trajectory_controller -- que interpola
        suavemente entre os pontos usando perfis de velocidade/
        aceleracao proprios.

        Alem disso, so' pede o PROXIMO chunk depois de confirmar
        (via feedback de /joint_states) que o robo REALMENTE chegou
        perto do ultimo ponto da trajetoria -- eliminando o atraso
        de rastreamento comando-vs-real que estavamos debugando.
        """
        step_counter = 0
        ARRIVAL_TOLERANCE = 0.01  # rad
        ARRIVAL_TIMEOUT_S = 0.2   # seguranca, nao trava para sempre

        while step_counter < horizon_steps:
            obs = None
            while obs is None:
                obs = self.ros_bridge.get_observation()
                if obs is None:
                    await asyncio.sleep(0.05)

            joint_pos, gripper_state, images = obs

            print(f"[CLIENTE] step={step_counter} | pedindo novo chunk "
                  f"(observacao fresca, robo parado aguardando)...")
            full_chunk = await self._request_inference(joint_pos, gripper_state, images)

            chunk_to_execute = list(full_chunk)[:EXECUTION_STEPS_PER_CHUNK]
            actions_6d = [a[:6] for a in chunk_to_execute]

            chunk_to_execute_gripper = list(full_chunk)[:]
            gripper_values = [a[6] for a in chunk_to_execute_gripper]
            gripper_mean = np.mean(gripper_values)

            # ── MUDANCA: em vez de seguir TODOS os pontos intermediarios
            # previstos (que podem ser ruidosos/oscilantes, ja' que sao
            # a parte mais incerta da previsao), manda o robo DIRETO
            # para o ULTIMO ponto previsto (10o), no mesmo espaco de
            # tempo total (400ms) -- deixando o proprio controlador
            # interpolar suavemente ate la', ignorando os pontos do
            # meio que podem estar "confusos" entre trajetorias
            # multimodais diferentes.
            # print(f"JOINT POSITIONS AT STEP {step_counter}: {np.round(joint_pos, 4)}")
            # print(f"Acoes previstas (6D) do chunk: {np.round(actions_6d, 4)}")
            print(f"Acoes previstas (gripper) do chunk: {np.round(gripper_mean, 4)}")
            final_target = actions_6d[-1]
            total_duration = EXECUTION_STEPS_PER_CHUNK * DT_CONTROL  # 400ms
            # self.ros_bridge.publish_smooth_trajectory([final_target], total_duration)

            # self.ros_bridge.publish_gripper_command(float(gripper_mean))

            print(f"[TRAJETORIA] step={step_counter} | indo DIRETO para o ultimo "
                  f"ponto previsto (10o), duracao={total_duration*1000:.0f}ms, "
                  f"alvo={np.round(final_target, 4)}")

            # ── Espera a trajetoria REALMENTE terminar (nao-bloqueante
            # para o event loop, so' aguarda ativamente) ────────────────
            target = actions_6d[-1]
            t_wait_start = time.perf_counter()
            arrived = False
            while (time.perf_counter() - t_wait_start) < ARRIVAL_TIMEOUT_S:
                current = self.ros_bridge.get_current_joint_positions()
                if current is not None:
                    error = np.max(np.abs(current - target))
                    if error < ARRIVAL_TOLERANCE:
                        arrived = True
                        break
                await asyncio.sleep(0.02)

            if not arrived:
                print(f"[AVISO] step={step_counter} | timeout esperando chegada "
                      f"(erro final={error:.4f} rad) -- prosseguindo mesmo assim.")

            step_counter += len(chunk_to_execute)
            print(f"[DRY-RUN] chunk concluido -- so' o ultimo ponto foi usado "
                  f"como alvo real: {np.round(final_target, 4)}")

        print("[CLIENTE] Horizon concluido.")


async def main_async():
    ros_bridge_holder = {}

    def start_ros():
        rclpy.init()
        node = UR5RosBridge()
        ros_bridge_holder["node"] = node
        ros_spin_thread(node)

    ros_thread = threading.Thread(target=start_ros, daemon=True)
    ros_thread.start()

    while "node" not in ros_bridge_holder:
        await asyncio.sleep(0.1)

    ros_bridge = ros_bridge_holder["node"]

    client = SyncInferenceClient(ros_bridge)
    await client.connect()
    await client.run(horizon_steps=2000)


if __name__ == "__main__":
    asyncio.run(main_async())