#!/usr/bin/env python3
"""
UR5 Object Position Tracker -- Cartesian Pose Tracking via Joint
Trajectory Controller

Adaptado do UR5 Dataset Player: em vez de reproduzir poses gravadas
de um arquivo JSON, este no assina o topico /object_position_yolo_world
(publicado pelo live_object_position_viewer.py) e calcula, em tempo
real, os angulos de junta necessarios para levar a garra ate' essa
posicao -- mantendo a garra SEMPRE NA HORIZONTAL (apontando para
baixo, perpendicular a mesa, o que deixa o flange/pulso paralelo ao
plano XY).

MODO TESTE: a publicacao real da trajetoria esta' COMENTADA -- o
script so' CALCULA e IMPRIME os angulos de junta resultantes, sem
mover o robo de verdade.

ATENCAO -- suposicao que fiz sobre "horizontal": assumi que
"garra na horizontal, paralela ao plano XY" significa o eixo Z da
ferramenta apontando para BAIXO (perpendicular a mesa, abordagem
tipica para pegar objetos de cima) -- isso deixa o FLANGE (plano do
pulso) paralelo ao plano XY. Se sua intencao era outra orientacao
(ex: abordagem lateral, com o eixo Z apontando horizontalmente),
ajuste a matriz R_TARGET_FIXED abaixo.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import sys
import threading

# ══════════════════════════════════════════════════════════════════════════════
# Configurações
# ══════════════════════════════════════════════════════════════════════════════

OBJECT_POSITION_TOPIC = "/object_position_yolo_world"

# NOVO: sem conceito de "frequencia de controle" -- o objeto esta'
# PARADO, entao so' precisamos de UMA solucao final de cinematica
# inversa (calculada internamente, iterando ate' convergir), e mandar
# o robo ate' la' de uma vez so', num movimento suave de MOVE_DURATION_S
# segundos (mesmo padrao usado no script de homing).
MOVE_DURATION_S = 5.0        # duracao do movimento ate' o alvo
MAX_IK_ITERATIONS = 200      # limite de iteracoes internas do solver
IK_CONVERGENCE_THRESHOLD = 1e-5  # norma do erro de pose considerada "convergido"






# NOVO: recalcula automaticamente a cada N segundos (so' informativo,
# NAO move o robo) -- o Enter agora dispara a EXECUCAO do ultimo
# calculo disponivel, nao o calculo em si.
PERIODIC_CALC_INTERVAL_S = 2.0

# NOVO: distancia de seguranca antes do alvo final -- o robo primeiro
# vai a este ponto (na MESMA linha base->objeto, MESMA orientacao),
# so' entao avanca os ultimos centimetros ate' o alvo. Evita "esmagar"
# o objeto chegando direto/rapido demais.
APPROACH_CLEARANCE_M = 0.10  # 10cm antes do alvo

# NOVO: distancia de seguranca para o ponto intermediario (pre-pega)
# -- o robo primeiro vai ate' este ponto, alinhado com a linha
# base->objeto, ANTES de avancar reto para o alvo final. Evita
# aproximacoes "tortas" que poderiam esbarrar/esmagar o objeto.
STAGE1_OFFSET_M = 0.10  # 10cm antes do alvo final, ao longo da linha de abordagem
STAGE1_DURATION_S = 4.0  # duracao do movimento ate' o ponto intermediario
STAGE2_DURATION_S = 2.0  # duracao do movimento final (intermediario -> alvo)

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# From cat ~/my_robot_calibration.yaml
UR5_DH = np.array([
    [0.089201,  0.0,       np.pi/2],  
    [0.0,      -0.425428,  0.0    ],  
    [0.0,      -0.392387,  0.0    ],
    [0.110225,  0.0,       np.pi/2], 
    [0.094859,  0.0,      -np.pi/2],  
    [0.082384,  0.0,       0.0    ],  
])

TCP_OFFSET_Z   = 0.150  # metros (offset da sua garra)
DAMPING        = 0.05   # Amortecimento para evitar singularidades

# NOVO: pesos por junta para influenciar QUAL configuracao o solver
# prefere, entre as multiplas validas para a mesma pose. Peso MAIOR
# = movimento mais "caro"/desencorajado; peso MENOR = mais "barato"/
# preferido. Aqui, desencorajamos o "shoulder_lift" (junta 2, indice
# 1) de se mover muito, incentivando o "elbow" (junta 3, indice 2) a
# fazer a maior parte do trabalho -- evitando a configuracao onde o
# ombro avanca e o cotovelo recua (que arriscava bater no movel).
JOINT_WEIGHTS = np.array([1.0, 5.0, 0.3, 1.0, 1.0, 1.0])

# NOVO: em vez de uma orientacao FIXA, a rotacao agora e' calculada
# DINAMICAMENTE por chamada -- o eixo Z da ferramenta (direcao de
# abordagem) fica SEMPRE dentro do plano XY (horizontal, sem
# inclinacao para cima/baixo), apontando da base do robo em direcao
# ao objeto -- pegando-o DE LADO, nao de cima.
def compute_horizontal_approach_rotation(current_rotation: np.ndarray) -> np.ndarray:
    """
    Constroi a matriz de rotacao da ferramenta para abordagem
    HORIZONTAL (de lado): o eixo Z da ferramenta fica SEMPRE alinhado
    ao eixo X da BASE do robo (fixo), independente de onde o objeto
    esteja -- diferente da versao anterior, que alinhava
    dinamicamente com a linha base->objeto.

    Existem DUAS orientacoes validas para essa abordagem, diferindo
    por uma rotacao de 180 graus em torno do proprio eixo de
    abordagem (Z) -- para uma garra simetrica de 2 dedos, ambas
    resultam na MESMA pegada fisica. Calcula as DUAS opcoes e escolhe
    a que fica MAIS PROXIMA da orientacao atual do robo, minimizando
    o movimento desnecessario do punho.
    """
    z_axis = np.array([1.0, 0.0, 0.0])  # sempre o eixo X da base, fixo
    world_up = np.array([0.0, 0.0, 1.0])

    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_option_a = np.column_stack([x_axis, y_axis, z_axis])
    # Opcao alternativa: rotacao de 180 graus em torno do eixo Z
    # (equivalente a inverter x_axis e y_axis)
    R_option_b = np.column_stack([-x_axis, -y_axis, z_axis])

    # Escolhe a opcao mais proxima da orientacao ATUAL do robo
    # (norma de Frobenius da diferenca -- menor = mais parecida)
    dist_a = np.linalg.norm(R_option_a - current_rotation)
    dist_b = np.linalg.norm(R_option_b - current_rotation)

    return R_option_a if dist_a <= dist_b else R_option_b

# ══════════════════════════════════════════════════════════════════════════════
# Funções de Álgebra e Cinemática Analítica (inalteradas)
# ══════════════════════════════════════════════════════════════════════════════

def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ],
    ])

def ur5_fk_frames(q: np.ndarray):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        d, a, alpha = UR5_DH[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
        frames.append(T.copy())
    return frames

def ur5_tcp_pose(q: np.ndarray) -> np.ndarray:
    frames = ur5_fk_frames(q)
    T_tool0 = frames[6]
    T_offset = np.eye(4)
    T_offset[2, 3] = TCP_OFFSET_Z
    return T_tool0 @ T_offset

def ur5_jacobian_tcp(q: np.ndarray) -> np.ndarray:
    frames = ur5_fk_frames(q)
    T_tcp  = ur5_tcp_pose(q)
    p_tcp  = T_tcp[:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_tcp - p_i)
        J[3:, i] = z_i
    return J

def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))

def weighted_damped_pinv(J: np.ndarray, lam: float, joint_weights: np.ndarray) -> np.ndarray:
    """
    Pseudo-inversa amortecida PONDERADA -- em vez de tratar o
    movimento de todas as juntas como igualmente "barato", pesos
    MAIORES tornam o movimento daquela junta mais "caro"
    (desencorajado), e pesos MENORES tornam mais "barato"
    (preferido). Isso permite influenciar QUAL configuracao entre
    varias validas o solver escolhe, sem mudar a garantia de
    convergencia (ainda converge para uma solucao valida, so'
    prefere uma configuracao especifica quando ha' mais de uma
    possivel).

    Formula: Jp = W^-1 @ J^T @ (J @ W^-1 @ J^T + lam^2*I)^-1
    """
    W_inv = np.diag(1.0 / joint_weights)
    return W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + lam**2 * np.eye(6))

def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """
    Envolve os angulos de volta para o intervalo [-pi, pi] --
    diferente de um simples clip/corte (que distorceria o valor),
    isso preserva o significado fisico real do angulo (ex: 3.2 rad
    vira aproximadamente -3.08 rad, nao e' cortado em pi).
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi

def compute_pose_error(T_target, T_current):
    """ Calcula o erro cartesiano 6D (posição + orientação) entre duas matrizes homogêneas """
    error = np.zeros(6)
    error[:3] = T_target[:3, 3] - T_current[:3, 3]

    R_target = T_target[:3, :3]
    R_current = T_current[:3, :3]
    R_error = R_target @ R_current.T

    acos_val = np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(acos_val)

    if theta > 1e-5:
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2.0 * np.sin(theta))
        error[3:] = axis * theta
    return error

def solve_ik(T_target: np.ndarray, q_start: np.ndarray):
    """
    Resolve a cinematica inversa iterativamente, partindo de
    q_start. Reaproveitada tanto para o alvo FINAL (partindo da pose
    ATUAL do robo, pode precisar de mais iteracoes) quanto para o
    ponto "10cm antes" (partindo da solucao FINAL ja calculada, que
    fica bem perto -- converge rapido, sem precisar refazer o solve
    completo do zero).
    """
    q_iter = q_start.copy()
    error_norm = None

    for iteration in range(MAX_IK_ITERATIONS):
        T_current = ur5_tcp_pose(q_iter)
        pose_error = compute_pose_error(T_target, T_current)

        error_norm = np.linalg.norm(pose_error)
        if error_norm < IK_CONVERGENCE_THRESHOLD:
            break

        J = ur5_jacobian_tcp(q_iter)
        Jp = weighted_damped_pinv(J, DAMPING, JOINT_WEIGHTS)
        dq = Jp @ pose_error

        q_iter = q_iter + dq

    return wrap_to_pi(q_iter), iteration + 1, error_norm

# ══════════════════════════════════════════════════════════════════════════════
# Nó de Rastreamento de Posição de Objeto
# ══════════════════════════════════════════════════════════════════════════════

class UR5ObjectPositionTracker(Node):

    def __init__(self):
        super().__init__("ur5_object_position_tracker")

        self._q = np.zeros(6)
        self._js_ok = False

        self.target_position = None  # (3,) -- ultima posicao recebida do topico
        self.latest_computed_joints_pre = None    # (6,) -- ponto 10cm antes do alvo
        self.latest_computed_joints_final = None  # (6,) -- alvo final

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._js_cb,
            qos_profile=qos_profile_sensor_data
        )

        self.create_subscription(
            Point,
            OBJECT_POSITION_TOPIC,
            self._object_position_cb,
            10
        )

        # NOVO: recalcula automaticamente em background, a cada
        # PERIODIC_CALC_INTERVAL_S segundos -- so' CALCULA e
        # ARMAZENA o resultado (nao move o robo). O Enter passa a
        # disparar a EXECUCAO do ultimo calculo disponivel.
        self.calc_timer = self.create_timer(PERIODIC_CALC_INTERVAL_S, self._periodic_calculate)

        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(
            f"No pronto. Calculando automaticamente a cada "
            f"{PERIODIC_CALC_INTERVAL_S}s -- aperte ENTER no terminal para "
            f"EXECUTAR o ultimo calculo e mover o robo ate' o objeto."
        )

    def _js_cb(self, msg: JointState):
        try:
            positions_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            self._q = np.array([positions_map[name] for name in UR5_JOINT_NAMES])
            self._js_ok = True
        except KeyError:
            pass

    def _object_position_cb(self, msg: Point):
        self.target_position = np.array([msg.x, msg.y, msg.z])

    def _keyboard_listener(self):
        """Thread separada, escutando o terminal -- a cada ENTER
        (linha vazia), EXECUTA (move o robo de verdade) usando o
        ULTIMO calculo disponivel (feito automaticamente em
        background pelo timer periodico)."""
        print(f"\nPronto! Calculando automaticamente a cada "
              f"{PERIODIC_CALC_INTERVAL_S}s. Aperte ENTER para EXECUTAR o "
              f"ultimo calculo (mover o robo de verdade ate' o objeto).\n")
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break
            self.execute_move()

    def _periodic_calculate(self):
        """Chamado automaticamente pelo timer -- so' CALCULA e
        ARMAZENA o resultado em self.latest_computed_joints, sem
        mover o robo."""
        self.compute_once()

    def execute_move(self):
        """Chamado quando o usuario aperta ENTER -- publica de
        verdade os DOIS waypoints (ponto 10cm antes, depois alvo
        final) numa unica trajetoria, movendo o robo."""
        if self.latest_computed_joints_pre is None or self.latest_computed_joints_final is None:
            print("[AVISO] Ainda nao ha' nenhum calculo disponivel -- aguarde.")
            return

        print(f"\n[EXECUTANDO] Indo primeiro ao ponto 10cm antes, depois ao "
              f"alvo final (duracao total={MOVE_DURATION_S}s)")
        self._publish_joint_trajectory(self.latest_computed_joints_pre, self.latest_computed_joints_final)

    def compute_once(self):
        """
        Calcula a SOLUCAO FINAL de cinematica inversa para a posicao
        atual do objeto (UMA VEZ, partindo da pose ATUAL do robo).

        Depois, deriva o ponto "10cm ANTES" do alvo (mesma linha
        base->objeto, MESMA orientacao) via um refinamento RAPIDO,
        partindo da solucao FINAL ja' calculada (que fica bem perto
        no espaco cartesiano) -- NAO precisa refazer o solve completo
        do zero para esse segundo ponto.
        """
        if not self._js_ok:
            print("[AVISO] Ainda nao recebi /joint_states -- aguarde e tente de novo.")
            return

        if self.target_position is None:
            print(f"[AVISO] Ainda nao recebi nenhuma posicao em "
                  f"{OBJECT_POSITION_TOPIC} -- aguarde e tente de novo.")
            return

        current_rotation = ur5_tcp_pose(self._q)[:3, :3]
        R_target = compute_horizontal_approach_rotation(current_rotation)

        # ── 1. Solve completo para o ALVO FINAL, partindo da pose ATUAL ──
        T_target = np.eye(4)
        T_target[:3, 3] = self.target_position
        T_target[:3, :3] = R_target

        q_target_joints, iterations_final, error_final = solve_ik(T_target, self._q)

        # ── 2. Deriva o ponto "10cm ANTES", MESMA orientacao, recuado
        # ao longo do eixo de abordagem (Z da orientacao calculada) ──
        approach_z_axis = R_target[:, 2]
        pre_target_position = self.target_position - APPROACH_CLEARANCE_M * approach_z_axis

        T_pre_target = np.eye(4)
        T_pre_target[:3, 3] = pre_target_position
        T_pre_target[:3, :3] = R_target

        # Refinamento RAPIDO, partindo da solucao FINAL ja calculada
        # (fica so' 10cm longe -- converge em poucas iteracoes)
        q_pre_target_joints, iterations_pre, error_pre = solve_ik(T_pre_target, q_target_joints)

        print(f"\n[CALCULADO] Posicao alvo do objeto usada: "
              f"{np.round(self.target_position, 4)}")
        print(f"[CALCULADO] Alvo final: convergiu em {iterations_final} iteracoes "
              f"(erro={error_final*1000:.4f}mm)")
        print(f"[CALCULADO] Ponto 10cm antes: convergiu em {iterations_pre} iteracoes "
              f"(erro={error_pre*1000:.4f}mm)")
        print(f"[CALCULADO] Angulos ATUAIS (rad):        {np.round(self._q, 4)}")
        print(f"[CALCULADO] Angulos PONTO ANTES (rad):   {np.round(q_pre_target_joints, 4)}")
        print(f"[CALCULADO] Angulos ALVO FINAL (rad):    {np.round(q_target_joints, 4)}")
        print(f"[CALCULADO] (aperte ENTER para EXECUTAR -- vai primeiro ao "
              f"ponto 10cm antes, depois ao alvo final)")

        # Armazena os DOIS waypoints -- a EXECUCAO real (publish) so'
        # acontece quando o usuario aperta ENTER, via execute_move()
        self.latest_computed_joints_pre = q_pre_target_joints
        self.latest_computed_joints_final = q_target_joints

    def _publish_joint_trajectory(self, q_pre: np.ndarray, q_final: np.ndarray):
        """
        Publica UMA trajetoria com DOIS pontos:
          1. Ponto 10cm ANTES do alvo (chega em ~80% do tempo total)
          2. Alvo FINAL (chega no tempo total MOVE_DURATION_S)

        O proprio scaled_joint_trajectory_controller interpola
        suavemente entre os 3 pontos (atual -> antes -> final).
        """
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = UR5_JOINT_NAMES

        # Ponto intermediario: "10cm antes" -- chega mais cedo,
        # deixando a aproximacao final (ultimos 10cm) mais lenta/
        # cuidadosa, evitando "esmagar" o objeto
        t_pre = MOVE_DURATION_S * 0.8

        point_pre = JointTrajectoryPoint()
        point_pre.positions = q_pre.tolist()
        point_pre.time_from_start = Duration(seconds=t_pre).to_msg()

        point_final = JointTrajectoryPoint()
        point_final.positions = q_final.tolist()
        point_final.time_from_start = Duration(seconds=MOVE_DURATION_S).to_msg()

        traj.points = [point_pre, point_final]
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = UR5ObjectPositionTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n[SINAL] Interrompido pelo usuário.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()