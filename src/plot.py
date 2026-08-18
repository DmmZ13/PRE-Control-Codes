#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ordem cinemática padrão do UR5
UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class UR5JointCollectorNode(Node):
    def __init__(self):
        super().__init__('ur5_joint_collector_node')
        
        # Buffer para armazenar as 1000 amostras
        self.joint_buffer = []
        self.max_samples = 10000
        
        # 🌟 O SUBSCRIBER PARA PEGAR OS VALORES DAS JUNTAS
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.get_logger().info(f"Nó iniciado. Aguardando a coleta de {self.max_samples} amostras de juntas...")

    def joint_callback(self, msg: JointState):
        try:
            # Garante a ordenação correta das juntas do UR5
            pos_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            ordered_joints = [float(pos_map[joint]) for joint in UR5_JOINT_NAMES]
            
            # Adiciona ao buffer
            self.joint_buffer.append(ordered_joints)
            
            # Mostrador de progresso no terminal
            sys.stdout.write(f"\r[Progresso] Coletando amostras: {len(self.joint_buffer)}/{self.max_samples}")
            sys.stdout.flush()
            
            # Quando atingir o limite de 1000 amostras, dispara o processamento
            if len(self.joint_buffer) >= self.max_samples:
                print("\n")
                self.get_logger().info("🎯 1000 amostras coletadas com sucesso! Encerrando subscriber e processando...")
                
                # Desativa o subscriber imediatamente para não receber mais dados
                self.destroy_subscription(self.joint_sub)
                
                # Executa a análise e gera o gráfico
                self.processar_e_plotar()
                
                # Encerra o nó ROS 2 de forma limpa
                rclpy.shutdown()
                
        except KeyError:
            # Ignora mensagens de outros componentes que não tenham as juntas do UR5
            pass

    def processar_e_plotar(self):
        # Converte o buffer para uma matriz numpy (shape: 100, 6)
        joint_data = np.array(self.joint_buffer)
        indices = np.arange(self.max_samples)
        
        # --- CÁLCULO DOS ERROS (Média como referência estática) ---
        mae_vector = np.zeros(6)
        max_vector = np.zeros(6)
        error_matrix = np.zeros_like(joint_data)
        
        print("\n" + "="*70)
        print("          RELATÓRIO DE ERROS BRUTOS EM ESPAÇO DE JUNTAS (RAD)         ")
        print("="*70)
        
        for i in range(6):
            referencia = np.mean(joint_data[:, i])
            error_matrix[:, i] = np.abs(joint_data[:, i] - referencia)
            mae_vector[i] = np.mean(error_matrix[:, i])
            max_vector[i] = np.max(error_matrix[:, i])
            
            print(f"Junta {i} -> MAE: {mae_vector[i]:.6f} rad | Erro Máx: {max_vector[i]:.6f} rad")
        print("="*70 + "\n")
        
        # --- RENDERIZAÇÃO DO GRÁFICO (Grid 3x2) ---
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        
        for i in range(6):
            row = i % 3
            col = i // 3
            ax = axes[row, col]
            
            # Plota os valores brutos em radianos
            ax.plot(indices, joint_data[:, i], marker='o', linestyle='-', markersize=2, color='#1f77b4', label=f'Junta {i}')
            ax.set_title(f"Junta {i} (MAE: {mae_vector[i]:.5f} | Max: {max_vector[i]:.5f})", fontsize=10, fontweight='bold')
            ax.set_ylabel("Radianos", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right', fontsize=8)
            
        axes[2, 0].set_xlabel("Índice da Amostra", fontsize=10)
        axes[2, 1].set_xlabel("Índice da Amostra", fontsize=10)
        
        plt.tight_layout()
        
        # Salva o gráfico em disco
        output_image_path = "ur5_joint_analysis_100.png"
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        self.get_logger().info(f"✅ Análise visual salva com sucesso em: {output_image_path}")

def main(args=None):
    rclpy.init(args=args)
    node = UR5JointCollectorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n[SIGNAL] Execução interrompida manualmente.")
    finally:
        print("Nó ROS 2 finalizado.")

if __name__ == '__main__':
    main()
    