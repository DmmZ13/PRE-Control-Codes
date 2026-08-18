#!/usr/bin/env python3
"""
Verifica, para cada synchronized_dataset_N/synchronized_metadata.json,
se a PRIMEIRA posicao de junta gravada (primeiro elemento de
robot_trajectory) difere da pose home de referencia
([0, -1.5708, 0, -1.5708, -1.5708, 0]) em pelo menos 0.0003 rad em
QUALQUER uma das 6 juntas.

Uso:
  python check_first_joint_position.py
"""

import glob
import json
import os

DATASET_ROOT = "/home/ziqi/pre_ws/dataset"
HOME_REFERENCE = [0.0, -1.5708, 0.0, -1.5708, 1.5708, 0.0]
TOLERANCE = 0.0003  # rad


def find_dataset_dirs(root):
    pattern = os.path.join(root, "synchronized_dataset_*")
    dirs = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.rsplit("_", 1)[-1]) if p.rsplit("_", 1)[-1].isdigit() else 0
    )
    return dirs


def main():
    dataset_dirs = find_dataset_dirs(DATASET_ROOT)
    print(f"Encontradas {len(dataset_dirs)} pastas synchronized_dataset_*\n")

    diverging = []
    ok_count = 0
    error_count = 0

    for ep_dir in dataset_dirs:
        json_path = os.path.join(ep_dir, "synchronized_metadata.json")
        dataset_name = os.path.basename(ep_dir)

        if not os.path.exists(json_path):
            print(f"[AVISO] {dataset_name}: synchronized_metadata.json nao encontrado, pulando.")
            error_count += 1
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERRO] {dataset_name}: falha ao ler JSON ({e})")
            error_count += 1
            continue

        robot_trajectory = data.get("robot_trajectory", [])
        if len(robot_trajectory) == 0:
            print(f"[AVISO] {dataset_name}: robot_trajectory vazio, pulando.")
            error_count += 1
            continue

        first_joint_pos = robot_trajectory[0].get("joint_positions", None)
        if first_joint_pos is None or len(first_joint_pos) != 6:
            print(f"[AVISO] {dataset_name}: joint_positions ausente/invalido no primeiro frame, pulando.")
            error_count += 1
            continue

        diffs = [abs(a - b) for a, b in zip(first_joint_pos, HOME_REFERENCE)]
        max_diff = max(diffs)

        if max_diff >= TOLERANCE:
            diverging.append({
                "dataset": dataset_name,
                "first_joint_pos": first_joint_pos,
                "diffs": diffs,
                "max_diff": max_diff,
            })
        else:
            ok_count += 1

    # ── Resultado ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESUMO: {len(dataset_dirs)} pastas | {ok_count} OK | "
          f"{len(diverging)} divergentes | {error_count} com erro/pulados")
    print("=" * 70 + "\n")

    if diverging:
        print(f"Datasets com primeira posicao de junta DIFERENTE do home "
              f"(tolerancia={TOLERANCE} rad):\n")
        for item in diverging:
            print(f"  {item['dataset']}: max_diff={item['max_diff']:.6f} rad")
            print(f"    joint_positions: {[round(v, 6) for v in item['first_joint_pos']]}")
            print(f"    diffs por junta: {[round(v, 6) for v in item['diffs']]}")
            print()
    else:
        print("Nenhum dataset divergente encontrado -- todos comecam perto do home.")

    # Salva tambem em arquivo, para consulta posterior
    output_path = os.path.join(DATASET_ROOT, "check_first_joint_position_report.json")
    with open(output_path, "w") as f:
        json.dump({
            "home_reference": HOME_REFERENCE,
            "tolerance": TOLERANCE,
            "total_datasets": len(dataset_dirs),
            "ok_count": ok_count,
            "error_count": error_count,
            "diverging": diverging,
        }, f, indent=2)
    print(f"\nRelatorio completo salvo em: {output_path}")


if __name__ == "__main__":
    main()