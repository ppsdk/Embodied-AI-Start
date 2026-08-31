#!/usr/bin/env python3
"""Solve eye-in-hand calibration from collect_samples.py CSV output."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_matrix(row, prefix: str) -> np.ndarray:
    values = [float(row[f"{prefix}_{i}{j}"]) for i in range(4) for j in range(4)]
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{prefix} contains NaN or infinity")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{prefix} bottom row is not [0, 0, 0, 1]")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError(f"{prefix} rotation is not orthonormal")
    if np.linalg.det(matrix[:3, :3]) < 0.0:
        raise ValueError(f"{prefix} rotation has negative determinant")
    return matrix


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = b[:3, :3].T @ a[:3, :3]
    value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def write_yaml(path: Path, matrix: np.ndarray, method: str, metrics: dict) -> None:
    lines = [f"method: {method}", "T_EC:"]
    lines += ["  - [" + ", ".join(f"{value:.12g}" for value in row) + "]" for row in matrix]
    lines += ["metrics:"]
    lines += [f"  {key}: {value:.9g}" for key, value in metrics.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve T^E_C for eye-in-hand calibration")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["DANIILIDIS", "TSAI", "PARK"], default="DANIILIDIS")
    parser.add_argument("--min-samples", type=int, default=6)
    args = parser.parse_args()
    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.csv_path.open("r", newline="", encoding="utf-8")))
    if len(rows) < args.min_samples:
        raise ValueError(f"Need at least {args.min_samples} samples, got {len(rows)}")
    frames = {(row["base_frame"], row["ee_frame"], row["camera_frame"], row["target_frame"]) for row in rows}
    if len(frames) != 1:
        raise ValueError(f"Frame names are inconsistent: {frames}")
    mountings = {row.get("mounting", "eye_in_hand") for row in rows}
    if mountings != {"eye_in_hand"}:
        raise ValueError(f"Expected eye_in_hand samples, got {mountings}")

    T_BE = [read_matrix(row, "T_BE") for row in rows]
    T_CT = [read_matrix(row, "T_CT") for row in rows]
    R_gripper2base = [matrix[:3, :3] for matrix in T_BE]
    t_gripper2base = [matrix[:3, 3].reshape(3, 1) for matrix in T_BE]
    R_target2cam = [matrix[:3, :3] for matrix in T_CT]
    t_target2cam = [matrix[:3, 3].reshape(3, 1) for matrix in T_CT]
    method_id = getattr(cv2, f"CALIB_HAND_EYE_{args.method}")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method_id
    )
    T_EC = np.eye(4, dtype=np.float64)
    T_EC[:3, :3] = R_cam2gripper
    T_EC[:3, 3] = np.asarray(t_cam2gripper).reshape(3)

    reconstructed = [T_BE_i @ T_EC @ T_CT_i for T_BE_i, T_CT_i in zip(T_BE, T_CT)]
    reference = reconstructed[0]
    translation_errors_mm = [float(np.linalg.norm(item[:3, 3] - reference[:3, 3]) * 1000.0) for item in reconstructed]
    rotation_errors_deg = [rotation_error_deg(item, reference) for item in reconstructed]
    metrics = {
        "translation_median_mm": float(np.median(translation_errors_mm)),
        "translation_p95_mm": float(np.percentile(translation_errors_mm, 95)),
        "translation_max_mm": float(np.max(translation_errors_mm)),
        "rotation_median_deg": float(np.median(rotation_errors_deg)),
        "rotation_p95_deg": float(np.percentile(rotation_errors_deg, 95)),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
    }
    np.save(output_dir / "T_EC.npy", T_EC)
    write_yaml(output_dir / "T_EC.yaml", T_EC, args.method, metrics)
    with (output_dir / "residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "translation_error_mm", "rotation_error_deg"])
        for row, trans, rot in zip(rows, translation_errors_mm, rotation_errors_deg):
            writer.writerow([row["sample_id"], f"{trans:.6f}", f"{rot:.6f}"])
    print(f"T^E_C ({args.method}) =\n{T_EC}")
    print("Residuals:", ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
