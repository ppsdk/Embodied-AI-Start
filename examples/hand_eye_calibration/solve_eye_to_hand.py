#!/usr/bin/env python3
"""Solve eye-to-hand calibration from the shared hand-eye CSV format.

Known per sample: T^B_E and T^C_T. Unknown fixed transforms:
  T^B_C: camera in base, and T^E_T: target in end-effector.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from solve_hand_eye import read_matrix, rotation_error_deg, write_yaml


def average_transforms(transforms):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([item[:3, 3] for item in transforms], axis=0)
    u, _, vh = np.linalg.svd(np.sum([item[:3, :3] for item in transforms], axis=0))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    result[:3, :3] = rotation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve T^B_C and T^E_T for eye-to-hand calibration")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["DANIILIDIS", "TSAI", "PARK"], default="DANIILIDIS")
    parser.add_argument("--min-samples", type=int, default=6)
    args = parser.parse_args()
    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < args.min_samples:
        raise ValueError(f"Need at least {args.min_samples} samples, got {len(rows)}")
    frames = {(row["base_frame"], row["ee_frame"], row["camera_frame"], row["target_frame"]) for row in rows}
    if len(frames) != 1:
        raise ValueError(f"Frame names are inconsistent: {frames}")
    mountings = {row.get("mounting", "eye_to_hand") for row in rows}
    if mountings != {"eye_to_hand"}:
        raise ValueError(f"Expected eye_to_hand samples, got {mountings}")

    T_BE = [read_matrix(row, "T_BE") for row in rows]
    T_CT = [read_matrix(row, "T_CT") for row in rows]
    # Eye-to-hand equation: T^B_C T^C_T = T^B_E T^E_T.
    # After inversion of the visual pose, the standard AX=XB solver returns T^E_T.
    T_TC = [np.linalg.inv(item) for item in T_CT]
    method_id = getattr(cv2, f"CALIB_HAND_EYE_{args.method}")
    r_et, t_et = cv2.calibrateHandEye(
        [item[:3, :3] for item in T_BE],
        [item[:3, 3, None] for item in T_BE],
        [item[:3, :3] for item in T_TC],
        [item[:3, 3, None] for item in T_TC],
        method=method_id,
    )
    T_ET = np.eye(4, dtype=np.float64)
    T_ET[:3, :3] = r_et
    T_ET[:3, 3] = np.asarray(t_et).reshape(3)
    T_BC = average_transforms([be @ T_ET @ np.linalg.inv(ct) for be, ct in zip(T_BE, T_CT)])

    reconstructed = [np.linalg.inv(T_BC) @ be @ T_ET for be in T_BE]
    translation_errors_mm = [float(np.linalg.norm(item[:3, 3] - ct[:3, 3]) * 1000.0) for item, ct in zip(reconstructed, T_CT)]
    rotation_errors_deg = [rotation_error_deg(item, ct) for item, ct in zip(reconstructed, T_CT)]
    metrics = {
        "target_in_camera_translation_median_mm": float(np.median(translation_errors_mm)),
        "target_in_camera_translation_p95_mm": float(np.percentile(translation_errors_mm, 95)),
        "target_in_camera_rotation_median_deg": float(np.median(rotation_errors_deg)),
        "target_in_camera_rotation_p95_deg": float(np.percentile(rotation_errors_deg, 95)),
    }
    np.save(output_dir / "T_BC.npy", T_BC)
    np.save(output_dir / "T_ET.npy", T_ET)
    write_yaml(output_dir / "T_BC.yaml", T_BC, args.method, metrics)
    write_yaml(output_dir / "T_ET.yaml", T_ET, args.method, metrics)
    with (output_dir / "residuals_eye_to_hand.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "translation_error_mm", "rotation_error_deg"])
        for row, trans, rot in zip(rows, translation_errors_mm, rotation_errors_deg):
            writer.writerow([row["sample_id"], f"{trans:.6f}", f"{rot:.6f}"])
    print(f"T^B_C ({args.method}) =\n{T_BC}")
    print(f"T^E_T ({args.method}) =\n{T_ET}")
    print("Residuals:", ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
