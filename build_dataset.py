#!/usr/bin/env python3
"""Build the hydra_eval/ dataset directory from the raw Hydra ICP benchmark.

Pre-requisite: run calibrate.py (without --dry-run) first so that
T_cam_base_pnp.npy exists under each realsense_april/ directory.
If a .npy is missing the calibration is re-run on the fly.

Output layout:
    <out>/
    |-- lbr/
    |   |-- measurement_0/
    |   |   |-- T_cam_base.npy      (4,4) float64  GT camera pose
    |   |   |-- camera_K.npy        (3,3) float64  RealSense intrinsics
    |   |   |-- images/             000.png ... 014.png
    |   |   `-- ground_truth.json   {frame_idx: {"joints": [...]}}
    |   |-- measurement_1/
    |   `-- measurement_2/
    |-- meca/
    `-- xarm/

Usage:
    python build_dataset.py --dataset /path/to/raw_dataset --out hydra_eval
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from calibrate import N_CONFIGS, ROBOTS, load_K, make_detector, solve_measurement


def _generate_measurement(
    meas_src: Path,
    meas_dst: Path,
    robot: str,
    detector,
    verbose: bool,
) -> bool:
    """Assemble one (robot, measurement) into the output layout. Returns True on success."""
    t_path = meas_src / "realsense_april" / "T_cam_base_pnp.npy"

    if t_path.exists():
        T_cam_base = np.load(t_path)
    else:
        print("  T_cam_base_pnp.npy missing, running calibration ...", end=" ", flush=True)
        T_cam_base, _, _ = solve_measurement(meas_src, robot, detector, verbose=verbose)
        if T_cam_base is None:
            print("FAILED")
            return False

    K = load_K(meas_src)

    hydra_dir = meas_src / "realsense_hydra"

    # Collect per-frame joint angles.
    frames_gt: dict = {}
    for i in range(N_CONFIGS):
        js_path = hydra_dir / f"joint_states_{i}.npy"
        if js_path.exists():
            frames_gt[i] = {"joints": np.load(js_path)}

    if not frames_gt:
        print("no joint_states_N.npy found, skipping")
        return False

    # Create output dirs.
    img_dst = meas_dst / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    # Copy RGB images (rename camera.image_N.png -> NNN.png).
    for i in range(N_CONFIGS):
        src = hydra_dir / f"camera.image_{i}.png"
        if src.exists():
            shutil.copy2(src, img_dst / f"{i:03d}.png")

    # Write GT files.
    np.save(meas_dst / "T_cam_base.npy", T_cam_base)
    np.save(meas_dst / "camera_K.npy",   K)
    gt_json = {str(idx): {"joints": v["joints"].tolist()} for idx, v in frames_gt.items()}
    with open(meas_dst / "ground_truth.json", "w") as fh:
        json.dump(gt_json, fh)

    return True


def generate(dataset_root: Path, out_root: Path, verbose: bool) -> None:
    detector = make_detector()
    ok = 0

    for robot in ROBOTS:
        for m in range(3):
            meas_src = dataset_root / robot / f"measurement_{m}"
            meas_dst = out_root     / robot / f"measurement_{m}"

            label = f"{robot}/measurement_{m}"
            print(f"  {label} ...", end=" ", flush=True)

            success = _generate_measurement(meas_src, meas_dst, robot, detector, verbose)
            if success:
                n = len(list((meas_dst / "images").glob("*.png")))
                print(f"{n} frames -> {meas_dst.relative_to(out_root)}")
                ok += 1
            # failure message already printed inside helper

    total = len(ROBOTS) * 3
    print(f"\n{ok}/{total} measurements written to {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the raw Hydra ICP dataset directory")
    parser.add_argument("--out", required=True,
                        help="Output root directory (will be created)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print calibration detail when re-running PnP")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    out_root     = Path(args.out).resolve()

    if not dataset_root.exists():
        sys.exit(f"ERROR: dataset not found: {dataset_root}")

    print(f"Source : {dataset_root}")
    print(f"Output : {out_root}")
    print()
    generate(dataset_root, out_root, args.verbose)


if __name__ == "__main__":
    main()
