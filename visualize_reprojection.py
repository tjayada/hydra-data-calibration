#!/usr/bin/env python3
"""Per-frame qualitative check for calibrate.py.

Runs the exact same calibration as calibrate.py (solve_measurement), then
for each detected frame draws:

  GREEN  detected 2D corners (pyapriltags)
  RED    projected FK corners (calibrated T_cam_base)
  BLUE   projected FK corners under the reference GT (base2cam_calib.npy,
         only present for meca/measurement_0)

Good calibration: RED lands on GREEN. FK noise (LBR): RED consistently
offset from GREEN by a few pixels. Annotated PNGs are written to
<out>/<robot>/measurement_<N>/frame_<NN>.png with a per-frame error table
on the console.

Usage:
    python visualize_reprojection.py --dataset /path/to/raw_dataset --robot meca --measurement 0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from calibrate import (
    ROBOTS,
    _corner_reproj_px,
    make_detector,
    rot_err_deg,
    solve_measurement,
)


def project_corners(T_cam_base: np.ndarray,
                    pts3d: np.ndarray,
                    K: np.ndarray) -> np.ndarray:
    """Project (4,3) base-frame corners to (4,2) pixel coords."""
    p_cam = (T_cam_base[:3, :3] @ pts3d.T + T_cam_base[:3, 3:]).T
    p_img = K @ (p_cam / p_cam[:, 2:]).T
    return p_img[:2].T


def draw_quad(img, pts, color, thickness=2):
    p = pts.astype(int)
    for i in range(4):
        cv2.line(img, tuple(p[i]), tuple(p[(i + 1) % 4]), color, thickness)
    cv2.circle(img, tuple(p.mean(0).astype(int)), 5, color, -1)


def annotate_frame(img_bgr: np.ndarray,
                   det_corners: np.ndarray,
                   proj_corners: np.ndarray | None,
                   ref_corners:  np.ndarray | None,
                   label: str) -> np.ndarray:
    out = img_bgr.copy()
    draw_quad(out, det_corners,  (0, 200, 0))          # detected: GREEN
    if proj_corners is not None:
        draw_quad(out, proj_corners, (0, 0, 220))      # projected: RED
    if ref_corners is not None:
        draw_quad(out, ref_corners,  (220, 140, 0))    # reference: BLUE
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 50), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True,
                        help="Path to the raw Hydra ICP dataset directory")
    parser.add_argument("--robot",       required=True, choices=ROBOTS)
    parser.add_argument("--measurement", type=int, default=0)
    parser.add_argument("--out",         default="reprojection_out")
    args = parser.parse_args()

    base     = Path(args.dataset).resolve()
    meas_dir = base / args.robot / f"measurement_{args.measurement}"
    out_dir  = Path(args.out) / args.robot / f"measurement_{args.measurement}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibrating {args.robot}/measurement_{args.measurement} ...")
    detector = make_detector()
    T_cam_base, frames, K = solve_measurement(meas_dir, args.robot, detector, verbose=True)
    if T_cam_base is None:
        sys.exit("Calibration failed, nothing to draw.")

    # Reference GT, if present (meca/measurement_0 only).
    ref_path = meas_dir / "realsense_april" / "base2cam_calib.npy"
    T_ref = None
    if ref_path.exists():
        T_ref = np.load(ref_path)
        R_err = rot_err_deg(T_cam_base[:3, :3], T_ref[:3, :3])
        t_err = np.linalg.norm(T_cam_base[:3, 3] - T_ref[:3, 3]) * 1000
        print(f"Reference GT (base2cam_calib.npy): "
              f"R_err = {R_err:.3f} deg   t_err = {t_err:.1f} mm vs our result")
        print(f"  reference reproj = {_corner_reproj_px(T_ref, frames, K):.2f} px   "
              f"calibrated = {_corner_reproj_px(T_cam_base, frames, K):.2f} px")

    print(f"\n{'Frame':>5}  {'reproj (calib)':>14}  {'reproj (ref)':>13}  saved")
    print("-" * 52)

    for f in frames:
        img_path = meas_dir / "realsense_hydra" / f"camera.image_{f.idx}.png"
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  {f.idx:3d}  cannot read image, skipping")
            continue

        proj     = project_corners(T_cam_base, f.pts3d, K)
        err_ours = float(np.mean(np.linalg.norm(proj - f.pts2d, axis=1)))

        ref_proj, err_ref = None, float("nan")
        if T_ref is not None:
            ref_proj = project_corners(T_ref, f.pts3d, K)
            err_ref  = float(np.mean(np.linalg.norm(ref_proj - f.pts2d, axis=1)))

        label = (f"f{f.idx:02d}  reproj={err_ours:.1f}px"
                 + (f"  ref={err_ref:.1f}px" if T_ref is not None else ""))
        annotated = annotate_frame(img_bgr, f.pts2d, proj, ref_proj, label)

        out_path = out_dir / f"frame_{f.idx:02d}.png"
        cv2.imwrite(str(out_path), annotated)
        ref_col = f"{err_ref:>12.1f}" if T_ref is not None else "           n/a"
        print(f"  {f.idx:3d}  {err_ours:>13.1f}  {ref_col}  {out_path.name}")

    print(f"\nAnnotated images saved to: {out_dir.resolve()}")
    if T_ref is not None:
        print("GREEN = detected   RED = calibrated reproj   BLUE = reference GT")
    else:
        print("GREEN = detected   RED = calibrated reproj")


if __name__ == "__main__":
    main()
