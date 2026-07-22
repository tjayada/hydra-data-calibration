#!/usr/bin/env python3
"""Bootstrap T_cam_base for each (robot, measurement) in the Hydra ICP
benchmark via AprilTag multi-frame PnP on RealSense RGB images.

Method (mirrors Hydra paper Section III-A.1, 'marker-based PnP' benchmark):
  For each of the 15 FK configurations, detect the 4 AprilTag corners.
  Compute corresponding 3D positions in the robot base frame:
      T_tag2base = inv(base2ee) @ tag2ee
  Stack all (3D, 2D) pairs and run a single global cv2.solvePnP (SQPnP).

Conventions:
  base2ee = T_{EE<-base}: p_ee = base2ee @ p_base   (FK output)
  tag2ee  = T_{EE<-tag}:  p_ee = tag2ee  @ p_tag    (CAD)
  Output T_cam_base: p_cam = T_cam_base @ p_base    (float64, 4x4)

Two corrections to the released tag geometry (see README for the evidence):
  xArm: a second tag (id 1) sits on the back of the tag plate;
    xarm/tagzero2tagone.npy relates the two tag frames. Frames where only
    tag 1 is visible (xarm/measurement_2, frames 11-14) use the tag-1
    geometry tag2ee @ tagzero2tagone.
  LBR: the released tag2ee_cad.npy describes the tag frame flipped 180
    degrees about the tag's x-axis (predicted corners land on the opposite
    corners; reversing the correspondence drops the corner residual from
    ~40 px to ~1.5 px and reproduces Hydra Table I). Corrected by
    tag2ee @ Rx(180).

Evaluation (Hydra paper Section III-A.3, Table I protocol):
  Monte Carlo cross-validation: calibrate on N=9 configs, measure tag-centre
  reprojection on 6 held-out configs, repeat 5 times. Results pooled across
  all 3 measurements per robot, directly comparable to Hydra Table I.

Usage:
    python calibrate.py --dataset /path/to/24_12_19_hydra_icp_eval
"""

import argparse
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import yaml
from pyapriltags import Detector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Paper, Section III-B.3: "Each of the three serial manipulators had a
# 5 x 5 cm AprilTag [23] mounted as an end-effector (see Fig. 3). We used
# the apriltag software for pose estimation."
# The paper never names the tag family; tag36h11 is the apriltag library's
# recommended default and detects all frames, so it is assumed here.
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.05
HALF       = TAG_SIZE_M / 2.0

# Paper, Section III: "Each serial manipulator was observed from three
# randomly selected camera poses across 15 randomly chosen joint space
# configurations, resulting in a total of 270 unique samples."
N_CONFIGS  = 15
ROBOTS     = ["lbr", "meca", "xarm"]

# Corner positions in the tag's own frame.
# Matches pyapriltags ordering: bottom-left, bottom-right, top-right, top-left.
TAG_CORNERS_3D = np.array([
    [-HALF, -HALF, 0],
    [ HALF, -HALF, 0],
    [ HALF,  HALF, 0],
    [-HALF,  HALF, 0],
], dtype=np.float64)

_DIST = np.zeros(5, dtype=np.float32)   # confirmed: dataset YAMLs store d=[0,0,0,0,0] (plumb_bob)

# One calibration observation: FK-derived 3D geometry + detected 2D positions.
FrameObs = namedtuple("FrameObs", [
    "pts3d",    # (4, 3) float64  FK tag corners in robot base frame
    "pts2d",    # (4, 2) float64  detected 2D corners in image
    "centre3d", # (3,)   float64  FK tag centre in base frame (= pts3d.mean(0))
    "centre2d", # (2,)   float64  detector's homography-derived tag centre (r.center)
    "idx",      # int             config index (0..N_CONFIGS-1) the frame came from
])

# Hydra Table I, PnP + RealSense D435, N=9 held-out (Huber et al. 2025).
_PAPER_PNP = {
    "lbr":  (1.5, 0.7, 1.8, 0.9),   # (mean_px, std_px, mean_mm, std_mm)
    "meca": (1.0, 0.6, 0.6, 0.3),
    "xarm": (4.6, 1.9, 3.9, 2.0),
}

# N values for the Monte Carlo sweep, matching Hydra Fig. 5 (N in {3, 6, 9, 12}).
N_TRAIN_VALUES = [3, 6, 9, 12]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _T44(R, t) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(R, dtype=np.float64)
    M[:3,  3] = np.asarray(t, dtype=np.float64).ravel()
    return M


def rot_err_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def load_K(meas_dir: Path) -> np.ndarray:
    p = meas_dir / "realsense_hydra" / "camera.image.camera_info_0.yaml"
    with open(p) as f:
        return np.array(yaml.safe_load(f)["k"], dtype=np.float64).reshape(3, 3)


def make_detector() -> Detector:
    return Detector(families=TAG_FAMILY, nthreads=4,
                    quad_decimate=1.0, refine_edges=1, decode_sharpening=0.25)


# 180-degree rotation about the tag's x-axis. Corrects the LBR's released
# tag2ee_cad.npy, which describes the tag frame viewed from the back (see the
# module docstring and README for the evidence).
_RX180 = np.diag([1.0, -1.0, -1.0, 1.0])
TAG2EE_CORRECTION = {"lbr": _RX180}


def load_tag_geometry(meas_dir: Path) -> dict:
    """Map tag_id -> tag2ee (T_{EE<-tag}) for this measurement.

    Tag 0 uses realsense_april/tag2ee_cad.npy, right-multiplied by the
    robot's entry in TAG2EE_CORRECTION if any (the robot is the name of the
    measurement's parent directory). If the robot directory has a
    tagzero2tagone.npy (only the xArm does), tag 1's geometry is derived as
    tag2ee @ tagzero2tagone. That matrix is a 180-degree flip about the tag's
    y-axis plus a 5 mm plate offset and is its own inverse, so its direction
    convention does not matter.
    """
    tag2ee = np.load(meas_dir / "realsense_april" / "tag2ee_cad.npy")
    correction = TAG2EE_CORRECTION.get(meas_dir.parent.name)
    if correction is not None:
        tag2ee = tag2ee @ correction
    geometry = {0: tag2ee}
    tz_path = meas_dir.parent / "tagzero2tagone.npy"
    if tz_path.exists():
        geometry[1] = tag2ee @ np.load(tz_path)
    return geometry


# ---------------------------------------------------------------------------
# FK geometry
# ---------------------------------------------------------------------------

def corners_in_base(base2ee: np.ndarray, tag2ee: np.ndarray) -> np.ndarray:
    """(4, 3) tag corner positions in the robot base frame.

    T_tag2base = inv(base2ee) @ tag2ee
      base2ee = T_{EE<-base}, so inv(base2ee) = T_{base<-EE}
      tag2ee  = T_{EE<-tag}
      product = T_{base<-tag}  (maps tag-frame corners to base frame)
    """
    T = np.linalg.inv(base2ee) @ tag2ee
    return (T[:3, :3] @ TAG_CORNERS_3D.T + T[:3, 3:]).T


# ---------------------------------------------------------------------------
# PnP solver
# ---------------------------------------------------------------------------

def _solvepnp(pts3d: np.ndarray, pts2d: np.ndarray,
              K: np.ndarray) -> np.ndarray | None:
    """Global PnP via SQPnP (Terzakis & Lourakis 2020).

    Reproduces the marker-based PnP of Paper Section III-A.1 (each of the 4
    AprilTag corners in each config is one 3D point). The paper does not name
    the solver; SQPnP needs no initial estimate or RANSAC threshold.
    """
    ok, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float64), pts2d.astype(np.float64),
        K, _DIST, flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return _T44(R, tvec.ravel())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _corner_reproj_px(T: np.ndarray, frames: list, K: np.ndarray) -> float:
    """Mean corner reprojection error over all frames (px)."""
    pts3d = np.vstack([f.pts3d for f in frames])
    pts2d = np.vstack([f.pts2d for f in frames])
    p     = (T[:3, :3] @ pts3d.T + T[:3, 3:]).T
    p_img = K @ (p / p[:, 2:]).T
    return float(np.mean(np.linalg.norm(p_img[:2].T - pts2d, axis=1)))


def _centre_reproj_px(T: np.ndarray, frames: list, K: np.ndarray) -> np.ndarray:
    """Per-frame tag-centre reprojection error (px).

    Paper Section III-A.2 (quasi task space metric): project the FK-predicted
    tag centre through T_cam_base and measure the pixel distance to the
    detector's homography-derived centre (r.center).
    """
    errs = []
    for f in frames:
        p     = T[:3, :3] @ f.centre3d + T[:3, 3]
        p_img = K @ (p / p[2])
        errs.append(float(np.linalg.norm(p_img[:2] - f.centre2d)))
    return np.array(errs)


def _tag_scale_mm_per_px(frame: FrameObs) -> float:
    """mm-per-pixel scale from the detected tag boundary of one frame.

    Paper, Section III-A.2: "we additionally express ca'_i in coordinates
    rescaled according to the AprilTag boundaries, centered at a'_i (see
    Fig. 3). Simply computing the average length of ca'_i in this
    AprilTag-centric coordinate system and scaling it by the AprilTag size
    then allow the estimation of task space accuracy."

    The scale is the physical tag size over the mean detected side length.
    """
    side_px = np.mean([np.linalg.norm(frame.pts2d[(i + 1) % 4] - frame.pts2d[i])
                       for i in range(4)])
    return TAG_SIZE_M * 1000.0 / float(side_px)


def _px_to_mm(reproj_px: float, frames: list) -> float:
    """Convert a centre reprojection error from px to mm.

    Uses the paper's AprilTag-centric scale (see _tag_scale_mm_per_px),
    averaged over the given frames.
    """
    return reproj_px * float(np.mean([_tag_scale_mm_per_px(f) for f in frames]))


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------

def _collect_frames(meas_dir: Path, detector: Detector, tag_geometry: dict,
                    verbose: bool = False) -> tuple[list, np.ndarray]:
    """Detect AprilTags in all N_CONFIGS images; return (frames, K).

    tag_geometry maps tag_id -> tag2ee; detections of unknown tag ids are
    ignored. Per frame the known-id detection with the highest decision
    margin wins (in practice each Hydra frame contains exactly one tag).
    """
    K      = load_K(meas_dir)
    params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    all_b2e = [np.load(meas_dir / "realsense_april" / f"base2ee_{i}.npy")
               for i in range(N_CONFIGS)]
    frames, n_miss = [], 0

    for i in range(N_CONFIGS):
        img = cv2.imread(
            str(meas_dir / "realsense_hydra" / f"camera.image_{i}.png"),
            cv2.IMREAD_GRAYSCALE)
        if img is None:
            if verbose:
                print(f"    [{i:2d}] cannot read image, skipping")
            n_miss += 1
            continue

        dets = detector.detect(img, estimate_tag_pose=False,
                               camera_params=params, tag_size=TAG_SIZE_M)
        dets = [d for d in dets if d.tag_id in tag_geometry]
        if not dets:
            if verbose:
                print(f"    [{i:2d}] no known tag detected, skipping")
            n_miss += 1
            continue

        r   = max(dets, key=lambda x: x.decision_margin)
        c3d = corners_in_base(all_b2e[i], tag_geometry[r.tag_id])
        c2d = np.array(r.corners, dtype=np.float64)
        frames.append(FrameObs(c3d, c2d, c3d.mean(0),
                               np.array(r.center, dtype=np.float64), i))

    if verbose:
        print(f"    {len(frames)}/{N_CONFIGS} configs detected "
              f"({n_miss} missed)")
    return frames, K


# ---------------------------------------------------------------------------
# Solve (pure, no side-effects)
# ---------------------------------------------------------------------------

def _solve(frames: list, K: np.ndarray) -> np.ndarray | None:
    """T_cam_base from a global PnP over the given frames."""
    if not frames:
        return None
    pts3d = np.vstack([f.pts3d for f in frames]).astype(np.float64)
    pts2d = np.vstack([f.pts2d for f in frames]).astype(np.float64)
    return _solvepnp(pts3d, pts2d, K)


# ---------------------------------------------------------------------------
# Per-measurement calibration
# ---------------------------------------------------------------------------

def solve_measurement(meas_dir: Path, robot: str, detector: Detector,
                      verbose: bool = False,
                      ) -> tuple[np.ndarray | None, list, np.ndarray]:
    """Calibrate T_cam_base from all detected configs; return (T, frames, K)."""
    tag_geometry = load_tag_geometry(meas_dir)
    frames, K = _collect_frames(meas_dir, detector, tag_geometry, verbose=verbose)

    if not frames:
        print("  No tags detected, skipping")
        return None, frames, K

    T = _solve(frames, K)
    if T is None:
        print("  solvePnP failed, skipping")
        return None, frames, K

    cor = _corner_reproj_px(T, frames, K)
    cen = float(np.mean(_centre_reproj_px(T, frames, K)))
    if cor > max(5.0 * cen, 10.0):
        # This is how both released tag-geometry errors (xArm tag 1, LBR
        # flip) manifest: corners pair with the wrong counterparts while
        # the centre still fits.
        print(f"  WARNING: corner reproj ({cor:.1f} px) far exceeds centre "
              f"reproj ({cen:.2f} px); possible tag-frame flip or wrong "
              f"corner correspondence")

    if verbose:
        mm = _px_to_mm(cen, frames)
        print(f"  |t_cam| = {np.linalg.norm(T[:3, 3]):.3f} m  "
              f"corner = {cor:.1f} px  "
              f"centre = {cen:.2f} px ({mm:.1f} mm)")

    return T, frames, K


# ---------------------------------------------------------------------------
# Monte Carlo cross-validation (Hydra paper Section III-A.3)
# ---------------------------------------------------------------------------

def monte_carlo_eval(frames: list, K: np.ndarray,
                     n_train: int = 9, n_splits: int = 5,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate on n_train configs; evaluate tag-centre reproj on held-out.

    Monte Carlo cross-validation from Paper Section III-A.3: draw n_splits
    random train sets of size n_train from the 15 configs and measure the
    quasi task space metric on the held-out configs.

    Returns (split_means_px, split_means_mm), one scalar per successful split
    (the mean over that split's held-out samples). Pool across all 3
    measurements per robot before comparing to Hydra Table I.
    """
    if len(frames) < n_train + 1:
        return np.array([]), np.array([])

    rng = np.random.default_rng(seed)
    split_means_px, split_means_mm = [], []

    for _ in range(n_splits):
        idx = rng.permutation(len(frames))
        T   = _solve([frames[i] for i in idx[:n_train]], K)
        if T is None:
            continue
        split_px, split_mm = [], []
        for i in idx[n_train:]:
            f      = frames[i]
            p      = T[:3, :3] @ f.centre3d + T[:3, 3]
            p_img  = K @ (p / p[2])
            err_px = float(np.linalg.norm(p_img[:2] - f.centre2d))
            err_mm = err_px * _tag_scale_mm_per_px(f)
            split_px.append(err_px)
            split_mm.append(err_mm)
        if split_px:
            split_means_px.append(float(np.mean(split_px)))
            split_means_mm.append(float(np.mean(split_mm)))

    return np.array(split_means_px), np.array(split_means_mm)


# ---------------------------------------------------------------------------
# Validation against reference GT
# ---------------------------------------------------------------------------

def validate_against_reference(T_ours: np.ndarray, ref_path: Path) -> None:
    """Compare T_cam_base against the pre-computed reference (meca/meas_0 only)."""
    ref = np.load(ref_path)
    print(f"\n  Reference validation ({ref_path.name}):")
    for label, ref_T in [("direct  ", ref), ("inv(ref)", np.linalg.inv(ref))]:
        R_err = rot_err_deg(T_ours[:3, :3], ref_T[:3, :3])
        t_err = np.linalg.norm(T_ours[:3, 3] - ref_T[:3, 3]) * 1000
        flag  = "  OK" if R_err < 0.5 and t_err < 2.0 else ""
        print(f"    [{label}]  R = {R_err:.3f} deg   t = {t_err:.2f} mm{flag}")
    print("  (Reference has ~12 px corner reproj; ours has ~1 px; the "
          "difference reflects reference uncertainty.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the raw Hydra ICP dataset directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame detection detail and per-measurement metrics")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calibrate and evaluate but do not write .npy files")
    args = parser.parse_args()

    base = Path(args.dataset).resolve()
    if not base.exists():
        sys.exit(f"ERROR: dataset not found: {base}")

    detector = make_detector()

    # Per-measurement calibration + Monte Carlo evaluation.
    results = {}   # (robot, m) -> {"gt": ..., "mc_by_n": {n: (px_arr, mm_arr)}}

    for robot in ROBOTS:
        for m in range(3):
            meas_dir = base / robot / f"measurement_{m}"

            if args.verbose:
                print(f"\n-- {robot}/measurement_{m} " + "-" * 44)
            else:
                print(f"  {robot}/measurement_{m} ...", end=" ", flush=True)

            T, frames, K = solve_measurement(
                meas_dir, robot, detector,
                verbose=args.verbose,
            )

            # Reference validation (meca/measurement_0 only, the one file that exists).
            ref_path = meas_dir / "realsense_april" / "base2cam_calib.npy"
            if T is not None and ref_path.exists():
                validate_against_reference(T, ref_path)

            # Monte Carlo evaluation, sweeping N in {3,6,9,12} (Hydra Fig. 5 protocol).
            mc_by_n: dict = {}
            if T is not None:
                for n in N_TRAIN_VALUES:
                    mc_by_n[n] = monte_carlo_eval(frames, K, n_train=n)

            # Final GT metrics (all frames).
            gt_info = None
            if T is not None:
                cor    = _corner_reproj_px(T, frames, K)
                cen    = float(np.mean(_centre_reproj_px(T, frames, K)))
                cen_mm = _px_to_mm(cen, frames)
                gt_info = dict(
                    T=T, n=len(frames),
                    corner_px=cor, centre_px=cen, centre_mm=cen_mm,
                    cam_dist=float(np.linalg.norm(T[:3, 3])),
                )

            results[(robot, m)] = dict(gt=gt_info, mc_by_n=mc_by_n)

            if T is not None and not args.dry_run:
                out = meas_dir / "realsense_april" / "T_cam_base_pnp.npy"
                np.save(out, T)
                if not args.verbose:
                    print("saved")
                else:
                    print(f"  saved -> {out.relative_to(base)}")
            elif not args.verbose:
                print("FAILED" if T is None else "done (dry-run)")

    W   = 78
    SEP = "=" * W

    def _mc_pool(robot: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Pool per-split means across all 3 measurements for a given robot and n_train.

        Each element is the mean error over one held-out set (one split of one
        measurement). mean +/- std of this array matches the paper's "averaged
        over the sets" aggregation.
        """
        px = np.concatenate([results[(robot, m)]["mc_by_n"].get(n, (np.array([]), np.array([])))[0]
                             for m in range(3)])
        mm = np.concatenate([results[(robot, m)]["mc_by_n"].get(n, (np.array([]), np.array([])))[1]
                             for m in range(3)])
        return px, mm

    # Table 1: Monte Carlo cross-validation (comparable to Hydra Table I).
    print(f"\n{SEP}")
    print("  Monte Carlo cross-validation: N=9 train / 6 held-out, 5 splits")
    print("  Pooled across 3 measurements per robot, comparable to Hydra Table I")
    print(SEP)
    print(f"  {'Robot':<6}  {'Our result (this dataset)':^28}  |  "
          f"{'Hydra Table I  [PnP + RealSense]':^28}")
    print(f"  {'':6}  {'Centre (px)':>13} {'Centre (mm)':>13}  |  "
          f"{'Centre (px)':>13} {'Centre (mm)':>13}")
    print("  " + "-" * (W - 2))
    for robot in ROBOTS:
        all_px, all_mm = _mc_pool(robot, 9)
        ppx, pspx, pmm, psmm = _PAPER_PNP[robot]
        if len(all_px) == 0:
            print(f"  {robot:<6}  {'FAILED':^28}  |  "
                  f"{ppx:.1f} +/- {pspx:.1f}        {pmm:.1f} +/- {psmm:.1f}")
            continue
        mpx, spx = all_px.mean(), all_px.std()
        mmm, smm = all_mm.mean(), all_mm.std()
        print(f"  {robot:<6}  {mpx:>6.2f} +/- {spx:<5.2f}  "
              f"{mmm:>6.2f} +/- {smm:<5.2f}  |  "
              f"{ppx:>6.1f} +/- {pspx:<5.1f}  "
              f"{pmm:>6.1f} +/- {psmm:<5.1f}")
    print()
    print("  Paper: Huber et al. 2025 (Hydra), same PnP protocol + RealSense D435.")
    print("  Solver: SQPnP (Terzakis & Lourakis 2020) via cv2.SOLVEPNP_SQPNP.")

    # Table 1b: repeatability vs N (reproduces Hydra Fig. 5 quantitatively).
    print(f"\n{SEP}")
    print("  Repeatability vs. training size  (cf. Hydra Fig. 5)")
    print("  Pooled across 3 measurements per robot, centre reprojection (px +/- std)")
    print(SEP)
    col_w = 16
    hdr = f"  {'Robot':<6}"
    for n in N_TRAIN_VALUES:
        hdr += f"  {'N=' + str(n) + ' (train)':>{col_w}}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for robot in ROBOTS:
        row = f"  {robot:<6}"
        for n in N_TRAIN_VALUES:
            px, _ = _mc_pool(robot, n)
            if len(px) == 0:
                row += f"  {'FAILED':>{col_w}}"
            else:
                row += f"  {px.mean():>5.2f} +/- {px.std():<5.2f} "
        print(row)
    print(SEP)

    # Table 2: final GT saved to .npy (all frames).
    print(f"\n{SEP}")
    print("  Final GT: all frames, saved to T_cam_base_pnp.npy")
    print(SEP)
    print(f"  {'Robot':<6} {'Meas':>4}  {'N':>3}  "
          f"{'Corner (px)':>11}  {'Centre (px)':>11}  "
          f"{'Centre (mm)':>11}  {'|t_cam| (m)':>11}")
    print("  " + "-" * (W - 2))
    for robot in ROBOTS:
        for m in range(3):
            g = results[(robot, m)]["gt"]
            if g is None:
                print(f"  {robot:<6} {m:>4}  {'':>3}  FAILED")
                continue
            print(f"  {robot:<6} {m:>4}  {g['n']:>3}  "
                  f"{g['corner_px']:>11.2f}  {g['centre_px']:>11.2f}  "
                  f"{g['centre_mm']:>11.2f}  {g['cam_dist']:>11.3f}")
    if args.dry_run:
        print("\n  [dry-run: no .npy files written]")
    print(SEP)


if __name__ == "__main__":
    main()
