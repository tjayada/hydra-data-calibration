#!/usr/bin/env python3
"""Download and unpack the raw Hydra ICP benchmark dataset (658 MB zip).

The data is hosted on Google Drive and was shared by the Hydra authors for
research use; see DATA_NOTICE.md for provenance and terms. The file is too
large for Google's virus scan, so a plain HTTP request hits a confirmation
page; gdown handles the confirmation token.

The zip is verified against a pinned SHA-256, unpacked, and normalised to
<repo>/24_12_19_hydra_icp_eval/ (the --dataset path the pipeline scripts
expect). Safe to re-run: skips when the dataset directory already exists.

Usage:
    python download_data.py
    python download_data.py --force       # re-download and re-unpack
    python download_data.py --keep-zip    # keep the zip after unpacking
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent
FILE_ID     = "1WAU9EgYlW-wVcVWL_4bldZYYWfXXX5eM"
ZIP_NAME    = "24_12_19_hydra_icp_eval.zip"
DATASET_DIR = REPO_ROOT / "24_12_19_hydra_icp_eval"

# Pinned on 2026-07-18 from a fresh download of the Drive file above.
ZIP_SHA256 = "1ff0bfc6f2a93977f43b96a8f355c636f283c83caae059fe49318ae980918818"

NOTICE = """
----------------------------------------------------------------------------
Data provenance notice

This dataset was created by the Hydra authors (Huber et al. 2025, KCL) and
has not been officially published by them. It is redistributed here with the
permission of Martin Huber (martin.huber@kcl.ac.uk) for research purposes.
All rights to the data remain with the Hydra authors. Please cite the Hydra
paper (arXiv:2504.20584) when using it, and contact the authors for any use
beyond reproducing this benchmark. Full notice: DATA_NOTICE.md
----------------------------------------------------------------------------
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(zip_path: Path) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("ERROR: gdown is not installed. Install it with:  pip install gdown")

    print(f"Downloading {ZIP_NAME} (658 MB) from Google Drive ...")
    out = gdown.download(id=FILE_ID, output=str(zip_path), quiet=False)
    if out is None or not zip_path.exists():
        sys.exit(
            "ERROR: download failed. Google Drive quota errors are common for "
            "large shared files; wait a day and retry, or download the file "
            f"manually (file id {FILE_ID}) and place it at:\n  {zip_path}\n"
            "then re-run this script."
        )


def verify_zip(zip_path: Path) -> None:
    print("Verifying SHA-256 ...")
    got = _sha256(zip_path)
    if got != ZIP_SHA256:
        sys.exit(
            f"ERROR: SHA-256 mismatch for {zip_path.name}:\n"
            f"  expected {ZIP_SHA256}\n"
            f"  got      {got}\n"
            "The hosted file has changed since this script was pinned, or the "
            "download is corrupt. Delete the zip and retry; if the mismatch "
            "persists, contact the maintainer."
        )
    print("  OK")


def unpack_zip(zip_path: Path) -> None:
    """Unpack and normalise the top-level folder name to DATASET_DIR."""
    print(f"Unpacking to {DATASET_DIR} ...")
    tmp_dir = DATASET_DIR.parent / (DATASET_DIR.name + ".unpacking")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    # The zip may contain a single wrapper directory of any name; the robot
    # folders (lbr/meca/xarm) are what the pipeline needs directly under the
    # dataset root.
    entries = [p for p in tmp_dir.iterdir() if not p.name.startswith(".")]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_dir
    if not (root / "lbr").is_dir():
        sys.exit(f"ERROR: unexpected zip layout, no lbr/ found under {root}")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    root.rename(DATASET_DIR)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print("  OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-unpack even if the dataset directory exists")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the downloaded zip after unpacking (default: delete)")
    args = parser.parse_args()

    if DATASET_DIR.exists() and not args.force:
        print(f"Dataset already present at {DATASET_DIR}  (use --force to re-download)")
        return

    zip_path = REPO_ROOT / ZIP_NAME
    if not zip_path.exists() or args.force:
        download_zip(zip_path)
    else:
        print(f"Using existing zip {zip_path.name}")

    verify_zip(zip_path)
    unpack_zip(zip_path)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Deleted {zip_path.name}  (use --keep-zip to keep it)")

    print(NOTICE)
    print(f"Done. Dataset at: {DATASET_DIR}")


if __name__ == "__main__":
    main()
