#!/usr/bin/env python3
"""Validate that pyproject.toml's package-data alone ships the required data files.

Two mechanisms can silently mask a broken ``[tool.setuptools.package-data]``
declaration and bundle the .pt/.npy model files anyway:

  1. The git file-finder — when ``.git/`` is present setuptools bundles ALL
     git-tracked files inside a discovered package, ignoring package-data.
  2. ``MANIFEST.in`` — with include_package_data (the default), files listed
     there that fall inside a package are included regardless of package-data.

To validate package-data on its own, this script copies the source WITHOUT
``.git`` and WITHOUT ``MANIFEST.in``, builds a wheel from that copy, and checks
that the required data files really landed inside it. A broken package-data path
is therefore flagged as a failure.

Cross-platform (used by CI on Linux and Windows). Fast: builds with --no-deps,
so no heavy dependencies (torch etc.) are downloaded.

Exit code 0 if all required files are present, 1 otherwise.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Data files that MUST be present in a correctly-built wheel, as they appear
# inside the wheel (i.e. relative to site-packages).
REQUIRED_DATA_FILES = [
    "pypef/plm/prosst_structure/static/AE.pt",
    "pypef/plm/prosst_structure/static/AE_CPU.pt",
    "pypef/plm/prosst_structure/static/2048_kmeans_cluster_centers.npy",
    "pypef/ml/AAindex/ANDN920101.txt",
    "pypef/ml/AAindex/Refined_cluster_indices_r0.93_r0.97/0.93_0.97_1_0_2.txt",
    "pypef/gui/assets/pypef_logo.jpg",
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pypef_pkg_check_"))
    try:
        src = tmp / "src"
        src.mkdir()

        # Copy only pyproject.toml + README + the pypef package.
        # Deliberately NOT copying .git or MANIFEST.in so that only
        # pyproject.toml's package-data governs what ships in the wheel.
        shutil.copy(REPO_ROOT / "pyproject.toml", src / "pyproject.toml")
        for optional in ("README.md", "requirements.txt"):
            p = REPO_ROOT / optional
            if p.exists():
                shutil.copy(p, src / optional)
        shutil.copytree(
            REPO_ROOT / "pypef",
            src / "pypef",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )

        wheel_dir = tmp / "wheel"
        wheel_dir.mkdir()

        print("Building wheel from git-less / MANIFEST-less source copy...")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             "-w", str(wheel_dir), str(src)],
            check=True,
        )

        wheels = list(wheel_dir.glob("pypef-*.whl"))
        if not wheels:
            print("FAIL: no pypef wheel was produced.", file=sys.stderr)
            return 1

        names = set(zipfile.ZipFile(wheels[0]).namelist())
        missing = [f for f in REQUIRED_DATA_FILES if f not in names]
        if missing:
            print("FAIL: required data files MISSING from the built wheel:")
            for m in missing:
                print(f"  - {m}")
            print("\n→ Fix [tool.setuptools.package-data] in pyproject.toml so that")
            print("  package-data alone includes these files (independent of git /")
            print("  MANIFEST.in).")
            return 1

        print(f"OK: all {len(REQUIRED_DATA_FILES)} required data files present in wheel:")
        for f in REQUIRED_DATA_FILES:
            print(f"  - {f}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
