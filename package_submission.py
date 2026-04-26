"""
Phase 9 — Submission Package Creator
====================================
Creates a clean, self-contained submission archive.

Usage:
    python package_submission.py
    python package_submission.py --format zip
    python package_submission.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import zipfile
from typing import List


SUBMISSION_FILES = [
    "inference.py",
    "server.py",
    "environment.py",
    "models.py",
    "tasks.py",
    "graders.py",
    "rewards.py",
    "domain_config.py",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "README.md",
    ".gitignore",
]

SUBMISSION_DIRS = [
    "data",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".pytest_cache",
    ".egg-info",
    ".env",
    "*.log",
    ".DS_Store",
]


def collect_files() -> List[str]:
    """Collect all files that should be in the submission."""
    files = []

    for f in SUBMISSION_FILES:
        if os.path.exists(f):
            files.append(f)
        else:
            print(f"  WARNING: Missing file: {f}")

    for d in SUBMISSION_DIRS:
        if os.path.isdir(d):
            for root, dirs, filenames in os.walk(d):
                dirs[:] = [dd for dd in dirs if not any(p in dd for p in EXCLUDE_PATTERNS)]
                for fn in filenames:
                    if not any(fn.endswith(p) for p in [".pyc", ".pyo"]):
                        files.append(os.path.join(root, fn))
        else:
            print(f"  WARNING: Missing directory: {d}")

    return sorted(files)


def create_tar(files: List[str], output: str):
    """Create a .tar.gz archive."""
    with tarfile.open(output, "w:gz") as tar:
        for f in files:
            tar.add(f)
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"  Created: {output} ({size_mb:.2f} MB, {len(files)} files)")


def create_zip(files: List[str], output: str):
    """Create a .zip archive."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f)
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"  Created: {output} ({size_mb:.2f} MB, {len(files)} files)")


def verify_package(archive_path: str) -> bool:
    """Verify a submission package has all required files."""
    print(f"\nVerifying: {archive_path}")

    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
    else:
        print("  ERROR: Unknown archive format")
        return False

    issues = 0

    for f in SUBMISSION_FILES:
        if f not in members:
            print(f"  ✗ Missing: {f}")
            issues += 1
        else:
            print(f"  ✓ Found: {f}")

    data_files = [m for m in members if m.startswith("data/")]
    if len(data_files) < 4:
        print(f"  ✗ Data directory incomplete ({len(data_files)} files)")
        issues += 1
    else:
        print(f"  ✓ Data directory: {len(data_files)} files")

    if "data/manifest.json" in members:
        print("  ✓ Manifest present")
    else:
        print("  ✗ Missing manifest")
        issues += 1

    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    if size_mb > 50:
        print(f"  ✗ Package too large: {size_mb:.1f} MB (limit: 50 MB)")
        issues += 1
    else:
        print(f"  ✓ Package size: {size_mb:.2f} MB")

    bad_files = [m for m in members if any(p in m for p in ["__pycache__", ".pyc", "phase", "test_"])]
    if bad_files:
        print(f"  ⚠ Potentially unwanted files: {bad_files[:5]}")

    total = len(SUBMISSION_FILES) + 2
    print(f"\n  Result: {total - issues}/{total} checks passed")

    return issues == 0


def main():
    parser = argparse.ArgumentParser(description="Create submission package")
    parser.add_argument("--format", choices=["tar", "zip"], default="tar")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--verify", help="Verify an existing package")
    parser.add_argument("--list", action="store_true", help="List files that would be included")
    args = parser.parse_args()

    print("=" * 60)
    print("SUBMISSION PACKAGE CREATOR")
    print("=" * 60)

    if args.verify:
        ok = verify_package(args.verify)
        sys.exit(0 if ok else 1)

    files = collect_files()

    if args.list:
        print(f"\n{len(files)} files would be included:")
        for f in files:
            size = os.path.getsize(f)
            print(f"  {f} ({size:,} bytes)")
        total = sum(os.path.getsize(f) for f in files)
        print(f"\nTotal: {total / 1024:.1f} KB uncompressed")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.format == "zip":
        output = args.output or f"contract-clause-review-{timestamp}.zip"
        create_zip(files, output)
    else:
        output = args.output or f"contract-clause-review-{timestamp}.tar.gz"
        create_tar(files, output)

    print("\nVerifying package...")
    ok = verify_package(output)

    if ok:
        print("\n✓ Package is ready for submission!")
    else:
        print("\n⚠ Package has issues — review before submitting!")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
