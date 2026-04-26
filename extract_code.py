"""
Phase 8 — Code Extraction Script
================================
Extracts all fenced code blocks from phase markdown files into actual
project files.  Each phase document contains the final version of each
file; later phases override earlier ones.

Usage:
    python extract_code.py
    python extract_code.py --dry-run     # Show what would be created
    python extract_code.py --phase 6     # Extract only from phase6.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


PHASE_ORDER = [
    "phase1.md",
    "phase2.md",
    "phase3.md",
    "phase4.md",
    "phase5.md",
    "phase6.md",
    "phase7.md",
]

HEADING_TO_FILE = {
    "domain_config.py": "domain_config.py",
    "models.py": "models.py",
    "tasks.py": "tasks.py",
    "rewards.py": "rewards.py",
    "graders.py": "graders.py",
    "environment.py": "environment.py",
    "server.py": "server.py",
    "inference.py": "inference.py",
    "validate.py": "validate.py",
    "run_submission.py": "run_submission.py",
    "calibrate.py": "calibrate.py",
    "stress_test.py": "stress_test.py",
    "submission_report.py": "submission_report.py",
    "requirements.txt": "requirements.txt",
    "Dockerfile": "Dockerfile",
    "README.md": "README.md",
    "openenv.yaml": "openenv.yaml",
    ".gitignore": ".gitignore",
    "manifest.json": "data/manifest.json",
    "tests/__init__.py": "tests/__init__.py",
    "tests/test_phase1.py": "tests/test_phase1.py",
    "tests/test_phase2.py": "tests/test_phase2.py",
    "tests/test_phase3.py": "tests/test_phase3.py",
    "tests/test_phase4.py": "tests/test_phase4.py",
    "tests/test_phase5.py": "tests/test_phase5.py",
    "tests/test_phase6.py": "tests/test_phase6.py",
    "tests/test_phase7.py": "tests/test_phase7.py",
    "tests/test_phase8.py": "tests/test_phase8.py",
}


def extract_blocks_from_md(md_path: str) -> List[Tuple[str, str, str]]:
    """Parse a markdown file and extract (heading, language, code) tuples."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks: List[Tuple[str, str, str]] = []
    lines = content.split("\n")
    current_heading = ""
    in_block = False
    block_lang = ""
    block_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^###\s+`?([^`\n]+)`?\s*$", line)
        if heading_match and not in_block:
            current_heading = heading_match.group(1).strip()
            continue

        fence_start = re.match(r"^```(\w*)\s*$", line)
        if fence_start and not in_block:
            in_block = True
            block_lang = fence_start.group(1) or "text"
            block_lines = []
            continue

        if line.strip() == "```" and in_block:
            in_block = False
            code = "\n".join(block_lines)
            if code.strip():
                blocks.append((current_heading, block_lang, code))
            continue

        if in_block:
            block_lines.append(line)

    return blocks


def match_heading_to_file(heading: str) -> Optional[str]:
    """Map a markdown heading to a target file path."""
    heading_clean = heading.strip().rstrip(":")

    if heading_clean in HEADING_TO_FILE:
        return HEADING_TO_FILE[heading_clean]

    for key, path in HEADING_TO_FILE.items():
        if key in heading_clean or heading_clean.endswith(key):
            return path

    scenario_match = re.search(
        r"(task_\d+_\w+)/?(scenario_\d+\.json)", heading_clean
    )
    if scenario_match:
        return f"data/{scenario_match.group(1)}/{scenario_match.group(2)}"

    return None


def extract_all(
    phases: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Extract code from phase markdown files."""
    files: Dict[str, str] = {}

    phase_list = phases or PHASE_ORDER

    for phase_file in phase_list:
        if not os.path.exists(phase_file):
            print(f"  SKIP: {phase_file} not found")
            continue

        blocks = extract_blocks_from_md(phase_file)
        print(f"\n  {phase_file}: {len(blocks)} code blocks found")

        for heading, lang, code in blocks:
            target = match_heading_to_file(heading)
            if target is None:
                if lang == "python" and "def " in code:
                    for sig, path in [
                        ("class ContractReviewEnv", "environment.py"),
                        ("class Action", "models.py"),
                        ("TASK_REGISTRY", "tasks.py"),
                        ("def grade_episode", "graders.py"),
                        ("def compute_classify_reward", "rewards.py"),
                        ("app = FastAPI", "server.py"),
                        ("def call_llm", "inference.py"),
                        ("class Validator", "validate.py"),
                        ("def run_calibration", "calibrate.py"),
                        ("def run_all_stress_tests", "stress_test.py"),
                        ("def generate_report", "submission_report.py"),
                    ]:
                        if sig in code:
                            target = path
                            break

            if target is None:
                if lang in ("bash", ""):
                    continue
                print(f"    ? Unmapped: '{heading}' ({lang}, {len(code)} chars)")
                continue

            files[target] = code
            action = "WOULD CREATE" if dry_run else "→"
            print(f"    {action} {target} ({len(code)} chars)")

    return files


def write_files(files: Dict[str, str]):
    """Write extracted files to disk."""
    created = 0
    for filepath, content in files.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
        created += 1

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Extract code from phase markdown files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing",
    )
    parser.add_argument(
        "--phase", type=int,
        help="Extract only from specific phase (e.g. --phase 6)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8 — CODE EXTRACTION")
    print("=" * 60)

    phases = None
    if args.phase:
        phases = [f"phase{args.phase}.md"]

    files = extract_all(phases=phases, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n  DRY RUN: {len(files)} files would be created.")
        return

    if not files:
        print("\n  No files to write.")
        return

    print(f"\nWriting {len(files)} files...")
    created = write_files(files)
    print(f"  {created} files created/updated.")

    init_path = "tests/__init__.py"
    if not os.path.exists(init_path):
        os.makedirs("tests", exist_ok=True)
        with open(init_path, "w") as f:
            f.write("")
        print(f"  Created {init_path}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python -m pytest tests/ -v")
    print("  3. python server.py")
    print("  4. python validate.py")
    print("  5. python calibrate.py")
    print("  6. python submission_report.py")


if __name__ == "__main__":
    main()
