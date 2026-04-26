"""
Phase 7 — Final Submission Readiness Report
===========================================
Runs calibration + stress tests + file validation, generates a
comprehensive report summarizing submission readiness.

Usage:
    python submission_report.py
    python submission_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from calibrate import run_calibration, CalibrationResult
from stress_test import run_all_stress_tests, StressResult


REQUIRED_FILES = [
    "inference.py", "server.py", "environment.py", "models.py",
    "tasks.py", "graders.py", "rewards.py", "Dockerfile",
    "requirements.txt", "openenv.yaml", "README.md",
    "validate.py", "run_submission.py", ".gitignore",
    "data/manifest.json",
]


def check_files() -> Dict[str, bool]:
    return {f: os.path.exists(f) for f in REQUIRED_FILES}


def check_scenario_files() -> Dict[str, bool]:
    results = {}
    try:
        with open("data/manifest.json") as f:
            manifest = json.load(f)
        for tid, entry in manifest.items():
            for sf in entry.get("scenario_files", []):
                path = os.path.join("data", sf)
                results[path] = os.path.exists(path)
    except Exception:
        results["data/manifest.json"] = False
    return results


def check_no_secrets() -> bool:
    import re
    pattern = re.compile(r'(hf_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{30,})')
    source_files = [
        "inference.py", "server.py", "environment.py",
        "models.py", "tasks.py", "graders.py", "rewards.py",
    ]
    for fp in source_files:
        if os.path.exists(fp):
            with open(fp) as f:
                if pattern.findall(f.read()):
                    return False
    return True


def generate_report(
    cal_results: List[CalibrationResult],
    stress_results: List[StressResult],
    file_checks: Dict[str, bool],
    scenario_checks: Dict[str, bool],
    no_secrets: bool,
) -> Dict[str, Any]:
    cal_ok = all(cr.monotonic and cr.spread > 0.3 for cr in cal_results)
    stress_ok = all(sr.passed for sr in stress_results)
    files_ok = all(file_checks.values())
    scenarios_ok = all(scenario_checks.values())

    sections = {
        "calibration": {
            "status": "PASS" if cal_ok else "FAIL",
            "tasks": [
                {
                    "task_id": cr.task_id,
                    "scores": {
                        "perfect": cr.perfect, "good": cr.good,
                        "partial": cr.partial, "poor": cr.poor,
                        "empty": cr.empty,
                    },
                    "monotonic": cr.monotonic,
                    "spread": cr.spread,
                }
                for cr in cal_results
            ],
        },
        "stress_tests": {
            "status": "PASS" if stress_ok else "FAIL",
            "passed": sum(1 for sr in stress_results if sr.passed),
            "failed": sum(1 for sr in stress_results if not sr.passed),
            "details": [
                {"name": sr.name, "passed": sr.passed, "message": sr.message}
                for sr in stress_results
            ],
        },
        "files": {
            "status": "PASS" if files_ok else "FAIL",
            "missing": [f for f, ok in file_checks.items() if not ok],
        },
        "scenarios": {
            "status": "PASS" if scenarios_ok else "FAIL",
            "missing": [f for f, ok in scenario_checks.items() if not ok],
        },
        "security": {
            "status": "PASS" if no_secrets else "FAIL",
            "no_hardcoded_secrets": no_secrets,
        },
    }

    overall = all([cal_ok, stress_ok, files_ok, scenarios_ok, no_secrets])

    return {
        "overall": "READY" if overall else "NOT READY",
        "sections": sections,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def print_report(report: Dict[str, Any]):
    print("\n" + "═" * 72)
    print("FINAL SUBMISSION READINESS REPORT")
    print("═" * 72)

    for section_name, section in report["sections"].items():
        status = section["status"]
        symbol = "✓" if status == "PASS" else "✗"
        print(f"\n  [{symbol}] {section_name.upper()}: {status}")

        if section_name == "calibration":
            for t in section.get("tasks", []):
                scores = t["scores"]
                mono = "✓" if t["monotonic"] else "✗"
                print(f"      {t['task_id']}: "
                      f"P={scores['perfect']:.3f} G={scores['good']:.3f} "
                      f"H={scores['partial']:.3f} B={scores['poor']:.3f} "
                      f"E={scores['empty']:.3f} spread={t['spread']:.3f} [{mono}]")

        elif section_name == "stress_tests":
            print(f"      {section['passed']} passed, {section['failed']} failed")
            for d in section.get("details", []):
                if not d["passed"]:
                    print(f"      ✗ {d['name']}: {d['message']}")

        elif section_name in ("files", "scenarios"):
            missing = section.get("missing", [])
            if missing:
                for m in missing:
                    print(f"      ✗ Missing: {m}")

    overall = report["overall"]
    symbol = "✓" if overall == "READY" else "✗"
    print(f"\n{'═' * 72}")
    print(f"  [{symbol}] OVERALL: {overall}")

    if overall == "READY":
        print("\n  ✓ Environment is READY for submission!")
        print("  Next steps:")
        print("    1. git add . && git commit -m 'Submission ready'")
        print("    2. Push to Hugging Face Space")
        print("    3. Verify Space deploys")
        print("    4. Run: python inference.py (with API credentials)")
        print("    5. Submit")
    else:
        print("\n  ⚠ Fix the issues above before submitting!")

    print("═" * 72)


def main():
    parser = argparse.ArgumentParser(description="Submission readiness report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("\n" + "═" * 72)
    print("PHASE 7 — SUBMISSION READINESS")
    print("═" * 72)

    print("\n[1/4] Running calibration...")
    cal_results = run_calibration(verbose=args.verbose)
    cal_ok = all(cr.monotonic and cr.spread > 0.3 for cr in cal_results)
    print(f"  Calibration: {'PASS' if cal_ok else 'FAIL'}")

    print("\n[2/4] Running stress tests...")
    stress_results = run_all_stress_tests(verbose=args.verbose)
    stress_ok = all(sr.passed for sr in stress_results)
    print(f"  Stress tests: {'PASS' if stress_ok else 'FAIL'}")

    print("\n[3/4] Checking files...")
    file_checks = check_files()
    files_ok = all(file_checks.values())
    print(f"  Files: {'PASS' if files_ok else 'FAIL'}")

    print("\n[4/4] Checking scenarios...")
    scenario_checks = check_scenario_files()
    scenarios_ok = all(scenario_checks.values())
    print(f"  Scenarios: {'PASS' if scenarios_ok else 'FAIL'}")

    no_secrets = check_no_secrets()
    print(f"  Security: {'PASS' if no_secrets else 'FAIL'}")

    report = generate_report(cal_results, stress_results, file_checks, scenario_checks, no_secrets)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    sys.exit(0 if report["overall"] == "READY" else 1)


if __name__ == "__main__":
    main()
