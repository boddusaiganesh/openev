"""
Phase 6 — Full Submission Pipeline
Runs everything needed for a clean submission:
  1. Local file validation
  2. Environment unit tests
  3. Docker build
  4. Server launch + server validation
  5. Dry-run inference (no LLM)
  6. Summary report

Usage:
    python run_submission.py
    python run_submission.py --skip-docker
    python run_submission.py --include-inference
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx


class StepResult:
    def __init__(self, name: str, passed: bool, message: str = "", elapsed: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.elapsed = elapsed


def run_cmd(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def wait_for_server(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def step_local_validation() -> StepResult:
    """Run local file checks via validate.py."""
    start = time.time()
    try:
        result = run_cmd([sys.executable, "validate.py", "--local"], timeout=30)
        ok = result.returncode == 0 and "FAIL" not in result.stdout
        return StepResult(
            "Local Validation",
            ok,
            result.stdout[-300:] if not ok else "All local checks passed.",
            time.time() - start,
        )
    except Exception as e:
        return StepResult("Local Validation", False, str(e), time.time() - start)


def step_env_validation() -> StepResult:
    """Run environment tests via validate.py."""
    start = time.time()
    try:
        result = run_cmd([sys.executable, "validate.py", "--env"], timeout=60)
        ok = result.returncode == 0 and "FAIL" not in result.stdout
        return StepResult(
            "Environment Tests",
            ok,
            result.stdout[-300:] if not ok else "All environment tests passed.",
            time.time() - start,
        )
    except Exception as e:
        return StepResult("Environment Tests", False, str(e), time.time() - start)


def step_pytest() -> StepResult:
    """Run the full pytest suite."""
    start = time.time()
    try:
        result = run_cmd(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            timeout=120,
        )
        ok = result.returncode == 0
        return StepResult(
            "Pytest Suite",
            ok,
            result.stdout[-500:] if not ok else "All tests passed.",
            time.time() - start,
        )
    except subprocess.TimeoutExpired:
        return StepResult("Pytest Suite", False, "Timed out after 120s", time.time() - start)
    except Exception as e:
        return StepResult("Pytest Suite", False, str(e), time.time() - start)


def step_docker_build() -> StepResult:
    """Build Docker image."""
    start = time.time()
    try:
        result = run_cmd(
            ["docker", "build", "-t", "contract-review-submission", "."],
            timeout=180,
        )
        ok = result.returncode == 0
        if ok:
            run_cmd(["docker", "rmi", "contract-review-submission"], timeout=30)
        return StepResult(
            "Docker Build",
            ok,
            "" if ok else result.stderr[-300:],
            time.time() - start,
        )
    except FileNotFoundError:
        return StepResult("Docker Build", False, "Docker not found", time.time() - start)
    except subprocess.TimeoutExpired:
        return StepResult("Docker Build", False, "Timed out", time.time() - start)


def step_server_test() -> StepResult:
    """Start server, run validation, stop server."""
    start = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "7865"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        url = "http://127.0.0.1:7865"
        if not wait_for_server(url, timeout=15):
            return StepResult("Server Test", False, "Server did not start", time.time() - start)

        result = run_cmd(
            [sys.executable, "validate.py", "--server", "--url", url],
            timeout=60,
        )
        ok = result.returncode == 0 and "FAIL" not in result.stdout

        return StepResult(
            "Server Tests",
            ok,
            result.stdout[-300:] if not ok else "All server tests passed.",
            time.time() - start,
        )
    except Exception as e:
        return StepResult("Server Test", False, str(e), time.time() - start)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def step_dry_run_inference() -> StepResult:
    """Run inference with mocked LLM (no API call)."""
    start = time.time()
    try:
        from environment import ContractReviewEnv
        from models import Action, ActionType
        from tasks import get_task_config

        env = ContractReviewEnv()
        results = {}

        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            obs = env.reset(tid)
            cfg = get_task_config(tid)
            required = cfg.required_action_types

            for i in range(obs.total_clauses):
                if env.done:
                    break
                gt = env.scenario.clauses[i]

                for at in required:
                    if env.done:
                        break
                    if at == ActionType.CLASSIFY:
                        env.step(Action(action_type=at, clause_type=gt.clause_type))
                    elif at == ActionType.RATE_SEVERITY:
                        env.step(Action(action_type=at, risk_level=gt.risk_level))
                    elif at == ActionType.FLAG:
                        env.step(Action(action_type=at, flags=gt.issues))
                    elif at == ActionType.SUGGEST:
                        env.step(Action(action_type=at, suggested_action=gt.recommended_action))
                    elif at == ActionType.REASON:
                        env.step(Action(action_type=at, reasoning=" ".join(gt.reasoning_keywords)))

                if not env.done and i < obs.total_clauses - 1:
                    env.step(Action(action_type=ActionType.NEXT_CLAUSE))

            if not env.done:
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            score = env.grader_result.score if env.grader_result else 0.0
            results[tid] = score

        msg = " | ".join(f"{k}={v:.4f}" for k, v in results.items())
        all_ok = all(v >= 0.5 for v in results.values())
        return StepResult(
            "Dry-Run Inference",
            all_ok,
            msg,
            time.time() - start,
        )
    except Exception as e:
        return StepResult("Dry-Run Inference", False, str(e), time.time() - start)


def step_live_inference() -> StepResult:
    """Run actual LLM inference (requires API credentials)."""
    start = time.time()
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME")

    if not api_key or not model:
        return StepResult(
            "Live Inference",
            False,
            "HF_TOKEN and MODEL_NAME required. Skipped.",
            0.0,
        )

    try:
        result = run_cmd(
            [sys.executable, "inference.py"],
            timeout=1200,
        )
        ok = result.returncode == 0
        return StepResult(
            "Live Inference",
            ok,
            result.stdout[-500:] if ok else result.stderr[-300:],
            time.time() - start,
        )
    except subprocess.TimeoutExpired:
        return StepResult("Live Inference", False, "Exceeded 20 min", time.time() - start)
    except Exception as e:
        return StepResult("Live Inference", False, str(e), time.time() - start)


def main():
    parser = argparse.ArgumentParser(description="Full submission pipeline")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker build")
    parser.add_argument("--skip-server", action="store_true", help="Skip server tests")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest")
    parser.add_argument("--include-inference", action="store_true", help="Run live LLM inference")
    args = parser.parse_args()

    print("=" * 72)
    print("CONTRACT CLAUSE REVIEW — SUBMISSION PIPELINE")
    print("=" * 72)

    steps: List[StepResult] = []

    print("\n[1/6] Running local validation...")
    steps.append(step_local_validation())
    _print_step(steps[-1])

    print("\n[2/6] Running environment tests...")
    steps.append(step_env_validation())
    _print_step(steps[-1])

    if not args.skip_pytest:
        print("\n[3/6] Running pytest suite...")
        steps.append(step_pytest())
        _print_step(steps[-1])
    else:
        print("\n[3/6] Pytest — SKIPPED")

    if not args.skip_docker:
        print("\n[4/6] Building Docker image...")
        steps.append(step_docker_build())
        _print_step(steps[-1])
    else:
        print("\n[4/6] Docker build — SKIPPED")

    if not args.skip_server:
        print("\n[5/6] Running server tests...")
        steps.append(step_server_test())
        _print_step(steps[-1])
    else:
        print("\n[5/6] Server tests — SKIPPED")

    print("\n[6/6] Running dry-run inference...")
    steps.append(step_dry_run_inference())
    _print_step(steps[-1])

    if args.include_inference:
        print("\n[BONUS] Running live LLM inference...")
        steps.append(step_live_inference())
        _print_step(steps[-1])

    _print_summary(steps)


def _print_step(result: StepResult):
    symbol = "✓" if result.passed else "✗"
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{symbol}] {status}: {result.name} ({result.elapsed:.1f}s)")
    if result.message and not result.passed:
        for line in result.message.strip().split("\n")[-3:]:
            print(f"       {line}")


def _print_summary(steps: List[StepResult]):
    print("\n" + "=" * 72)
    print("SUBMISSION PIPELINE — SUMMARY")
    print("=" * 72)

    passed = sum(1 for s in steps if s.passed)
    failed = sum(1 for s in steps if not s.passed)
    total_time = sum(s.elapsed for s in steps)

    for s in steps:
        symbol = "✓" if s.passed else "✗"
        status = "PASS" if s.passed else "FAIL"
        print(f"  [{symbol}] {status}: {s.name} ({s.elapsed:.1f}s)")

    print(f"\n  Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print(f"  Total time: {total_time:.1f}s")

    if failed == 0:
        print("\n  ✓ ALL CHECKS PASSED — Ready for submission!")
        print("  Next steps:")
        print("    1. Push to Hugging Face Space")
        print("    2. Verify Space deploys and responds")
        print("    3. Run: python inference.py  (with LLM credentials)")
        print("    4. Submit")
    else:
        print(f"\n  ⚠ {failed} CHECK(S) FAILED — Fix before submitting!")

    print("=" * 72)


if __name__ == "__main__":
    main()
