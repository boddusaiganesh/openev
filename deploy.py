"""
Phase 10 — Hugging Face Spaces Deployment Helper
=============================================
Automates the deployment process to HF Spaces.

Usage:
    python deploy.py --space YOUR_USERNAME/contract-clause-review
    python deploy.py --verify-only
    python deploy.py --check-local
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import httpx


DEPLOY_FILES = [
    "inference.py", "server.py", "environment.py", "models.py",
    "tasks.py", "graders.py", "rewards.py", "domain_config.py",
    "Dockerfile", "requirements.txt", "openenv.yaml", "README.md",
    ".gitignore",
]

DEPLOY_DIRS = ["data"]


def run_pre_deploy_checks() -> bool:
    """Run all pre-deployment checks."""
    print("\n  Pre-Deployment Checks:")
    issues = 0

    for f in DEPLOY_FILES:
        if not os.path.exists(f):
            print(f"    ✗ Missing: {f}")
            issues += 1

    if not os.path.isdir("data"):
        print("    ✗ Missing: data/")
        issues += 1

    try:
        with open("data/manifest.json") as f:
            manifest = json.load(f)
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            if tid not in manifest:
                print(f"    ✗ Missing task in manifest: {tid}")
                issues += 1
    except Exception as e:
        print(f"    ✗ Manifest error: {e}")
        issues += 1

    try:
        with open("README.md") as f:
            content = f.read()
        if not content.startswith("---"):
            print("    ✗ README missing YAML frontmatter")
            issues += 1
        if "sdk: docker" not in content:
            print("    ✗ README missing 'sdk: docker'")
            issues += 1
        if "app_port: 7860" not in content:
            print("    ✗ README missing 'app_port: 7860'")
            issues += 1
    except Exception as e:
        print(f"    ✗ README error: {e}")
        issues += 1

    print("    Testing Docker build...")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "cr-deploy-test", "."],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            print(f"    ✗ Docker build failed")
            issues += 1
        else:
            print("    ✓ Docker build OK")
            subprocess.run(["docker", "rmi", "cr-deploy-test"], capture_output=True, timeout=30)
    except FileNotFoundError:
        print("    ⚠ Docker not found")
    except subprocess.TimeoutExpired:
        print("    ✗ Docker build timed out")
        issues += 1

    try:
        from environment import ContractReviewEnv
        from models import Action, ActionType
        env = ContractReviewEnv()
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            obs = env.reset(tid)
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            assert env.grader_result is not None
            assert 0.0 <= env.grader_result.score <= 1.0
        print("    ✓ Environment loads and runs")
    except Exception as e:
        print(f"    ✗ Environment error: {e}")
        issues += 1

    if issues == 0:
        print("\n  ✓ All pre-deployment checks PASSED")
    else:
        print(f"\n  ✗ {issues} check(s) FAILED")

    return issues == 0


def verify_deployment(space_url: str) -> bool:
    """Verify a deployed HF Space is working correctly."""
    base = space_url.rstrip("/")
    print(f"\n  Verifying deployment: {base}")
    issues = 0
    client = httpx.Client(timeout=30.0)

    try:
        r = client.get(f"{base}/")
        if r.status_code != 200:
            print(f"    ✗ Health check returned {r.status_code}")
            issues += 1
        else:
            data = r.json()
            if data.get("status") != "ok":
                print(f"    ✗ Health check status: {data.get('status')}")
                issues += 1
            else:
                print(f"    ✓ Health check: OK")
    except Exception as e:
        print(f"    ✗ Health check failed: {e}")
        issues += 1
        client.close()
        return False

    for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        try:
            r = client.post(f"{base}/reset", json={"task_id": tid})
            if r.status_code != 200:
                print(f"    ✗ Reset {tid}: HTTP {r.status_code}")
                issues += 1
            else:
                obs = r.json()
                if obs.get("total_clauses", 0) < 3:
                    print(f"    ✗ Reset {tid}: too few clauses")
                    issues += 1
                else:
                    print(f"    ✓ Reset {tid}: OK ({obs['total_clauses']} clauses)")
        except Exception as e:
            print(f"    ✗ Reset {tid}: {e}")
            issues += 1

    try:
        client.post(f"{base}/reset", json={"task_id": "task_1_easy"})
        r = client.post(f"{base}/step",
                        json={"action_type": "classify", "clause_type": "confidentiality"})
        if r.status_code == 200:
            data = r.json()
            if "reward" in data and "observation" in data:
                print(f"    ✓ Step: OK (reward={data['reward']['score']:+.3f})")
            else:
                print(f"    ✗ Step: missing fields")
                issues += 1
        else:
            print(f"    ✗ Step: HTTP {r.status_code}")
            issues += 1
    except Exception as e:
        print(f"    ✗ Step: {e}")
        issues += 1

    try:
        r = client.get(f"{base}/state")
        if r.status_code == 200:
            state = r.json()
            if "ground_truth" in state and "task_id" in state:
                print(f"    ✓ State: OK")
            else:
                print(f"    ✗ State: missing fields")
                issues += 1
        else:
            print(f"    ✗ State: HTTP {r.status_code}")
            issues += 1
    except Exception as e:
        print(f"    ✗ State: {e}")
        issues += 1

    try:
        client.post(f"{base}/reset", json={"task_id": "task_1_easy"})
        state = client.get(f"{base}/state").json()
        gt = state["ground_truth"][0]

        client.post(f"{base}/step",
                    json={"action_type": "classify", "clause_type": gt["clause_type"]})
        r = client.post(f"{base}/step",
                        json={"action_type": "complete_review"})
        data = r.json()

        if data.get("done") and data.get("info", {}).get("grader_score") is not None:
            score = data["info"]["grader_score"]
            print(f"    ✓ Grader: OK (score={score:.4f})")
        else:
            print(f"    ✗ Grader: no score returned")
            issues += 1
    except Exception as e:
        print(f"    ✗ Grader: {e}")
        issues += 1

    if issues == 0:
        print(f"\n  ✓ Deployment VERIFIED!")
    else:
        print(f"\n  ✗ {issues} issue(s) found")

    client.close()
    return issues == 0


def main():
    parser = argparse.ArgumentParser(description="HF Spaces deployment helper")
    parser.add_argument("--space", help="HF Space name (user/repo)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify deployment")
    parser.add_argument("--check-local", action="store_true", help="Pre-deployment checks only")
    parser.add_argument("--url", help="Custom Space URL to verify")
    args = parser.parse_args()

    print("=" * 60)
    print("HF SPACES DEPLOYMENT HELPER")
    print("=" * 60)

    if args.check_local:
        ok = run_pre_deploy_checks()
        sys.exit(0 if ok else 1)

    if args.verify_only or args.url:
        url = args.url
        if not url and args.space:
            url = f"https://{args.space.replace('/', '-')}.hf.space"
        if not url:
            print("ERROR: Provide --space or --url")
            sys.exit(1)
        ok = verify_deployment(url)
        sys.exit(0 if ok else 1)

    if not args.space:
        print("ERROR: --space required")
        print("\nAvailable commands:")
        print("  python deploy.py --check-local")
        print("  python deploy.py --space USER/REPO")
        print("  python deploy.py --verify-only --url https://...")
        sys.exit(1)

    print(f"\nTarget: {args.space}")

    print("\n[1/2] Running pre-deployment checks...")
    if not run_pre_deploy_checks():
        print("\n  ✗ Fix issues above before deploying!")
        sys.exit(1)

    space_url = f"https://huggingface.co/spaces/{args.space}"
    clone_url = f"https://huggingface.co/spaces/{args.space}"

    print(f"\n[2/2] Deploy to HF Spaces:")
    print(f"""
  If the Space doesn't exist yet:
    1. Go to https://huggingface.co/new-space
    2. Name: {args.space.split('/')[-1]}
    3. SDK: Docker
    4. Create Space

  Then push your code:
    git clone {clone_url} hf-space-deploy
    cp -r {' '.join(DEPLOY_FILES)} hf-space-deploy/
    cp -r data/ hf-space-deploy/data/
    cd hf-space-deploy
    git add .
    git commit -m "Deploy Contract Clause Review Environment"
    git push

  Or use the HF CLI:
    huggingface-cli upload {args.space} . . --repo-type space
""")

    deploy_url = f"https://{args.space.replace('/', '-')}.hf.space"
    print(f"After deployment, verify with:")
    print(f"  python deploy.py --verify-only --url {deploy_url}")

    input("\n  Press Enter after pushing to HF Space...")

    print(f"\n  Waiting for Space to build (checking every 15s)...")
    for attempt in range(20):
        time.sleep(15)
        try:
            r = httpx.get(f"{deploy_url}/", timeout=10.0)
            if r.status_code == 200:
                print(f"  Space is live! Verifying...")
                ok = verify_deployment(deploy_url)
                if ok:
                    print(f"\n{'=' * 60}")
                    print(f"  ✓ DEPLOYMENT SUCCESSFUL!")
                    print(f"  URL: {deploy_url}")
                    print(f"{'=' * 60}")
                sys.exit(0 if ok else 1)
        except Exception:
            pass
        print(f"  Attempt {attempt + 1}/20 — not ready yet...")

    print("  ✗ Timed out waiting for Space to build.")
    sys.exit(1)


if __name__ == "__main__":
    main()
