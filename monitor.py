"""
Phase 10 — Post-Deployment Health Monitor
=====================================
Periodically checks the deployed environment for health, latency,
and correctness. Designed to run as a cron job or background task.

Usage:
    python monitor.py --url https://YOUR-SPACE.hf.space
    python monitor.py --url URL --interval 300
    python monitor.py --url URL --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict

import httpx


def check_health(base_url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """Run all health checks against a deployed environment."""
    client = httpx.Client(timeout=timeout)
    result = {
        "timestamp": datetime.now().isoformat(),
        "url": base_url,
        "healthy": True,
        "checks": {},
        "latency_ms": {},
    }

    try:
        start = time.time()
        r = client.get(f"{base_url}/")
        elapsed = (time.time() - start) * 1000
        result["latency_ms"]["health"] = round(elapsed, 1)
        result["checks"]["health"] = r.status_code == 200
        if r.status_code != 200:
            result["healthy"] = False
    except Exception as e:
        result["checks"]["health"] = False
        result["healthy"] = False
        result["error"] = str(e)
        client.close()
        return result

    try:
        start = time.time()
        r = client.post(f"{base_url}/reset", json={"task_id": "task_1_easy"})
        elapsed = (time.time() - start) * 1000
        result["latency_ms"]["reset"] = round(elapsed, 1)
        result["checks"]["reset"] = r.status_code == 200
        if r.status_code != 200:
            result["healthy"] = False
    except Exception:
        result["checks"]["reset"] = False
        result["healthy"] = False

    try:
        start = time.time()
        r = client.post(f"{base_url}/step",
                        json={"action_type": "classify", "clause_type": "confidentiality"})
        elapsed = (time.time() - start) * 1000
        result["latency_ms"]["step"] = round(elapsed, 1)
        result["checks"]["step"] = r.status_code == 200
        if r.status_code != 200:
            result["healthy"] = False
    except Exception:
        result["checks"]["step"] = False
        result["healthy"] = False

    try:
        client.post(f"{base_url}/reset", json={"task_id": "task_1_easy"})
        r = client.post(f"{base_url}/step", json={"action_type": "complete_review"})
        data = r.json()
        score = data.get("info", {}).get("grader_score")
        result["checks"]["grader"] = score is not None and 0.0 <= score <= 1.0
        if not result["checks"]["grader"]:
            result["healthy"] = False
    except Exception:
        result["checks"]["grader"] = False
        result["healthy"] = False

    try:
        r = client.get(f"{base_url}/state")
        result["checks"]["state"] = r.status_code == 200
        if r.status_code != 200:
            result["healthy"] = False
    except Exception:
        result["checks"]["state"] = False
        result["healthy"] = False

    client.close()
    return result


def print_check_result(result: Dict[str, Any]):
    """Print a formatted health check result."""
    status = "HEALTHY" if result["healthy"] else "UNHEALTHY"
    symbol = "✓" if result["healthy"] else "✗"

    print(f"  [{symbol}] {result['timestamp']} — {status}")

    for check, ok in result.get("checks", {}).items():
        s = "✓" if ok else "✗"
        latency = result.get("latency_ms", {}).get(check, "")
        lat_str = f" ({latency}ms)" if latency else ""
        print(f"      [{s}] {check}{lat_str}")

    if "error" in result:
        print(f"      ERROR: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description="Post-deployment health monitor")
    parser.add_argument("--url", required=True, help="Deployed environment URL")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--once", action="store_true", help="Run a single check")
    parser.add_argument("--log", help="Append results to a JSON log file")
    args = parser.parse_args()

    print("=" * 60)
    print("POST-DEPLOYMENT HEALTH MONITOR")
    print("=" * 60)
    print(f"  URL: {args.url}")
    print(f"  Interval: {args.interval}s")

    log_entries = []

    try:
        while True:
            result = check_health(args.url)
            print_check_result(result)
            log_entries.append(result)

            if args.log:
                with open(args.log, "w") as f:
                    json.dump(log_entries[-100:], f, indent=2)

            if args.once:
                sys.exit(0 if result["healthy"] else 1)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        if log_entries:
            healthy_pct = sum(1 for e in log_entries if e["healthy"]) / len(log_entries) * 100
            print(f"  Uptime: {healthy_pct:.1f}% ({len(log_entries)} checks)")


if __name__ == "__main__":
    main()
