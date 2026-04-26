"""
Phase 9 — Score Analysis & Report Generator
===========================================
Analyzes benchmark and inference results, verifies score distributions
match expected ranges, and generates a summary report.

Usage:
    python analyze_results.py
    python analyze_results.py --inference results.json
    python analyze_results.py --update-readme
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from tasks import get_task_config, list_task_ids


EXPECTED_RANGES = {
    "task_1_easy": {
        "optimal": (0.85, 1.00),
        "partial_correct": (0.40, 0.80),
        "classify_only": (0.60, 1.00),
        "all_wrong": (0.00, 0.15),
        "empty": (0.00, 0.10),
    },
    "task_2_medium": {
        "optimal": (0.75, 1.00),
        "partial_correct": (0.30, 0.65),
        "classify_only": (0.20, 0.50),
        "all_wrong": (0.00, 0.15),
        "empty": (0.00, 0.10),
    },
    "task_3_hard": {
        "optimal": (0.50, 1.00),
        "partial_correct": (0.20, 0.60),
        "classify_only": (0.10, 0.35),
        "all_wrong": (0.00, 0.15),
        "empty": (0.00, 0.10),
    },
}


def analyze_benchmark(results_path: str = "benchmark_results.json") -> Dict[str, Any]:
    """Analyze benchmark results and check against expected ranges."""
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run benchmark.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    results = data.get("benchmark_results", [])
    analysis = {
        "source": results_path,
        "total_runs": len(results),
        "tasks": {},
        "violations": [],
        "checks_passed": 0,
        "checks_failed": 0,
    }

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        tid = r["task_id"]
        strat = r["strategy"]
        if tid not in grouped:
            grouped[tid] = {}
        if strat not in grouped[tid]:
            grouped[tid][strat] = []
        grouped[tid][strat].append(r["grader_score"])

    for tid in list_task_ids():
        task_data = grouped.get(tid, {})
        task_analysis = {}

        for strat, scores in task_data.items():
            avg = sum(scores) / len(scores)
            mn = min(scores)
            mx = max(scores)

            task_analysis[strat] = {
                "avg": round(avg, 4),
                "min": round(mn, 4),
                "max": round(mx, 4),
                "count": len(scores),
            }

            expected = EXPECTED_RANGES.get(tid, {}).get(strat)
            if expected:
                lo, hi = expected
                in_range = lo <= avg <= hi
                if in_range:
                    analysis["checks_passed"] += 1
                else:
                    analysis["checks_failed"] += 1
                    analysis["violations"].append({
                        "task": tid,
                        "strategy": strat,
                        "avg_score": round(avg, 4),
                        "expected": f"[{lo:.2f}, {hi:.2f}]",
                    })

        analysis["tasks"][tid] = task_analysis

    for tid in list_task_ids():
        task_data = analysis["tasks"].get(tid, {})
        optimal_avg = task_data.get("optimal", {}).get("avg", 0)
        empty_avg = task_data.get("empty", {}).get("avg", 1)

        if optimal_avg > empty_avg:
            analysis["checks_passed"] += 1
        else:
            analysis["checks_failed"] += 1
            analysis["violations"].append({
                "task": tid,
                "strategy": "monotonicity",
                "avg_score": optimal_avg,
                "expected": f"optimal ({optimal_avg:.4f}) > empty ({empty_avg:.4f})",
            })

    return analysis


def analyze_inference(results_path: str) -> Dict[str, Any]:
    """Analyze LLM inference results."""
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    analysis = {
        "source": results_path,
        "model": data.get("model", "unknown"),
        "mode": data.get("mode", "unknown"),
        "total_runtime": data.get("total_runtime_seconds", 0),
        "tasks": {},
    }

    for r in results:
        tid = r["task_id"]
        analysis["tasks"][tid] = {
            "score": r.get("grader_score", 0),
            "steps": r.get("total_steps", 0),
            "llm_calls": r.get("llm_calls", 0),
            "clauses": f"{r.get('clauses_reviewed', 0)}/{r.get('total_clauses', 0)}",
            "time": r.get("elapsed_seconds", 0),
            "breakdown": r.get("grader_breakdown", {}),
        }

    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    print("\n" + "=" * 72)
    print("SCORE ANALYSIS REPORT")
    print("=" * 72)
    print(f"Source: {analysis['source']}")
    print(f"Total runs: {analysis.get('total_runs', 'N/A')}")

    for tid, task_data in analysis.get("tasks", {}).items():
        print(f"\n  {tid}:")
        if isinstance(task_data, dict) and "score" in task_data:
            print(f"    Score: {task_data['score']:.4f}")
            print(f"    Steps: {task_data['steps']} | LLM calls: {task_data['llm_calls']}")
            print(f"    Clauses: {task_data['clauses']} | Time: {task_data['time']:.1f}s")
            if task_data.get("breakdown"):
                for k, v in task_data["breakdown"].items():
                    print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
        else:
            for strat, scores in task_data.items():
                if isinstance(scores, dict):
                    expected = EXPECTED_RANGES.get(tid, {}).get(strat)
                    range_str = f" (expected: [{expected[0]:.2f}-{expected[1]:.2f}])" if expected else ""
                    in_range = ""
                    if expected:
                        avg = scores.get("avg", 0)
                        in_range = " ✓" if expected[0] <= avg <= expected[1] else " ✗"
                    print(f"    {strat:<20} avg={scores['avg']:.4f}  "
                          f"[{scores['min']:.4f}-{scores['max']:.4f}]  "
                          f"n={scores['count']}{range_str}{in_range}")

    violations = analysis.get("violations", [])
    if violations:
        print(f"\n  ⚠ {len(violations)} VIOLATIONS:")
        for v in violations:
            print(f"    ✗ {v['task']} / {v['strategy']}: "
                  f"got {v['avg_score']:.4f}, expected {v['expected']}")
    else:
        p = analysis.get("checks_passed", 0)
        print(f"\n  ✓ All {p} range checks PASSED")

    print("=" * 72)


def generate_readme_scores(analysis: Dict[str, Any]) -> str:
    """Generate a markdown table for README.md baseline scores section."""
    lines = [
        "## Baseline Benchmark Scores",
        "",
        "Deterministic benchmark results (no LLM, ground-truth trajectories):",
        "",
        "| Task | Strategy | Avg Score | Min | Max |",
        "|---|---|---|---|---|",
    ]

    for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        task_data = analysis.get("tasks", {}).get(tid, {})
        for strat in ["optimal", "partial_correct", "classify_only", "all_wrong", "empty"]:
            scores = task_data.get(strat, {})
            if scores:
                lines.append(
                    f"| `{tid}` | {strat} | "
                    f"{scores.get('avg', 0):.4f} | "
                    f"{scores.get('min', 0):.4f} | "
                    f"{scores.get('max', 0):.4f} |"
                )

    lines.append("")
    lines.append("Run `python benchmark.py` to reproduce these scores.")
    return "\n".join(lines)


def update_readme(analysis: Dict[str, Any]):
    """Insert benchmark scores into README.md."""
    scores_section = generate_readme_scores(analysis)

    if not os.path.exists("README.md"):
        print("ERROR: README.md not found")
        return

    with open("README.md") as f:
        content = f.read()

    marker_start = "## Baseline Scores"
    marker_alt = "## Baseline Benchmark Scores"

    for marker in [marker_start, marker_alt]:
        if marker in content:
            idx = content.index(marker)
            rest = content[idx + len(marker):]
            next_section = rest.find("\n## ")
            if next_section > 0:
                content = content[:idx] + scores_section + "\n" + rest[next_section + 1:]
            else:
                content = content[:idx] + scores_section
            break
    else:
        content += "\n\n" + scores_section

    with open("README.md", "w") as f:
        f.write(content)

    print("README.md updated with benchmark scores.")


def main():
    parser = argparse.ArgumentParser(description="Score analysis & report generator")
    parser.add_argument("--benchmark", default="benchmark_results.json", help="Benchmark results file")
    parser.add_argument("--inference", help="Inference results file to analyze")
    parser.add_argument("--update-readme", action="store_true", help="Update README with scores")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.inference:
        analysis = analyze_inference(args.inference)
    else:
        analysis = analyze_benchmark(args.benchmark)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_analysis(analysis)

    if args.update_readme:
        bench_analysis = analyze_benchmark(args.benchmark)
        update_readme(bench_analysis)


if __name__ == "__main__":
    main()
