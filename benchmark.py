"""
Phase 9 — Multi-Scenario Benchmark Runner
=========================================
Runs deterministic benchmark trajectories across ALL scenarios in the manifest.
Produces structured results for analysis.

Usage:
    python benchmark.py
    python benchmark.py --task task_1_easy
    python benchmark.py --scenario 0
    python benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from environment import ContractReviewEnv
from models import (
    Action,
    ActionType,
    ClauseGroundTruth,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
)
from tasks import get_task_config, list_task_ids


RISK_LIST = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def _other_type(ct: str) -> str:
    for t in CLAUSE_TAXONOMY:
        if t != ct:
            return t
    return ct


def _flip_risk(rl: RiskLevel) -> RiskLevel:
    idx = RISK_LIST.index(rl)
    return RISK_LIST[3 - idx]


def _collect_result(env: ContractReviewEnv, strategy: str) -> Dict[str, Any]:
    gr = env.grader_result
    return {
        "strategy": strategy,
        "grader_score": round(gr.score, 4) if gr else 0.0,
        "breakdown": gr.breakdown if gr else {},
        "penalties": gr.penalties if gr else {},
        "steps_used": env.step_number,
        "max_steps": env.task_config.max_steps if env.task_config else 0,
        "clauses_reviewed": sum(1 for r in env.clause_records if r.action_count > 0),
        "total_clauses": len(env.scenario.clauses) if env.scenario else 0,
    }


def strategy_optimal(env: ContractReviewEnv) -> Dict[str, Any]:
    """Perfect answers using ground truth."""
    cfg = env.task_config
    required = cfg.required_action_types
    obs_total = len(env.scenario.clauses)

    for i in range(obs_total):
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
        if not env.done and i < obs_total - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))
    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return _collect_result(env, "optimal")


def strategy_partial_correct(env: ContractReviewEnv) -> Dict[str, Any]:
    """Correct on even clauses, wrong on odd clauses."""
    cfg = env.task_config
    required = cfg.required_action_types
    obs_total = len(env.scenario.clauses)

    for i in range(obs_total):
        if env.done:
            break
        gt = env.scenario.clauses[i]
        correct = (i % 2 == 0)
        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                ct = gt.clause_type if correct else _other_type(gt.clause_type)
                env.step(Action(action_type=at, clause_type=ct))
            elif at == ActionType.RATE_SEVERITY:
                rl = gt.risk_level if correct else _flip_risk(gt.risk_level)
                env.step(Action(action_type=at, risk_level=rl))
            elif at == ActionType.FLAG:
                flags = gt.issues if correct else []
                env.step(Action(action_type=at, flags=flags))
            elif at == ActionType.SUGGEST:
                sa = gt.recommended_action if correct else SuggestedActionType.FLAG_FOR_NEGOTIATION
                env.step(Action(action_type=at, suggested_action=sa))
            elif at == ActionType.REASON:
                text = " ".join(gt.reasoning_keywords) if correct else "No significant issues."
                env.step(Action(action_type=at, reasoning=text))
        if not env.done and i < obs_total - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))
    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return _collect_result(env, "partial_correct")


def strategy_all_wrong(env: ContractReviewEnv) -> Dict[str, Any]:
    """Deliberately wrong answers for everything."""
    cfg = env.task_config
    required = cfg.required_action_types
    obs_total = len(env.scenario.clauses)

    for i in range(obs_total):
        if env.done:
            break
        gt = env.scenario.clauses[i]
        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                env.step(Action(action_type=at, clause_type=_other_type(gt.clause_type)))
            elif at == ActionType.RATE_SEVERITY:
                env.step(Action(action_type=at, risk_level=_flip_risk(gt.risk_level)))
            elif at == ActionType.FLAG:
                env.step(Action(action_type=at, flags=["unreasonable_penalty"]))
            elif at == ActionType.SUGGEST:
                env.step(Action(action_type=at, suggested_action=SuggestedActionType.ACCEPT_AS_IS))
            elif at == ActionType.REASON:
                env.step(Action(action_type=at, reasoning="No issues found."))
        if not env.done and i < obs_total - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))
    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return _collect_result(env, "all_wrong")


def strategy_empty(env: ContractReviewEnv) -> Dict[str, Any]:
    """Immediately complete without reviewing."""
    env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return _collect_result(env, "empty")


def strategy_classify_only(env: ContractReviewEnv) -> Dict[str, Any]:
    """Only classify (correct), skip all other actions."""
    obs_total = len(env.scenario.clauses)
    for i in range(obs_total):
        if env.done:
            break
        gt = env.scenario.clauses[i]
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
        if not env.done and i < obs_total - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))
    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return _collect_result(env, "classify_only")


STRATEGIES = {
    "optimal": strategy_optimal,
    "partial_correct": strategy_partial_correct,
    "classify_only": strategy_classify_only,
    "all_wrong": strategy_all_wrong,
    "empty": strategy_empty,
}


def run_benchmark(
    task_filter: Optional[str] = None,
    scenario_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run all strategies across all tasks and scenarios."""
    results: List[Dict[str, Any]] = []

    with open("data/manifest.json") as f:
        manifest = json.load(f)

    tasks = [task_filter] if task_filter else list_task_ids()

    for tid in tasks:
        entry = manifest.get(tid, {})
        scenario_files = entry.get("scenario_files", [])

        if scenario_index is not None:
            scenario_files = [scenario_files[scenario_index]] if scenario_index < len(scenario_files) else []

        for si, sf in enumerate(scenario_files):
            for strat_name, strat_fn in STRATEGIES.items():
                env = ContractReviewEnv()
                obs = env.reset(tid)

                start = time.time()
                result = strat_fn(env)
                elapsed = time.time() - start

                result.update({
                    "task_id": tid,
                    "difficulty": env.task_config.difficulty.value,
                    "scenario_file": sf,
                    "scenario_index": si,
                    "elapsed_seconds": round(elapsed, 4),
                })
                results.append(result)

    return results


def print_benchmark_table(results: List[Dict[str, Any]]):
    """Print results as a formatted table."""
    try:
        from tabulate import tabulate
    except ImportError:
        for r in results:
            print(f"  {r['task_id']:<18} {r['strategy']:<18} {r['grader_score']:.4f}")
        return

    headers = ["Task", "Scenario", "Strategy", "Score", "Steps", "Clauses", "Time"]
    rows = []
    for r in results:
        rows.append([
            r["task_id"],
            r["scenario_index"],
            r["strategy"],
            f"{r['grader_score']:.4f}",
            f"{r['steps_used']}/{r['max_steps']}",
            f"{r['clauses_reviewed']}/{r['total_clauses']}",
            f"{r['elapsed_seconds']:.3f}s",
        ])

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="Multi-scenario benchmark runner")
    parser.add_argument("--task", help="Run only this task ID")
    parser.add_argument("--scenario", type=int, help="Run only this scenario index")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), help="Run only this strategy")
    args = parser.parse_args()

    print("=" * 72)
    print("CONTRACT CLAUSE REVIEW — BENCHMARK SUITE")
    print("=" * 72)

    start = time.time()
    results = run_benchmark(task_filter=args.task, scenario_index=args.scenario)

    if args.strategy:
        results = [r for r in results if r["strategy"] == args.strategy]

    elapsed = time.time() - start

    print_benchmark_table(results)

    print(f"\n{'─' * 72}")
    print("SUMMARY BY STRATEGY")
    print(f"{'─' * 72}")

    strategies_seen = {}
    for r in results:
        key = r["strategy"]
        if key not in strategies_seen:
            strategies_seen[key] = []
        strategies_seen[key].append(r["grader_score"])

    for strat, scores in strategies_seen.items():
        avg = sum(scores) / len(scores)
        mn = min(scores)
        mx = max(scores)
        print(f"  {strat:<20} avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}  n={len(scores)}")

    print(f"\nTotal runtime: {elapsed:.1f}s")

    with open(args.output, "w") as f:
        json.dump({
            "benchmark_results": results,
            "total_runtime_seconds": round(elapsed, 2),
            "summary": {
                strat: {
                    "avg": round(sum(s) / len(s), 4),
                    "min": round(min(s), 4),
                    "max": round(max(s), 4),
                    "count": len(s),
                }
                for strat, s in strategies_seen.items()
            },
        }, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
