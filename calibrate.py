"""
Phase 7 — Grader Calibration Verification
=========================================
Runs synthetic trajectories at 5 quality tiers (perfect, good, partial,
poor, empty) across all 3 tasks, verifies score monotonicity, range
adherence, and sensitivity.  Prints a calibration report.

Usage:
    python calibrate.py
    python calibrate.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from environment import ContractReviewEnv
from models import (
    Action,
    ActionType,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import get_task_config, list_task_ids


def run_perfect_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """All correct answers for every clause, then complete."""
    obs = env.reset(task_id)
    cfg = env.task_config
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

    return env.grader_result.score if env.grader_result else 0.0


def run_good_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """80% correct — wrong on every 5th clause."""
    obs = env.reset(task_id)
    cfg = env.task_config
    required = cfg.required_action_types

    for i in range(obs.total_clauses):
        if env.done:
            break
        gt = env.scenario.clauses[i]
        is_wrong = (i % 5 == 4)

        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                ct = gt.clause_type if not is_wrong else _wrong_type(gt.clause_type)
                env.step(Action(action_type=at, clause_type=ct))
            elif at == ActionType.RATE_SEVERITY:
                rl = gt.risk_level if not is_wrong else _adjacent_risk(gt.risk_level)
                env.step(Action(action_type=at, risk_level=rl))
            elif at == ActionType.FLAG:
                flags = gt.issues if not is_wrong else []
                env.step(Action(action_type=at, flags=flags))
            elif at == ActionType.SUGGEST:
                sa = gt.recommended_action if not is_wrong else SuggestedActionType.FLAG_FOR_NEGOTIATION
                env.step(Action(action_type=at, suggested_action=sa))
            elif at == ActionType.REASON:
                text = " ".join(gt.reasoning_keywords) if not is_wrong else "Generic analysis."
                env.step(Action(action_type=at, reasoning=text))

        if not env.done and i < obs.total_clauses - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))

    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

    return env.grader_result.score if env.grader_result else 0.0


def run_partial_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """50% correct — alternate right/wrong per clause."""
    obs = env.reset(task_id)
    cfg = env.task_config
    required = cfg.required_action_types

    for i in range(obs.total_clauses):
        if env.done:
            break
        gt = env.scenario.clauses[i]
        is_wrong = (i % 2 == 1)

        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                ct = gt.clause_type if not is_wrong else _wrong_type(gt.clause_type)
                env.step(Action(action_type=at, clause_type=ct))
            elif at == ActionType.RATE_SEVERITY:
                rl = gt.risk_level if not is_wrong else _opposite_risk(gt.risk_level)
                env.step(Action(action_type=at, risk_level=rl))
            elif at == ActionType.FLAG:
                flags = gt.issues if not is_wrong else ["vague_language"]
                env.step(Action(action_type=at, flags=flags))
            elif at == ActionType.SUGGEST:
                sa = gt.recommended_action if not is_wrong else SuggestedActionType.REJECT_CLAUSE
                env.step(Action(action_type=at, suggested_action=sa))
            elif at == ActionType.REASON:
                text = " ".join(gt.reasoning_keywords) if not is_wrong else "Unable to assess."
                env.step(Action(action_type=at, reasoning=text))

        if not env.done and i < obs.total_clauses - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))

    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

    return env.grader_result.score if env.grader_result else 0.0


def run_poor_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """All wrong answers for every clause."""
    obs = env.reset(task_id)
    cfg = env.task_config
    required = cfg.required_action_types

    for i in range(obs.total_clauses):
        if env.done:
            break
        gt = env.scenario.clauses[i]

        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                env.step(Action(action_type=at, clause_type=_wrong_type(gt.clause_type)))
            elif at == ActionType.RATE_SEVERITY:
                env.step(Action(action_type=at, risk_level=_opposite_risk(gt.risk_level)))
            elif at == ActionType.FLAG:
                env.step(Action(action_type=at, flags=["unreasonable_penalty"]))
            elif at == ActionType.SUGGEST:
                env.step(Action(action_type=at, suggested_action=SuggestedActionType.ACCEPT_AS_IS))
            elif at == ActionType.REASON:
                env.step(Action(action_type=at, reasoning="No issues identified."))

        if not env.done and i < obs.total_clauses - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))

    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

    return env.grader_result.score if env.grader_result else 0.0


def run_empty_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """Immediately complete without reviewing any clause."""
    env.reset(task_id)
    env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
    return env.grader_result.score if env.grader_result else 0.0


def _wrong_type(correct: str) -> str:
    for ct in CLAUSE_TAXONOMY:
        if ct != correct:
            return ct
    return correct


RISK_LIST = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def _adjacent_risk(level: RiskLevel) -> RiskLevel:
    idx = RISK_LIST.index(level)
    return RISK_LIST[max(0, idx - 1)] if idx > 0 else RISK_LIST[1]


def _opposite_risk(level: RiskLevel) -> RiskLevel:
    idx = RISK_LIST.index(level)
    return RISK_LIST[3 - idx]


@dataclass
class CalibrationResult:
    task_id: str
    perfect: float = 0.0
    good: float = 0.0
    partial: float = 0.0
    poor: float = 0.0
    empty: float = 0.0
    monotonic: bool = False
    spread: float = 0.0


def run_calibration(verbose: bool = False) -> List[CalibrationResult]:
    env = ContractReviewEnv()
    results: List[CalibrationResult] = []

    tiers = [
        ("perfect", run_perfect_trajectory),
        ("good", run_good_trajectory),
        ("partial", run_partial_trajectory),
        ("poor", run_poor_trajectory),
        ("empty", run_empty_trajectory),
    ]

    for tid in list_task_ids():
        cr = CalibrationResult(task_id=tid)

        if verbose:
            print(f"\n  Calibrating {tid}...")

        for tier_name, tier_fn in tiers:
            score = tier_fn(env, tid)
            setattr(cr, tier_name, round(score, 4))
            if verbose:
                print(f"    {tier_name:>10}: {score:.4f}")

        cr.monotonic = (
            cr.perfect >= cr.good >= cr.partial >= cr.poor >= cr.empty
        )
        cr.spread = round(cr.perfect - cr.empty, 4)

        results.append(cr)

    return results


def print_calibration_report(results: List[CalibrationResult]):
    print("\n" + "=" * 80)
    print("GRADER CALIBRATION REPORT")
    print("=" * 80)
    print(f"\n{'Task':<18} {'Perfect':>8} {'Good':>8} {'Partial':>8} {'Poor':>8} {'Empty':>8} {'Spread':>8} {'Mono':>6}")
    print("-" * 80)

    all_mono = True
    for cr in results:
        mono_str = "✓" if cr.monotonic else "✗"
        if not cr.monotonic:
            all_mono = False
        print(f"{cr.task_id:<18} {cr.perfect:>8.4f} {cr.good:>8.4f} {cr.partial:>8.4f} "
              f"{cr.poor:>8.4f} {cr.empty:>8.4f} {cr.spread:>8.4f} {mono_str:>6}")

    print("-" * 80)

    checks = []
    for cr in results:
        checks.append(("Monotonicity", cr.task_id, cr.monotonic))
        checks.append(("Spread > 0.3", cr.task_id, cr.spread > 0.3))
        checks.append(("Perfect > 0.5", cr.task_id, cr.perfect > 0.5))
        checks.append(("Empty < 0.3", cr.task_id, cr.empty < 0.3))
        checks.append(("All in [0,1]", cr.task_id,
                        all(0.0 <= s <= 1.0 for s in [cr.perfect, cr.good, cr.partial, cr.poor, cr.empty])))

    passed = sum(1 for _, _, ok in checks if ok)
    failed = sum(1 for _, _, ok in checks if not ok)

    print(f"\nCalibration Checks: {passed} passed, {failed} failed")
    for name, tid, ok in checks:
        if not ok:
            print(f"  ✗ FAIL: {name} — {tid}")

    if failed == 0:
        print("  ✓ All calibration checks PASSED!")
    print("=" * 80)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Grader calibration verification")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    start = time.time()
    results = run_calibration(verbose=args.verbose)
    elapsed = time.time() - start

    if args.json:
        data = [
            {
                "task_id": cr.task_id,
                "perfect": cr.perfect, "good": cr.good, "partial": cr.partial,
                "poor": cr.poor, "empty": cr.empty,
                "monotonic": cr.monotonic, "spread": cr.spread,
            }
            for cr in results
        ]
        print(json.dumps({"calibration": data, "elapsed_seconds": round(elapsed, 2)}, indent=2))
    else:
        ok = print_calibration_report(results)
        print(f"\nCompleted in {elapsed:.1f}s")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
