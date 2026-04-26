"""
Phase 7 — Adversarial & Stress Test Runner
==========================================
Tests the environment against adversarial inputs, concurrent resets,
rapid-fire steps, boundary conditions, and degenerate agent behaviors.

Usage:
    python stress_test.py
    python stress_test.py --verbose
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Any, Dict, List, Tuple

from environment import ContractReviewEnv
from models import (
    Action,
    ActionType,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import list_task_ids, get_task_config


class StressResult:
    def __init__(self, name: str, passed: bool, message: str = "", elapsed: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.elapsed = elapsed


def test_random_agent(verbose: bool = False) -> StressResult:
    """Random valid actions for 200 episodes — should never crash."""
    start = time.time()
    env = ContractReviewEnv()
    episodes = 200
    crashes = 0

    for ep in range(episodes):
        tid = random.choice(list_task_ids())
        try:
            env.reset(tid)
            for _ in range(50):
                if env.done:
                    break
                action = _random_action()
                env.step(action)
            if not env.done:
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            if env.grader_result is None:
                crashes += 1
            elif not (0.0 <= env.grader_result.score <= 1.0):
                crashes += 1

        except Exception as e:
            crashes += 1
            if verbose:
                print(f"  Crash on episode {ep}: {e}")

    elapsed = time.time() - start
    return StressResult(
        f"Random agent ({episodes} episodes)",
        crashes == 0,
        f"{crashes} crashes" if crashes > 0 else f"All clean in {elapsed:.1f}s",
        elapsed,
    )


def test_rapid_resets(verbose: bool = False) -> StressResult:
    """Rapid alternate resets without stepping — no leakage."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for _ in range(500):
        tid = random.choice(list_task_ids())
        obs = env.reset(tid)
        if obs.step_number != 0 or obs.accumulated_score != 0.0 or obs.done is not False:
            issues += 1

    elapsed = time.time() - start
    return StressResult(
        "Rapid resets (500x)",
        issues == 0,
        f"{issues} state leaks" if issues > 0 else "Clean",
        elapsed,
    )


def test_mid_episode_reset(verbose: bool = False) -> StressResult:
    """Reset mid-episode to different task — clean state."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for _ in range(100):
        t1, t2 = random.sample(list_task_ids(), 2)
        env.reset(t1)
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        obs2 = env.reset(t2)
        if (
            obs2.step_number != 0
            or obs2.clause_index != 0
            or obs2.accumulated_score != 0.0
            or obs2.task_id != t2
            or obs2.done is not False
        ):
            issues += 1

    elapsed = time.time() - start
    return StressResult(
        "Mid-episode reset (100x)",
        issues == 0,
        f"{issues} leaks" if issues > 0 else "Clean",
        elapsed,
    )


def test_degenerate_spam(verbose: bool = False) -> StressResult:
    """Spam the same action 50 times — escalating penalties, no crash."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        env.reset(tid)
        rewards = []
        for _ in range(50):
            if env.done:
                break
            _, r, _, _ = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
            )
            rewards.append(r.score)
            if not (-1.0 <= r.score <= 1.0):
                issues += 1

        if len(rewards) > 3 and rewards[-1] >= rewards[0]:
            issues += 1

        if env.grader_result is None:
            issues += 1
        elif not (0.0 <= env.grader_result.score <= 1.0):
            issues += 1

    elapsed = time.time() - start
    return StressResult(
        "Degenerate spam (50 repeats/task)",
        issues == 0,
        f"{issues} issues" if issues > 0 else "Penalties escalate correctly",
        elapsed,
    )


def test_boundary_max_steps(verbose: bool = False) -> StressResult:
    """Push to exactly max_steps — verify termination."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        env.reset(tid)
        max_s = env.task_config.max_steps

        for _ in range(max_s + 10):
            if env.done:
                break
            env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

        if not env.done:
            issues += 1
        if env.step_number > max_s:
            issues += 1
        if env.grader_result is None:
            issues += 1

    elapsed = time.time() - start
    return StressResult(
        "Max-steps boundary",
        issues == 0,
        f"{issues} issues" if issues > 0 else "All tasks terminate correctly at max_steps",
        elapsed,
    )


def test_all_action_combos(verbose: bool = False) -> StressResult:
    """Every action type on every task — no crash."""
    start = time.time()
    env = ContractReviewEnv()
    crashes = 0

    actions = [
        Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"),
        Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.HIGH),
        Action(action_type=ActionType.FLAG, flags=["vague_language", "missing_liability_cap"]),
        Action(action_type=ActionType.SUGGEST, suggested_action=SuggestedActionType.REQUEST_MODIFICATION),
        Action(action_type=ActionType.REASON, reasoning="This clause presents significant risk."),
        Action(action_type=ActionType.NEXT_CLAUSE),
        Action(action_type=ActionType.COMPLETE_REVIEW),
    ]

    for tid in list_task_ids():
        for action in actions:
            try:
                env.reset(tid)
                env.step(action)
            except Exception as e:
                crashes += 1
                if verbose:
                    print(f"  Crash: {tid} + {action.action_type.value}: {e}")

    elapsed = time.time() - start
    return StressResult(
        "All action combos (7 actions × 3 tasks)",
        crashes == 0,
        f"{crashes} crashes" if crashes > 0 else "All accepted",
        elapsed,
    )


def test_grader_determinism(verbose: bool = False) -> StressResult:
    """Same trajectory 10 times — identical score each time."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        scores = []
        for _ in range(10):
            env.reset(tid)
            gt = env.scenario.clauses[0]
            env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            scores.append(env.grader_result.score)

        unique = len(set(scores))
        if unique != 1:
            issues += 1
            if verbose:
                print(f"  Non-deterministic: {tid} → {set(scores)}")

    elapsed = time.time() - start
    return StressResult(
        "Grader determinism (10 runs × 3 tasks)",
        issues == 0,
        f"{issues} non-deterministic tasks" if issues > 0 else "All deterministic",
        elapsed,
    )


def test_step_after_done(verbose: bool = False) -> StressResult:
    """20 steps after done — all return done=True, reward=0."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        env.reset(tid)
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        for _ in range(20):
            obs, r, done, _ = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
            )
            if done is not True:
                issues += 1
            if r.score != 0.0:
                issues += 1

    elapsed = time.time() - start
    return StressResult(
        "Step after done (20 steps × 3 tasks)",
        issues == 0,
        f"{issues} issues" if issues > 0 else "All correct",
        elapsed,
    )


def test_state_serialization_every_step(verbose: bool = False) -> StressResult:
    """State must be JSON-serializable at every step through a full episode."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        env.reset(tid)
        for _ in range(env.task_config.max_steps + 5):
            if env.done:
                break
            try:
                state = env.state()
                d = state.model_dump()
                j = json.dumps(d)
                json.loads(j)
            except Exception:
                issues += 1
            env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

    elapsed = time.time() - start
    return StressResult(
        "State serialization every step",
        issues == 0,
        f"{issues} serialization failures" if issues > 0 else "All serializable",
        elapsed,
    )


def test_accumulated_score_consistency(verbose: bool = False) -> StressResult:
    """accumulated_score == sum of all reward.scores."""
    start = time.time()
    env = ContractReviewEnv()
    issues = 0

    for tid in list_task_ids():
        env.reset(tid)
        total = 0.0
        for _ in range(env.task_config.max_steps):
            if env.done:
                break
            _, r, _, _ = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
            )
            total += r.score

        diff = abs(env.accumulated_score - total)
        if diff > 0.001:
            issues += 1
            if verbose:
                print(f"  Score mismatch {tid}: env={env.accumulated_score:.4f} sum={total:.4f}")

    elapsed = time.time() - start
    return StressResult(
        "Accumulated score consistency",
        issues == 0,
        f"{issues} mismatches" if issues > 0 else "All consistent",
        elapsed,
    )


def _random_action() -> Action:
    at = random.choice(list(ActionType))
    if at == ActionType.CLASSIFY:
        return Action(action_type=at, clause_type=random.choice(CLAUSE_TAXONOMY))
    elif at == ActionType.RATE_SEVERITY:
        return Action(action_type=at, risk_level=random.choice(list(RiskLevel)))
    elif at == ActionType.FLAG:
        n = random.randint(0, 4)
        flags = random.sample(ISSUE_FLAGS, min(n, len(ISSUE_FLAGS)))
        return Action(action_type=at, flags=flags)
    elif at == ActionType.SUGGEST:
        return Action(action_type=at, suggested_action=random.choice(list(SuggestedActionType)))
    elif at == ActionType.REASON:
        return Action(action_type=at, reasoning="Random analysis text for stress testing purposes.")
    else:
        return Action(action_type=at)


def run_all_stress_tests(verbose: bool = False) -> List[StressResult]:
    tests = [
        test_random_agent,
        test_rapid_resets,
        test_mid_episode_reset,
        test_degenerate_spam,
        test_boundary_max_steps,
        test_all_action_combos,
        test_grader_determinism,
        test_step_after_done,
        test_state_serialization_every_step,
        test_accumulated_score_consistency,
    ]

    results: List[StressResult] = []
    for test_fn in tests:
        print(f"  Running: {test_fn.__name__}...", end="", flush=True)
        r = test_fn(verbose=verbose)
        results.append(r)
        symbol = "✓" if r.passed else "✗"
        print(f" [{symbol}] {r.elapsed:.1f}s")

    return results


def print_stress_report(results: List[StressResult]) -> bool:
    print("\n" + "=" * 72)
    print("STRESS TEST REPORT")
    print("=" * 72)

    passed = 0
    failed = 0
    for r in results:
        symbol = "✓" if r.passed else "✗"
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{symbol}] {status}: {r.name} ({r.elapsed:.1f}s)")
        if r.message and not r.passed:
            print(f"         {r.message}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    total_time = sum(r.elapsed for r in results)
    print(f"\n  Total: {passed + failed} | Passed: {passed} | Failed: {failed} | Time: {total_time:.1f}s")

    if failed == 0:
        print("  ✓ All stress tests PASSED!")
    else:
        print(f"  ⚠ {failed} stress test(s) FAILED!")

    print("=" * 72)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Adversarial & stress tests")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("PHASE 7 — STRESS TESTING")
    print("=" * 72 + "\n")

    results = run_all_stress_tests(verbose=args.verbose)
    ok = print_stress_report(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
