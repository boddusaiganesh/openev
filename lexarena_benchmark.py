"""
LexArena — Full Benchmark
Runs all deterministic baseline strategies across all 6 tiers
and produces a complete Legal IQ leaderboard.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from lexarena_runner import LexArenaRunner
from lexarena_models import LexArenaConfig, LegalIQScore
from lexarena_scorer import compute_legal_iq, compare_scores, print_legal_iq
from tier1_runner import BASELINES


# ---------------------------------------------------------------------------
# Strategy grid for the full benchmark
# ---------------------------------------------------------------------------

BENCHMARK_STRATEGIES = [
    {
        "name": "passive_baseline",
        "t1": "always_no_clause",       # Worst T1 (lazy)
        "crisis": "passive",            # Worst crisis (do nothing)
        "t3_submit_empty": True,
    },
    {
        "name": "first_sentence_passive",
        "t1": "first_sentence",         # Basic T1
        "crisis": "passive",
        "t3_submit_empty": True,
    },
    {
        "name": "full_context_deadline",
        "t1": "full_context",           # Max recall T1
        "crisis": "deadline_first",     # Best crisis strategy
        "t3_submit_empty": True,
    },
    {
        "name": "best_available",
        "t1": "first_sentence",         # Reasonable T1
        "crisis": "deadline_first",     # Best crisis
        "t3_submit_empty": True,
    },
]


def run_full_benchmark(
    output_dir: str = "artifacts",
    verbose: bool = True,
) -> Dict[str, LegalIQScore]:
    """
    Run the full LexArena benchmark across all strategy combinations.
    Returns a dict of strategy_name -> LegalIQScore.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_scores: Dict[str, LegalIQScore] = {}

    print(f"\n{'='*65}")
    print("  L E X A R E N A   F U L L   B E N C H M A R K")
    print(f"  {len(BENCHMARK_STRATEGIES)} strategies × 6 tiers")
    print(f"{'='*65}")

    for strat in BENCHMARK_STRATEGIES:
        name = strat["name"]
        print(f"\n--- Strategy: {name} ---")

        config = LexArenaConfig(
            model_name=name,
            run_probes=False,  # Probes need cascade env compat — run separately
            tier1_max_samples=15,  # Fast run with built-in samples
        )
        runner = LexArenaRunner(config=config)

        # Tier 1
        t1_fn = BASELINES.get(strat["t1"])
        t1 = runner.run_tier1(infer_fn=t1_fn, verbose=verbose)

        # Tier 2 (baseline placeholder)
        t2 = runner.run_tier2(verbose=verbose)

        # Tier 3 (zero-edge submission as baseline)
        t3 = runner.run_tier3(verbose=verbose)

        # Crisis tiers 4-6
        crisis = runner.run_crisis_tiers(strategy=strat["crisis"], verbose=verbose)

        # Composite
        score = compute_legal_iq(
            t1_score=t1,
            t2_score=t2,
            t3_score=t3,
            t4_score=crisis["t4"],
            t5_score=crisis["t5"],
            t6_score=crisis["t6"],
        )

        all_scores[name] = score

        if verbose:
            print(f"  Legal IQ: {score.legal_iq:.4f} | {score.label}")

    # Print leaderboard
    compare_scores(all_scores)

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"lexarena_full_benchmark_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(
            {name: s.model_dump() for name, s in all_scores.items()},
            f, indent=2, default=str,
        )
    print(f"\nResults saved to: {out_path}")
    return all_scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexArena Full Benchmark")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    scores = run_full_benchmark(output_dir=args.output, verbose=not args.quiet)
    best = max(scores.items(), key=lambda x: x[1].legal_iq)
    print(f"\nBest strategy: {best[0]} with Legal IQ = {best[1].legal_iq:.4f}")
