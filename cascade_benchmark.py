"""
LexDomino — Cascade Benchmark
Deterministic multi-strategy benchmark for cascade tasks 4, 5, 6.
Mirrors the structure of benchmark.py for the clause-review tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from cascade_environment import LexDominoCrisisEnv
from cascade_models import CascadeAction, CascadeActionType


# ---------------------------------------------------------------------------
# Built-in deterministic strategies
# ---------------------------------------------------------------------------

def _strategy_passive(obs) -> CascadeAction:
    """Always advance day — do nothing. Measures worst-case cascade damage."""
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_investigate_first(obs, step: int) -> CascadeAction:
    """
    Round-robin investigation before acting:
    steps 0-1: cross-reference all contract pairs
    step 2: review deadlines
    step 3+: advance day
    """
    contracts = [c["contract_id"] for c in obs.contracts_summary]
    if step == 0 and len(contracts) >= 2:
        return CascadeAction(
            action_type=CascadeActionType.CROSS_REFERENCE_CONTRACTS,
            contract_ids=contracts[:2],
        )
    if step == 1 and len(contracts) >= 4:
        return CascadeAction(
            action_type=CascadeActionType.CROSS_REFERENCE_CONTRACTS,
            contract_ids=contracts[2:4],
        )
    if step == 2:
        return CascadeAction(action_type=CascadeActionType.REVIEW_DEADLINE_STATUS)
    if step == 3:
        return CascadeAction(action_type=CascadeActionType.ANALYZE_FINANCIAL_IMPACT)
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_deadline_first(obs) -> CascadeAction:
    """
    Urgency-driven: address the soonest expiring deadline each step.
    Sends formal notice on the highest-priority deadline contract, then advances.
    """
    deadlines = sorted(obs.active_deadlines, key=lambda d: d["days_remaining"])
    if deadlines:
        d = deadlines[0]
        if d["days_remaining"] <= 2:
            return CascadeAction(
                action_type=CascadeActionType.SEND_FORMAL_NOTICE,
                contract_id=d["contract_id"],
                justification=f"Proactive notice before deadline: {d['description']}",
            )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_financial_triage(obs) -> CascadeAction:
    """
    Finance-first: draw credit facility if cash near covenant, else advance.
    """
    gap = obs.cash_balance - obs.covenant_min_cash
    if gap < 500_000 and obs.cash_balance > 0:
        # Try to draw enough to create a buffer
        draw_amount = max(100_000.0, 500_000.0 - gap)
        return CascadeAction(
            action_type=CascadeActionType.DRAW_CREDIT_FACILITY,
            amount=draw_amount,
        )
    if not obs.insurance_active:
        # Attempt a claim on first contract
        if obs.contracts_summary:
            return CascadeAction(
                action_type=CascadeActionType.FILE_INSURANCE_CLAIM,
                contract_id=obs.contracts_summary[0]["contract_id"],
                amount=obs.cash_balance * 0.2,
            )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    task_id: str
    scenario_index: int
    strategy: str
    grader_score: float
    final_cash: float
    initial_cash: float
    cash_ratio: float
    deadlines_met: int
    deadlines_total: int
    deadline_ratio: float
    bankruptcy: bool
    steps_taken: int
    duration_seconds: float
    cascade_depth_max: int
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    task_id: str
    total_scenarios: int
    strategies_tested: List[str]
    results: List[BenchmarkResult] = field(default_factory=list)
    strategy_averages: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

class CascadeBenchmark:

    CASCADE_TASKS = ["task_4_cascade_easy", "task_5_cascade_medium", "task_6_cascade_hard"]
    STRATEGIES = ["passive", "investigate_first", "deadline_first", "financial_triage"]

    def __init__(self, data_dir: str = "data", output_dir: str = "artifacts"):
        self.env = LexDominoCrisisEnv(data_dir=data_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single(
        self,
        task_id: str,
        scenario_index: int,
        strategy: str,
    ) -> BenchmarkResult:
        t0 = time.time()
        try:
            obs = self.env.reset(task_id, scenario_index)
            initial_cash = self.env.scenario.initial_cash
            step = 0
            while not obs.done:
                action = self._choose(strategy, obs, step)
                obs, _, done, _ = self.env.step(action)
                step += 1

            gr = self.env.grader_result
            state = self.env.state()
            return BenchmarkResult(
                task_id=task_id,
                scenario_index=scenario_index,
                strategy=strategy,
                grader_score=float(gr.score) if gr else 0.0,
                final_cash=state.cash_balance,
                initial_cash=initial_cash,
                cash_ratio=float(gr.normalized_cash_ratio) if gr else 0.0,
                deadlines_met=state.deadlines_met,
                deadlines_total=state.deadlines_total,
                deadline_ratio=float(gr.deadlines_met_ratio) if gr else 0.0,
                bankruptcy=state.bankruptcy,
                steps_taken=step,
                duration_seconds=round(time.time() - t0, 3),
                cascade_depth_max=self.env.cascade_depth_max,
            )
        except Exception as e:
            return BenchmarkResult(
                task_id=task_id,
                scenario_index=scenario_index,
                strategy=strategy,
                grader_score=0.0,
                final_cash=0.0,
                initial_cash=0.0,
                cash_ratio=0.0,
                deadlines_met=0,
                deadlines_total=0,
                deadline_ratio=0.0,
                bankruptcy=True,
                steps_taken=0,
                duration_seconds=round(time.time() - t0, 3),
                cascade_depth_max=0,
                error=str(e),
            )

    def _choose(self, strategy: str, obs, step: int) -> CascadeAction:
        if strategy == "passive":
            return _strategy_passive(obs)
        if strategy == "investigate_first":
            return _strategy_investigate_first(obs, step)
        if strategy == "deadline_first":
            return _strategy_deadline_first(obs)
        if strategy == "financial_triage":
            return _strategy_financial_triage(obs)
        return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)

    def run_task(self, task_id: str, strategies: Optional[List[str]] = None) -> BenchmarkSummary:
        strategies = strategies or self.STRATEGIES
        manifest = self.env.manifest
        task_entry = manifest.get(task_id, {})
        n_scenarios = len(task_entry.get("scenario_files", []))

        summary = BenchmarkSummary(
            task_id=task_id,
            total_scenarios=n_scenarios,
            strategies_tested=strategies,
        )

        for strategy in strategies:
            scores = []
            for idx in range(n_scenarios):
                r = self.run_single(task_id, idx, strategy)
                summary.results.append(r)
                scores.append(r.grader_score)
                status = "BANKRUPT" if r.bankruptcy else f"score={r.grader_score:.4f}"
                print(f"  [{task_id}] scenario={idx} strategy={strategy:<20} {status} "
                      f"cash_ratio={r.cash_ratio:.2%} deadlines={r.deadlines_met}/{r.deadlines_total}")
            summary.strategy_averages[strategy] = round(sum(scores) / len(scores), 4) if scores else 0.0

        return summary

    def run_all(self, tasks: Optional[List[str]] = None, strategies: Optional[List[str]] = None) -> List[BenchmarkSummary]:
        tasks = tasks or self.CASCADE_TASKS
        all_summaries = []
        for task_id in tasks:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {task_id}")
            print(f"{'='*60}")
            try:
                summary = self.run_task(task_id, strategies)
                all_summaries.append(summary)
                print(f"\nStrategy averages for {task_id}:")
                for strat, avg in summary.strategy_averages.items():
                    print(f"  {strat:<25} avg_score={avg:.4f}")
            except Exception as e:
                print(f"ERROR running {task_id}: {e}")
        return all_summaries

    def save_results(self, summaries: List[BenchmarkSummary], filename: str = "cascade_benchmark_results.json") -> str:
        output_path = os.path.join(self.output_dir, filename)
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summaries": [
                {
                    "task_id": s.task_id,
                    "total_scenarios": s.total_scenarios,
                    "strategies_tested": s.strategies_tested,
                    "strategy_averages": s.strategy_averages,
                    "results": [asdict(r) for r in s.results],
                }
                for s in summaries
            ],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        return output_path

    def print_leaderboard(self, summaries: List[BenchmarkSummary]) -> None:
        print(f"\n{'='*60}")
        print("LEADERBOARD — Strategy Performance Across All Tasks")
        print(f"{'='*60}")
        # Aggregate per strategy
        strategy_scores: Dict[str, List[float]] = {}
        for summary in summaries:
            for strat, avg in summary.strategy_averages.items():
                strategy_scores.setdefault(strat, []).append(avg)
        ranked = sorted(
            [(strat, sum(scores) / len(scores)) for strat, scores in strategy_scores.items()],
            key=lambda x: -x[1],
        )
        for rank, (strat, avg) in enumerate(ranked, 1):
            print(f"  #{rank} {strat:<25} overall_avg={avg:.4f}")
        print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexDomino Cascade Benchmark")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs to benchmark (default: all 3 cascade tasks)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies: passive investigate_first deadline_first financial_triage")
    parser.add_argument("--output", default="artifacts", help="Output directory for results JSON")
    args = parser.parse_args()

    bench = CascadeBenchmark(output_dir=args.output)
    summaries = bench.run_all(tasks=args.tasks, strategies=args.strategies)
    bench.print_leaderboard(summaries)
    bench.save_results(summaries)
