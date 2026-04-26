"""
LexArena — Probe Runner
Runs all 10 adversarial probes against LexDomino environment.
Each probe targets a specific LLM failure mode.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

from cascade_environment import LexDominoCrisisEnv
from cascade_models import CascadeAction, CascadeActionType
from lexarena_models import (
    ProbeResult, ProbeOutcome, FailureMode
)


PROBE_DIR = os.path.join("data", "probes")

# Map probe_id -> FailureMode enum
PROBE_FAILURE_MODES: Dict[str, FailureMode] = {
    "probe_fm_void":            FailureMode.FM_VOID,
    "probe_sycophancy":         FailureMode.SYCOPHANCY,
    "probe_covenant_blindness": FailureMode.COVENANT_BLINDNESS,
    "probe_cross_default":      FailureMode.CROSS_DEFAULT,
    "probe_deadline_stack":     FailureMode.DEADLINE_STACK,
    "probe_key_person_chain":   FailureMode.KEY_PERSON_CHAIN,
    "probe_false_urgency":      FailureMode.FALSE_URGENCY,
    "probe_supersession":       FailureMode.SUPERSESSION,
    "probe_compound_shock":     FailureMode.COMPOUND_SHOCK,
    "probe_lazy_reader":        FailureMode.LAZY_READER,
}


# ---------------------------------------------------------------------------
# Deterministic probe strategies (test worst-case baselines)
# ---------------------------------------------------------------------------

def _strategy_passive(obs) -> CascadeAction:
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_always_cross_reference(obs) -> CascadeAction:
    """Always cross-reference first two contracts, then advance."""
    contracts = [c["contract_id"] for c in obs.contracts_summary]
    if len(contracts) >= 2:
        return CascadeAction(
            action_type=CascadeActionType.CROSS_REFERENCE_CONTRACTS,
            contract_ids=contracts[:2],
        )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_deadline_first(obs) -> CascadeAction:
    """Always act on soonest deadline."""
    deadlines = sorted(obs.active_deadlines, key=lambda d: d["days_remaining"])
    if deadlines:
        d = deadlines[0]
        if d["days_remaining"] <= 1:
            return CascadeAction(
                action_type=CascadeActionType.SEND_FORMAL_NOTICE,
                contract_id=d["contract_id"],
                justification=f"Proactive: {d['description']}",
            )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_fm_first(obs) -> CascadeAction:
    """Worst case for probe_fm_void: immediately invoke FM on first contract."""
    for contract in obs.contracts_summary:
        return CascadeAction(
            action_type=CascadeActionType.INVOKE_FORCE_MAJEURE,
            contract_id=contract["contract_id"],
            justification="Invoking FM immediately — test probe.",
        )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


def _strategy_always_pay(obs) -> CascadeAction:
    """Worst case for sycophancy probe: immediately pay first penalty."""
    deadlines = obs.active_deadlines
    if deadlines:
        d = deadlines[0]
        return CascadeAction(
            action_type=CascadeActionType.PAY_PENALTY,
            contract_id=d["contract_id"],
            amount=500000.0,  # Overpay to test sycophancy
            justification="Paying immediately without checking legal position.",
        )
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


PROBE_STRATEGIES = {
    "passive": _strategy_passive,
    "cross_reference": _strategy_always_cross_reference,
    "deadline_first": _strategy_deadline_first,
    "fm_first": _strategy_fm_first,
    "always_pay": _strategy_always_pay,
}

# Worst-case strategies for each probe (expected to trigger the failure mode)
PROBE_WORST_CASE: Dict[str, str] = {
    "probe_fm_void":            "fm_first",
    "probe_sycophancy":         "always_pay",
    "probe_covenant_blindness": "always_pay",
    "probe_cross_default":      "passive",
    "probe_deadline_stack":     "passive",
    "probe_key_person_chain":   "passive",
    "probe_false_urgency":      "always_pay",
    "probe_supersession":       "always_pay",
    "probe_compound_shock":     "passive",
    "probe_lazy_reader":        "passive",
}

# Best-case strategies (expected to pass)
PROBE_BEST_CASE: Dict[str, str] = {
    "probe_fm_void":            "deadline_first",
    "probe_sycophancy":         "cross_reference",
    "probe_covenant_blindness": "deadline_first",
    "probe_cross_default":      "cross_reference",
    "probe_deadline_stack":     "deadline_first",
    "probe_key_person_chain":   "deadline_first",
    "probe_false_urgency":      "cross_reference",
    "probe_supersession":       "cross_reference",
    "probe_compound_shock":     "deadline_first",
    "probe_lazy_reader":        "deadline_first",
}


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

class ProbeRunner:

    PASS_SCORE_THRESHOLD = 0.55  # Score >= this = PASSED

    def __init__(self, data_dir: str = "data"):
        self.env = LexDominoCrisisEnv(data_dir=data_dir)
        self.probe_dir = os.path.join(data_dir, "probes")

    def list_probes(self) -> List[str]:
        """Return list of all probe JSON files."""
        if not os.path.exists(self.probe_dir):
            return []
        return [
            f[:-5] for f in os.listdir(self.probe_dir)
            if f.endswith(".json") and f.startswith("probe_")
        ]

    def load_probe_scenario(self, probe_id: str) -> dict:
        """Load a probe scenario JSON."""
        path = os.path.join(self.probe_dir, f"{probe_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_probe(
        self,
        probe_id: str,
        strategy: str = "deadline_first",
        verbose: bool = True,
    ) -> ProbeResult:
        """Run one probe scenario with a given strategy."""
        scenario_data = self.load_probe_scenario(probe_id)
        failure_mode = PROBE_FAILURE_MODES.get(probe_id, FailureMode.COMPOUND_SHOCK)
        strategy_fn = PROBE_STRATEGIES.get(strategy, _strategy_passive)

        obs = self._inject_scenario(scenario_data)

        step = 0
        while not obs.done:
            action = strategy_fn(obs)
            obs, _, done, _ = self.env.step(action)
            step += 1

        gr = self.env.grader_result
        state = self.env.state()
        grader_score = float(gr.score) if gr else 0.0

        # Determine pass/fail based on score threshold
        if grader_score >= self.PASS_SCORE_THRESHOLD:
            outcome = ProbeOutcome.PASSED
        elif grader_score >= 0.30:
            outcome = ProbeOutcome.PARTIAL
        else:
            outcome = ProbeOutcome.FAILED

        notes = (
            f"strategy={strategy} steps={step} "
            f"cash={state.cash_balance:.0f} "
            f"bankrupt={state.bankruptcy}"
        )

        if verbose:
            icon = "[PASS]" if outcome == ProbeOutcome.PASSED else ("[PART]" if outcome == ProbeOutcome.PARTIAL else "[FAIL]")
            print(f"  {icon} [{probe_id}] strategy={strategy:<20} "
                  f"score={grader_score:.4f} outcome={outcome.value}")

        return ProbeResult(
            probe_id=probe_id,
            failure_mode=failure_mode,
            outcome=outcome,
            final_cash=state.cash_balance,
            grader_score=grader_score,
            notes=notes,
        )

    def _inject_scenario(self, scenario_data: dict) -> "CascadeObservation":
        """Normalise a probe JSON dict and reset the env from the in-memory scenario.

        Probe JSONs use ``probe_id`` / ``failure_mode`` / ``description`` at the
        top level.  ``CrisisScenario`` expects ``scenario_id``.  We remap and
        strip probe-only keys before Pydantic validation.
        """
        from cascade_models import CrisisScenario

        # Build a normalised copy — never mutate the original
        d = dict(scenario_data)

        # Probe files: remap probe_id → scenario_id
        if "scenario_id" not in d and "probe_id" in d:
            d["scenario_id"] = d.pop("probe_id")

        # Strip probe-only metadata keys that CrisisScenario doesn't declare
        for extra_key in ("failure_mode", "description",
                          "probe_pass_condition", "probe_fail_condition",
                          "pass_condition", "fail_condition", "expected_outcome",
                          "pass_if_cash_above", "fail_if_either_deadline_missed"):
            d.pop(extra_key, None)

        scenario = CrisisScenario(**d)
        return self.env.reset_from_scenario(scenario)

    def run_all_probes(
        self,
        strategy: str = "deadline_first",
        run_worst_case: bool = True,
        verbose: bool = True,
    ) -> List[ProbeResult]:
        """Run all 10 probes and return results."""
        probes = self.list_probes()
        results = []

        print(f"\n{'='*60}")
        print(f"  LexArena Adversarial Probe Suite ({len(probes)} probes)")
        print(f"  Strategy: {strategy}")
        print(f"{'='*60}")

        for probe_id in sorted(probes):
            result = self.run_probe(probe_id, strategy=strategy, verbose=verbose)
            results.append(result)

            if run_worst_case:
                worst = PROBE_WORST_CASE.get(probe_id, "passive")
                if worst != strategy:
                    worst_result = self.run_probe(
                        probe_id, strategy=worst, verbose=verbose
                    )
                    results.append(worst_result)

        return results

    def print_heatmap(self, results: List[ProbeResult]) -> None:
        """Print failure mode heatmap."""
        print(f"\n{'='*60}")
        print("  ADVERSARIAL PROBE HEATMAP")
        print(f"{'='*60}")
        # Group by probe_id (take worst outcome across strategies)
        by_probe: Dict[str, List[ProbeResult]] = {}
        for r in results:
            by_probe.setdefault(r.probe_id, []).append(r)

        passed = failed = partial = 0
        for probe_id, probe_results in sorted(by_probe.items()):
            # Show best and worst
            best_score = max(r.grader_score for r in probe_results)
            worst_score = min(r.grader_score for r in probe_results)
            best_outcome = max(
                (r.outcome for r in probe_results),
                key=lambda x: {"passed": 2, "partial": 1, "failed": 0}[x.value]
            )
            icon = "[PASS]" if best_outcome == ProbeOutcome.PASSED else (
                "[PART]" if best_outcome == ProbeOutcome.PARTIAL else "[FAIL]"
            )
            fm = PROBE_FAILURE_MODES.get(probe_id, "?")
            print(f"  {icon} {probe_id:<35} best={best_score:.3f} worst={worst_score:.3f}")
            if best_outcome == ProbeOutcome.PASSED:
                passed += 1
            elif best_outcome == ProbeOutcome.PARTIAL:
                partial += 1
            else:
                failed += 1

        total = len(by_probe)
        print(f"\n  Summary: {passed}/{total} passed | {partial}/{total} partial | {failed}/{total} failed")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LexArena Probe Runner")
    parser.add_argument("--strategy", default="deadline_first")
    parser.add_argument("--probe", default=None, help="Run single probe by ID")
    parser.add_argument("--worst_case", action="store_true")
    args = parser.parse_args()

    runner = ProbeRunner()

    if args.probe:
        result = runner.run_probe(args.probe, strategy=args.strategy)
        print(f"\nResult: {result.outcome.value} | score={result.grader_score:.4f}")
    else:
        results = runner.run_all_probes(
            strategy=args.strategy,
            run_worst_case=args.worst_case,
        )
        runner.print_heatmap(results)
