"""
LexArena — Tier 3 Environment
Dependency Graph Mapping: agent maps cross-contract dependency edges
before the crisis starts. Scored against LexDomino ground-truth edges.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from lexarena_models import (
    PredictedEdge, Tier3Action, Tier3Observation,
    Tier3SampleResult, Tier3Score
)
from cascade_models import DependencyEdge, EdgeType


# ---------------------------------------------------------------------------
# Tier 3 Environment
# ---------------------------------------------------------------------------

class Tier3MappingEnv:
    """
    Environment for dependency graph mapping.

    The agent is shown a set of contracts (from a LexDomino scenario) and
    must identify all hidden dependency edges before being scored.
    The agent submits its predicted edge list and receives a score.
    """

    INSTRUCTIONS = (
        "You are mapping the dependency graph of a contract portfolio. "
        "Read the contracts below carefully and identify ALL hidden dependency edges — "
        "places where an event in one contract automatically triggers a consequence "
        "in another. Use the 5 edge types:\n"
        "  - cascade_trigger: breach in A automatically constitutes breach in B\n"
        "  - mutual_exclusion: invoking clause in A legally prevents invoking clause in B\n"
        "  - condition_precedent: clause A only activates after clause B is satisfied\n"
        "  - supersession: clause A overrides clause B under specific conditions\n"
        "  - temporal_gate: clause A's enforceability expires after deadline in clause B\n\n"
        "Submit your complete edge list using the submit_dependency_map action. "
        "Be thorough — missing a cascade edge costs more than a false positive."
    )

    def __init__(self, time_budget: int = 10):
        self.time_budget = time_budget
        self.scenario_id: Optional[str] = None
        self.contracts: List[Dict] = []
        self.ground_truth_edges: List[DependencyEdge] = []
        self.step_number: int = 0
        self.done: bool = False
        self.submitted_edges: Optional[List[PredictedEdge]] = None

    def reset(self, scenario_data: Dict[str, Any]) -> Tier3Observation:
        """Load a LexDomino scenario and prepare the mapping task."""
        self.scenario_id = scenario_data.get("scenario_id", "unknown")
        self.step_number = 0
        self.done = False
        self.submitted_edges = None

        # Load contracts summary (clause IDs and types only — no full text to force reading)
        self.contracts = []
        for c in scenario_data.get("contracts", []):
            self.contracts.append({
                "contract_id": c["contract_id"],
                "contract_type": c["contract_type"],
                "parties": c["parties"],
                "jurisdiction": c.get("jurisdiction", ""),
                "clauses": [
                    {
                        "clause_id": cl["clause_id"],
                        "clause_type": cl["clause_type"],
                        "text": cl["text"],  # Full clause text — agent must read it
                    }
                    for cl in c.get("clauses", [])
                ],
            })

        # Load ground truth edges (hidden from agent)
        self.ground_truth_edges = []
        for e in scenario_data.get("dependency_edges", []):
            try:
                self.ground_truth_edges.append(DependencyEdge(**e))
            except Exception:
                pass  # Skip malformed edges

        return self._make_obs()

    def step(self, action: Tier3Action) -> Tuple[Tier3Observation, Tier3SampleResult]:
        """Process one agent action. Only submit_dependency_map terminates the episode."""
        self.step_number += 1

        if action.action_type == "submit_dependency_map":
            self.submitted_edges = action.predicted_edges or []
            self.done = True
            result = self._grade()
            return self._make_obs(), result

        # request_more_time — no change except consuming a step
        if self.step_number >= self.time_budget:
            self.done = True
            self.submitted_edges = []
            result = self._grade()
            return self._make_obs(), result

        return self._make_obs(), None  # type: ignore

    def _make_obs(self) -> Tier3Observation:
        return Tier3Observation(
            scenario_id=self.scenario_id or "",
            contracts_summary=self.contracts,
            time_budget_remaining=max(0, self.time_budget - self.step_number),
            step_number=self.step_number,
            instructions=self.INSTRUCTIONS,
            done=self.done,
        )

    def _grade(self) -> Tier3SampleResult:
        """Score predicted edges against ground truth."""
        pred = self.submitted_edges or []
        gt = self.ground_truth_edges

        if not gt:
            # No ground truth edges in this scenario
            return Tier3SampleResult(
                scenario_id=self.scenario_id or "",
                ground_truth_edges=0,
                predicted_edges=len(pred),
                true_positives=0,
                false_positives=len(pred),
                false_negatives=0,
                precision=1.0 if not pred else 0.0,
                recall=1.0,
                f1=1.0 if not pred else 0.0,
                edge_type_accuracy=0.0,
                severity_order_score=0.0,
            )

        # Match predicted edges to ground truth
        gt_keys = {
            (e.source_clause_id, e.target_clause_id): e
            for e in gt
        }
        pred_keys = {
            (e.source_clause_id, e.target_clause_id): e
            for e in pred
        }

        tp_keys = set(gt_keys) & set(pred_keys)
        fp = len(pred_keys) - len(tp_keys)
        fn = len(gt_keys) - len(tp_keys)
        tp = len(tp_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Bonus: edge type accuracy for true positives
        type_correct = 0
        for key in tp_keys:
            gt_type = gt_keys[key].edge_type.value
            pred_type = pred_keys[key].edge_type
            if pred_type == gt_type:
                type_correct += 1
        edge_type_acc = type_correct / tp if tp > 0 else 0.0

        # Bonus: severity ordering (cascade_trigger > condition_precedent > mutual_exclusion)
        severity_order_score = _compute_severity_order(pred, gt)

        return Tier3SampleResult(
            scenario_id=self.scenario_id or "",
            ground_truth_edges=len(gt),
            predicted_edges=len(pred),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            edge_type_accuracy=round(edge_type_acc, 4),
            severity_order_score=round(severity_order_score, 4),
        )


# ---------------------------------------------------------------------------
# Severity ordering scorer
# ---------------------------------------------------------------------------

_SEVERITY = {
    "cascade_trigger": 4,
    "mutual_exclusion": 3,
    "condition_precedent": 2,
    "supersession": 1,
    "temporal_gate": 0,
}


def _compute_severity_order(
    pred: List[PredictedEdge],
    gt: List[DependencyEdge],
) -> float:
    """
    Reward agents that list higher-severity edges first.
    Score = fraction of consecutive pairs in correct severity order.
    """
    if len(pred) < 2:
        return 1.0
    correct = 0
    for i in range(len(pred) - 1):
        a = _SEVERITY.get(pred[i].edge_type, 0)
        b = _SEVERITY.get(pred[i + 1].edge_type, 0)
        if a >= b:
            correct += 1
    return correct / (len(pred) - 1)


# ---------------------------------------------------------------------------
# Batch Tier 3 grader
# ---------------------------------------------------------------------------

def grade_tier3_batch(results: List[Tier3SampleResult]) -> Tier3Score:
    """Aggregate Tier 3 scores across multiple scenarios."""
    if not results:
        return Tier3Score(
            total_scenarios=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            mean_edge_type_accuracy=0.0,
            tier_score=0.0,
        )

    n = len(results)
    mean_precision = sum(r.precision for r in results) / n
    mean_recall = sum(r.recall for r in results) / n
    mean_f1 = sum(r.f1 for r in results) / n
    mean_eta = sum(r.edge_type_accuracy for r in results) / n
    mean_sev = sum(r.severity_order_score for r in results) / n

    # Composite Tier 3 score — recall-heavy (missing an edge = missing a crisis)
    tier_score = (
        0.50 * mean_recall
        + 0.25 * mean_precision
        + 0.15 * mean_eta
        + 0.10 * mean_sev
    )
    tier_score = max(0.0, min(1.0, round(tier_score, 4)))

    return Tier3Score(
        total_scenarios=n,
        mean_precision=round(mean_precision, 4),
        mean_recall=round(mean_recall, 4),
        mean_f1=round(mean_f1, 4),
        mean_edge_type_accuracy=round(mean_eta, 4),
        tier_score=tier_score,
    )


# ---------------------------------------------------------------------------
# Load all scenarios for Tier 3 from LexDomino data dir
# ---------------------------------------------------------------------------

def load_tier3_scenarios(data_dir: str = "data") -> List[Dict]:
    """Load all LexDomino scenarios to use as Tier 3 mapping tasks."""
    scenarios = []
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cascade_tasks = [k for k in manifest if "cascade" in k]
    for task_id in cascade_tasks:
        for rel_path in manifest[task_id].get("scenario_files", []):
            full_path = os.path.join(data_dir, rel_path)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    scenarios.append(json.load(f))

    return scenarios
