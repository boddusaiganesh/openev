"""
LexDomino — Cascade Graders
Trajectory-level graders for the crisis simulator tasks.
"""

from __future__ import annotations

from typing import List

from cascade_models import (
    CascadeGraderResult,
    Deadline,
    DeadlineStatus,
    DependencyEdge,
)
from cascade_rewards import compute_final_score


def _clamp(v: float) -> float:
    return max(0.001, min(0.999, v))


def grade_cascade_episode(
    task_id: str,
    cash_final: float,
    cash_initial: float,
    deadlines: List[Deadline],
    dependency_edges: List[DependencyEdge],
    cascade_depth_max: int,
    bankruptcy: bool,
) -> CascadeGraderResult:
    """Unified grader for all LexDomino cascade tasks."""

    deadlines_met = sum(1 for d in deadlines if d.status == DeadlineStatus.MET)
    deadlines_total = len(deadlines)

    # Rights preserved = correctly discovered + used dependency edges
    discovered = sum(1 for e in dependency_edges if e.discovered)
    rights_total = len(dependency_edges)

    metrics = compute_final_score(
        cash_final=cash_final,
        cash_initial=cash_initial,
        deadlines_met=deadlines_met,
        deadlines_total=deadlines_total,
        rights_invoked_correctly=discovered,
        rights_total=rights_total,
        cascade_depth_max=cascade_depth_max,
        bankruptcy=bankruptcy,
    )

    score = _clamp(metrics["score"])

    # Human-readable message
    if bankruptcy:
        msg = "BANKRUPT — company failed before survival target."
    elif metrics["normalized_cash_ratio"] >= 0.90:
        msg = f"Excellent crisis management. {int(metrics['normalized_cash_ratio']*100)}% capital preserved."
    elif metrics["normalized_cash_ratio"] >= 0.60:
        msg = f"Adequate response. {int(metrics['normalized_cash_ratio']*100)}% capital preserved."
    else:
        msg = f"Significant damage. Only {int(metrics['normalized_cash_ratio']*100)}% capital preserved."

    msg += (
        f" Deadlines met: {deadlines_met}/{deadlines_total}."
        f" Dependencies discovered: {discovered}/{rights_total}."
    )

    return CascadeGraderResult(
        score=score,
        normalized_cash_ratio=metrics["normalized_cash_ratio"],
        deadlines_met_ratio=metrics["deadlines_met_ratio"],
        rights_preserved_ratio=metrics["rights_preserved_ratio"],
        cascade_depth_max=cascade_depth_max,
        breakdown={
            "cash_ratio": metrics["normalized_cash_ratio"],
            "deadline_ratio": metrics["deadlines_met_ratio"],
            "rights_ratio": metrics["rights_preserved_ratio"],
            "cascade_penalty": round(min(0.10, 0.02 * max(0, cascade_depth_max - 1)), 4),
        },
        message=msg,
    )
