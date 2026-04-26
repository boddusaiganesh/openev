"""
LexArena — Legal IQ Composite Scorer
======================================
Aggregates per-tier scores into the composite Legal IQ score.

Formula:
    Legal_IQ = 0.15 * T1  +  0.15 * T2  +  0.20 * T3
              + 0.50 * (0.25 * T4 + 0.35 * T5 + 0.40 * T6)

Labels:
    0.85–1.00  Expert CRO Level
    0.70–0.84  Senior Lawyer Level
    0.50–0.69  Junior Associate Level
    0.30–0.49  Paralegal Level
    0.00–0.29  Fails Legal Practice Bar
"""

from __future__ import annotations

from typing import Dict, List, Optional

from models import LegalIQScore


# ---------------------------------------------------------------------------
# Weight constants (matches openenv.yaml)
# ---------------------------------------------------------------------------

TIER_WEIGHTS = {
    "t1": 0.15,
    "t2": 0.15,
    "t3": 0.20,
}

CRISIS_WEIGHTS = {
    "t4": 0.25,
    "t5": 0.35,
    "t6": 0.40,
}

CRISIS_SHARE = 0.50   # crisis management gets 50% of Legal IQ

LABELS = [
    (0.85, 1.01, "Expert CRO Level"),
    (0.70, 0.85, "Senior Lawyer Level"),
    (0.50, 0.70, "Junior Associate Level"),
    (0.30, 0.50, "Paralegal Level"),
    (0.00, 0.30, "Fails Legal Practice Bar"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_legal_iq(
    t1_score: float = 0.0,
    t2_score: float = 0.0,
    t3_score: float = 0.0,
    t4_score: float = 0.0,
    t5_score: float = 0.0,
    t6_score: float = 0.0,
    model_name: str = "",
) -> LegalIQScore:
    """
    Compute the composite Legal IQ score from per-tier scores.

    All inputs are expected in [0.0, 1.0]. Missing tiers should be 0.0.
    Returns a LegalIQScore with breakdown and label.
    """
    # Clamp inputs
    def c(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    t1, t2, t3 = c(t1_score), c(t2_score), c(t3_score)
    t4, t5, t6 = c(t4_score), c(t5_score), c(t6_score)

    # Crisis sub-score (weighted average of Tiers 4-6)
    crisis = (
        CRISIS_WEIGHTS["t4"] * t4
        + CRISIS_WEIGHTS["t5"] * t5
        + CRISIS_WEIGHTS["t6"] * t6
    )

    # Full composite
    legal_iq = (
        TIER_WEIGHTS["t1"] * t1
        + TIER_WEIGHTS["t2"] * t2
        + TIER_WEIGHTS["t3"] * t3
        + CRISIS_SHARE * crisis
    )
    legal_iq = max(0.0, min(1.0, legal_iq))

    label = _get_label(legal_iq)

    return LegalIQScore(
        model_name   = model_name,
        t1_score     = round(t1, 4),
        t2_score     = round(t2, 4),
        t3_score     = round(t3, 4),
        t4_score     = round(t4, 4),
        t5_score     = round(t5, 4),
        t6_score     = round(t6, 4),
        crisis_score = round(crisis, 4),
        legal_iq     = round(legal_iq, 4),
        label        = label,
        tier_breakdown={
            "t1_contribution": round(TIER_WEIGHTS["t1"] * t1, 4),
            "t2_contribution": round(TIER_WEIGHTS["t2"] * t2, 4),
            "t3_contribution": round(TIER_WEIGHTS["t3"] * t3, 4),
            "t4_contribution": round(CRISIS_SHARE * CRISIS_WEIGHTS["t4"] * t4, 4),
            "t5_contribution": round(CRISIS_SHARE * CRISIS_WEIGHTS["t5"] * t5, 4),
            "t6_contribution": round(CRISIS_SHARE * CRISIS_WEIGHTS["t6"] * t6, 4),
            "crisis_contribution": round(CRISIS_SHARE * crisis, 4),
        },
    )


def score_from_results(
    results: List[Dict],
    model_name: str = "",
) -> LegalIQScore:
    """
    Build a LegalIQScore from a list of run_task result dicts.

    Each dict must have 'task_id' and 'grader_score'.
    Tiers not present default to 0.0.
    """
    tier_scores: Dict[str, float] = {}
    for r in results:
        tid   = r.get("task_id", "")
        score = float(r.get("grader_score", 0.0))
        # Map task_id to tier variable
        if tid == "tier1_clause_reading":
            tier_scores["t1"] = score
        elif tid in ("task_1_easy", "task_2_medium", "task_3_hard"):
            # T2 = mean of task_1/2/3
            tier_scores.setdefault("t2_list", [])  # type: ignore[assignment]
            tier_scores["t2_list"].append(score)    # type: ignore[index]
        elif tid == "tier3_dependency_mapping":
            tier_scores["t3"] = score
        elif tid == "task_4_cascade_easy":
            tier_scores["t4"] = score
        elif tid == "task_5_cascade_medium":
            tier_scores["t5"] = score
        elif tid == "task_6_cascade_hard":
            tier_scores["t6"] = score

    # Aggregate T2
    t2_list = tier_scores.pop("t2_list", [])  # type: ignore[arg-type]
    t2 = sum(t2_list) / len(t2_list) if t2_list else 0.0

    return compute_legal_iq(
        t1_score   = tier_scores.get("t1", 0.0),
        t2_score   = t2,
        t3_score   = tier_scores.get("t3", 0.0),
        t4_score   = tier_scores.get("t4", 0.0),
        t5_score   = tier_scores.get("t5", 0.0),
        t6_score   = tier_scores.get("t6", 0.0),
        model_name = model_name,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_label(score: float) -> str:
    for lo, hi, label in LABELS:
        if lo <= score < hi:
            return label
    return "Fails Legal Practice Bar"
