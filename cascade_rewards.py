"""
LexDomino — Cascade Rewards
Per-step and trajectory reward functions for the crisis simulator.
"""

from __future__ import annotations

from typing import Tuple

from cascade_models import (
    CascadeAction,
    CascadeActionType,
    CascadeReward,
)


# ---------------------------------------------------------------------------
# Step-level informational signal
# (score is primarily trajectory-level; step rewards are small signals)
# ---------------------------------------------------------------------------

def reward_for_deadline_met(penalty_avoided: float, initial_cash: float) -> CascadeReward:
    """Agent met a legal deadline, avoiding a penalty."""
    norm = min(0.15, penalty_avoided / max(initial_cash, 1.0))
    return CascadeReward(
        score=round(norm, 4),
        cash_delta=0.0,
        breakdown={"deadline_met": norm},
        message=f"Deadline met — avoided penalty of ${penalty_avoided:,.0f}.",
    )


def reward_for_deadline_missed(penalty: float, initial_cash: float) -> CascadeReward:
    """Agent missed a legal deadline; penalty is now due."""
    norm = -min(0.30, penalty / max(initial_cash, 1.0))
    return CascadeReward(
        score=round(norm, 4),
        cash_delta=-penalty,
        breakdown={"deadline_missed": norm},
        message=f"Deadline missed — ${penalty:,.0f} penalty applied.",
    )


def reward_for_cash_change(delta: float, initial_cash: float) -> CascadeReward:
    """Reward/penalty proportional to cash delta as fraction of initial cash."""
    norm = max(-0.25, min(0.10, delta / max(initial_cash, 1.0)))
    polarity = "gained" if delta >= 0 else "lost"
    return CascadeReward(
        score=round(norm, 4),
        cash_delta=delta,
        breakdown={"cash_change": norm},
        message=f"Cash {polarity} ${abs(delta):,.0f}.",
    )


def reward_for_bankruptcy() -> CascadeReward:
    return CascadeReward(
        score=-1.0,
        cash_delta=0.0,
        breakdown={"bankruptcy": -1.0},
        message="BANKRUPTCY — episode terminated. Company insolvent.",
    )


def reward_for_covenant_violation(initial_cash: float, min_cash: float) -> CascadeReward:
    """Debt covenant violated — bank may freeze credit."""
    fraction = min(0.20, min_cash / max(initial_cash, 1.0))
    return CascadeReward(
        score=round(-fraction, 4),
        cash_delta=0.0,
        breakdown={"covenant_violated": -fraction},
        message="Debt covenant violated — bank credit facility at risk.",
    )


def reward_for_discovery(edges_found: int) -> CascadeReward:
    """Agent discovered hidden contract dependency edges."""
    score = min(0.08, 0.02 * edges_found)
    return CascadeReward(
        score=round(score, 4),
        cash_delta=0.0,
        breakdown={"discovery": score},
        message=f"Discovered {edges_found} hidden contract dependency edge(s).",
    )


def reward_for_invalid_action(reason: str) -> CascadeReward:
    return CascadeReward(
        score=-0.05,
        cash_delta=0.0,
        breakdown={"invalid_action": -0.05},
        message=f"Invalid action: {reason}",
    )


def reward_for_insurance_voided(policy_value: float, initial_cash: float) -> CascadeReward:
    fraction = -min(0.35, policy_value / max(initial_cash, 1.0))
    return CascadeReward(
        score=round(fraction, 4),
        cash_delta=0.0,
        breakdown={"insurance_voided": fraction},
        message=f"Insurance policy voided — lost coverage worth ${policy_value:,.0f}.",
    )


def reward_for_counterparty_appeasement(relationship_gain: float) -> CascadeReward:
    score = min(0.06, relationship_gain * 0.1)
    return CascadeReward(
        score=round(score, 4),
        cash_delta=0.0,
        breakdown={"relationship_gain": score},
        message="Counterparty relationship improved through proactive action.",
    )


def reward_neutral(message: str = "Action recorded.") -> CascadeReward:
    return CascadeReward(
        score=0.0,
        cash_delta=0.0,
        breakdown={},
        message=message,
    )


# ---------------------------------------------------------------------------
# Trajectory-level score (primary metric)
# ---------------------------------------------------------------------------

def compute_final_score(
    cash_final: float,
    cash_initial: float,
    deadlines_met: int,
    deadlines_total: int,
    rights_invoked_correctly: int,
    rights_total: int,
    cascade_depth_max: int,
    bankruptcy: bool,
) -> dict:
    """
    Primary score = normalized_cash_ratio (cash preserved vs initial).
    Secondary metrics are for diagnostic breakdown only.

    Returns dict with 'score' and all breakdown values.
    """
    if bankruptcy or cash_final <= 0:
        return {
            "score": 0.001,
            "normalized_cash_ratio": 0.0,
            "deadlines_met_ratio": 0.0,
            "rights_preserved_ratio": 0.0,
            "cascade_depth_max": cascade_depth_max,
        }

    # Primary: cash preservation ratio (clamped to [0, 1])
    cash_ratio = max(0.0, min(1.0, cash_final / max(cash_initial, 1.0)))

    # Secondary: deadline compliance (weight 0.15)
    deadline_ratio = deadlines_met / max(deadlines_total, 1)

    # Secondary: rights preservation (weight 0.10)
    rights_ratio = (
        rights_invoked_correctly / max(rights_total, 1)
        if rights_total > 0 else 1.0
    )

    # Weighted composite score (primary = 0.75, secondaries = 0.25)
    composite = (
        0.75 * cash_ratio
        + 0.15 * deadline_ratio
        + 0.10 * rights_ratio
    )

    # Cascade depth penalty: each extra link in cascade costs 2%
    cascade_penalty = min(0.10, 0.02 * max(0, cascade_depth_max - 1))
    composite = max(0.001, min(0.999, composite - cascade_penalty))

    return {
        "score": round(composite, 4),
        "normalized_cash_ratio": round(cash_ratio, 4),
        "deadlines_met_ratio": round(deadline_ratio, 4),
        "rights_preserved_ratio": round(rights_ratio, 4),
        "cascade_depth_max": cascade_depth_max,
    }
