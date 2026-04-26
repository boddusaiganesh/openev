"""
Phase 3 — Reward Computation
Per-step reward functions.
"""

from __future__ import annotations

from typing import Optional, Tuple

from models import (
    Action,
    ActionType,
    ClauseActionRecord,
    ClauseGroundTruth,
    Reward,
    RiskLevel,
    SuggestedActionType,
)


RISK_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def compute_classify_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Compute reward for classify action."""
    if record.classify_action is not None:
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_classify": -0.02},
                message="Already classified this clause.",
            ),
            "Already classified this clause.",
        )

    correct = action.clause_type == gt.clause_type

    if correct:
        score = 0.15
        msg = f"Correct classification: {action.clause_type}."
    elif _clause_type_family_match(action.clause_type, gt.clause_type):
        score = 0.05
        msg = f"Partial match. You said {action.clause_type}, correct is {gt.clause_type}."
    else:
        score = -0.05
        msg = f"Incorrect. You said {action.clause_type}, correct is {gt.clause_type}."

    breakdown = {"type_accuracy": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_risk_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Compute reward for rate_severity action."""
    if record.risk_action is not None:
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_risk": -0.02},
                message="Already rated risk for this clause.",
            ),
            "Already rated risk for this clause.",
        )

    if action.risk_level is None:
        return Reward(score=-0.05, message="No risk level provided."), "No risk level."

    try:
        agent_idx = RISK_ORDER.index(action.risk_level)
        truth_idx = RISK_ORDER.index(gt.risk_level)
        distance = abs(agent_idx - truth_idx)
    except ValueError:
        distance = 3

    if distance == 0:
        score = 0.15
        msg = f"Correct risk level: {action.risk_level.value}."
    elif distance == 1:
        score = 0.05
        msg = f"Close. You said {action.risk_level.value}, correct is {gt.risk_level.value}."
    else:
        score = -0.05
        msg = f"Incorrect. You said {action.risk_level.value}, correct is {gt.risk_level.value}."

    breakdown = {"risk_accuracy": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_flag_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Compute reward for flag action."""
    if record.flag_action is not None:
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_flag": -0.02},
                message="Already flagged issues for this clause.",
            ),
            "Already flagged issues for this clause.",
        )

    agent_set = set(action.flags or [])
    truth_set = set(gt.issues)

    if not truth_set and not agent_set:
        score = 0.10
        msg = "Correct: no issues to flag."
    elif not truth_set and agent_set:
        score = -0.05
        msg = f"False positives: {agent_set}. No issues expected."
    elif truth_set and not agent_set:
        score = -0.05
        msg = f"Missed issues: {truth_set}."
    else:
        tp = len(agent_set & truth_set)
        fp = len(agent_set - truth_set)
        fn = len(truth_set - agent_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        score = round(f1 * 0.15 - 0.05, 4)
        msg = f"F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f})."

    breakdown = {"flag_accuracy": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_suggest_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Compute reward for suggest action."""
    if record.suggest_action is not None:
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_suggest": -0.02},
                message="Already suggested action for this clause.",
            ),
            "Already suggested action for this clause.",
        )

    if action.suggested_action is None:
        return Reward(score=-0.05, message="No suggestion provided."), "No suggestion."

    if action.suggested_action == gt.recommended_action:
        score = 0.10
        msg = f"Correct suggestion: {action.suggested_action.value}."
    elif _is_acceptable_alternative(action.suggested_action, gt.recommended_action):
        score = 0.04
        msg = (
            f"Acceptable alternative. You said {action.suggested_action.value}, "
            f"ideal is {gt.recommended_action.value}."
        )
    else:
        score = -0.05
        msg = (
            f"Incorrect. You said {action.suggested_action.value}, "
            f"correct is {gt.recommended_action.value}."
        )

    breakdown = {"suggest_accuracy": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_reason_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Compute reward for reason action."""
    if record.reason_action is not None:
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_reason": -0.02},
                message="Already provided reasoning for this clause.",
            ),
            "Already provided reasoning for this clause.",
        )

    if not action.reasoning:
        return Reward(score=-0.05, message="No reasoning provided."), "No reasoning."

    reasoning_lower = action.reasoning.lower()
    keywords = gt.reasoning_keywords
    matched = [kw for kw in keywords if kw.lower() in reasoning_lower]
    ratio = len(matched) / len(keywords) if keywords else 0.0

    if ratio >= 0.8:
        score = 0.05
        msg = "Strong reasoning — covers key concepts."
    elif ratio >= 0.4:
        score = 0.02
        msg = f"Partial reasoning. Matched {len(matched)}/{len(keywords)} keywords."
    else:
        score = -0.02
        msg = f"Weak reasoning. Matched {len(matched)}/{len(keywords)} keywords."

    breakdown = {"reasoning_quality": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_progress_reward(
    record: ClauseActionRecord,
    required_actions: list,
    clause_index: int,
) -> Tuple[Reward, str]:
    """Compute reward for next_clause action."""
    completed = _count_completed_actions(record, required_actions)
    total_required = len(required_actions)

    if completed < total_required:
        completeness = completed / total_required
        score = 0.02 * completeness
        msg = (
            f"Moving to next clause. Completed {completed}/{total_required} "
            f"required assessments on clause {clause_index}."
        )
    else:
        score = 0.05
        msg = f"All assessments complete for clause {clause_index}. Moving on."

    breakdown = {"progress": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_completion_reward(
    total_clauses: int,
    clauses_reviewed: int,
    step_number: int,
    max_steps: int,
) -> Tuple[Reward, str]:
    """Compute reward for complete_review action."""
    ratio = clauses_reviewed / total_clauses if total_clauses > 0 else 0.0

    if ratio >= 1.0:
        score = 0.10
        msg = f"Review complete. All {total_clauses} clauses reviewed."
    elif ratio >= 0.5:
        score = 0.03
        msg = f"Review ended early. {clauses_reviewed}/{total_clauses} clauses reviewed."
    else:
        score = -0.05
        msg = f"Review ended very early. Only {clauses_reviewed}/{total_clauses} clauses reviewed."

    if step_number < max_steps * 0.8:
        score += 0.05
        msg += " Efficiency bonus applied."

    breakdown = {"completion": score}
    return Reward(score=score, breakdown=breakdown, message=msg), msg


def compute_invalid_action_penalty(error_msg: str) -> Tuple[Reward, str]:
    """Compute penalty for invalid action."""
    return (
        Reward(
            score=-0.05,
            breakdown={"invalid_action_penalty": -0.05},
            message=error_msg,
        ),
        error_msg,
    )


def _clause_type_family_match(agent_type: Optional[str], truth_type: str) -> bool:
    """Check if two clause types belong to the same broad family."""
    if agent_type is None:
        return False
    families = {
        "liability": {"indemnification", "limitation_of_liability", "warranty", "insurance"},
        "restrictive": {"non_compete", "confidentiality", "intellectual_property"},
        "governance": {"governing_law", "dispute_resolution", "assignment"},
        "commercial": {"payment_terms", "representations", "termination"},
        "external": {"force_majeure", "data_protection"},
    }
    for members in families.values():
        if agent_type in members and truth_type in members:
            return True
    return False


def _is_acceptable_alternative(
    agent_action: Optional[SuggestedActionType],
    truth_action: SuggestedActionType,
) -> bool:
    """Some suggestions are acceptable even if not exact match."""
    if agent_action is None:
        return False
    acceptable_map = {
        SuggestedActionType.REQUEST_MODIFICATION: {
            SuggestedActionType.FLAG_FOR_NEGOTIATION,
        },
        SuggestedActionType.FLAG_FOR_NEGOTIATION: {
            SuggestedActionType.REQUEST_MODIFICATION,
        },
        SuggestedActionType.ESCALATE_TO_SENIOR_COUNSEL: {
            SuggestedActionType.FLAG_FOR_NEGOTIATION,
            SuggestedActionType.REJECT_CLAUSE,
        },
        SuggestedActionType.REJECT_CLAUSE: {
            SuggestedActionType.ESCALATE_TO_SENIOR_COUNSEL,
        },
    }
    alternatives = acceptable_map.get(truth_action, set())
    return agent_action in alternatives


def _count_completed_actions(record: ClauseActionRecord, required: list) -> int:
    """Count how many required actions were completed."""
    completed = 0
    for at in required:
        if at == ActionType.CLASSIFY and record.classify_action is not None:
            completed += 1
        elif at == ActionType.RATE_SEVERITY and record.risk_action is not None:
            completed += 1
        elif at == ActionType.FLAG and record.flag_action is not None:
            completed += 1
        elif at == ActionType.SUGGEST and record.suggest_action is not None:
            completed += 1
        elif at == ActionType.REASON and record.reason_action is not None:
            completed += 1
    return completed


def compute_degenerate_penalty(consecutive_count: int) -> float:
    """Escalating penalty for repeated identical actions on same clause."""
    if consecutive_count <= 0:
        return 0.0
    penalty = -0.02 * (2 ** (consecutive_count - 1))
    return max(penalty, -0.50)


def compute_no_clause_penalty() -> Tuple[Reward, str]:
    """Penalty for acting when no clause is available."""
    return (
        Reward(
            score=-0.05,
            breakdown={"no_clause_penalty": -0.05},
            message="No clause to act on. Use complete_review.",
        ),
        "No clause available. Use complete_review.",
    )
