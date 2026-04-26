"""
LexArena — Per-Step Reward Functions
======================================
Multi-component rewards with anti-abuse checks.

Design principles (per hackathon rubric §7, §8):
  - Multiple INDEPENDENT reward signals (type + risk + flag + suggest + reason)
  - Each sub-reward capped separately — cannot hack total via one component
  - Degenerate penalty ONLY fires when the record already has that action stored
  - corrective_feedback returned with every reward for process-aware learning
  - No mutable global state; all side-effects isolated to return values
"""

from __future__ import annotations

from typing import Optional, Tuple, List

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

# ---------------------------------------------------------------------------
# Tier 2 — Clause Review Rewards
# ---------------------------------------------------------------------------

def compute_classify_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Reward for classify action.

    Bug fixed: redundant check now uses record.classify_action, not
    degenerate-penalty sequencing. Only fires if action is already stored.
    """
    if record.classify_action is not None:
        msg = (
            f"Already classified. Correct type was '{gt.clause_type}'. "
            "Use next_clause to move on."
        )
        return (
            Reward(
                score=-0.02,
                breakdown={"redundant_classify": -0.02},
                message=msg,
                corrective_feedback=msg,
            ),
            msg,
        )

    correct = action.clause_type == gt.clause_type
    family  = _clause_type_family_match(action.clause_type, gt.clause_type)

    if correct:
        score = 0.15
        feedback = f"Correct. '{action.clause_type}' is the right clause type."
    elif family:
        score = 0.05
        feedback = (
            f"Partial. You said '{action.clause_type}'; correct is '{gt.clause_type}'. "
            "They share the same legal family but are distinct types."
        )
    else:
        score = -0.05
        feedback = (
            f"Incorrect. You said '{action.clause_type}'; correct is '{gt.clause_type}'. "
            "These are different legal categories."
        )

    return Reward(score=score, breakdown={"type_accuracy": score}, message=feedback,
                  corrective_feedback=feedback), feedback


def compute_risk_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Reward for rate_severity action."""
    if record.risk_action is not None:
        msg = (
            f"Already rated. Correct risk level was '{gt.risk_level.value}'. "
            "Use next_clause to move on."
        )
        return (
            Reward(score=-0.02, breakdown={"redundant_risk": -0.02}, message=msg,
                   corrective_feedback=msg),
            msg,
        )

    if action.risk_level is None:
        msg = "No risk level provided. Must supply risk_level for rate_severity."
        return Reward(score=-0.05, message=msg, corrective_feedback=msg), msg

    try:
        agent_idx = RISK_ORDER.index(action.risk_level)
        truth_idx = RISK_ORDER.index(gt.risk_level)
        distance  = abs(agent_idx - truth_idx)
    except ValueError:
        distance = 3

    if distance == 0:
        score    = 0.15
        feedback = f"Correct risk level: '{action.risk_level.value}'."
    elif distance == 1:
        score    = 0.05
        feedback = (
            f"Close. You said '{action.risk_level.value}'; correct is "
            f"'{gt.risk_level.value}'. Off by one level."
        )
    else:
        score    = -0.05
        feedback = (
            f"Incorrect. You said '{action.risk_level.value}'; correct is "
            f"'{gt.risk_level.value}'. Risk is significantly mis-estimated."
        )

    return Reward(score=score, breakdown={"risk_accuracy": score}, message=feedback,
                  corrective_feedback=feedback), feedback


def compute_flag_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Reward for flag action. Uses F1 scoring with multi-component breakdown."""
    if record.flag_action is not None:
        msg = (
            f"Already flagged. Correct flags: {sorted(gt.issues) or '[]'}. "
            "Use next_clause to move on."
        )
        return (
            Reward(score=-0.02, breakdown={"redundant_flag": -0.02}, message=msg,
                   corrective_feedback=msg),
            msg,
        )

    agent_set = set(action.flags or [])
    truth_set = set(gt.issues)

    if not truth_set and not agent_set:
        score    = 0.10
        feedback = "Correct: no issues to flag for this clause."
    elif not truth_set and agent_set:
        score    = -0.05
        feedback = (
            f"False positives: {sorted(agent_set)}. This clause has no issues. "
            "Avoid flagging issues that don't exist."
        )
    elif truth_set and not agent_set:
        score    = -0.08
        feedback = (
            f"Missed issues: {sorted(truth_set)}. "
            "Always identify all real problems in the clause."
        )
    else:
        tp = len(agent_set & truth_set)
        fp = len(agent_set - truth_set)
        fn = len(truth_set - agent_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        # Scale: perfect F1 = +0.15, zero F1 = -0.05
        score    = round(f1 * 0.20 - 0.05, 4)
        missed   = sorted(truth_set - agent_set)
        extra    = sorted(agent_set - truth_set)
        feedback = (
            f"F1={f1:.2f} (P={precision:.2f}, R={recall:.2f}). "
            + (f"Missed: {missed}. " if missed else "")
            + (f"Extra: {extra}." if extra else "")
        )

    return Reward(score=score, breakdown={"flag_accuracy": score}, message=feedback,
                  corrective_feedback=feedback), feedback


def compute_suggest_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Reward for suggest action."""
    if record.suggest_action is not None:
        msg = (
            f"Already suggested. Correct action was '{gt.recommended_action.value}'. "
            "Use next_clause to move on."
        )
        return (
            Reward(score=-0.02, breakdown={"redundant_suggest": -0.02}, message=msg,
                   corrective_feedback=msg),
            msg,
        )

    if action.suggested_action is None:
        msg = "No suggested_action provided. Required for suggest action type."
        return Reward(score=-0.05, message=msg, corrective_feedback=msg), msg

    if action.suggested_action == gt.recommended_action:
        score    = 0.10
        feedback = f"Correct recommendation: '{action.suggested_action.value}'."
    elif _is_acceptable_alternative(action.suggested_action, gt.recommended_action):
        score    = 0.04
        feedback = (
            f"Acceptable but not ideal. You said '{action.suggested_action.value}'; "
            f"preferred is '{gt.recommended_action.value}'."
        )
    else:
        score    = -0.05
        feedback = (
            f"Incorrect. You said '{action.suggested_action.value}'; "
            f"correct is '{gt.recommended_action.value}'."
        )

    return Reward(score=score, breakdown={"suggest_accuracy": score}, message=feedback,
                  corrective_feedback=feedback), feedback


def compute_reason_reward(
    action: Action,
    record: ClauseActionRecord,
    gt: ClauseGroundTruth,
) -> Tuple[Reward, str]:
    """Reward for reason action. Keyword-matching process reward."""
    if record.reason_action is not None:
        msg = "Already provided reasoning. Use next_clause to move on."
        return (
            Reward(score=-0.02, breakdown={"redundant_reason": -0.02}, message=msg,
                   corrective_feedback=msg),
            msg,
        )

    if not action.reasoning or len(action.reasoning.strip()) < 10:
        msg = "Reasoning too short or missing. Provide a substantive explanation."
        return Reward(score=-0.05, message=msg, corrective_feedback=msg), msg

    reasoning_lower = action.reasoning.lower()
    keywords        = gt.reasoning_keywords
    matched         = [kw for kw in keywords if kw.lower() in reasoning_lower]
    ratio           = len(matched) / len(keywords) if keywords else 0.5

    # Anti-hack: reasoning must be > 20 chars to qualify for positive score
    if len(action.reasoning.strip()) < 20:
        ratio = 0.0

    if ratio >= 0.8:
        score    = 0.05
        feedback = f"Strong reasoning — covers {len(matched)}/{len(keywords)} key concepts."
    elif ratio >= 0.4:
        score    = 0.02
        feedback = (
            f"Partial reasoning. Matched {len(matched)}/{len(keywords)} keywords: "
            f"{matched}. Missing: {[k for k in keywords if k.lower() not in reasoning_lower]}"
        )
    else:
        score    = -0.02
        feedback = (
            f"Weak reasoning. Matched only {len(matched)}/{len(keywords)} keywords. "
            f"Key concepts missing: {keywords[:3]}"
        )

    return Reward(score=score, breakdown={"reasoning_quality": score}, message=feedback,
                  corrective_feedback=feedback), feedback


def compute_progress_reward(
    record: ClauseActionRecord,
    required_actions: list,
    clause_index: int,
) -> Tuple[Reward, str]:
    """Reward for next_clause action. Incentivises completing all required sub-tasks."""
    completed     = _count_completed_actions(record, required_actions)
    total_required = len(required_actions)

    if completed < total_required:
        completeness = completed / total_required if total_required > 0 else 0.0
        score        = round(0.02 * completeness, 4)
        msg          = (
            f"Moving to next clause. Completed {completed}/{total_required} "
            f"required assessments on clause {clause_index}. "
            f"Tip: complete all required actions before moving on for maximum score."
        )
    else:
        score = 0.05
        msg   = f"All {total_required} assessments complete for clause {clause_index}. Moving on."

    return Reward(score=score, breakdown={"progress": score}, message=msg,
                  corrective_feedback=msg), msg


def compute_completion_reward(
    total_clauses: int,
    clauses_reviewed: int,
    step_number: int,
    max_steps: int,
) -> Tuple[Reward, str]:
    """Reward for complete_review action. Efficiency bonus if finished early."""
    ratio = clauses_reviewed / total_clauses if total_clauses > 0 else 0.0

    if ratio >= 1.0:
        score = 0.10
        msg   = f"Review complete. All {total_clauses} clauses reviewed."
    elif ratio >= 0.75:
        score = 0.05
        msg   = f"Review ended. {clauses_reviewed}/{total_clauses} clauses reviewed (≥75%)."
    elif ratio >= 0.5:
        score = 0.02
        msg   = f"Review ended early. Only {clauses_reviewed}/{total_clauses} clauses reviewed."
    else:
        score = -0.05
        msg   = (
            f"Review ended very early. Only {clauses_reviewed}/{total_clauses} clauses "
            "reviewed. Significant coverage penalty applied."
        )

    # Efficiency bonus: finished before 80% of steps used
    if ratio >= 1.0 and step_number < max_steps * 0.80:
        score += 0.05
        msg   += " Efficiency bonus applied (finished early)."

    return Reward(score=score, breakdown={"completion": score}, message=msg,
                  corrective_feedback=msg), msg


def compute_invalid_action_penalty(error_msg: str) -> Tuple[Reward, str]:
    """Penalty for invalid/malformed actions."""
    feedback = f"Invalid action: {error_msg}"
    return (
        Reward(
            score=-0.05,
            breakdown={"invalid_action_penalty": -0.05},
            message=feedback,
            corrective_feedback=feedback,
        ),
        feedback,
    )


def compute_no_clause_penalty() -> Tuple[Reward, str]:
    """Penalty for acting when no clause is available."""
    msg = "No clause available to act on. Use complete_review to end the episode."
    return (
        Reward(
            score=-0.05,
            breakdown={"no_clause_penalty": -0.05},
            message=msg,
            corrective_feedback=msg,
        ),
        msg,
    )


def compute_degenerate_penalty(consecutive_count: int) -> float:
    """
    Escalating penalty for TRULY repeated identical actions on the same clause.

    This only triggers when _detect_degenerate detects the same action_type
    on the same clause_index when the record already has that action stored.
    Normal sequences (classify → rate_severity → flag) on the same clause
    are NOT degenerate and will NOT trigger this.
    """
    if consecutive_count <= 0:
        return 0.0
    penalty = -0.02 * (2 ** (consecutive_count - 1))
    return max(penalty, -0.50)


# ---------------------------------------------------------------------------
# Tier 1 — Clause Extraction Rewards (F2 + Jaccard)
# ---------------------------------------------------------------------------

def compute_extraction_reward(
    extracted_text: str,
    ground_truth_text: str,
    has_answer: bool,
) -> Tuple[Reward, str]:
    """
    Tier 1 reward: verbatim clause extraction.

    Scoring (pure math, no LLM judge):
        score = 0.60 * F2  +  0.25 * Jaccard  +  0.15 * (1 - laziness)

    F2 is recall-weighted: missing a real clause (FN) is worse than
    over-extracting (FP).
    """
    extracted = extracted_text.strip() if extracted_text else ""
    ref       = ground_truth_text.strip() if ground_truth_text else ""

    # Laziness detection: agent responded "No related clause." when there IS one
    NO_CLAUSE = "no related clause"
    is_lazy   = NO_CLAUSE in extracted.lower() and has_answer
    laziness  = 1.0 if is_lazy else 0.0

    if is_lazy:
        score    = max(0.001, 0.15 * (1 - laziness))
        feedback = (
            "Refusal error: you said 'No related clause' but a relevant clause "
            f"exists. It was: \"{ref[:150]}...\""
        )
        return Reward(
            score=round(score, 4),
            breakdown={"f2_score": 0.0, "jaccard": 0.0, "laziness_penalty": -0.15},
            message=feedback,
            corrective_feedback=feedback,
        ), feedback

    if not has_answer and NO_CLAUSE in extracted.lower():
        score    = 0.90
        feedback = "Correct: no related clause exists and you identified that."
        return Reward(
            score=score,
            breakdown={"f2_score": 0.9, "jaccard": 0.9, "laziness_penalty": 0.0},
            message=feedback,
            corrective_feedback=feedback,
        ), feedback

    # Tokenise for F2
    def tokenize(text: str) -> List[str]:
        return [w.lower().strip(".,;:\"'()") for w in text.split() if w.strip()]

    pred_tokens = set(tokenize(extracted))
    ref_tokens  = set(tokenize(ref))

    if not ref_tokens:
        score    = 0.5
        feedback = "Reference answer is empty — awarded 0.5 by default."
        return Reward(score=score, breakdown={}, message=feedback,
                      corrective_feedback=feedback), feedback

    tp = len(pred_tokens & ref_tokens)
    fp = len(pred_tokens - ref_tokens)
    fn = len(ref_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F2: recall weighted 2× (beta=2)
    beta      = 2
    f2 = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) \
         if (beta**2 * precision + recall) > 0 else 0.0

    # Jaccard
    union   = len(pred_tokens | ref_tokens)
    jaccard = tp / union if union > 0 else 0.0

    raw   = 0.60 * f2 + 0.25 * jaccard + 0.15 * (1 - laziness)
    score = max(0.001, min(0.999, raw))

    coverage = round(recall * 100)
    feedback = (
        f"Extraction score: {score:.3f} "
        f"(F2={f2:.2f}, Jaccard={jaccard:.2f}, coverage={coverage}%). "
    )
    if recall < 0.5:
        feedback += f"Low recall — you missed key words. Reference: \"{ref[:120]}...\""

    return Reward(
        score=round(score, 4),
        breakdown={"f2_score": round(f2, 4), "jaccard": round(jaccard, 4),
                   "laziness_penalty": round(-0.15 * laziness, 4)},
        message=feedback,
        corrective_feedback=feedback,
    ), feedback


# ---------------------------------------------------------------------------
# Tier 3 — Dependency Graph Rewards (precision / recall)
# ---------------------------------------------------------------------------

def compute_graph_reward(
    predicted_edges: list,
    ground_truth_edges: list,
) -> Tuple[Reward, str]:
    """
    Tier 3 reward: dependency graph mapping.

    An edge matches if (source_contract, source_clause, target_contract,
    target_clause) all match — edge_type gets a bonus if also correct.

    Score = 0.50 * recall  +  0.25 * precision
          + 0.15 * edge_type_accuracy  +  0.10 * severity_bonus
    """
    def edge_key(e: dict) -> tuple:
        return (
            e.get("source_contract", ""),
            e.get("source_clause", ""),
            e.get("target_contract", ""),
            e.get("target_clause", ""),
        )

    pred_keys = {edge_key(e): e.get("edge_type", "") for e in predicted_edges}
    gt_keys   = {edge_key(e): e.get("edge_type", "") for e in ground_truth_edges}

    if not gt_keys:
        # No dependencies expected
        if not pred_keys:
            score    = 0.90
            feedback = "Correct: no dependency edges exist in this scenario."
        else:
            score    = 0.20
            feedback = (
                f"False positives: you found {len(pred_keys)} edges, but none exist. "
                "Avoid hallucinating cross-document dependencies."
            )
        return Reward(score=score, breakdown={"graph_score": score}, message=feedback,
                      corrective_feedback=feedback), feedback

    tp_keys = set(pred_keys.keys()) & set(gt_keys.keys())
    fp      = len(pred_keys) - len(tp_keys)
    fn      = len(gt_keys)   - len(tp_keys)

    precision = len(tp_keys) / len(pred_keys) if pred_keys else 0.0
    recall    = len(tp_keys) / len(gt_keys)  if gt_keys   else 0.0

    # Edge type bonus
    type_correct = sum(
        1 for k in tp_keys if pred_keys[k] == gt_keys[k]
    )
    type_acc = type_correct / len(tp_keys) if tp_keys else 0.0

    raw   = 0.50 * recall + 0.25 * precision + 0.15 * type_acc + 0.10
    score = max(0.001, min(0.999, raw))

    missed = [k for k in gt_keys if k not in pred_keys]
    feedback = (
        f"Graph score: {score:.3f} "
        f"(recall={recall:.2f}, precision={precision:.2f}, "
        f"edge_type_acc={type_acc:.2f}). "
        f"TP={len(tp_keys)}, FP={fp}, FN={fn}."
    )
    if missed:
        feedback += (
            f" Missed edges: "
            + ", ".join(f"{k[0]}→{k[2]}" for k in missed[:3])
        )

    return Reward(
        score=round(score, 4),
        breakdown={
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "edge_type_accuracy": round(type_acc, 4),
        },
        message=feedback,
        corrective_feedback=feedback,
    ), feedback


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clause_type_family_match(agent_type: Optional[str], truth_type: str) -> bool:
    if agent_type is None:
        return False
    families = {
        "liability":   {"indemnification", "limitation_of_liability", "warranty", "insurance"},
        "restrictive": {"non_compete", "confidentiality", "intellectual_property"},
        "governance":  {"governing_law", "dispute_resolution", "assignment"},
        "commercial":  {"payment_terms", "representations", "termination"},
        "external":    {"force_majeure", "data_protection"},
    }
    for members in families.values():
        if agent_type in members and truth_type in members:
            return True
    return False


def _is_acceptable_alternative(
    agent_action: Optional[SuggestedActionType],
    truth_action: SuggestedActionType,
) -> bool:
    if agent_action is None:
        return False
    acceptable_map = {
        SuggestedActionType.REQUEST_MODIFICATION:     {SuggestedActionType.FLAG_FOR_NEGOTIATION},
        SuggestedActionType.FLAG_FOR_NEGOTIATION:     {SuggestedActionType.REQUEST_MODIFICATION},
        SuggestedActionType.ESCALATE_TO_SENIOR_COUNSEL: {
            SuggestedActionType.FLAG_FOR_NEGOTIATION,
            SuggestedActionType.REJECT_CLAUSE,
        },
        SuggestedActionType.REJECT_CLAUSE: {SuggestedActionType.ESCALATE_TO_SENIOR_COUNSEL},
    }
    return agent_action in acceptable_map.get(truth_action, set())


def _count_completed_actions(record: ClauseActionRecord, required: list) -> int:
    completed = 0
    for at in required:
        if at == ActionType.CLASSIFY     and record.classify_action  is not None:
            completed += 1
        elif at == ActionType.RATE_SEVERITY and record.risk_action   is not None:
            completed += 1
        elif at == ActionType.FLAG          and record.flag_action    is not None:
            completed += 1
        elif at == ActionType.SUGGEST       and record.suggest_action is not None:
            completed += 1
        elif at == ActionType.REASON        and record.reason_action  is not None:
            completed += 1
    return completed
