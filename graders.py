"""
LexArena — Trajectory-Level Graders
=====================================
Graders for all tiers. Pure-math, deterministic, no LLM-as-judge.

Tiers:
  Tier 1 (task_1_easy / task_2_medium / task_3_hard) — clause review graders
  Tier 1 extraction — F2 + Jaccard + laziness
  Tier 3 dependency — precision + recall + edge_type_accuracy

Design principles (hackathon rubric §7, §8):
  - Multiple independent scoring components (type, risk, flag, suggest, reason)
  - Scores strictly in (0, 1) — clamped to [0.001, 0.999]
  - No global state mutation
  - Grader failures return safe score 0.001, never crash
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import (
    ClauseActionRecord,
    ClauseGroundTruth,
    EpisodeMeta,
    GraderResult,
    RiskLevel,
    SuggestedActionType,
    TaskConfig,
    Tier1ExtractionSample,
)


RISK_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


# ===========================================================================
# Main dispatch
# ===========================================================================

def grade_episode(
    task_id: str,
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Dispatch to the correct task grader."""
    graders = {
        "task_1_easy":   _grade_task_1,
        "task_2_medium": _grade_task_2,
        "task_3_hard":   _grade_task_3,
    }
    grader_fn = graders.get(task_id)
    if grader_fn is None:
        return GraderResult(score=0.001, message=f"No grader for {task_id}")
    return grader_fn(clause_records, ground_truth, task_config, episode_meta)


# ===========================================================================
# Tier 2a — Clause Classification (Easy)
# ===========================================================================

def _grade_task_1(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Score = type_accuracy + completion component + penalties."""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    per_clause: List[Dict[str, Any]] = []
    type_scores: List[float] = []

    for i in range(total_clauses):
        gt     = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts = _score_type_accuracy(record, gt)
        type_scores.append(ts)

        per_clause.append({
            "clause_index": i,
            "type_score":   round(ts, 4),
            "agent_type":   record.classify_action if record else None,
            "truth_type":   gt.clause_type,
        })

    avg_type = sum(type_scores) / total_clauses

    # Coverage bonus: reward reviewing more clauses
    coverage_ratio = episode_meta.clauses_reviewed / total_clauses if total_clauses > 0 else 0.0
    coverage_bonus = 0.10 * coverage_ratio

    penalties     = _compute_penalties(episode_meta)
    penalty_total = sum(penalties.values())

    raw_score   = avg_type * 0.90 + coverage_bonus + penalty_total
    final_score = _clamp(raw_score)

    return GraderResult(
        score=round(final_score, 4),
        breakdown={
            "type_accuracy":   round(avg_type, 4),
            "coverage_bonus":  round(coverage_bonus, 4),
        },
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
        tier=2,
    )


# ===========================================================================
# Tier 2b — Risk Assessment (Medium)
# ===========================================================================

def _grade_task_2(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Score = 0.30*type + 0.30*risk + 0.40*flag + coverage + penalties."""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    weights = task_config.grader_weights
    w_type  = weights.get("type_accuracy", 0.30)
    w_risk  = weights.get("risk_accuracy", 0.30)
    w_flag  = weights.get("flag_accuracy", 0.40)

    per_clause: List[Dict[str, Any]] = []
    type_scores:  List[float] = []
    risk_scores:  List[float] = []
    flag_scores:  List[float] = []

    for i in range(total_clauses):
        gt     = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts = _score_type_accuracy(record, gt)
        rs = _score_risk_accuracy(record, gt)
        fs = _score_flag_accuracy(record, gt)

        type_scores.append(ts)
        risk_scores.append(rs)
        flag_scores.append(fs)

        per_clause.append({
            "clause_index": i,
            "type_score":   round(ts, 4),
            "risk_score":   round(rs, 4),
            "flag_score":   round(fs, 4),
            "agent_type":   record.classify_action if record else None,
            "agent_risk":   (record.risk_action.value if record and record.risk_action else None),
            "agent_flags":  (sorted(record.flag_action) if record and record.flag_action else None),
            "truth_type":   gt.clause_type,
            "truth_risk":   gt.risk_level.value,
            "truth_flags":  sorted(gt.issues),
        })

    avg_type = sum(type_scores) / total_clauses
    avg_risk = sum(risk_scores) / total_clauses
    avg_flag = sum(flag_scores) / total_clauses

    weighted = w_type * avg_type + w_risk * avg_risk + w_flag * avg_flag

    coverage_ratio = episode_meta.clauses_reviewed / total_clauses if total_clauses > 0 else 0.0
    coverage_bonus = 0.10 * coverage_ratio

    penalties     = _compute_penalties(episode_meta)
    penalty_total = sum(penalties.values())

    raw_score   = weighted * 0.90 + coverage_bonus + penalty_total
    final_score = _clamp(raw_score)

    return GraderResult(
        score=round(final_score, 4),
        breakdown={
            "type_accuracy":            round(avg_type, 4),
            "risk_accuracy":            round(avg_risk, 4),
            "flag_accuracy":            round(avg_flag, 4),
            "weighted_before_penalties": round(weighted, 4),
            "coverage_bonus":           round(coverage_bonus, 4),
        },
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
        tier=2,
    )


# ===========================================================================
# Tier 2c — Full Contract Review (Hard)
# ===========================================================================

def _grade_task_3(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Full grading: type, risk, flag, suggest, reasoning, priority, coverage."""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    weights   = task_config.grader_weights
    w_type    = weights.get("type_accuracy",    0.15)
    w_risk    = weights.get("risk_accuracy",    0.20)
    w_flag    = weights.get("flag_accuracy",    0.20)
    w_suggest = weights.get("suggest_accuracy", 0.20)
    w_reason  = weights.get("reasoning_quality", 0.15)
    w_priority = weights.get("priority_ordering", 0.10)

    per_clause: List[Dict[str, Any]] = []
    type_scores:    List[float] = []
    risk_scores:    List[float] = []
    flag_scores:    List[float] = []
    suggest_scores: List[float] = []
    reason_scores:  List[float] = []

    for i in range(total_clauses):
        gt     = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts  = _score_type_accuracy(record, gt)
        rs  = _score_risk_accuracy(record, gt)
        fs  = _score_flag_accuracy(record, gt)
        ss  = _score_suggest_accuracy(record, gt)
        qs  = _score_reasoning_quality(record, gt)

        type_scores.append(ts)
        risk_scores.append(rs)
        flag_scores.append(fs)
        suggest_scores.append(ss)
        reason_scores.append(qs)

        per_clause.append({
            "clause_index":  i,
            "type_score":    round(ts, 4),
            "risk_score":    round(rs, 4),
            "flag_score":    round(fs, 4),
            "suggest_score": round(ss, 4),
            "reason_score":  round(qs, 4),
        })

    avg_type    = sum(type_scores)    / total_clauses
    avg_risk    = sum(risk_scores)    / total_clauses
    avg_flag    = sum(flag_scores)    / total_clauses
    avg_suggest = sum(suggest_scores) / total_clauses
    avg_reason  = sum(reason_scores)  / total_clauses
    priority_score = _score_priority_ordering(clause_records, ground_truth)

    weighted = (
        w_type    * avg_type
        + w_risk    * avg_risk
        + w_flag    * avg_flag
        + w_suggest * avg_suggest
        + w_reason  * avg_reason
        + w_priority * priority_score
    )

    coverage_ratio = episode_meta.clauses_reviewed / total_clauses if total_clauses > 0 else 0.0
    coverage_bonus = 0.10 * coverage_ratio

    penalties     = _compute_penalties(episode_meta, clause_records, ground_truth)
    penalty_total = sum(penalties.values())

    raw_score   = weighted * 0.90 + coverage_bonus + penalty_total
    final_score = _clamp(raw_score)

    return GraderResult(
        score=round(final_score, 4),
        breakdown={
            "type_accuracy":            round(avg_type, 4),
            "risk_accuracy":            round(avg_risk, 4),
            "flag_accuracy":            round(avg_flag, 4),
            "suggest_accuracy":         round(avg_suggest, 4),
            "reasoning_quality":        round(avg_reason, 4),
            "priority_ordering":        round(priority_score, 4),
            "weighted_before_penalties": round(weighted, 4),
            "coverage_bonus":           round(coverage_bonus, 4),
        },
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
        tier=2,
    )


# ===========================================================================
# Tier 1 — Clause Extraction Grader (F2 + Jaccard + laziness)
# ===========================================================================

def grade_tier1_extraction(
    sample: Tier1ExtractionSample,
    extracted_text: str,
) -> GraderResult:
    """
    Grade a single Tier 1 extraction.

    Score = 0.60 * F2  +  0.25 * Jaccard  +  0.15 * (1 - laziness_rate)

    F2 recall-weighted: missing a real clause (FN) > false extraction (FP).
    """
    extracted = extracted_text.strip() if extracted_text else ""
    ref       = sample.answer_text.strip() if sample.answer_text else ""
    NO_CLAUSE = "no related clause"

    is_lazy = NO_CLAUSE in extracted.lower() and sample.has_answer
    laziness = 1.0 if is_lazy else 0.0

    if is_lazy:
        raw   = 0.15 * (1 - laziness)
        score = max(0.001, min(0.999, raw))
        return GraderResult(
            score=round(score, 4),
            breakdown={"f2_score": 0.0, "jaccard": 0.0, "laziness_penalty": -0.15},
            message=f"Refusal error — clause exists: \"{ref[:100]}...\"",
            tier=1,
        )

    if not sample.has_answer and NO_CLAUSE in extracted.lower():
        return GraderResult(
            score=0.90,
            breakdown={"f2_score": 0.9, "jaccard": 0.9, "laziness_penalty": 0.0},
            message="Correct: no related clause.",
            tier=1,
        )

    def tokenize(text: str):
        return [w.lower().strip(".,;:\"'()") for w in text.split() if w.strip()]

    pred_tokens = set(tokenize(extracted))
    ref_tokens  = set(tokenize(ref))

    if not ref_tokens:
        return GraderResult(score=0.5, breakdown={}, message="Empty reference.", tier=1)

    tp = len(pred_tokens & ref_tokens)
    fp = len(pred_tokens - ref_tokens)
    fn = len(ref_tokens  - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 2
    f2 = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) \
         if (beta**2 * precision + recall) > 0 else 0.0
    union   = len(pred_tokens | ref_tokens)
    jaccard = tp / union if union > 0 else 0.0

    raw   = 0.60 * f2 + 0.25 * jaccard + 0.15 * (1 - laziness)
    score = max(0.001, min(0.999, raw))

    return GraderResult(
        score=round(score, 4),
        breakdown={
            "f2_score":        round(f2, 4),
            "jaccard":         round(jaccard, 4),
            "laziness_penalty": 0.0,
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
        },
        message=f"T1 score {score:.3f}: F2={f2:.2f}, Jaccard={jaccard:.2f}",
        tier=1,
    )


# ===========================================================================
# Tier 3 — Dependency Graph Grader
# ===========================================================================

def grade_tier3_dependency(
    predicted_edges: List[Dict],
    ground_truth_edges: List[Dict],
) -> GraderResult:
    """
    Grade a Tier 3 dependency graph submission.

    Score = 0.50*recall + 0.25*precision + 0.15*edge_type_accuracy + 0.10
    An edge matches if (src_contract, src_clause, tgt_contract, tgt_clause) match.
    Edge type correct gives extra credit.
    """
    def edge_key(e: dict) -> tuple:
        return (
            e.get("source_contract", ""),
            e.get("source_clause",   ""),
            e.get("target_contract", ""),
            e.get("target_clause",   ""),
        )

    pred_dict = {edge_key(e): e.get("edge_type", "") for e in predicted_edges}
    gt_dict   = {edge_key(e): e.get("edge_type", "") for e in ground_truth_edges}

    if not gt_dict:
        if not pred_dict:
            return GraderResult(
                score=0.90,
                breakdown={"recall": 1.0, "precision": 1.0, "edge_type_accuracy": 1.0},
                message="Correct: no dependency edges exist.",
                tier=3,
            )
        return GraderResult(
            score=0.20,
            breakdown={"recall": 0.0, "precision": 0.0},
            message=f"False positives: found {len(pred_dict)} spurious edges.",
            tier=3,
        )

    tp_keys   = set(pred_dict) & set(gt_dict)
    precision = len(tp_keys) / len(pred_dict) if pred_dict else 0.0
    recall    = len(tp_keys) / len(gt_dict)

    type_correct = sum(1 for k in tp_keys if pred_dict[k] == gt_dict[k])
    type_acc     = type_correct / len(tp_keys) if tp_keys else 0.0

    raw   = 0.50 * recall + 0.25 * precision + 0.15 * type_acc + 0.10
    score = max(0.001, min(0.999, raw))

    fn = len(gt_dict) - len(tp_keys)
    fp = len(pred_dict) - len(tp_keys)

    return GraderResult(
        score=round(score, 4),
        breakdown={
            "recall":             round(recall,   4),
            "precision":          round(precision, 4),
            "edge_type_accuracy": round(type_acc,  4),
            "tp":                 len(tp_keys),
            "fp":                 fp,
            "fn":                 fn,
        },
        message=(
            f"T3 score {score:.3f}: recall={recall:.2f}, "
            f"precision={precision:.2f}, edge_type_acc={type_acc:.2f}. "
            f"TP={len(tp_keys)}, FP={fp}, FN={fn}."
        ),
        tier=3,
    )


# ===========================================================================
# Public wrappers (backward-compatible)
# ===========================================================================

def grade_task_1(clause_records, ground_truth, task_config, episode_meta):
    return _grade_task_1(clause_records, ground_truth, task_config, episode_meta)

def grade_task_2(clause_records, ground_truth, task_config, episode_meta):
    return _grade_task_2(clause_records, ground_truth, task_config, episode_meta)

def grade_task_3(clause_records, ground_truth, task_config, episode_meta):
    return _grade_task_3(clause_records, ground_truth, task_config, episode_meta)


# ===========================================================================
# Private scoring helpers
# ===========================================================================

def _score_type_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    if record is None or record.classify_action is None:
        return 0.001
    if record.classify_action == gt.clause_type:
        return 0.999
    if _clause_type_family_match(record.classify_action, gt.clause_type):
        return 0.5
    return 0.001


def _score_risk_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    if record is None or record.risk_action is None:
        return 0.001
    if record.risk_action == gt.risk_level:
        return 0.999
    try:
        agent_idx = RISK_ORDER.index(record.risk_action)
        truth_idx = RISK_ORDER.index(gt.risk_level)
        if abs(agent_idx - truth_idx) == 1:
            return 0.5
    except ValueError:
        pass
    return 0.001


def _score_flag_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    if record is None or record.flag_action is None:
        return 0.001
    agent_set = set(record.flag_action)
    truth_set = set(gt.issues)
    if not truth_set and not agent_set:
        return 0.999
    if not truth_set and agent_set:
        return 0.001
    if truth_set and not agent_set:
        return 0.001
    tp = len(agent_set & truth_set)
    fp = len(agent_set - truth_set)
    fn = len(truth_set - agent_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.001
    f1 = 2 * precision * recall / (precision + recall)
    return max(0.001, min(0.999, f1))


def _score_suggest_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    if record is None or record.suggest_action is None:
        return 0.001
    if record.suggest_action == gt.recommended_action:
        return 0.999
    if _is_acceptable_alternative(record.suggest_action, gt.recommended_action):
        return 0.5
    return 0.001


def _score_reasoning_quality(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    if record is None or record.reason_action is None:
        return 0.001
    reasoning_lower = record.reason_action.lower()
    keywords = gt.reasoning_keywords
    if not keywords:
        return 0.5
    matched = [kw for kw in keywords if kw.lower() in reasoning_lower]
    ratio   = len(matched) / len(keywords)
    return max(0.001, min(ratio, 0.999))


def _score_priority_ordering(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
) -> float:
    priority_indices = [
        i for i, gt in enumerate(ground_truth)
        if gt.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
    ]
    if not priority_indices:
        return 0.999
    reviewed = sum(
        1 for i in priority_indices
        if i < len(clause_records) and clause_records[i].action_count > 0
    )
    return max(0.001, min(0.999, reviewed / len(priority_indices)))


def _clause_type_family_match(agent_type: str, truth_type: str) -> bool:
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
    agent_action: SuggestedActionType,
    truth_action: SuggestedActionType,
) -> bool:
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


def _compute_penalties(
    episode_meta: EpisodeMeta,
    clause_records: Optional[List[ClauseActionRecord]] = None,
    ground_truth: Optional[List[ClauseGroundTruth]] = None,
) -> Dict[str, float]:
    """Multi-component penalty system (anti-reward-hacking §8)."""
    penalties: Dict[str, float] = {}

    # Penalty for invalid actions (capped at -0.30 to avoid kill-shot)
    if episode_meta.total_invalid_actions > 0:
        raw = -0.05 * episode_meta.total_invalid_actions
        penalties["invalid_actions"] = max(raw, -0.30)

    # Small penalty for redundant actions (already-taken)
    if episode_meta.total_redundant_actions > 0:
        raw = -0.02 * episode_meta.total_redundant_actions
        penalties["redundant_actions"] = max(raw, -0.20)

    # Penalty for skipped critical clauses
    if (
        clause_records is not None
        and ground_truth is not None
        and episode_meta.completed_normally
    ):
        missed_critical = sum(
            1 for i, gt in enumerate(ground_truth)
            if gt.risk_level == RiskLevel.CRITICAL
            and (i >= len(clause_records) or clause_records[i].action_count == 0)
        )
        if missed_critical > 0:
            penalties["missed_critical_clauses"] = -0.10 * missed_critical

    return penalties


def _clamp(score: float) -> float:
    return max(0.001, min(0.999, score))


def _build_message(score: float, total_clauses: int, episode_meta: EpisodeMeta) -> str:
    percentage = int(score * 100)
    msg = f"Graded {total_clauses} clauses. Final score: {percentage}%."
    if episode_meta.total_invalid_actions > 0:
        msg += f" {episode_meta.total_invalid_actions} invalid action(s) penalized."
    if episode_meta.total_redundant_actions > 0:
        msg += f" {episode_meta.total_redundant_actions} redundant action(s) penalized."
    if not episode_meta.completed_normally:
        msg += " Episode did not complete normally."
    return msg
