"""
Phase 3 — Grading Functions
Trajectory-level graders for each task.
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
)


RISK_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def grade_episode(
    task_id: str,
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Dispatch to the correct task grader."""
    graders = {
        "task_1_easy": _grade_task_1,
        "task_2_medium": _grade_task_2,
        "task_3_hard": _grade_task_3,
    }
    grader_fn = graders.get(task_id)
    if grader_fn is None:
        return GraderResult(score=0.001, message=f"No grader for {task_id}")
    return grader_fn(clause_records, ground_truth, task_config, episode_meta)


def _grade_task_1(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Score = average type_accuracy across all clauses."""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    per_clause: List[Dict[str, Any]] = []
    type_scores: List[float] = []

    for i in range(total_clauses):
        gt = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts = _score_type_accuracy(record, gt)
        type_scores.append(ts)

        per_clause.append(
            {
                "clause_index": i,
                "type_score": round(ts, 4),
                "agent_type": record.classify_action if record else None,
                "truth_type": gt.clause_type,
            }
        )

    avg_type = sum(type_scores) / total_clauses

    penalties = _compute_penalties(episode_meta)
    penalty_total = sum(penalties.values())

    raw_score = avg_type + penalty_total
    final_score = _clamp(raw_score)

    breakdown = {
        "type_accuracy": round(avg_type, 4),
    }

    return GraderResult(
        score=round(final_score, 4),
        breakdown=breakdown,
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
    )


def _grade_task_2(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Score = 0.30 * avg_type + 0.30 * avg_risk + 0.40 * avg_flag"""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    weights = task_config.grader_weights
    w_type = weights.get("type_accuracy", 0.30)
    w_risk = weights.get("risk_accuracy", 0.30)
    w_flag = weights.get("flag_accuracy", 0.40)

    per_clause: List[Dict[str, Any]] = []
    type_scores: List[float] = []
    risk_scores: List[float] = []
    flag_scores: List[float] = []

    for i in range(total_clauses):
        gt = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts = _score_type_accuracy(record, gt)
        rs = _score_risk_accuracy(record, gt)
        fs = _score_flag_accuracy(record, gt)

        type_scores.append(ts)
        risk_scores.append(rs)
        flag_scores.append(fs)

        per_clause.append(
            {
                "clause_index": i,
                "type_score": round(ts, 4),
                "risk_score": round(rs, 4),
                "flag_score": round(fs, 4),
                "agent_type": record.classify_action if record else None,
                "agent_risk": (
                    record.risk_action.value if record and record.risk_action else None
                ),
                "agent_flags": (
                    sorted(record.flag_action)
                    if record and record.flag_action
                    else None
                ),
                "truth_type": gt.clause_type,
                "truth_risk": gt.risk_level.value,
                "truth_flags": sorted(gt.issues),
            }
        )

    avg_type = sum(type_scores) / total_clauses
    avg_risk = sum(risk_scores) / total_clauses
    avg_flag = sum(flag_scores) / total_clauses

    weighted = w_type * avg_type + w_risk * avg_risk + w_flag * avg_flag

    penalties = _compute_penalties(episode_meta)
    penalty_total = sum(penalties.values())

    raw_score = weighted + penalty_total
    final_score = _clamp(raw_score)

    breakdown = {
        "type_accuracy": round(avg_type, 4),
        "risk_accuracy": round(avg_risk, 4),
        "flag_accuracy": round(avg_flag, 4),
        "weighted_before_penalties": round(weighted, 4),
    }

    return GraderResult(
        score=round(final_score, 4),
        breakdown=breakdown,
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
    )


def _grade_task_3(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Full grading with type, risk, flag, suggest, reasoning."""
    total_clauses = len(ground_truth)
    if total_clauses == 0:
        return GraderResult(score=0.001, message="No clauses to grade.")

    weights = task_config.grader_weights
    w_type = weights.get("type_accuracy", 0.15)
    w_risk = weights.get("risk_accuracy", 0.20)
    w_flag = weights.get("flag_accuracy", 0.20)
    w_suggest = weights.get("suggest_accuracy", 0.20)
    w_reason = weights.get("reasoning_quality", 0.15)
    w_priority = weights.get("priority_ordering", 0.10)

    per_clause: List[Dict[str, Any]] = []
    type_scores: List[float] = []
    risk_scores: List[float] = []
    flag_scores: List[float] = []
    suggest_scores: List[float] = []
    reason_scores: List[float] = []

    for i in range(total_clauses):
        gt = ground_truth[i]
        record = clause_records[i] if i < len(clause_records) else None

        ts = _score_type_accuracy(record, gt)
        rs = _score_risk_accuracy(record, gt)
        fs = _score_flag_accuracy(record, gt)
        ss = _score_suggest_accuracy(record, gt)
        rs_reason = _score_reasoning_quality(record, gt)

        type_scores.append(ts)
        risk_scores.append(rs)
        flag_scores.append(fs)
        suggest_scores.append(ss)
        reason_scores.append(rs_reason)

        per_clause.append(
            {
                "clause_index": i,
                "type_score": round(ts, 4),
                "risk_score": round(rs, 4),
                "flag_score": round(fs, 4),
                "suggest_score": round(ss, 4),
                "reason_score": round(rs_reason, 4),
            }
        )

    avg_type = sum(type_scores) / total_clauses
    avg_risk = sum(risk_scores) / total_clauses
    avg_flag = sum(flag_scores) / total_clauses
    avg_suggest = sum(suggest_scores) / total_clauses
    avg_reason = sum(reason_scores) / total_clauses
    priority_score = _score_priority_ordering(clause_records, ground_truth)

    weighted = (
        w_type * avg_type
        + w_risk * avg_risk
        + w_flag * avg_flag
        + w_suggest * avg_suggest
        + w_reason * avg_reason
        + w_priority * priority_score
    )

    penalties = _compute_penalties(episode_meta, clause_records, ground_truth)
    penalty_total = sum(penalties.values())

    raw_score = weighted + penalty_total
    final_score = _clamp(raw_score)

    breakdown = {
        "type_accuracy": round(avg_type, 4),
        "risk_accuracy": round(avg_risk, 4),
        "flag_accuracy": round(avg_flag, 4),
        "suggest_accuracy": round(avg_suggest, 4),
        "reasoning_quality": round(avg_reason, 4),
        "priority_ordering": round(priority_score, 4),
        "weighted_before_penalties": round(weighted, 4),
    }

    return GraderResult(
        score=round(final_score, 4),
        breakdown=breakdown,
        per_clause_scores=per_clause,
        penalties=penalties,
        message=_build_message(final_score, total_clauses, episode_meta),
    )


def _score_type_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    """Score for clause type classification."""
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
    """Score for risk level assessment."""
    if record is None or record.risk_action is None:
        return 0.001
    if record.risk_action == gt.risk_level:
        return 0.999
    try:
        agent_idx = RISK_ORDER.index(record.risk_action)
        truth_idx = RISK_ORDER.index(gt.risk_level)
        distance = abs(agent_idx - truth_idx)
        if distance == 1:
            return 0.5
    except ValueError:
        pass
    return 0.001


def _score_flag_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    """Score for issue flagging using F1."""
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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.001
    f1 = 2 * precision * recall / (precision + recall)
    return max(0.001, min(0.999, f1))


def _score_suggest_accuracy(
    record: Optional[ClauseActionRecord], gt: ClauseGroundTruth
) -> float:
    """Score for recommended action."""
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
    """Score for reasoning based on keyword matching."""
    if record is None or record.reason_action is None:
        return 0.001
    reasoning_lower = record.reason_action.lower()
    keywords = gt.reasoning_keywords
    if not keywords:
        return 0.5
    matched = [kw for kw in keywords if kw.lower() in reasoning_lower]
    ratio = len(matched) / len(keywords)
    return max(0.001, min(ratio, 0.999))


def _score_priority_ordering(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
) -> float:
    """Score whether high-priority clauses were actually reviewed."""
    priority_indices = [
        i
        for i, gt in enumerate(ground_truth)
        if gt.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
    ]
    if not priority_indices:
        return 0.999

    reviewed = 0
    for i in priority_indices:
        if i < len(clause_records) and clause_records[i].action_count > 0:
            reviewed += 1
    return max(0.001, min(0.999, reviewed / len(priority_indices)))


def _clause_type_family_match(agent_type: str, truth_type: str) -> bool:
    """Check if two clause types belong to the same broad family."""
    families = {
        "liability": {
            "indemnification",
            "limitation_of_liability",
            "warranty",
            "insurance",
        },
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
    agent_action: SuggestedActionType,
    truth_action: SuggestedActionType,
) -> bool:
    """Some suggestions are acceptable even if not exact match."""
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


def _compute_penalties(
    episode_meta: EpisodeMeta,
    clause_records: Optional[List[ClauseActionRecord]] = None,
    ground_truth: Optional[List[ClauseGroundTruth]] = None,
) -> Dict[str, float]:
    """Compute penalties for invalid/redundant actions and skipped clauses."""
    penalties: Dict[str, float] = {}

    if episode_meta.total_invalid_actions > 0:
        penalties["invalid_actions"] = -0.05 * episode_meta.total_invalid_actions

    if episode_meta.total_redundant_actions > 0:
        penalties["redundant_actions"] = -0.02 * episode_meta.total_redundant_actions

    if (
        clause_records is not None
        and ground_truth is not None
        and episode_meta.completed_normally
    ):
        skipped = 0
        missed_critical = 0
        for i in range(min(len(clause_records), len(ground_truth))):
            if clause_records[i].action_count == 0:
                skipped += 1
                if ground_truth[i].risk_level == RiskLevel.CRITICAL:
                    missed_critical += 1
        if skipped > 0:
            penalties["skipped_clauses"] = -0.1 * skipped
        if missed_critical > 0:
            penalties["missed_critical_clauses"] = -0.1 * missed_critical

    return penalties


def grade_task_1(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Public task-specific grader wrapper."""
    return _grade_task_1(clause_records, ground_truth, task_config, episode_meta)


def grade_task_2(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Public task-specific grader wrapper."""
    return _grade_task_2(clause_records, ground_truth, task_config, episode_meta)


def grade_task_3(
    clause_records: List[ClauseActionRecord],
    ground_truth: List[ClauseGroundTruth],
    task_config: TaskConfig,
    episode_meta: EpisodeMeta,
) -> GraderResult:
    """Public task-specific grader wrapper."""
    return _grade_task_3(clause_records, ground_truth, task_config, episode_meta)


def _clamp(score: float) -> float:
    """Clamp score strictly between (0.0, 1.0)."""
    return max(0.001, min(0.999, score))


def _build_message(score: float, total_clauses: int, episode_meta: EpisodeMeta) -> str:
    """Build human-readable message."""
    percentage = int(score * 100)
    msg = f"Graded {total_clauses} clauses. Final score: {percentage}%."

    if episode_meta.total_invalid_actions > 0:
        msg += f" {episode_meta.total_invalid_actions} invalid actions penalized."
    if episode_meta.total_redundant_actions > 0:
        msg += f" {episode_meta.total_redundant_actions} redundant actions penalized."

    if not episode_meta.completed_normally:
        msg += " Episode did not complete normally."

    return msg
