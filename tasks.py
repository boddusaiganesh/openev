"""
Phase 3 — Task Registry & Instructions
Central task configuration and instruction templates.
"""

from __future__ import annotations

from typing import Dict, List

from models import (
    ActionType,
    Difficulty,
    TaskConfig,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "task_1_easy": TaskConfig(
        task_id="task_1_easy",
        name="Clause Classification",
        difficulty=Difficulty.EASY,
        description=(
            "Classify each clause in a simple contract by type. "
            "Clauses use standard legal language with no ambiguity."
        ),
        max_steps=10,
        scenario_file="task_1_easy/scenario_1.json",
        required_action_types=[ActionType.CLASSIFY],
        grader_weights={"type_accuracy": 1.0},
    ),
    "task_2_medium": TaskConfig(
        task_id="task_2_medium",
        name="Risk Assessment",
        difficulty=Difficulty.MEDIUM,
        description=(
            "Classify clause type, assign risk level, and flag specific "
            "issues. Some clauses have non-obvious risks."
        ),
        max_steps=20,
        scenario_file="task_2_medium/scenario_1.json",
        required_action_types=[
            ActionType.CLASSIFY,
            ActionType.RATE_SEVERITY,
            ActionType.FLAG,
        ],
        grader_weights={
            "type_accuracy": 0.30,
            "risk_accuracy": 0.30,
            "flag_accuracy": 0.40,
        },
    ),
    "task_3_hard": TaskConfig(
        task_id="task_3_hard",
        name="Full Contract Review",
        difficulty=Difficulty.HARD,
        description=(
            "Full contract review: classify, assess risk, flag issues, "
            "suggest actions, provide reasoning. Contains conflicting "
            "clauses, red herrings, and sleeper clauses."
        ),
        max_steps=40,
        scenario_file="task_3_hard/scenario_1.json",
        required_action_types=[
            ActionType.CLASSIFY,
            ActionType.RATE_SEVERITY,
            ActionType.FLAG,
            ActionType.SUGGEST,
            ActionType.REASON,
        ],
        grader_weights={
            "type_accuracy": 0.15,
            "risk_accuracy": 0.20,
            "flag_accuracy": 0.20,
            "suggest_accuracy": 0.20,
            "reasoning_quality": 0.15,
            "priority_ordering": 0.10,
        },
    ),
}


TASK_INSTRUCTIONS: Dict[str, str] = {
    "task_1_easy": (
        "Review each clause and classify its type from the following taxonomy: "
        + ", ".join(CLAUSE_TAXONOMY)
        + ". Use the 'classify' action with clause_type set to the correct type. "
        "After classifying a clause, use 'next_clause' to proceed. "
        "When all clauses are reviewed, use 'complete_review'."
    ),
    "task_2_medium": (
        "Review each clause and perform three assessments:\n"
        "1. Classify the clause type using 'classify' with clause_type.\n"
        "2. Rate the risk level using 'rate_severity' with risk_level "
        "(low/medium/high/critical).\n"
        "3. Flag any issues using 'flag' with a list of applicable flags from: "
        + ", ".join(ISSUE_FLAGS)
        + ".\nAfter completing all assessments for a clause, use 'next_clause'. "
        "When finished, use 'complete_review'."
    ),
    "task_3_hard": (
        "Perform a full contract review. For each clause:\n"
        "1. Classify the clause type using 'classify'.\n"
        "2. Rate the risk level using 'rate_severity'.\n"
        "3. Flag any issues using 'flag'.\n"
        "4. Suggest an action using 'suggest' (accept_as_is / request_modification / "
        "escalate_to_senior_counsel / reject_clause / flag_for_negotiation).\n"
        "5. Provide reasoning using 'reason' explaining your assessment.\n"
        "After completing all assessments for a clause, use 'next_clause'. "
        "Pay attention to cross-clause conflicts and prioritize critical issues. "
        "When finished, use 'complete_review'."
    ),
}


def get_task_instruction(task_id: str) -> str:
    """Get instruction string for a task."""
    return TASK_INSTRUCTIONS.get(task_id, "Review the clause.")


def get_task_config(task_id: str) -> TaskConfig:
    """Get task configuration or raise for unknown task IDs."""
    config = TASK_REGISTRY.get(task_id)
    if config is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return config


def list_task_ids() -> List[str]:
    """List all available task IDs."""
    return list(TASK_REGISTRY.keys())
