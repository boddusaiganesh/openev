"""
Phase 3 — Complete Pydantic Model Definitions
All typed contracts for the API surface.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    CLASSIFY = "classify"
    RATE_SEVERITY = "rate_severity"
    FLAG = "flag"
    SUGGEST = "suggest"
    REASON = "reason"
    NEXT_CLAUSE = "next_clause"
    COMPLETE_REVIEW = "complete_review"


ACTION_TYPES: List[str] = [a.value for a in ActionType]


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SuggestedActionType(str, Enum):
    ACCEPT_AS_IS = "accept_as_is"
    REQUEST_MODIFICATION = "request_modification"
    ESCALATE_TO_SENIOR_COUNSEL = "escalate_to_senior_counsel"
    REJECT_CLAUSE = "reject_clause"
    FLAG_FOR_NEGOTIATION = "flag_for_negotiation"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


CLAUSE_TAXONOMY: List[str] = [
    "indemnification",
    "limitation_of_liability",
    "termination",
    "confidentiality",
    "non_compete",
    "force_majeure",
    "assignment",
    "governing_law",
    "warranty",
    "intellectual_property",
    "payment_terms",
    "representations",
    "dispute_resolution",
    "data_protection",
    "insurance",
]

ISSUE_FLAGS: List[str] = [
    "vague_language",
    "missing_liability_cap",
    "one_sided_obligation",
    "unusual_term",
    "market_standard",
    "overly_broad_scope",
    "missing_time_limit",
    "ambiguous_definition",
    "conflicting_with_other_clause",
    "missing_carve_out",
    "automatic_renewal",
    "unreasonable_penalty",
    "silent_on_key_issue",
]


class Action(BaseModel):
    """The input the agent submits each step."""

    action_type: ActionType
    clause_type: Optional[str] = Field(
        default=None,
        description="Required when action_type is classify. Must be from CLAUSE_TAXONOMY.",
    )
    risk_level: Optional[RiskLevel] = Field(
        default=None,
        description="Required when action_type is rate_severity.",
    )
    flags: Optional[List[str]] = Field(
        default=None,
        description="Required when action_type is flag. Must be from ISSUE_FLAGS.",
    )
    suggested_action: Optional[SuggestedActionType] = Field(
        default=None,
        description="Required when action_type is suggest.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Required when action_type is reason. Free text.",
    )

    @field_validator("clause_type")
    @classmethod
    def validate_clause_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in CLAUSE_TAXONOMY:
            raise ValueError(
                f"clause_type '{v}' not in taxonomy. "
                f"Valid: {CLAUSE_TAXONOMY}"
            )
        return v

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            invalid = [f for f in v if f not in ISSUE_FLAGS]
            if invalid:
                raise ValueError(
                    f"Invalid flags: {invalid}. Valid: {ISSUE_FLAGS}"
                )
        return v


class Reward(BaseModel):
    """Signal returned after every step.

    score is strictly between 0.001 and 0.999 (exclusive of both bounds)
    per the OpenEnv specification. Raw scores are clamped automatically.
    """

    score: float = Field(
        default=0.1,
        description="Step-level reward in (0.001, 0.999) exclusive.",
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown.",
    )
    message: str = Field(
        default="",
        description="Human-readable feedback.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to strictly (0.001, 0.999).

        Negative rewards and zero are raised to 0.001 (still a signal of failure).
        Perfect scores are lowered to 0.999 (a score of 1.0 implies mathematical
        certainty that is unachievable in real-world contract review).
        """
        return max(0.001, min(0.999, float(v)))


class Observation(BaseModel):
    """What the agent sees each step."""

    task_id: str
    step_number: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    clause_text: str
    clause_index: int = Field(ge=0)
    total_clauses: int = Field(ge=0)
    contract_type: str
    parties: List[str]
    jurisdiction: str
    instructions: str
    available_actions: List[str]
    last_action_feedback: Optional[str] = None
    accumulated_score: float = 0.0
    done: bool = False


class ClauseGroundTruth(BaseModel):
    """Hidden ground truth for one clause."""

    text: str
    clause_type: str
    risk_level: RiskLevel
    issues: List[str]
    recommended_action: SuggestedActionType
    reasoning_keywords: List[str]
    difficulty_note: str = ""


class ContractMeta(BaseModel):
    """Metadata about the contract being reviewed."""

    contract_type: str
    parties: List[str]
    jurisdiction: str
    effective_date: str = ""


class ScenarioData(BaseModel):
    """A single scenario loaded from JSON."""

    contract_meta: ContractMeta
    clauses: List[ClauseGroundTruth]


class TaskConfig(BaseModel):
    """Configuration for a single task."""

    task_id: str
    name: str
    difficulty: Difficulty
    description: str
    max_steps: int = Field(gt=0)
    scenario_file: str
    required_action_types: List[ActionType]
    grader_weights: Dict[str, float] = Field(default_factory=dict)


class ClauseActionRecord(BaseModel):
    """Tracks what the agent did for a specific clause."""

    clause_index: int
    classify_action: Optional[str] = None
    risk_action: Optional[RiskLevel] = None
    flag_action: Optional[List[str]] = None
    suggest_action: Optional[SuggestedActionType] = None
    reason_action: Optional[str] = None
    action_count: int = 0


class EpisodeMeta(BaseModel):
    """Episode-level metadata for grading."""
    total_steps: int = 0
    max_steps: int = 10
    total_invalid_actions: int = 0
    total_redundant_actions: int = 0
    clauses_reviewed: int = 0
    total_clauses: int = 0
    completed_normally: bool = False


class GraderResult(BaseModel):
    """Trajectory-level grading result. Score is strictly (0.001, 0.999)."""
    score: float = Field(default=0.001, description="Final episode score in (0.001, 0.999) exclusive.")
    breakdown: Dict[str, float] = Field(default_factory=dict)
    per_clause_scores: List[Dict[str, Any]] = Field(default_factory=list)
    penalties: Dict[str, float] = Field(default_factory=dict)
    message: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def clamp_grader_score(cls, v: float) -> float:
        """Clamp to (0.001, 0.999): no perfect scores, no zero scores."""
        return max(0.001, min(0.999, float(v)))


class EnvironmentState(BaseModel):
    """Full internal state exposed via GET /state."""

    task_id: str
    difficulty: str = ""
    step_number: int = 0
    max_steps: int = 10
    clause_index: int = 0
    total_clauses: int = 0
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    rewards_given: List[Dict[str, Any]] = Field(default_factory=list)
    clause_records: List[Dict[str, Any]] = Field(default_factory=list)
    ground_truth: List[Dict[str, Any]] = Field(default_factory=list)
    accumulated_score: float = 0.0
    done: bool = False
    episode_start_time: Optional[str] = None
    grader_result: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    """POST /reset body."""
    task_id: str


class StepResponse(BaseModel):
    """POST /step response."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
