"""
LexArena — Complete Pydantic Model Definitions
All typed contracts for the Environment API surface.

Changes vs. v1:
  - Reward now includes corrective_feedback (process-aware learning signal)
  - Observation includes corrective_feedback field
  - Added Tier1/Tier3 models: Tier1ExtractionTask, DependencyEdge, etc.
  - TaskConfig extended with task_tier
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLASSIFY        = "classify"
    RATE_SEVERITY   = "rate_severity"
    FLAG            = "flag"
    SUGGEST         = "suggest"
    REASON          = "reason"
    NEXT_CLAUSE     = "next_clause"
    COMPLETE_REVIEW = "complete_review"


ACTION_TYPES: List[str] = [a.value for a in ActionType]


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class SuggestedActionType(str, Enum):
    ACCEPT_AS_IS              = "accept_as_is"
    REQUEST_MODIFICATION      = "request_modification"
    ESCALATE_TO_SENIOR_COUNSEL = "escalate_to_senior_counsel"
    REJECT_CLAUSE             = "reject_clause"
    FLAG_FOR_NEGOTIATION      = "flag_for_negotiation"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Legal taxonomy constants
# ---------------------------------------------------------------------------

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

EDGE_TYPES: List[str] = [
    "cascade_trigger",
    "mutual_exclusion",
    "condition_precedent",
    "supersession",
    "temporal_gate",
]


# ---------------------------------------------------------------------------
# Core action model (Tier 2)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """The input the agent submits each step."""

    action_type:      ActionType
    clause_type:      Optional[str]               = Field(default=None)
    risk_level:       Optional[RiskLevel]         = Field(default=None)
    flags:            Optional[List[str]]         = Field(default=None)
    suggested_action: Optional[SuggestedActionType] = Field(default=None)
    reasoning:        Optional[str]               = Field(default=None)

    @field_validator("clause_type")
    @classmethod
    def validate_clause_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in CLAUSE_TAXONOMY:
            raise ValueError(
                f"clause_type '{v}' not in taxonomy. Valid: {CLAUSE_TAXONOMY}"
            )
        return v

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            invalid = [f for f in v if f not in ISSUE_FLAGS]
            if invalid:
                raise ValueError(f"Invalid flags: {invalid}. Valid: {ISSUE_FLAGS}")
        return v


# ---------------------------------------------------------------------------
# Reward — now includes corrective_feedback
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Signal returned after every step."""

    score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Step-level reward in [-1.0, 1.0].",
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown.",
    )
    message: str = Field(default="", description="Human-readable feedback.")
    corrective_feedback: str = Field(
        default="",
        description=(
            "Expert explanation of what was correct/incorrect and why — "
            "the legal principle involved. Used for process-aware learning."
        ),
    )


# ---------------------------------------------------------------------------
# Observation — includes corrective_feedback (§9 process-aware feedback)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees each step."""

    task_id:              str
    step_number:          int  = Field(ge=0)
    max_steps:            int  = Field(gt=0)
    clause_text:          str
    clause_index:         int  = Field(ge=0)
    total_clauses:        int  = Field(ge=0)
    contract_type:        str
    parties:              List[str]
    jurisdiction:         str
    instructions:         str
    available_actions:    List[str]
    last_action_feedback: Optional[str] = None
    # Process-aware learning: injected corrective feedback from last step
    corrective_feedback:  str = Field(
        default="",
        description="Expert explanation of last step's correctness and legal reasoning.",
    )
    accumulated_score:    float = 0.0
    done:                 bool  = False


# ---------------------------------------------------------------------------
# Ground truth & scenario models (Tier 2)
# ---------------------------------------------------------------------------

class ClauseGroundTruth(BaseModel):
    """Hidden ground truth for one clause."""

    text:              str
    clause_type:       str
    risk_level:        RiskLevel
    issues:            List[str]
    recommended_action: SuggestedActionType
    reasoning_keywords: List[str]
    difficulty_note:   str = ""


class ContractMeta(BaseModel):
    """Metadata about the contract being reviewed."""

    contract_type:  str
    parties:        List[str]
    jurisdiction:   str
    effective_date: str = ""


class ScenarioData(BaseModel):
    """A single scenario loaded from JSON."""

    contract_meta: ContractMeta
    clauses:       List[ClauseGroundTruth]


# ---------------------------------------------------------------------------
# Tier 1 — Clause Extraction models
# ---------------------------------------------------------------------------

class Tier1ExtractionSample(BaseModel):
    """One CUAD-style extraction sample."""

    sample_id:         str
    contract_name:     str
    question_category: str          # e.g. "Force Majeure"
    question:          str          # natural language question
    context:           str          # the clause text provided to the agent
    answer_text:       str          # verbatim correct extraction
    has_answer:        bool         # False → correct response is "No related clause."
    difficulty_note:   str = ""


class Tier1Observation(BaseModel):
    """Observation returned to the agent for Tier 1 tasks."""

    task_id:            str = "tier1_clause_reading"
    sample_id:          str
    contract_name:      str
    question_category:  str
    question:           str
    context:            str
    has_answer:         bool
    step_number:        int = 0
    max_steps:          int = 1
    instructions:       str = (
        "Extract the VERBATIM sentence from the contract that answers "
        "the question. If no clause applies, respond exactly: "
        "'No related clause.' — no other text."
    )
    last_action_feedback:  Optional[str] = None
    corrective_feedback:   str = ""
    done:                  bool = False


class Tier1Action(BaseModel):
    """Agent action for Tier 1 extraction."""

    extracted_text: str = Field(
        description="Verbatim sentence extracted from the contract, "
                    "or exactly 'No related clause.'"
    )


# ---------------------------------------------------------------------------
# Tier 3 — Dependency Graph Mapping models
# ---------------------------------------------------------------------------

class DependencyEdge(BaseModel):
    """One cross-document dependency edge."""

    source_contract: str
    source_clause:   str
    target_contract: str
    target_clause:   str
    edge_type:       str
    reasoning:       str = ""

    @field_validator("edge_type")
    @classmethod
    def validate_edge_type(cls, v: str) -> str:
        if v not in EDGE_TYPES:
            raise ValueError(f"edge_type '{v}' not in {EDGE_TYPES}")
        return v


class Tier3ContractSummary(BaseModel):
    """Summary of a single contract presented to the agent."""

    contract_id:   str
    contract_type: str
    parties:       List[str]
    jurisdiction:  str
    key_clauses:   List[Dict[str, str]]  # [{clause_id, clause_type, text_excerpt}]


class Tier3ScenarioData(BaseModel):
    """A Tier 3 dependency mapping scenario."""

    scenario_id:       str
    contracts:         List[Tier3ContractSummary]
    ground_truth_edges: List[Dict[str, Any]]   # raw dicts for flexibility
    difficulty_note:   str = ""


class Tier3Observation(BaseModel):
    """Observation returned to the agent for Tier 3 tasks."""

    task_id:         str = "tier3_dependency_mapping"
    scenario_id:     str
    contracts:       List[Dict[str, Any]]
    steps_remaining: int
    edges_found_so_far: int = 0
    instructions:    str = (
        "Identify ALL hidden cross-document dependency edges between the contracts. "
        "Output a JSON array of edge objects. "
        "Fields: source_contract, source_clause, target_contract, target_clause, "
        "edge_type (cascade_trigger|mutual_exclusion|condition_precedent|"
        "supersession|temporal_gate), reasoning."
    )
    last_feedback:       Optional[str] = None
    corrective_feedback: str = ""
    done:                bool = False


class Tier3Action(BaseModel):
    """Agent action for Tier 3 dependency mapping."""

    dependencies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of dependency edge objects.",
    )


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    """Configuration for a single task."""

    task_id:              str
    name:                 str
    difficulty:           Difficulty
    description:          str
    max_steps:            int           = Field(gt=0)
    scenario_file:        str
    required_action_types: List[ActionType]
    grader_weights:       Dict[str, float] = Field(default_factory=dict)
    task_tier:            int           = 2   # 1, 2, or 3


# ---------------------------------------------------------------------------
# Clause action record (tracks what was done per clause)
# ---------------------------------------------------------------------------

class ClauseActionRecord(BaseModel):
    """Tracks what the agent did for a specific clause."""

    clause_index:    int
    classify_action:  Optional[str]               = None
    risk_action:      Optional[RiskLevel]         = None
    flag_action:      Optional[List[str]]         = None
    suggest_action:   Optional[SuggestedActionType] = None
    reason_action:    Optional[str]               = None
    action_count:     int                         = 0


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------

class EpisodeMeta(BaseModel):
    """Episode-level metadata for grading."""

    total_steps:            int  = 0
    max_steps:              int  = 10
    total_invalid_actions:  int  = 0
    total_redundant_actions: int = 0
    clauses_reviewed:       int  = 0
    total_clauses:          int  = 0
    completed_normally:     bool = False


# ---------------------------------------------------------------------------
# Grader result
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    """Trajectory-level grading result."""

    score:            float              = Field(gt=0.0, lt=1.0)
    breakdown:        Dict[str, float]  = Field(default_factory=dict)
    per_clause_scores: List[Dict[str, Any]] = Field(default_factory=list)
    penalties:        Dict[str, float]  = Field(default_factory=dict)
    message:          str               = ""
    tier:             int               = 2


# ---------------------------------------------------------------------------
# Full environment state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state exposed via GET /state."""

    task_id:             str
    difficulty:          str                   = ""
    step_number:         int                   = 0
    max_steps:           int                   = 10
    clause_index:        int                   = 0
    total_clauses:       int                   = 0
    actions_taken:       List[Dict[str, Any]]  = Field(default_factory=list)
    rewards_given:       List[Dict[str, Any]]  = Field(default_factory=list)
    clause_records:      List[Dict[str, Any]]  = Field(default_factory=list)
    ground_truth:        List[Dict[str, Any]]  = Field(default_factory=list)
    accumulated_score:   float                 = 0.0
    done:                bool                  = False
    episode_start_time:  Optional[str]         = None
    grader_result:       Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# API request / response types
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """POST /reset body."""
    task_id:        str
    scenario_index: Optional[int] = None


class StepResponse(BaseModel):
    """POST /step response."""
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legal IQ composite score
# ---------------------------------------------------------------------------

class LegalIQScore(BaseModel):
    """Composite Legal IQ score across all tiers."""

    model_name:   str  = ""
    t1_score:     float = 0.0
    t2_score:     float = 0.0
    t3_score:     float = 0.0
    t4_score:     float = 0.0
    t5_score:     float = 0.0
    t6_score:     float = 0.0
    crisis_score: float = 0.0   # weighted avg of T4/T5/T6
    legal_iq:     float = 0.0   # composite
    label:        str   = ""
    tier_breakdown: Dict[str, float] = Field(default_factory=dict)
