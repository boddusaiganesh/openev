"""
LexArena — Shared Pydantic Models
Data models shared across all 6 tiers of the LexArena benchmark.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TierName(str, Enum):
    TIER1_READING = "tier1_reading"
    TIER2_CLASSIFICATION = "tier2_classification"
    TIER3_DEPENDENCY = "tier3_dependency"
    TIER4_CASCADE_EASY = "tier4_cascade_easy"
    TIER5_CASCADE_MEDIUM = "tier5_cascade_medium"
    TIER6_CASCADE_HARD = "tier6_cascade_hard"


class FailureMode(str, Enum):
    FM_VOID = "fm_void"                         # Invoked FM while it voids insurance
    SYCOPHANCY = "sycophancy"                   # Paid invalid aggressive demand
    COVENANT_BLINDNESS = "covenant_blindness"   # Missed debt covenant cascade
    DEADLINE_STACK = "deadline_stack"           # Failed to prioritise stacked deadlines
    KEY_PERSON_CHAIN = "key_person_chain"       # Missed 3-hop dependency chain
    LAZY_READER = "lazy_reader"                 # Said "No related clause" when one exists
    FALSE_URGENCY = "false_urgency"             # Acted on non-binding demand
    SUPERSESSION = "supersession"               # Missed later clause overriding earlier
    COMPOUND_SHOCK = "compound_shock"           # Failed to handle two simultaneous shocks
    CROSS_DEFAULT = "cross_default"             # Missed cross-default cascade


class ProbeOutcome(str, Enum):
    PASSED = "passed"       # Agent avoided the failure mode
    FAILED = "failed"       # Agent triggered the failure mode
    PARTIAL = "partial"     # Agent partially handled it


# ---------------------------------------------------------------------------
# Tier 1 — Clause Reading Models
# ---------------------------------------------------------------------------

class Tier1Sample(BaseModel):
    """One CUAD-derived clause extraction sample."""
    sample_id: str
    contract_name: str
    question_category: str      # CUAD question type e.g. "Exclusivity"
    context: str                # The clause text
    question: str               # The full CUAD question
    ground_truth: List[str]     # List of correct answer strings (may be empty)
    has_answer: bool            # True if a relevant clause exists


class Tier1Output(BaseModel):
    """Agent's response to a Tier 1 sample."""
    sample_id: str
    extracted_text: str         # What the agent extracted (or "No related clause.")
    is_no_clause: bool          # True if agent said "No related clause"


class Tier1SampleResult(BaseModel):
    """Graded result for one Tier 1 sample."""
    sample_id: str
    question_category: str
    is_true_positive: bool
    is_true_negative: bool
    is_false_positive: bool
    is_false_negative: bool
    jaccard_score: float        # 0.0 if no ground truth
    is_lazy: bool               # True if "No related clause" when answer exists


class Tier1Score(BaseModel):
    """Aggregated Tier 1 score across all samples."""
    total_samples: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float                   # Primary metric — recall-weighted
    jaccard_mean: float
    laziness_rate: float        # false "No related clause" rate
    false_no_clause_rate: float # FN / total positive samples
    tier_score: float           # Composite T1 score [0,1]


# ---------------------------------------------------------------------------
# Tier 3 — Dependency Mapping Models
# ---------------------------------------------------------------------------

class PredictedEdge(BaseModel):
    """One dependency edge predicted by the agent."""
    source_contract: str
    source_clause_id: str
    target_contract: str
    target_clause_id: str
    edge_type: str              # One of the 5 taxonomy types
    description: str = ""


class Tier3Action(BaseModel):
    """Agent action during Tier 3: submit the dependency map."""
    action_type: str = "submit_dependency_map"  # or "request_more_time"
    predicted_edges: List[PredictedEdge] = Field(default_factory=list)
    reasoning: Optional[str] = None


class Tier3Observation(BaseModel):
    """What the agent sees during Tier 3."""
    scenario_id: str
    contracts_summary: List[Dict[str, Any]]     # Contract IDs + clause summaries
    time_budget_remaining: int                   # Steps left to submit
    step_number: int
    instructions: str
    done: bool = False


class Tier3SampleResult(BaseModel):
    """Graded result for one Tier 3 scenario."""
    scenario_id: str
    ground_truth_edges: int
    predicted_edges: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    edge_type_accuracy: float   # Bonus: correct type for TP edges
    severity_order_score: float # Bonus: edges ordered by severity


class Tier3Score(BaseModel):
    """Aggregated Tier 3 score."""
    total_scenarios: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_edge_type_accuracy: float
    tier_score: float           # Composite T3 score [0,1]


# ---------------------------------------------------------------------------
# Legal IQ — Composite Score
# ---------------------------------------------------------------------------

class TierScore(BaseModel):
    """Score for a single tier."""
    tier: TierName
    raw_score: float            # [0,1]
    weighted_contribution: float
    weight: float
    details: Dict[str, Any] = Field(default_factory=dict)


class ProbeResult(BaseModel):
    """Result of one adversarial probe."""
    probe_id: str
    failure_mode: FailureMode
    outcome: ProbeOutcome
    final_cash: float
    grader_score: float
    notes: str = ""


class LegalIQScore(BaseModel):
    """The composite Legal IQ score across all tiers."""
    tier_scores: List[TierScore]
    probe_results: List[ProbeResult] = Field(default_factory=list)

    # Weighted composite
    legal_iq: float             # Primary score [0,1]

    # Tier breakdown
    t1_reading: float
    t2_classification: float
    t3_dependency: float
    t4_crisis_easy: float
    t5_crisis_medium: float
    t6_crisis_hard: float
    t4_t5_t6_combined: float

    # Probe summary
    probes_passed: int = 0
    probes_failed: int = 0
    probes_total: int = 0
    failure_modes_triggered: List[str] = Field(default_factory=list)

    # Label
    label: str                  # "Expert CRO", "Senior Lawyer", etc.

    @classmethod
    def compute_label(cls, score: float) -> str:
        if score >= 0.85:
            return "Expert CRO Level"
        elif score >= 0.70:
            return "Senior Lawyer Level"
        elif score >= 0.50:
            return "Junior Associate Level"
        elif score >= 0.30:
            return "Paralegal Level"
        else:
            return "Fails Legal Practice Bar"


# ---------------------------------------------------------------------------
# LexArena Run Config
# ---------------------------------------------------------------------------

class LexArenaConfig(BaseModel):
    """Configuration for a full LexArena run."""
    model_name: str
    run_tier1: bool = True
    run_tier2: bool = True
    run_tier3: bool = True
    run_tier4: bool = True
    run_tier5: bool = True
    run_tier6: bool = True
    run_probes: bool = True

    # Tier 1 config
    tier1_max_samples: int = 200    # Limit CUAD samples for speed
    tier1_categories: Optional[List[str]] = None  # None = all 41 categories

    # Tier 3 config
    tier3_time_budget: int = 10     # Steps to submit dependency map

    # Tier 2 / 4-6 config
    tier2_task_ids: List[str] = Field(default_factory=lambda: [
        "task_1_easy", "task_2_medium", "task_3_hard"
    ])
    crisis_task_ids: List[str] = Field(default_factory=lambda: [
        "task_4_cascade_easy", "task_5_cascade_medium", "task_6_cascade_hard"
    ])

    # Weights (must sum to 1.0)
    weight_t1: float = 0.15
    weight_t2: float = 0.15
    weight_t3: float = 0.20
    weight_t4_t5_t6: float = 0.50

    # Crisis sub-weights (must sum to 1.0)
    weight_t4: float = 0.25
    weight_t5: float = 0.35
    weight_t6: float = 0.40
