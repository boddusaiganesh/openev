"""
LexDomino — Cascade Models
Pydantic models for the Systemic Legal Cascade Simulator.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CascadeActionType(str, Enum):
    # Legal actions
    INVOKE_FORCE_MAJEURE = "invoke_force_majeure"
    SEND_BREACH_NOTICE = "send_breach_notice"
    FILE_INSURANCE_CLAIM = "file_insurance_claim"
    TERMINATE_CONTRACT = "terminate_contract"
    INVOKE_INDEMNIFICATION = "invoke_indemnification"
    REQUEST_WAIVER = "request_waiver"
    # Financial triage
    PAY_PENALTY = "pay_penalty"
    NEGOTIATE_PAYMENT_PLAN = "negotiate_payment_plan"
    DRAW_CREDIT_FACILITY = "draw_credit_facility"
    ACCELERATE_RECEIVABLE = "accelerate_receivable"
    # Communication
    SEND_FORMAL_NOTICE = "send_formal_notice"
    REQUEST_INFORMATION = "request_information"
    PROPOSE_AMENDMENT = "propose_amendment"
    # Investigation
    CROSS_REFERENCE_CONTRACTS = "cross_reference_contracts"
    ANALYZE_FINANCIAL_IMPACT = "analyze_financial_impact"
    REVIEW_DEADLINE_STATUS = "review_deadline_status"
    ASSESS_COUNTERPARTY_RISK = "assess_counterparty_risk"
    # Control
    ADVANCE_DAY = "advance_day"
    COMPLETE_CRISIS = "complete_crisis"


CASCADE_ACTION_TYPES: List[str] = [a.value for a in CascadeActionType]


class EdgeType(str, Enum):
    SUPERSESSION = "supersession"
    CONDITION_PRECEDENT = "condition_precedent"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    CASCADE_TRIGGER = "cascade_trigger"
    TEMPORAL_GATE = "temporal_gate"


class CounterpartyProfile(str, Enum):
    COOPERATIVE = "cooperative"
    AGGRESSIVE = "aggressive"
    LITIGIOUS = "litigious"
    BUREAUCRATIC = "bureaucratic"
    OPPORTUNISTIC = "opportunistic"


class ShockCategory(str, Enum):
    SUPPLY_CHAIN = "supply_chain"
    CYBERSECURITY = "cybersecurity"
    LABOR = "labor"
    REGULATORY = "regulatory"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"


class DeadlineStatus(str, Enum):
    ACTIVE = "active"
    MET = "met"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Contract graph primitives
# ---------------------------------------------------------------------------

class DependencyEdge(BaseModel):
    """Directed edge between two clauses across (possibly different) contracts."""
    source_contract: str
    source_clause_id: str
    target_contract: str
    target_clause_id: str
    edge_type: EdgeType
    description: str
    discovered: bool = False          # Becomes True once agent cross-references


class ClauseNode(BaseModel):
    """A single clause inside a contract document."""
    clause_id: str
    clause_type: str
    text: str
    # Financial consequence if obligation is breached
    penalty_amount: float = 0.0
    # Hard deadline (in simulation days) by which notice/action must occur
    deadline_days: Optional[int] = None
    # Whether invoking this clause voids another (populated by edges)
    is_invokable: bool = True
    invoked: bool = False


class ContractDocument(BaseModel):
    """A full contract with embedded clauses and metadata."""
    contract_id: str
    contract_type: str                 # e.g. "Master Service Agreement"
    parties: List[str]
    jurisdiction: str
    effective_date: str = ""
    clauses: List[ClauseNode]


# ---------------------------------------------------------------------------
# Financial state machine
# ---------------------------------------------------------------------------

class FinancialState(BaseModel):
    cash_balance: float
    accounts_receivable: float = 0.0
    accounts_payable: float = 0.0
    credit_facility_remaining: float = 0.0
    debt_covenant_min_cash: float = 0.0   # cash must stay above this
    insurance_coverage_active: bool = True
    reputation_score: float = 1.0         # 0.0 – 1.0

    @property
    def covenant_violated(self) -> bool:
        return self.cash_balance < self.debt_covenant_min_cash

    @property
    def solvent(self) -> bool:
        return self.cash_balance > 0.0


# ---------------------------------------------------------------------------
# Counterparty
# ---------------------------------------------------------------------------

class Counterparty(BaseModel):
    counterparty_id: str
    name: str
    role: str                    # e.g. "primary_lender", "key_client"
    profile: CounterpartyProfile
    relationship_health: float = 1.0   # 0.0 – 1.0; decays with bad interactions
    # Risk revealed once agent runs assess_counterparty_risk
    risk_profile_revealed: bool = False


# ---------------------------------------------------------------------------
# Deadline tracker
# ---------------------------------------------------------------------------

class Deadline(BaseModel):
    deadline_id: str
    description: str
    contract_id: str
    clause_id: str
    due_day: int
    penalty_if_missed: float = 0.0
    status: DeadlineStatus = DeadlineStatus.ACTIVE
    consequence_description: str = ""


# ---------------------------------------------------------------------------
# Inbox / event stream
# ---------------------------------------------------------------------------

class InboxMessage(BaseModel):
    message_id: str
    day_received: int
    sender: str
    subject: str
    body: str
    is_legal_notice: bool = False
    requires_response_by_day: Optional[int] = None
    read: bool = False


# ---------------------------------------------------------------------------
# Shock event
# ---------------------------------------------------------------------------

class ShockEvent(BaseModel):
    shock_id: str
    category: ShockCategory
    title: str
    description: str
    occurs_on_day: int = 1
    # Direct financial hit on occurrence
    immediate_cash_impact: float = 0.0
    # Contract obligations triggered by this shock
    triggered_obligations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full crisis scenario
# ---------------------------------------------------------------------------

class CrisisScenario(BaseModel):
    scenario_id: str
    company_name: str
    industry: str
    initial_cash: float
    survival_target_days: int          # agent must survive this many days
    contracts: List[ContractDocument]
    dependency_edges: List[DependencyEdge]
    counterparties: List[Counterparty]
    deadlines: List[Deadline]
    shock_events: List[ShockEvent]
    initial_inbox: List[InboxMessage] = Field(default_factory=list)
    # Optional financial state fields (present in probe JSONs and bridge scenarios)
    initial_ar: float = 0.0            # accounts receivable
    initial_ap: float = 0.0            # accounts payable
    credit_facility: float = 0.0       # revolving credit line
    debt_covenant_min_cash: float = 0.0  # minimum cash covenant


# ---------------------------------------------------------------------------
# Agent action
# ---------------------------------------------------------------------------

class CascadeAction(BaseModel):
    action_type: CascadeActionType
    # Contextual parameters (all optional; validated at env level)
    contract_id: Optional[str] = None
    clause_id: Optional[str] = None
    counterparty_id: Optional[str] = None
    amount: Optional[float] = None
    justification: Optional[str] = None
    proposed_terms: Optional[str] = None
    query: Optional[str] = None
    contract_ids: Optional[List[str]] = None    # for cross_reference


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class CascadeObservation(BaseModel):
    task_id: str
    current_day: int
    max_days: int
    step_number: int
    max_steps: int

    # Financial summary
    cash_balance: float
    covenant_min_cash: float
    covenant_violated: bool
    insurance_active: bool
    reputation_score: float

    # Active deadlines (sorted by urgency)
    active_deadlines: List[Dict[str, Any]] = Field(default_factory=list)

    # Inbox (unread messages)
    inbox_messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Discovered dependency edges
    discovered_edges: List[Dict[str, Any]] = Field(default_factory=list)

    # Available contracts summary
    contracts_summary: List[Dict[str, Any]] = Field(default_factory=list)

    # Last action feedback
    last_action_feedback: Optional[str] = None
    last_action_financial_impact: float = 0.0

    # Counterparty statuses
    counterparty_statuses: List[Dict[str, Any]] = Field(default_factory=list)

    available_actions: List[str] = Field(default_factory=list)
    done: bool = False
    bankruptcy: bool = False


# ---------------------------------------------------------------------------
# Reward & grader
# ---------------------------------------------------------------------------

class CascadeReward(BaseModel):
    """Step reward for crisis management actions.

    score is strictly between 0.001 and 0.999 per the OpenEnv spec.
    """
    score: float = Field(default=0.1, description="Step reward in (0.001, 0.999) exclusive.")
    cash_delta: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to strictly (0.001, 0.999). Bankruptcy → 0.001, perfect survival → 0.999."""
        return max(0.001, min(0.999, float(v)))


class CascadeGraderResult(BaseModel):
    """Final trajectory grader result. Score is strictly (0.001, 0.999)."""
    score: float = Field(default=0.001, description="Final score in (0.001, 0.999) exclusive.")
    normalized_cash_ratio: float = 0.0
    deadlines_met_ratio: float = 0.0
    rights_preserved_ratio: float = 0.0
    cascade_depth_max: int = 0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp to (0.001, 0.999)."""
        return max(0.001, min(0.999, float(v)))


# ---------------------------------------------------------------------------
# Environment state (for /state endpoint)
# ---------------------------------------------------------------------------

class CascadeEnvironmentState(BaseModel):
    task_id: str
    current_day: int
    max_days: int
    step_number: int
    cash_balance: float
    initial_cash: float
    bankruptcy: bool
    done: bool
    deadlines_met: int
    deadlines_expired: int
    deadlines_total: int
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    financial_history: List[Dict[str, Any]] = Field(default_factory=list)
    grader_result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Request / response wrappers
# ---------------------------------------------------------------------------

class CascadeStepResponse(BaseModel):
    observation: CascadeObservation
    reward: CascadeReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
