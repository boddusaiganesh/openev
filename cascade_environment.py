"""
LexDomino — Cascade Environment (Part 1: imports, class definition, init)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from cascade_models import (
    CascadeAction, CascadeActionType, CascadeEnvironmentState,
    CascadeGraderResult, CascadeObservation, CascadeReward,
    ClauseNode, ContractDocument, Counterparty, CounterpartyProfile,
    CrisisScenario, Deadline, DeadlineStatus, DependencyEdge,
    FinancialState, InboxMessage, ShockEvent,
)
from cascade_graders import grade_cascade_episode
from cascade_rewards import (
    reward_for_bankruptcy, reward_for_cash_change,
    reward_for_covenant_violation, reward_for_deadline_missed,
    reward_for_deadline_met, reward_for_discovery,
    reward_for_invalid_action, reward_neutral,
)

logger = logging.getLogger("lexdomino.cascade_environment")


class LexDominoCrisisEnv:
    """
    LexDomino — Systemic Legal Cascade Simulator.
    OpenEnv-compliant environment with temporal crisis management.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.manifest: Dict = self._load_manifest()

        # Episode state (reset each episode)
        self.scenario: Optional[CrisisScenario] = None
        self.task_id: Optional[str] = None
        self.financial_state: Optional[FinancialState] = None
        self.current_day: int = 0
        self.step_number: int = 0
        self.max_days: int = 0
        self.max_steps: int = 0
        self.done: bool = False
        self.bankruptcy: bool = False
        self.episode_start_time: Optional[str] = None
        self.grader_result: Optional[CascadeGraderResult] = None

        # Mutable copies of scenario objects
        self.deadlines: List[Deadline] = []
        self.dependency_edges: List[DependencyEdge] = []
        self.counterparties: List[Counterparty] = []
        self.inbox: List[InboxMessage] = []
        self.contracts: List[ContractDocument] = []

        # Tracking
        self.actions_taken: List[Dict] = []
        self.financial_history: List[Dict] = []
        self.cascade_depth_current: int = 0
        self.cascade_depth_max: int = 0
        self._invalid_action_count: int = 0
        self._active_scenario_file: str = ""

    def _load_manifest(self) -> Dict:
        path = os.path.join(self.data_dir, "manifest.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(path) as f:
            return json.load(f)

    def _load_scenario(self, task_id: str, scenario_rel_path: str) -> CrisisScenario:
        full_path = os.path.join(self.data_dir, scenario_rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Scenario not found: {full_path}")
        with open(full_path) as f:
            raw = json.load(f)
        return CrisisScenario(**raw)

    def reset(self, task_id: str, scenario_index: int = 0) -> CascadeObservation:
        """Load scenario, reset all state, apply Day-0 shock, return first obs."""
        task_entry = self.manifest.get(task_id)
        if not task_entry:
            raise ValueError(f"Unknown task_id: {task_id}")
        files = task_entry.get("scenario_files", [])
        if not files:
            raise ValueError(f"No scenario files for {task_id}")
        if scenario_index >= len(files):
            scenario_index = 0
        rel_path = files[scenario_index]
        self._active_scenario_file = rel_path

        self.scenario = self._load_scenario(task_id, rel_path)
        self.task_id = task_id
        self.max_days = self.scenario.survival_target_days
        # max_steps derived from task config stored in manifest
        self.max_steps = task_entry.get("max_steps", self.max_days * 3)

        # Deep-copy mutable objects
        self.contracts = [c.model_copy(deep=True) for c in self.scenario.contracts]
        self.deadlines = [d.model_copy(deep=True) for d in self.scenario.deadlines]
        self.dependency_edges = [e.model_copy(deep=True) for e in self.scenario.dependency_edges]
        self.counterparties = [cp.model_copy(deep=True) for cp in self.scenario.counterparties]
        self.inbox = [m.model_copy(deep=True) for m in self.scenario.initial_inbox]

        # Financial state
        s = self.scenario
        self.financial_state = FinancialState(
            cash_balance=s.initial_cash,
            accounts_receivable=getattr(s, "initial_ar", 0.0),
            accounts_payable=getattr(s, "initial_ap", 0.0),
            credit_facility_remaining=getattr(s, "credit_facility", 0.0),
            debt_covenant_min_cash=getattr(s, "debt_covenant_min_cash", 0.0),
            insurance_coverage_active=True,
            reputation_score=1.0,
        )

        # Episode tracking
        self.current_day = 1
        self.step_number = 0
        self.done = False
        self.bankruptcy = False
        self.grader_result = None
        self.actions_taken = []
        self.financial_history = []
        self.cascade_depth_current = 0
        self.cascade_depth_max = 0
        self._invalid_action_count = 0
        self.episode_start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Apply Day-1 shocks
        day1_shocks = [sh for sh in self.scenario.shock_events if sh.occurs_on_day <= 1]
        for shock in day1_shocks:
            self._apply_shock(shock)

        self._record_financial_snapshot("reset")
        return self._build_observation("Episode started. A crisis is unfolding. Review your inbox and act.")

    def reset_from_scenario(self, scenario: "CrisisScenario", max_steps: int = 0) -> "CascadeObservation":
        """Reset the environment from an in-memory CrisisScenario object.

        Used by the probe runner to inject probe scenarios without needing
        a manifest entry or a file on disk.
        """
        self.scenario = scenario
        self.task_id = scenario.scenario_id
        self.max_days = scenario.survival_target_days
        self.max_steps = max_steps if max_steps > 0 else self.max_days * 3
        self._active_scenario_file = f"<in-memory:{scenario.scenario_id}>"

        # Deep-copy mutable objects
        self.contracts = [c.model_copy(deep=True) for c in scenario.contracts]
        self.deadlines = [d.model_copy(deep=True) for d in scenario.deadlines]
        self.dependency_edges = [e.model_copy(deep=True) for e in scenario.dependency_edges]
        self.counterparties = [cp.model_copy(deep=True) for cp in scenario.counterparties]
        self.inbox = [m.model_copy(deep=True) for m in scenario.initial_inbox]

        # Financial state
        self.financial_state = FinancialState(
            cash_balance=scenario.initial_cash,
            accounts_receivable=scenario.initial_ar,
            accounts_payable=scenario.initial_ap,
            credit_facility_remaining=scenario.credit_facility,
            debt_covenant_min_cash=scenario.debt_covenant_min_cash,
            insurance_coverage_active=True,
            reputation_score=1.0,
        )

        # Episode tracking
        self.current_day = 1
        self.step_number = 0
        self.done = False
        self.bankruptcy = False
        self.grader_result = None
        self.actions_taken = []
        self.financial_history = []
        self.cascade_depth_current = 0
        self.cascade_depth_max = 0
        self._invalid_action_count = 0
        self.episode_start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Apply Day-1 shocks
        day1_shocks = [sh for sh in scenario.shock_events if sh.occurs_on_day <= 1]
        for shock in day1_shocks:
            self._apply_shock(shock)

        self._record_financial_snapshot("reset")
        return self._build_observation("Probe episode started. A crisis is unfolding.")

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action: CascadeAction) -> Tuple[CascadeObservation, CascadeReward, bool, Dict]:
        if self.scenario is None or self.financial_state is None:
            obs = self._empty_observation()
            return obs, reward_for_invalid_action("No active episode."), True, {}

        if self.done:
            obs = self._build_observation("Episode already finished.")
            return obs, reward_neutral("Episode done."), True, self._build_info()

        self.step_number += 1
        cash_before = self.financial_state.cash_balance

        # Validate
        err = self._validate_action(action)
        if err:
            self._invalid_action_count += 1
            rwd = reward_for_invalid_action(err)
            self._check_termination()
            return self._build_observation(err), rwd, self.done, self._build_info()

        # Dispatch
        rwd, feedback = self._dispatch_action(action)

        # Record
        self.actions_taken.append({
            "day": self.current_day,
            "step": self.step_number,
            "action": action.action_type.value,
            "contract_id": action.contract_id,
            "amount": action.amount,
            "reward": rwd.score,
        })

        # Check financial consequences
        cash_after = self.financial_state.cash_balance
        if not self.done and self.financial_state.covenant_violated:
            cov_rwd = reward_for_covenant_violation(
                self.scenario.initial_cash,
                self.financial_state.debt_covenant_min_cash,
            )
            rwd.score = max(-1.0, rwd.score + cov_rwd.score)
            feedback += " " + cov_rwd.message

        if not self.done and not self.financial_state.solvent:
            self.bankruptcy = True
            self.done = True
            rwd = reward_for_bankruptcy()
            feedback = rwd.message

        self._check_termination()
        if self.done and self.grader_result is None:
            self.grader_result = self._run_grader()

        self._record_financial_snapshot(action.action_type.value)
        obs = self._build_observation(feedback)
        return obs, rwd, self.done, self._build_info()

    # ------------------------------------------------------------------
    # VALIDATION & DISPATCH
    # ------------------------------------------------------------------

    def _validate_action(self, action: CascadeAction) -> Optional[str]:
        at = action.action_type
        needs_contract = {
            CascadeActionType.INVOKE_FORCE_MAJEURE,
            CascadeActionType.SEND_BREACH_NOTICE,
            CascadeActionType.FILE_INSURANCE_CLAIM,
            CascadeActionType.TERMINATE_CONTRACT,
            CascadeActionType.INVOKE_INDEMNIFICATION,
            CascadeActionType.REQUEST_WAIVER,
            CascadeActionType.PAY_PENALTY,
            CascadeActionType.NEGOTIATE_PAYMENT_PLAN,
            CascadeActionType.PROPOSE_AMENDMENT,
        }
        if at in needs_contract and not action.contract_id:
            return f"{at.value} requires contract_id."
        if at == CascadeActionType.PAY_PENALTY:
            if not action.amount or action.amount <= 0:
                return "pay_penalty requires a positive amount."
            if self.financial_state and action.amount > self.financial_state.cash_balance:
                return "Insufficient cash to pay that penalty."
        if at == CascadeActionType.DRAW_CREDIT_FACILITY:
            if not action.amount or action.amount <= 0:
                return "draw_credit_facility requires a positive amount."
            if self.financial_state and action.amount > self.financial_state.credit_facility_remaining:
                return "Amount exceeds remaining credit facility."
        if at == CascadeActionType.CROSS_REFERENCE_CONTRACTS:
            if not action.contract_ids or len(action.contract_ids) < 2:
                return "cross_reference_contracts requires at least 2 contract_ids."
        return None

    def _dispatch_action(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        at = action.action_type
        fs = self.financial_state

        if at == CascadeActionType.INVOKE_FORCE_MAJEURE:
            return self._handle_force_majeure(action)
        if at == CascadeActionType.SEND_BREACH_NOTICE:
            return self._handle_breach_notice(action)
        if at == CascadeActionType.FILE_INSURANCE_CLAIM:
            return self._handle_insurance_claim(action)
        if at == CascadeActionType.TERMINATE_CONTRACT:
            return self._handle_terminate(action)
        if at == CascadeActionType.INVOKE_INDEMNIFICATION:
            rwd = reward_neutral("Indemnification invoked. Outcome pending.")
            return rwd, rwd.message
        if at == CascadeActionType.REQUEST_WAIVER:
            return self._handle_waiver_request(action)
        if at == CascadeActionType.PAY_PENALTY:
            fs.cash_balance -= action.amount
            rwd = reward_for_cash_change(-action.amount, self.scenario.initial_cash)
            msg = f"Paid penalty ${action.amount:,.0f}. Cash now ${fs.cash_balance:,.0f}."
            rwd.message = msg
            return rwd, msg
        if at == CascadeActionType.DRAW_CREDIT_FACILITY:
            fs.cash_balance += action.amount
            fs.credit_facility_remaining -= action.amount
            rwd = reward_for_cash_change(action.amount * 0.5, self.scenario.initial_cash)
            msg = f"Drew ${action.amount:,.0f} from credit facility. Cash now ${fs.cash_balance:,.0f}."
            rwd.message = msg
            return rwd, msg
        if at == CascadeActionType.NEGOTIATE_PAYMENT_PLAN:
            rwd = reward_neutral("Payment plan proposal sent.")
            return rwd, rwd.message
        if at == CascadeActionType.ACCELERATE_RECEIVABLE:
            collected = (action.amount or 0) * 0.95
            fs.cash_balance += collected
            rwd = reward_for_cash_change(collected, self.scenario.initial_cash)
            msg = f"Collected ${collected:,.0f} from accelerated receivable."
            rwd.message = msg
            return rwd, msg
        if at == CascadeActionType.SEND_FORMAL_NOTICE:
            return self._handle_formal_notice(action)
        if at == CascadeActionType.REQUEST_INFORMATION:
            rwd = reward_neutral("Information request sent.")
            return rwd, rwd.message
        if at == CascadeActionType.PROPOSE_AMENDMENT:
            rwd = reward_neutral("Amendment proposed to counterparty.")
            return rwd, rwd.message
        if at == CascadeActionType.CROSS_REFERENCE_CONTRACTS:
            return self._handle_cross_reference(action)
        if at == CascadeActionType.ANALYZE_FINANCIAL_IMPACT:
            return self._handle_financial_analysis(action)
        if at == CascadeActionType.REVIEW_DEADLINE_STATUS:
            return self._handle_review_deadlines()
        if at == CascadeActionType.ASSESS_COUNTERPARTY_RISK:
            return self._handle_assess_counterparty(action)
        if at == CascadeActionType.ADVANCE_DAY:
            return self._handle_advance_day()
        if at == CascadeActionType.COMPLETE_CRISIS:
            self.done = True
            msg = f"Crisis management concluded on Day {self.current_day}."
            return reward_neutral(msg), msg
        return reward_neutral("Unknown action."), "Unknown action."

    # ------------------------------------------------------------------
    # SPECIFIC ACTION HANDLERS
    # ------------------------------------------------------------------

    def _handle_force_majeure(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        contract = self._find_contract(action.contract_id)
        if not contract:
            return reward_for_invalid_action(f"Contract {action.contract_id} not found."), ""
        # Check if any deadline on this contract is protected by FM
        protected = [d for d in self.deadlines
                     if d.contract_id == action.contract_id and d.status == DeadlineStatus.ACTIVE]
        for d in protected:
            d.status = DeadlineStatus.MET  # FM suspends obligations
        # FM may void insurance — check edges
        void_insurance = any(
            e.edge_type.value == "mutual_exclusion"
            and e.source_contract == action.contract_id
            and "insurance" in e.target_contract.lower()
            for e in self.dependency_edges
        )
        if void_insurance:
            self.financial_state.insurance_coverage_active = False
            msg = (f"Force Majeure invoked on {action.contract_id}. "
                   f"{len(protected)} obligation(s) suspended. "
                   f"WARNING: This voids your insurance coverage per cross-contract clause.")
        else:
            msg = (f"Force Majeure invoked on {action.contract_id}. "
                   f"{len(protected)} obligation(s) suspended.")
        self.cascade_depth_current += 1
        self.cascade_depth_max = max(self.cascade_depth_max, self.cascade_depth_current)
        return reward_neutral(msg), msg

    def _handle_breach_notice(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        contract = self._find_contract(action.contract_id)
        if not contract:
            return reward_for_invalid_action(f"Contract {action.contract_id} not found."), ""
        # Find deadlines on this contract, mark any that require notice as met
        met = [d for d in self.deadlines
               if d.contract_id == action.contract_id
               and "notice" in d.description.lower()
               and d.status == DeadlineStatus.ACTIVE]
        for d in met:
            d.status = DeadlineStatus.MET
        rwd = reward_for_deadline_met(
            sum(d.penalty_if_missed for d in met),
            self.scenario.initial_cash,
        ) if met else reward_neutral("Breach notice sent. No active notice deadlines affected.")
        msg = f"Formal breach notice sent on {action.contract_id}. {len(met)} deadline(s) satisfied."
        rwd.message = msg
        return rwd, msg

    def _handle_insurance_claim(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        if not self.financial_state.insurance_coverage_active:
            msg = "Insurance policy is VOID — claim rejected."
            return reward_for_invalid_action(msg), msg
        # Find related deadlines
        met = [d for d in self.deadlines
               if "insurance" in d.description.lower() and d.status == DeadlineStatus.ACTIVE]
        for d in met:
            d.status = DeadlineStatus.MET
        # Claim adds cash (simplified: 50% recovery)
        recovery = (action.amount or 0) * 0.5
        self.financial_state.cash_balance += recovery
        msg = f"Insurance claim filed on {action.contract_id}. Estimated recovery: ${recovery:,.0f}."
        rwd = reward_for_cash_change(recovery, self.scenario.initial_cash)
        rwd.message = msg
        return rwd, msg

    def _handle_terminate(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        contract = self._find_contract(action.contract_id)
        if not contract:
            return reward_for_invalid_action(f"Contract {action.contract_id} not found."), ""
        # Termination triggers cascades through edges
        cascade_targets = [
            e for e in self.dependency_edges
            if e.source_contract == action.contract_id
            and e.edge_type.value == "cascade_trigger"
        ]
        self.cascade_depth_current += len(cascade_targets)
        self.cascade_depth_max = max(self.cascade_depth_max, self.cascade_depth_current)
        msg = (f"Contract {action.contract_id} terminated. "
               f"{len(cascade_targets)} downstream cascade(s) triggered.")
        return reward_neutral(msg), msg

    def _handle_waiver_request(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        # Cooperative counterparties grant waiver 70% of the time (deterministic on contract_id hash)
        cp = self._find_counterparty_for_contract(action.contract_id)
        granted = cp and cp.profile == CounterpartyProfile.COOPERATIVE
        if granted:
            # Reset covenant temporarily
            msg = f"Waiver granted by lender for {action.contract_id}. Covenant suspended 5 days."
            return reward_neutral(msg), msg
        msg = f"Waiver request denied for {action.contract_id}. Covenant still active."
        return reward_for_invalid_action(msg), msg

    def _handle_formal_notice(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        met = [d for d in self.deadlines
               if (action.contract_id and d.contract_id == action.contract_id)
               and "notice" in d.description.lower()
               and d.status == DeadlineStatus.ACTIVE]
        for d in met:
            d.status = DeadlineStatus.MET
        rwd = reward_for_deadline_met(
            sum(d.penalty_if_missed for d in met),
            self.scenario.initial_cash,
        ) if met else reward_neutral("Formal notice sent. No active notice deadlines affected.")
        msg = f"Formal notice sent. {len(met)} deadline(s) satisfied."
        rwd.message = msg
        return rwd, msg

    def _handle_cross_reference(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        ids = set(action.contract_ids or [])
        # Find undiscovered edges between these contracts
        found = [
            e for e in self.dependency_edges
            if not e.discovered
            and e.source_contract in ids
            and e.target_contract in ids
        ]
        for e in found:
            e.discovered = True
        rwd = reward_for_discovery(len(found))
        descs = "; ".join(f"[{e.edge_type.value}] {e.source_clause_id}->{e.target_clause_id}" for e in found)
        msg = (f"Cross-referenced {len(ids)} contracts. "
               f"Discovered {len(found)} hidden dependency edge(s). {descs}")
        rwd.message = msg
        return rwd, msg

    def _handle_financial_analysis(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        fs = self.financial_state
        upcoming_penalties = sum(
            d.penalty_if_missed for d in self.deadlines
            if d.status == DeadlineStatus.ACTIVE
            and d.due_day <= self.current_day + 7
        )
        days_left = self.max_days - self.current_day
        proj_cash = fs.cash_balance - upcoming_penalties
        msg = (
            f"Financial Impact Analysis:\n"
            f"  Current cash: ${fs.cash_balance:,.0f}\n"
            f"  Upcoming penalties (7 days): ${upcoming_penalties:,.0f}\n"
            f"  Projected cash: ${proj_cash:,.0f}\n"
            f"  Covenant threshold: ${fs.debt_covenant_min_cash:,.0f}\n"
            f"  Days remaining: {days_left}\n"
            f"  Covenant status: {'VIOLATED' if fs.covenant_violated else 'OK'}"
        )
        return reward_neutral(msg), msg

    def _handle_review_deadlines(self) -> Tuple[CascadeReward, str]:
        active = [d for d in self.deadlines if d.status == DeadlineStatus.ACTIVE]
        if not active:
            msg = "No active deadlines."
            return reward_neutral(msg), msg
        lines = [
            f"  Day {d.due_day} [{d.due_day - self.current_day}d left]: {d.description} "
            f"(penalty: ${d.penalty_if_missed:,.0f}) [{d.contract_id}]"
            for d in sorted(active, key=lambda x: x.due_day)
        ]
        msg = f"Active deadlines ({len(active)}):\n" + "\n".join(lines)
        return reward_neutral(msg), msg

    def _handle_assess_counterparty(self, action: CascadeAction) -> Tuple[CascadeReward, str]:
        cp = next((c for c in self.counterparties if c.counterparty_id == action.counterparty_id), None)
        if not cp:
            return reward_for_invalid_action(f"Counterparty {action.counterparty_id} not found."), ""
        cp.risk_profile_revealed = True
        msg = (
            f"Counterparty Assessment — {cp.name}:\n"
            f"  Role: {cp.role}\n"
            f"  Behavior profile: {cp.profile.value}\n"
            f"  Relationship health: {int(cp.relationship_health * 100)}%\n"
        )
        if cp.profile == CounterpartyProfile.AGGRESSIVE:
            msg += "  WARNING: This counterparty will enforce all rights immediately."
        elif cp.profile == CounterpartyProfile.LITIGIOUS:
            msg += "  WARNING: High litigation risk — send pre-emptive notices."
        elif cp.profile == CounterpartyProfile.COOPERATIVE:
            msg += "  NOTE: Willing to negotiate extensions and waivers."
        return reward_neutral(msg), msg

    def _handle_advance_day(self) -> Tuple[CascadeReward, str]:
        """Advance the simulation clock by 1 day."""
        self.current_day += 1
        msgs = []

        # Process expired deadlines
        newly_expired = [
            d for d in self.deadlines
            if d.status == DeadlineStatus.ACTIVE and d.due_day < self.current_day
        ]
        total_penalty = 0.0
        for d in newly_expired:
            d.status = DeadlineStatus.EXPIRED
            self.financial_state.cash_balance -= d.penalty_if_missed
            total_penalty += d.penalty_if_missed
            msgs.append(f"DEADLINE EXPIRED: {d.description} — penalty ${d.penalty_if_missed:,.0f}")
            self.cascade_depth_current += 1
        self.cascade_depth_max = max(self.cascade_depth_max, self.cascade_depth_current)

        # Deliver new shock events for this day
        day_shocks = [sh for sh in self.scenario.shock_events if sh.occurs_on_day == self.current_day]
        for shock in day_shocks:
            self._apply_shock(shock)
            msgs.append(f"NEW SHOCK: {shock.title}")

        # Deliver new inbox messages
        new_msgs = [m for m in self.scenario.initial_inbox if m.day_received == self.current_day]
        for m in new_msgs:
            if not any(x.message_id == m.message_id for x in self.inbox):
                self.inbox.append(m.model_copy(deep=True))
                msgs.append(f"NEW MESSAGE: {m.subject}")

        msg = f"Advanced to Day {self.current_day}. "
        if total_penalty > 0:
            msg += f"${total_penalty:,.0f} in expired deadline penalties. "
        msg += " | ".join(msgs) if msgs else "No new events."

        self._check_termination()
        rwd = reward_for_cash_change(-total_penalty, self.scenario.initial_cash) if total_penalty > 0 else reward_neutral(msg)
        rwd.message = msg
        return rwd, msg

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _find_contract(self, contract_id: Optional[str]) -> Optional[ContractDocument]:
        if not contract_id:
            return None
        return next((c for c in self.contracts if c.contract_id == contract_id), None)

    def _find_counterparty_for_contract(self, contract_id: Optional[str]) -> Optional[Counterparty]:
        if not contract_id:
            return None
        contract = self._find_contract(contract_id)
        if not contract:
            return None
        return next((cp for cp in self.counterparties if cp.name in contract.parties), None)

    def _apply_shock(self, shock: ShockEvent) -> None:
        if shock.immediate_cash_impact != 0.0 and self.financial_state:
            self.financial_state.cash_balance += shock.immediate_cash_impact
        msg = InboxMessage(
            message_id=f"shock_{shock.shock_id}",
            day_received=shock.occurs_on_day,
            sender="Crisis Alert System",
            subject=f"ALERT: {shock.title}",
            body=shock.description,
            is_legal_notice=False,
        )
        if not any(m.message_id == msg.message_id for m in self.inbox):
            self.inbox.append(msg)

    def _check_termination(self) -> None:
        if self.done:
            return
        if not self.financial_state:
            return
        if not self.financial_state.solvent:
            self.bankruptcy = True
            self.done = True
            return
        if self.current_day > self.max_days:
            self.done = True
            return
        if self.step_number >= self.max_steps:
            self.done = True

    def _record_financial_snapshot(self, event: str) -> None:
        if not self.financial_state:
            return
        self.financial_history.append({
            "day": self.current_day,
            "step": self.step_number,
            "event": event,
            "cash": round(self.financial_state.cash_balance, 2),
            "covenant_violated": self.financial_state.covenant_violated,
            "insurance_active": self.financial_state.insurance_coverage_active,
        })

    def _run_grader(self) -> CascadeGraderResult:
        if not self.scenario or not self.financial_state:
            return CascadeGraderResult(score=0.001, message="No scenario loaded.")
        return grade_cascade_episode(
            task_id=self.task_id or "",
            cash_final=self.financial_state.cash_balance,
            cash_initial=self.scenario.initial_cash,
            deadlines=self.deadlines,
            dependency_edges=self.dependency_edges,
            cascade_depth_max=self.cascade_depth_max,
            bankruptcy=self.bankruptcy,
        )

    # ------------------------------------------------------------------
    # OBSERVATION & STATE
    # ------------------------------------------------------------------

    def _build_observation(self, feedback: str) -> CascadeObservation:
        fs = self.financial_state
        active_deadlines = [
            {
                "deadline_id": d.deadline_id,
                "description": d.description,
                "contract_id": d.contract_id,
                "due_day": d.due_day,
                "days_remaining": d.due_day - self.current_day,
                "penalty_if_missed": d.penalty_if_missed,
                "status": d.status.value,
            }
            for d in sorted(self.deadlines, key=lambda x: x.due_day)
            if d.status == DeadlineStatus.ACTIVE
        ]
        unread_inbox = [
            {
                "message_id": m.message_id,
                "sender": m.sender,
                "subject": m.subject,
                "body": m.body[:300],
                "day_received": m.day_received,
                "is_legal_notice": m.is_legal_notice,
                "requires_response_by_day": m.requires_response_by_day,
            }
            for m in self.inbox if not m.read
        ]
        for m in self.inbox:
            m.read = True
        discovered = [
            {
                "source": f"{e.source_contract}.{e.source_clause_id}",
                "target": f"{e.target_contract}.{e.target_clause_id}",
                "type": e.edge_type.value,
                "description": e.description,
            }
            for e in self.dependency_edges if e.discovered
        ]
        contracts_summary = [
            {"contract_id": c.contract_id, "type": c.contract_type,
             "parties": c.parties, "clause_count": len(c.clauses)}
            for c in self.contracts
        ]
        cp_statuses = [
            {
                "counterparty_id": cp.counterparty_id,
                "name": cp.name,
                "role": cp.role,
                "relationship_health": round(cp.relationship_health, 2),
                "profile_revealed": cp.risk_profile_revealed,
                "profile": cp.profile.value if cp.risk_profile_revealed else "unknown",
            }
            for cp in self.counterparties
        ]
        from cascade_models import CASCADE_ACTION_TYPES
        return CascadeObservation(
            task_id=self.task_id or "none",
            current_day=self.current_day,
            max_days=self.max_days,
            step_number=self.step_number,
            max_steps=self.max_steps,
            cash_balance=round(fs.cash_balance if fs else 0.0, 2),
            covenant_min_cash=fs.debt_covenant_min_cash if fs else 0.0,
            covenant_violated=fs.covenant_violated if fs else False,
            insurance_active=fs.insurance_coverage_active if fs else False,
            reputation_score=round(fs.reputation_score if fs else 0.0, 2),
            active_deadlines=active_deadlines,
            inbox_messages=unread_inbox,
            discovered_edges=discovered,
            contracts_summary=contracts_summary,
            last_action_feedback=feedback,
            counterparty_statuses=cp_statuses,
            available_actions=CASCADE_ACTION_TYPES,
            done=self.done,
            bankruptcy=self.bankruptcy,
        )

    def _empty_observation(self) -> CascadeObservation:
        from cascade_models import CASCADE_ACTION_TYPES
        return CascadeObservation(
            task_id="none", current_day=0, max_days=1, step_number=0, max_steps=1,
            cash_balance=0.0, covenant_min_cash=0.0, covenant_violated=False,
            insurance_active=False, reputation_score=0.0,
            available_actions=[], done=True, bankruptcy=False,
            last_action_feedback="No active episode. Call reset() first.",
        )

    def state(self) -> CascadeEnvironmentState:
        fs = self.financial_state
        return CascadeEnvironmentState(
            task_id=self.task_id or "none",
            current_day=self.current_day,
            max_days=self.max_days,
            step_number=self.step_number,
            cash_balance=round(fs.cash_balance if fs else 0.0, 2),
            initial_cash=self.scenario.initial_cash if self.scenario else 0.0,
            bankruptcy=self.bankruptcy,
            done=self.done,
            deadlines_met=sum(1 for d in self.deadlines if d.status == DeadlineStatus.MET),
            deadlines_expired=sum(1 for d in self.deadlines if d.status == DeadlineStatus.EXPIRED),
            deadlines_total=len(self.deadlines),
            actions_taken=self.actions_taken,
            financial_history=self.financial_history,
            grader_result=self.grader_result.model_dump() if self.grader_result else None,
        )

    def _build_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "current_day": self.current_day,
            "step_number": self.step_number,
            "cash_balance": round(self.financial_state.cash_balance if self.financial_state else 0.0, 2),
            "bankruptcy": self.bankruptcy,
        }
        if self.done and self.grader_result:
            info["grader_score"] = float(self.grader_result.score)
            info["grader_result"] = self.grader_result.model_dump()
        return info
