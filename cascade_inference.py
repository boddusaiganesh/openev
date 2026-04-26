"""
LexDomino — Cascade Inference Agent
LLM-powered agent for the crisis simulator.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

from cascade_environment import LexDominoCrisisEnv
from cascade_models import CascadeAction, CascadeActionType, CascadeObservation


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Autonomous Chief Risk Officer (CRO) of a corporation facing a systemic legal and financial crisis.

You manage a portfolio of interconnected legal contracts. Events cascade: a breach in one contract triggers penalties in another, which may violate debt covenants, freeze credit, and lead to bankruptcy.

YOUR MISSION: Keep the company solvent (cash > 0) for the entire simulation period.

STRATEGIC PRINCIPLES:
1. INVESTIGATE FIRST — always cross_reference_contracts and review_deadline_status before major decisions
2. DEADLINES ARE FATAL — a missed notice window can void insurance or trigger $500K+ penalties
3. HIDDEN DEPENDENCIES EXIST — Force Majeure on Contract A may void insurance Policy B
4. COUNTERPARTIES HAVE PROFILES — assess_counterparty_risk before negotiating
5. CASH IS SURVIVAL — every action has a financial consequence; model it with analyze_financial_impact

AVAILABLE ACTIONS:
Legal: invoke_force_majeure, send_breach_notice, file_insurance_claim, terminate_contract, invoke_indemnification, request_waiver
Financial: pay_penalty, negotiate_payment_plan, draw_credit_facility, accelerate_receivable
Communication: send_formal_notice, request_information, propose_amendment
Investigation: cross_reference_contracts, analyze_financial_impact, review_deadline_status, assess_counterparty_risk
Control: advance_day, complete_crisis

OUTPUT FORMAT (JSON only, no markdown):
{
  "reasoning": "brief explanation of your decision",
  "action_type": "<action_name>",
  "contract_id": "<contract_id or null>",
  "counterparty_id": "<counterparty_id or null>",
  "amount": <number or null>,
  "contract_ids": ["id1", "id2"] or null,
  "justification": "<string or null>"
}
"""


def _obs_to_prompt(obs: CascadeObservation) -> str:
    """Convert observation to a compact LLM-readable string."""
    lines = [
        f"=== DAY {obs.current_day}/{obs.max_days} | STEP {obs.step_number}/{obs.max_steps} ===",
        f"Cash: ${obs.cash_balance:,.0f} | Covenant Min: ${obs.covenant_min_cash:,.0f} | "
        f"Covenant: {'VIOLATED' if obs.covenant_violated else 'OK'} | "
        f"Insurance: {'ACTIVE' if obs.insurance_active else 'VOID'}",
        "",
    ]
    if obs.last_action_feedback:
        lines += [f"LAST ACTION: {obs.last_action_feedback}", ""]

    if obs.inbox_messages:
        lines.append("--- INBOX (UNREAD) ---")
        for m in obs.inbox_messages:
            lines.append(f"[{m['sender']}] {m['subject']}")
            lines.append(f"  {m['body'][:200]}")
        lines.append("")

    if obs.active_deadlines:
        lines.append("--- ACTIVE DEADLINES ---")
        for d in obs.active_deadlines:
            lines.append(
                f"  [{d['days_remaining']}d left] {d['description']} "
                f"| Penalty: ${d['penalty_if_missed']:,.0f} | Contract: {d['contract_id']}"
            )
        lines.append("")

    lines.append("--- CONTRACTS ---")
    for c in obs.contracts_summary:
        lines.append(f"  {c['contract_id']}: {c['type']} [{', '.join(c['parties'])}]")
    lines.append("")

    if obs.discovered_edges:
        lines.append("--- DISCOVERED DEPENDENCIES ---")
        for e in obs.discovered_edges:
            lines.append(f"  [{e['type']}] {e['source']} → {e['target']}: {e['description']}")
        lines.append("")

    lines.append("--- COUNTERPARTIES ---")
    for cp in obs.counterparty_statuses:
        profile = cp['profile'] if cp['profile_revealed'] else '(unknown — use assess_counterparty_risk)'
        lines.append(f"  {cp['counterparty_id']}: {cp['name']} [{cp['role']}] health={int(cp['relationship_health']*100)}% profile={profile}")

    return "\n".join(lines)


def _parse_action(raw: str) -> Optional[CascadeAction]:
    """Parse LLM JSON output into a CascadeAction."""
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        at = CascadeActionType(data["action_type"])
        return CascadeAction(
            action_type=at,
            contract_id=data.get("contract_id"),
            counterparty_id=data.get("counterparty_id"),
            amount=data.get("amount"),
            contract_ids=data.get("contract_ids"),
            justification=data.get("justification"),
            proposed_terms=data.get("proposed_terms"),
        )
    except Exception as e:
        return None


def _fallback_action(obs: CascadeObservation) -> CascadeAction:
    """Deterministic fallback: review deadlines, then advance day."""
    if obs.step_number % 5 == 0:
        return CascadeAction(action_type=CascadeActionType.REVIEW_DEADLINE_STATUS)
    if obs.step_number % 3 == 0 and len(obs.contracts_summary) >= 2:
        ids = [c["contract_id"] for c in obs.contracts_summary[:2]]
        return CascadeAction(action_type=CascadeActionType.CROSS_REFERENCE_CONTRACTS, contract_ids=ids)
    return CascadeAction(action_type=CascadeActionType.ADVANCE_DAY)


class CascadeAgent:
    """LLM-powered crisis management agent."""

    def __init__(self, model: str = "gpt-4o", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI() if _openai_available else None
        self.history: List[Dict] = []

    def choose_action(self, obs: CascadeObservation) -> CascadeAction:
        if not self.client:
            return _fallback_action(obs)

        prompt = _obs_to_prompt(obs)
        self.history.append({"role": "user", "content": prompt})
        # Keep history bounded
        if len(self.history) > 20:
            self.history = self.history[-20:]

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
                    temperature=0.2,
                    max_tokens=512,
                )
                content = resp.choices[0].message.content or ""
                self.history.append({"role": "assistant", "content": content})
                action = _parse_action(content)
                if action:
                    return action
            except Exception as e:
                time.sleep(1)
        return _fallback_action(obs)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_episode(
    task_id: str = "task_4_cascade_easy",
    scenario_index: int = 0,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Dict[str, Any]:
    env = LexDominoCrisisEnv()
    agent = CascadeAgent(model=model)
    obs = env.reset(task_id, scenario_index)
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"LexDomino — {task_id} | Scenario {scenario_index}")
        print(f"Company: {env.scenario.company_name if env.scenario else '?'}")
        print(f"Initial Cash: ${env.scenario.initial_cash:,.0f}" if env.scenario else "")
        print(f"{'='*70}\n")

    while not obs.done:
        action = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.score
        step += 1

        if verbose:
            cash = obs.cash_balance
            print(f"Day {obs.current_day:>2} | Step {step:>3} | Action: {action.action_type.value:<35} | "
                  f"Reward: {reward.score:+.3f} | Cash: ${cash:,.0f}")
            if reward.message:
                print(f"         {reward.message[:120]}")

    grader_score = info.get("grader_score", 0.0)
    result = env.grader_result

    if verbose:
        print(f"\n{'='*70}")
        status = "BANKRUPT" if obs.bankruptcy else "SURVIVED"
        print(f"EPISODE COMPLETE — {status}")
        print(f"  Grader Score: {grader_score:.4f}")
        print(f"  Final Cash:   ${obs.cash_balance:,.0f}")
        if result:
            print(f"  Cash Ratio:   {result.normalized_cash_ratio:.2%}")
            print(f"  Deadlines Met:{result.deadlines_met_ratio:.2%}")
            print(f"  {result.message}")
        print(f"{'='*70}\n")

    return {
        "task_id": task_id,
        "scenario_index": scenario_index,
        "grader_score": grader_score,
        "final_cash": obs.cash_balance,
        "bankruptcy": obs.bankruptcy,
        "steps": step,
        "total_reward": round(total_reward, 4),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_4_cascade_easy")
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()
    run_episode(args.task, args.scenario, args.model)
