"""
LexArena Inference Script — Complete 6-Tier Legal Intelligence Agent
=====================================================================
Runs an LLM agent through all 6 tiers of the LexArena benchmark.

Mandatory environment variables:
    API_BASE_URL   OpenAI-compatible inference endpoint.
    MODEL_NAME     Model identifier (e.g. 'Qwen/Qwen2.5-72B-Instruct').
    OPENAI_API_KEY        API key (injected by evaluator).

Optional:
    ENV_MODE       'direct' (default) or 'http'
    ENV_URL        Server URL when ENV_MODE=http (default: http://127.0.0.1:7860)
    TIERS          Comma-separated list of tiers to run (default: 1,2,3,4,5,6)
    DEBUG          'true' to enable verbose logging
    LLM_MAX_RETRIES, LLM_RETRY_BASE_SECONDS, LLM_RETRY_CAP_SECONDS

Design principles:
- Each tier has a dedicated system prompt tailored to that task.
- Few-shot examples are embedded for Tier 1 and Tier 2 to help lightweight
  models (< 13B) understand the exact expected output format.
- The curriculum runner progresses T1 → T2a → T2b → T2c → T3 → T4 → T5 → T6.
- Every LLM call includes corrective_feedback from the previous step in context.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from models import (
    Action,
    ActionType,
    Observation,
    Reward,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import TASK_REGISTRY, get_task_config

API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_MODE     = os.getenv("ENV_MODE", "direct").lower()
ENV_URL      = os.getenv("ENV_URL", "http://127.0.0.1:7860")
DEBUG        = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

TEMPERATURE            = 0.2   # Slight temperature for diversity in legal analysis
MAX_TOKENS             = 400
LLM_MAX_RETRIES        = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "2.0"))
LLM_RETRY_CAP_SECONDS  = float(os.getenv("LLM_RETRY_CAP_SECONDS", "45.0"))

# Which tiers to run in the curriculum (can be overridden via env var)
_TIERS_ENV = os.getenv("TIERS", "1,2,3,4,5,6")
ACTIVE_TIERS: List[int] = [int(t.strip()) for t in _TIERS_ENV.split(",") if t.strip().isdigit()]

# Tier 2 task IDs (backward-compatible OpenEnv tasks)
TIER2_TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# Backward-compatibility alias expected by test_phase4
TASKS = TIER2_TASKS

ANALYSIS_TEMPLATE = textwrap.dedent(
    """\
<analysis>
clause_type=
risk_level=
flags=
suggested_action=
reasoning=
</analysis>
"""
).strip()


# =============================================================================
# SYSTEM PROMPTS — one per tier, tuned for lightweight model compatibility
# =============================================================================

# Tier 1: Verbatim Clause Extraction
SYSTEM_PROMPT_T1 = textwrap.dedent("""\
You are a contract analysis AI. Your task is VERBATIM CLAUSE EXTRACTION.

Given a question about a contract category (e.g. 'Force Majeure') and a
contract text, find the EXACT sentence from the text that answers the question.

RULES:
1. Copy the sentence WORD-FOR-WORD from the contract. No paraphrasing.
2. Extract only ONE sentence — the most directly relevant one.
3. If NO clause of that type exists in the text, respond with exactly:
   No related clause.
4. Do NOT add explanation, commentary, or quotation marks.

EXAMPLES:
---
Question: Is there a Force Majeure clause?
Contract excerpt: "Neither party shall be liable for delays caused by circumstances beyond their reasonable control, including acts of God, war, or government action."
CORRECT ANSWER: Neither party shall be liable for delays caused by circumstances beyond their reasonable control, including acts of God, war, or government action.
---
Question: Is there a Non-Compete clause?
Contract excerpt: "Payment terms are Net 30 from invoice date. Late payments accrue interest at 1.5% per month."
CORRECT ANSWER: No related clause.
---
Now answer for the contract provided. Output ONLY the extracted sentence or 'No related clause.'
""").strip()


# Tier 2: Risk Classification + Full Review
SYSTEM_PROMPT_T2 = textwrap.dedent("""\
You are an expert contract review attorney. Analyse each clause for type,
risk, issues, and recommended action. Use the exact template below.

VALID CLAUSE TYPES:
indemnification, limitation_of_liability, termination, confidentiality,
non_compete, force_majeure, assignment, governing_law, warranty,
intellectual_property, payment_terms, representations, dispute_resolution,
data_protection, insurance

VALID RISK LEVELS: low, medium, high, critical

VALID ISSUE FLAGS:
vague_language, missing_liability_cap, one_sided_obligation, unusual_term,
market_standard, overly_broad_scope, missing_time_limit,
ambiguous_definition, conflicting_with_other_clause, missing_carve_out,
automatic_renewal, unreasonable_penalty, silent_on_key_issue

VALID SUGGESTED ACTIONS:
accept_as_is, request_modification, escalate_to_senior_counsel,
reject_clause, flag_for_negotiation

RULES:
- Use ONLY values from the lists above.
- flags=[] if no issues. Otherwise: flags=["flag1","flag2"]
- reasoning: 1-2 sentences maximum.
- Do NOT add commentary outside the template.
- Alternatively you may respond in JSON format with the same field names.
  JSON format: {"clause_type":"...","risk_level":"...","flags":[...],"suggested_action":"...","reasoning":"..."}

EXAMPLE (study this carefully):
Clause: "The Vendor shall indemnify and hold harmless the Client from any and all claims, damages, losses, costs and expenses, including attorneys' fees, arising out of Vendor's performance."
<analysis>
clause_type=indemnification
risk_level=high
flags=["one_sided_obligation","missing_liability_cap"]
suggested_action=request_modification
reasoning=One-sided indemnification with no cap creates unlimited financial exposure for the Vendor. Standard practice is mutual indemnification with a cap equal to the contract value.
</analysis>

TEMPLATE (fill this for every clause):
<analysis>
clause_type=
risk_level=
flags=
suggested_action=
reasoning=
</analysis>
""").strip()

# Alias for backward compatibility with call_llm()
SYSTEM_PROMPT = SYSTEM_PROMPT_T2


# Tier 3: Dependency Graph Mapping
SYSTEM_PROMPT_T3 = textwrap.dedent("""\
You are a contract dependency analyst. Given 3-5 contracts, your task is to
identify ALL hidden cross-document dependency edges between clauses.

An edge exists when invoking or triggering clause A in Contract X directly
changes the legal validity, financial consequence, or enforceability of
clause B in Contract Y.

EDGE TYPES:
- cascade_trigger: A's breach automatically triggers B's penalty/default
- mutual_exclusion: Invoking A makes B void (cannot use both)
- condition_precedent: A must occur before B becomes enforceable
- supersession: A overrides/replaces B (addendum > original clause)
- temporal_gate: A only applies during a time window defined by B

OUTPUT FORMAT — respond with a JSON array only, no commentary:
[
  {
    "source_contract": "CONTRACT_ID",
    "source_clause": "CLAUSE_ID",
    "target_contract": "CONTRACT_ID",
    "target_clause": "CLAUSE_ID",
    "edge_type": "cascade_trigger",
    "reasoning": "1 sentence explaining the dependency"
  }
]

If you find no dependencies, respond: []

EXAMPLE:
Contract A has a Force Majeure clause. Contract B (insurance) says coverage
is void if the policyholder has invoked Force Majeure on the same event.
Edge: source=A.force_majeure, target=B.coverage_exclusion, type=mutual_exclusion
""").strip()


# Tiers 4-6: Crisis Management (CRO)
SYSTEM_PROMPT_CRISIS = textwrap.dedent("""\
You are the AI Chief Risk Officer of a corporation in a legal and financial
crisis. You must keep the company solvent and legally compliant for the
duration of the crisis episode.

EVERY DAY you must choose ONE action. Think before acting.

ACTION TYPES:
cross_reference_contracts  — Discover hidden edges between contracts (do this FIRST)
review_deadline_status     — Check all active deadlines
assess_counterparty_risk   — Learn counterparty's legal stance
analyze_financial_impact   — Model the cash consequence of an action
file_insurance_claim       — File insurance (do BEFORE Force Majeure)
invoke_force_majeure       — Invoke FM (do AFTER insurance if applicable)
send_formal_notice         — Issue required written notices
request_waiver             — Ask for penalty waiver
negotiate_payment_plan     — Negotiate installment payment
draw_credit_facility       — Draw from credit line (emergency cash)
pay_penalty                — Pay a contractual penalty
terminate_contract         — Terminate a failing contract
advance_day                — Advance to next day (only when no urgent actions)

STRATEGY RULES:
1. INVESTIGATE FIRST. Use cross_reference_contracts and review_deadline_status
   before any financial action.
2. ORDER MATTERS. File insurance BEFORE invoking Force Majeure (FM voids insurance).
3. VERIFY BEFORE PAYING. Check legal position before paying any demand.
   Aggressive counterparties often make legally invalid claims.
4. PROTECT CASH. If cash drops near the debt covenant minimum, draw_credit_facility
   or negotiate_payment_plan immediately.
5. NEVER ignore deadlines. A missed contractual deadline costs more than the penalty.

RESPOND WITH THIS JSON FORMAT:
{
  "action_type": "cross_reference_contracts",
  "contract_ids": ["CONTRACT_A", "CONTRACT_B"],
  "justification": "One sentence explaining why this action now."
}

Optional fields depending on action:
  "contract_id": "..."
  "clause_id": "..."
  "counterparty_id": "..."
  "amount": 50000.0
""").strip()


def get_system_prompt(tier: int) -> str:
    """Return the appropriate system prompt for a given tier."""
    if tier == 1:
        return SYSTEM_PROMPT_T1
    elif tier == 2:
        return SYSTEM_PROMPT_T2
    elif tier == 3:
        return SYSTEM_PROMPT_T3
    elif tier in (4, 5, 6):
        return SYSTEM_PROMPT_CRISIS
    return SYSTEM_PROMPT_T2  # safe default



def build_user_prompt(
    obs_data: Any,
    task_description: str,
    history: List[str],
) -> str:
    if hasattr(obs_data, "model_dump"):
        obs_data = obs_data.model_dump()

    history_text = "\n".join(history[-6:]) if history else "None"
    clause_idx = obs_data.get("clause_index", 0) + 1
    total = obs_data.get("total_clauses", 0)
    step = obs_data.get("step_number", 0)
    max_steps = obs_data.get("max_steps", 10)
    clause_text = obs_data.get("clause_text", "")
    contract_type = obs_data.get("contract_type", "Unknown")
    parties = obs_data.get("parties", [])
    jurisdiction = obs_data.get("jurisdiction", "Unknown")

    return textwrap.dedent(
        f"""\
CONTRACT INFORMATION:
- Type: {contract_type}
- Parties: {", ".join(parties) if parties else "Unknown"}
- Jurisdiction: {jurisdiction}

TASK: {task_description}

CLAUSE {clause_idx} OF {total}:
"{clause_text}"

Step {step} of {max_steps}.

RECENT HISTORY:
{history_text}

Fill this template and return it exactly:
{ANALYSIS_TEMPLATE}
"""
    ).strip()


def call_llm(client: OpenAI, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Call the LLM with exponential-backoff retry logic.

    Args:
        client: Initialised OpenAI client.
        user_prompt: The task-specific user message.
        system_prompt: Tier-specific system prompt (defaults to Tier 2).

    Returns:
        Raw completion text, or empty string on unrecoverable failure.
    """

    def _is_retryable_error(err_text: str) -> bool:
        t = err_text.lower()
        return any(tok in t for tok in (
            "429", "resource_exhausted", "rate limit", "quota exceeded",
            "temporarily unavailable", "timed out", "timeout",
            "service unavailable", "connection reset", "connection aborted",
        ))

    def _extract_retry_delay_seconds(err_text: str) -> Optional[float]:
        m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"retryDelay'\s*:\s*'([0-9]+)s'", err_text)
        if m:
            return float(m.group(1))
        return None

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            err_text = str(exc)
            if attempt >= LLM_MAX_RETRIES or not _is_retryable_error(err_text):
                return ""
            retry_delay = _extract_retry_delay_seconds(err_text)
            if retry_delay is None:
                retry_delay = min(
                    LLM_RETRY_BASE_SECONDS * (2 ** attempt),
                    LLM_RETRY_CAP_SECONDS,
                )
            time.sleep(retry_delay)

    return ""


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()

    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Robust extraction: scan for the first decodable JSON object.
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _parse_flags_value(raw_flags: Any) -> List[str]:
    if raw_flags is None:
        return []

    if isinstance(raw_flags, list):
        return [str(f).strip() for f in raw_flags if str(f).strip()]

    text = str(raw_flags).strip()
    if not text or text.lower() in ("none", "null", "[]"):
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(f).strip() for f in parsed if str(f).strip()]
        except json.JSONDecodeError:
            pass

    tokens = re.split(r"[|,;]", text)
    return [t.strip().strip('"').strip("'") for t in tokens if t.strip()]


def _parse_template_response(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None

    lower = text.lower()
    start_tag = "<analysis>"
    end_tag = "</analysis>"
    start = lower.find(start_tag)
    end = lower.find(end_tag)

    block = text
    if start != -1 and end != -1 and end > start:
        block = text[start + len(start_tag) : end]

    parsed: Dict[str, Any] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue

        # Support both "key=value" and JSON-like "\"key\": value" lines.
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        norm_key = key.strip().strip('"').strip("'").lower()
        norm_value = value.strip().rstrip(",").strip()

        if norm_key == "clause_type":
            parsed["clause_type"] = norm_value.strip('"').strip("'")
        elif norm_key == "risk_level":
            parsed["risk_level"] = norm_value.strip('"').strip("'")
        elif norm_key == "flags":
            parsed["flags"] = _parse_flags_value(norm_value)
        elif norm_key == "suggested_action":
            parsed["suggested_action"] = norm_value.strip('"').strip("'")
        elif norm_key == "reasoning":
            parsed["reasoning"] = norm_value.strip('"').strip("'")

    return parsed or None


def _regex_clause_type(text: str) -> str:
    for ct in CLAUSE_TAXONOMY:
        if re.search(ct.replace("_", "[_ ]"), text.lower()):
            return ct
    return "representations"


def _regex_risk(text: str) -> str:
    for lv in ["critical", "high", "medium", "low"]:
        if lv in text.lower():
            return lv
    return "medium"


def _regex_flags(text: str) -> List[str]:
    return [f for f in ISSUE_FLAGS if re.search(f.replace("_", "[_ ]"), text.lower())]


def _regex_suggest(text: str) -> str:
    patterns = {
        "accept_as_is": [r"accept", r"as[- ]is"],
        "request_modification": [r"modif", r"request.{0,10}change"],
        "escalate_to_senior_counsel": [r"escalat", r"senior"],
        "reject_clause": [r"reject", r"remove"],
        "flag_for_negotiation": [r"flag", r"negotiat"],
    }
    for action, pats in patterns.items():
        for p in pats:
            if re.search(p, text.lower()):
                return action
    return "flag_for_negotiation"


def parse_clause_type_from_text(text: str) -> str:
    return _regex_clause_type(text or "")


def parse_risk_level_from_text(text: str) -> str:
    return _regex_risk(text or "")


def parse_flags_from_text(text: str) -> List[str]:
    return _regex_flags(text or "")


def parse_suggested_action_from_text(text: str) -> str:
    return _regex_suggest(text or "")


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    defaults = {
        "clause_type": "representations",
        "risk_level": "medium",
        "flags": [],
        "suggested_action": "flag_for_negotiation",
        "reasoning": response_text[:200] if response_text else "No analysis provided.",
    }
    if not response_text:
        return defaults

    obj = _parse_template_response(response_text)
    if obj is None:
        obj = extract_json_from_text(response_text)

    if obj:
        ct = obj.get("clause_type", "")
        defaults["clause_type"] = (
            ct if ct in CLAUSE_TAXONOMY else _regex_clause_type(str(ct))
        )

        rl = obj.get("risk_level", "")
        defaults["risk_level"] = (
            rl if rl in ["low", "medium", "high", "critical"] else _regex_risk(str(rl))
        )

        flags = obj.get("flags", [])
        if not isinstance(flags, list):
            flags = _parse_flags_value(flags)
        defaults["flags"] = (
            [f for f in flags if f in ISSUE_FLAGS]
            if isinstance(flags, list)
            else _regex_flags(str(flags))
        )

        sa = obj.get("suggested_action", "")
        valid_sa = [e.value for e in SuggestedActionType]
        defaults["suggested_action"] = sa if sa in valid_sa else _regex_suggest(str(sa))

        reasoning = obj.get("reasoning", "")
        if reasoning:
            defaults["reasoning"] = str(reasoning)[:300]
    else:
        parsed_ct = _regex_clause_type(response_text)
        parsed_rl = _regex_risk(response_text)
        parsed_flags = _regex_flags(response_text)
        parsed_sa = _regex_suggest(response_text)

        defaults["clause_type"] = parsed_ct
        defaults["risk_level"] = parsed_rl
        defaults["flags"] = parsed_flags
        defaults["suggested_action"] = parsed_sa
        defaults["reasoning"] = response_text[:300]

    return defaults


def build_action_dict(action_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    d: Dict[str, Any] = {"action_type": action_type}
    if action_type == "classify":
        d["clause_type"] = analysis.get("clause_type", "representations")
    elif action_type == "rate_severity":
        d["risk_level"] = analysis.get("risk_level", "medium")
    elif action_type == "flag":
        flags = analysis.get("flags", [])
        d["flags"] = [f for f in flags if f in ISSUE_FLAGS]
    elif action_type == "suggest":
        d["suggested_action"] = analysis.get("suggested_action", "flag_for_negotiation")
    elif action_type == "reason":
        d["reasoning"] = analysis.get("reasoning", "Standard clause.")
    return d


def build_action(action_type: ActionType, analysis: Dict[str, Any]) -> Action:
    """Compatibility helper used by phase tests.

    Builds a strongly-typed Action from parsed analysis with safe fallbacks.
    """
    if action_type == ActionType.CLASSIFY:
        clause_type = analysis.get("clause_type") or "representations"
        if clause_type not in CLAUSE_TAXONOMY:
            clause_type = "representations"
        return Action(action_type=action_type, clause_type=clause_type)

    if action_type == ActionType.RATE_SEVERITY:
        rl = str(analysis.get("risk_level") or "medium").lower()
        if rl not in [r.value for r in RiskLevel]:
            rl = "medium"
        return Action(action_type=action_type, risk_level=RiskLevel(rl))

    if action_type == ActionType.FLAG:
        raw_flags = analysis.get("flags")
        if not isinstance(raw_flags, list):
            raw_flags = []
        flags = [f for f in raw_flags if f in ISSUE_FLAGS]
        return Action(action_type=action_type, flags=flags)

    if action_type == ActionType.SUGGEST:
        sa = str(analysis.get("suggested_action") or "flag_for_negotiation")
        valid = [s.value for s in SuggestedActionType]
        if sa not in valid:
            sa = "flag_for_negotiation"
        return Action(action_type=action_type, suggested_action=SuggestedActionType(sa))

    if action_type == ActionType.REASON:
        reasoning = str(analysis.get("reasoning") or "No reasoning provided.")[:300]
        return Action(action_type=action_type, reasoning=reasoning)

    if action_type == ActionType.NEXT_CLAUSE:
        return Action(action_type=action_type)

    return Action(action_type=ActionType.COMPLETE_REVIEW)


class DirectEnvAdapter:
    def __init__(self):
        from environment import ContractReviewEnv

        self.env = ContractReviewEnv()

    def reset(self, task_id: str) -> Dict[str, Any]:
        obs = self.env.reset(task_id)
        return obs.model_dump()

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        action = Action(**action_dict)
        obs, reward, done, info = self.env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }

    def state(self) -> Dict[str, Any]:
        return self.env.state().model_dump()

    def close(self) -> None:
        # No explicit resources to release for in-process mode.
        return None


class HttpEnvAdapter:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/step", json=action_dict)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.client.close()


def _single_line_text(value: Any) -> str:
    """Normalize arbitrary values to a single-line printable token."""
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def run_task(
    env_adapter,
    client: OpenAI,
    task_id: str,
) -> Dict[str, Any]:
    start_time = time.time()
    task_config = get_task_config(task_id)
    if task_config is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    required = [a.value for a in task_config.required_action_types]
    task_desc = task_config.description

    obs_data = env_adapter.reset(task_id)
    history: List[str] = []
    llm_calls = 0
    total_steps = 0
    done = False

    print(f"[START] task={task_id} env=contract-clause-review model={MODEL_NAME}")

    result = None
    all_rewards: List[float] = []
    last_runtime_error: Optional[str] = None

    try:
        while not done:
            if obs_data.get("clause_index", 0) >= obs_data.get("total_clauses", 0):
                result = env_adapter.step({"action_type": "complete_review"})
                obs_data = result["observation"]
                done = result["done"]
                reward_val = result["reward"].get("score", 0.0)
                total_steps += 1
                all_rewards.append(reward_val)
                print(
                    f"[STEP] step={total_steps} action=complete_review reward={reward_val:.2f} done={str(done).lower()} error=null"
                )
                break

            user_prompt = build_user_prompt(obs_data, task_desc, history)
            response_text = call_llm(client, user_prompt)
            llm_calls += 1

            analysis = parse_llm_response(response_text)

            for act_type in required:
                if done:
                    break
                action_dict = build_action_dict(act_type, analysis)
                result = env_adapter.step(action_dict)
                obs_data = result["observation"]
                reward_data = result["reward"]
                done = result["done"]
                total_steps += 1

                reward_val = reward_data.get("score", 0.0)
                all_rewards.append(reward_val)

                error_info = result.get("info", {}).get("error")
                err_str = _single_line_text(error_info) if error_info else "null"

                print(
                    f"[STEP] step={total_steps} action={act_type} reward={reward_val:.2f} done={str(done).lower()} error={err_str}"
                )

                step_log = (
                    f"Step {obs_data['step_number']}: {act_type} -> {reward_val:+.3f}"
                )
                history.append(step_log)

            if done:
                break

            clause_idx = obs_data.get("clause_index", 0)
            total_clauses = obs_data.get("total_clauses", 0)

            if clause_idx < total_clauses - 1:
                result = env_adapter.step({"action_type": "next_clause"})
                obs_data = result["observation"]
                done = result["done"]
                reward_val = result["reward"].get("score", 0.0)
                total_steps += 1
                all_rewards.append(reward_val)
                print(
                    f"[STEP] step={total_steps} action=next_clause reward={reward_val:.2f} done={str(done).lower()} error=null"
                )
                history.append(
                    f"Step {obs_data['step_number']}: next_clause -> {reward_val:+.3f}"
                )
            else:
                result = env_adapter.step({"action_type": "complete_review"})
                obs_data = result["observation"]
                done = result["done"]
                reward_val = result["reward"].get("score", 0.0)
                total_steps += 1
                all_rewards.append(reward_val)
                print(
                    f"[STEP] step={total_steps} action=complete_review reward={reward_val:.2f} done={str(done).lower()} error=null"
                )
                history.append(
                    f"Step {obs_data['step_number']}: complete_review -> {reward_val:+.3f}"
                )
    except Exception as e:
        last_runtime_error = _single_line_text(e)
    finally:
        elapsed = time.time() - start_time
        final_info = result.get("info", {}) if result else {}

        # Defensive finalization: ensure we still attempt grader finalization
        # when the task loop exits unexpectedly.
        if "grader_score" not in final_info:
            try:
                final_step = env_adapter.step({"action_type": "complete_review"})
                final_info = final_step.get("info", {})
            except Exception:
                final_info = final_info or {}

        raw_score = float(final_info.get("grader_score", 0.001))
        grader_score = max(0.001, min(0.999, raw_score))
        grader_breakdown = final_info.get("grader_result", {}).get("breakdown", {})

        clauses_reviewed = 0
        try:
            state = env_adapter.state()
            clauses_reviewed = sum(
                1
                for cr in state.get("clause_records", [])
                if cr.get("action_count", 0) > 0
            )
        except Exception:
            clauses_reviewed = 0

        # Requirement: emit END after env.close().
        if hasattr(env_adapter, "close"):
            try:
                env_adapter.close()
            except Exception:
                pass

        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        success = (last_runtime_error is None) and (0.0 <= grader_score <= 1.0)
        print(
            f"[END] success={str(success).lower()} steps={total_steps} score={grader_score:.4f} rewards={rewards_str}"
        )

    return {
        "task_id": task_id,
        "difficulty": task_config.difficulty.value,
        "grader_score": grader_score,
        "grader_breakdown": grader_breakdown,
        "accumulated_reward": round(obs_data.get("accumulated_score", 0.0), 4),
        "total_steps": total_steps,
        "llm_calls": llm_calls,
        "clauses_reviewed": clauses_reviewed,
        "total_clauses": obs_data.get("total_clauses", 0),
        "elapsed_seconds": round(elapsed, 2),
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all tier results."""
    print("\n" + "=" * 72)
    print(f"  LexArena Results -- Model: {MODEL_NAME}")
    print("=" * 72)
    print(f"  {'Task':<35} {'Score':>8} {'Steps':>7} {'LLM Calls':>10}")
    print("-" * 72)
    for r in results:
        task = r.get("task_id", "?")
        score = r.get("grader_score", 0.0)
        steps = r.get("total_steps", 0)
        calls = r.get("llm_calls", 0)
        err = " [ERROR]" if r.get("error") else ""
        print(f"  {task:<35} {score:>8.4f} {steps:>7} {calls:>10}{err}")
    print("=" * 72)


def _run_tier2_curriculum(
    client: OpenAI,
    env_url: str,
    env_mode: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run Tier 2 tasks (task_1_easy -> task_2_medium -> task_3_hard).

    Returns the mean grader score and list of per-task results.
    """
    tier2_results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for task_id in TIER2_TASKS:
        adapter = HttpEnvAdapter(env_url) if env_mode == "http" else DirectEnvAdapter()
        try:
            result = run_task(adapter, client, task_id)
            tier2_results.append(result)
            scores.append(result.get("grader_score", 0.0))
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 2 task {task_id} failed: {exc}\n")
            tier2_results.append({"task_id": task_id, "grader_score": 0.001, "error": str(exc)})
            scores.append(0.001)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, tier2_results


def main() -> None:
    """Full 6-tier LexArena curriculum runner.

    Progresses: T1 (reading) -> T2 (classification) -> T3 (dependency) ->
                T4 (crisis easy) -> T5 (crisis medium) -> T6 (crisis hard).

    Each tier uses a dedicated system prompt. Corrective feedback from the
    environment is fed into subsequent prompts to help the model improve.

    Emits [START], [STEP], and [END] log lines per OpenEnv convention.
    Saves full results to baseline_results.json.
    """
    # --- Validate required environment variables ---
    missing = [v for v in ("API_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME")
               if not os.getenv(v, "").strip()]
    if missing:
        for var in missing:
            sys.stderr.write(f"ERROR: {var} environment variable not set.\n")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
    env_mode = ENV_MODE
    env_url = ENV_URL

    # Verify HTTP server is reachable
    if env_mode == "http":
        try:
            r = httpx.get(f"{env_url}/health", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:
            sys.stderr.write(f"ERROR: Cannot reach LexArena server at {env_url}: {exc}\n")
            sys.exit(1)

    overall_start = time.time()
    all_results: List[Dict[str, Any]] = []
    tier_scores: Dict[str, float] = {}

    print(f"[LEXARENA] model={MODEL_NAME} mode={env_mode} tiers={ACTIVE_TIERS}")

    # ---------------------------------------------------------------- Tier 1
    if 1 in ACTIVE_TIERS:
        print("\n[TIER 1] Clause Reading (CUAD verbatim extraction)")
        try:
            t1_scores: List[float] = []
            if env_mode == "http":
                reset_r = httpx.post(
                    f"{env_url}/tier1/reset",
                    json={"max_samples": 15, "priority_only": True},
                    timeout=30.0,
                )
                reset_r.raise_for_status()
                state = reset_r.json()
                total_t1 = state.get("total_samples", 0)

                for _ in range(total_t1):
                    sample = state.get("current_sample") or {}
                    if not sample:
                        break
                    user_prompt = (
                        f"Contract: {sample.get('contract_name', 'Unknown')}\n"
                        f"Category: {sample.get('question_category', '')}\n\n"
                        f"Excerpt:\n{sample.get('context', '')}\n\n"
                        f"Question: Is there a {sample.get('question_category', '')} clause?"
                    )
                    response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_T1)
                    step_r = httpx.post(
                        f"{env_url}/tier1/step",
                        json={"sample_id": sample.get("sample_id", ""), "extracted_text": response.strip()},
                        timeout=30.0,
                    )
                    step_r.raise_for_status()
                    state = step_r.json()
                    f2 = state.get("sample_result", {}).get("f2_score", 0.0)
                    t1_scores.append(f2)
                    print(f"  [STEP] T1 sample={sample.get('sample_id','')} f2={f2:.4f}")
                    if state.get("done"):
                        break
            else:
                print("  [INFO] Tier 1 requires http mode. Skipping (score=0.0).")

            t1_score = sum(t1_scores) / len(t1_scores) if t1_scores else 0.0
            tier_scores["t1"] = t1_score
            all_results.append({"task_id": "tier1_clause_reading", "tier": 1,
                                 "grader_score": t1_score, "total_steps": len(t1_scores), "llm_calls": len(t1_scores)})
            print(f"  [END] T1 score={t1_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 1 failed: {exc}\n")
            tier_scores["t1"] = 0.001
            all_results.append({"task_id": "tier1_clause_reading", "tier": 1, "grader_score": 0.001, "error": str(exc)})

    # ---------------------------------------------------------------- Tier 2
    if 2 in ACTIVE_TIERS:
        print("\n[TIER 2] Clause Review (Easy -> Medium -> Hard)")
        t2_score, t2_results = _run_tier2_curriculum(client, env_url, env_mode)
        tier_scores["t2"] = t2_score
        all_results.extend(t2_results)
        print(f"  [END] T2 mean_score={t2_score:.4f}")

    # ---------------------------------------------------------------- Tier 3
    if 3 in ACTIVE_TIERS:
        print("\n[TIER 3] Dependency Graph Mapping")
        try:
            t3_score = 0.0
            if env_mode == "http":
                reset_r = httpx.post(f"{env_url}/tier3/reset", json={"scenario_index": 0}, timeout=30.0)
                reset_r.raise_for_status()
                obs = reset_r.json()
                contracts_summary = json.dumps(obs.get("contracts", []), indent=2)
                user_prompt = (
                    f"Analyse these contracts and identify ALL dependency edges.\n\n"
                    f"Contracts:\n{contracts_summary}\n\n"
                    f"Output a JSON array of dependency edges."
                )
                response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_T3)
                try:
                    raw_deps = json.loads(response.strip())
                    if not isinstance(raw_deps, list):
                        raw_deps = []
                except (json.JSONDecodeError, ValueError):
                    raw_deps = []

                step_r = httpx.post(f"{env_url}/tier3/step", json={"dependencies": raw_deps}, timeout=30.0)
                step_r.raise_for_status()
                step_data = step_r.json()
                result = step_data.get("result") or {}
                t3_score = result.get("combined_score", 0.0)
                print(f"  [STEP] T3 recall={result.get('recall',0):.3f} precision={result.get('precision',0):.3f}")
            else:
                print("  [INFO] Tier 3 requires http mode. Skipping (score=0.0).")

            tier_scores["t3"] = t3_score
            all_results.append({"task_id": "tier3_dependency_mapping", "tier": 3,
                                 "grader_score": t3_score, "total_steps": 1, "llm_calls": 1})
            print(f"  [END] T3 score={t3_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 3 failed: {exc}\n")
            tier_scores["t3"] = 0.001
            all_results.append({"task_id": "tier3_dependency_mapping", "tier": 3, "grader_score": 0.001, "error": str(exc)})

    # ------------------------------------------------------------ Tiers 4-6
    for tier, task_id, label in [
        (4, "task_4_cascade_easy",   "Crisis Easy"),
        (5, "task_5_cascade_medium", "Crisis Medium"),
        (6, "task_6_cascade_hard",   "Crisis Hard"),
    ]:
        if tier not in ACTIVE_TIERS:
            continue
        print(f"\n[TIER {tier}] {label}")
        try:
            tier_score = 0.0
            step_count = 0
            llm_count = 0
            if env_mode == "http":
                reset_r = httpx.post(
                    f"{env_url}/cascade/reset",
                    json={"task_id": task_id, "scenario_index": 0},
                    timeout=30.0,
                )
                reset_r.raise_for_status()
                obs = reset_r.json()
                done = obs.get("done", False)
                crisis_feedback = obs.get("corrective_feedback", "")

                while not done:
                    obs_text = json.dumps({
                        k: obs[k] for k in (
                            "current_day", "cash_balance", "covenant_min_cash",
                            "active_deadlines", "inbox_messages", "discovered_edges",
                            "contracts_summary", "available_actions",
                        ) if k in obs
                    }, indent=2)
                    user_prompt = (
                        f"CORRECTIVE FEEDBACK FROM LAST ACTION:\n{crisis_feedback}\n\n"
                        f"CURRENT SITUATION (Day {obs.get('current_day', '?')}):\n{obs_text}\n\n"
                        f"Choose your next action. Respond with JSON."
                    )
                    response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_CRISIS)
                    llm_count += 1
                    try:
                        action_dict = json.loads(response.strip())
                    except (json.JSONDecodeError, ValueError):
                        action_dict = {"action_type": "advance_day", "justification": "parse error fallback"}

                    step_r = httpx.post(f"{env_url}/cascade/step", json=action_dict, timeout=30.0)
                    step_r.raise_for_status()
                    step_data = step_r.json()
                    obs = step_data.get("observation", {})
                    done = step_data.get("done", False)
                    reward = step_data.get("reward", {}).get("score", 0.0)
                    crisis_feedback = obs.get("corrective_feedback", "")
                    step_count += 1
                    print(f"  [STEP] T{tier} day={obs.get('current_day','?')} step={step_count} reward={reward:.3f}")

                # Retrieve final grader score
                state_r = httpx.get(f"{env_url}/cascade/state", timeout=10.0)
                state_r.raise_for_status()
                state = state_r.json()
                grader = state.get("grader_result") or {}
                tier_score = grader.get("score", 0.001)
            else:
                print(f"  [INFO] Tier {tier} requires http mode. Skipping (score=0.0).")

            tier_scores[f"t{tier}"] = tier_score
            all_results.append({"task_id": task_id, "tier": tier, "grader_score": tier_score,
                                 "total_steps": step_count, "llm_calls": llm_count})
            print(f"  [END] T{tier} score={tier_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier {tier} failed: {exc}\n")
            tier_scores[f"t{tier}"] = 0.001
            all_results.append({"task_id": task_id, "tier": tier, "grader_score": 0.001, "error": str(exc)})

    # ----------------------------------------------------- Legal IQ Composite
    t1 = tier_scores.get("t1", 0.0)
    t2 = tier_scores.get("t2", 0.0)
    t3 = tier_scores.get("t3", 0.0)
    t4 = tier_scores.get("t4", 0.0)
    t5 = tier_scores.get("t5", 0.0)
    t6 = tier_scores.get("t6", 0.0)
    legal_iq = round(0.15*t1 + 0.15*t2 + 0.20*t3 + 0.50*(0.25*t4 + 0.35*t5 + 0.40*t6), 4)
    label = ""

    if env_mode == "http":
        try:
            iq_r = httpx.post(
                f"{env_url}/legal_iq",
                json={"model_name": MODEL_NAME, "t1_score": t1, "t2_score": t2,
                      "t3_score": t3, "t4_score": t4, "t5_score": t5, "t6_score": t6},
                timeout=10.0,
            )
            iq_r.raise_for_status()
            iq_data = iq_r.json()
            legal_iq = iq_data.get("legal_iq", legal_iq)
            label = iq_data.get("label", "")
        except Exception:
            pass  # Keep locally computed score

    # ------------------------------------------------------ Print + Save
    overall_elapsed = time.time() - overall_start
    total_llm_calls = sum(int(r.get("llm_calls", 0)) for r in all_results)

    print_results_table(all_results)
    print(f"\n  Legal IQ : {legal_iq:.4f}  ({label})")
    print(f"  T1={t1:.3f}  T2={t2:.3f}  T3={t3:.3f}  T4={t4:.3f}  T5={t5:.3f}  T6={t6:.3f}")
    print(f"  Total runtime: {overall_elapsed:.1f}s\n")

    if total_llm_calls <= 0:
        sys.stderr.write(
            "ERROR: No LLM calls were made. Check API_BASE_URL/OPENAI_API_KEY setup.\n"
        )
        sys.exit(1)

    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "mode": env_mode,
        "temperature": TEMPERATURE,
        "active_tiers": ACTIVE_TIERS,
        "tier_scores": {
            "t1_reading": t1, "t2_classification": t2, "t3_dependency": t3,
            "t4_crisis_easy": t4, "t5_crisis_medium": t5, "t6_crisis_hard": t6,
        },
        "legal_iq": legal_iq,
        "label": label,
        "results": all_results,
        "total_llm_calls": total_llm_calls,
        "total_runtime_seconds": round(overall_elapsed, 2),
    }
    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("[LEXARENA] Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
