"""
LexArena — Unified API Server
==============================
Single FastAPI process serving ALL 6 tiers from one HF Space on port 7860.

Tier 1  — Clause Reading        (CUAD, F2-weighted)
Tier 2  — Clause Review         (OpenEnv tasks 1/2/3)
Tier 3  — Dependency Mapping    (novel graph task)
Tier 4  — Crisis Easy           (LexDomino in-process)
Tier 5  — Crisis Medium
Tier 6  — Crisis Hard
        + Adversarial Probes
        + Legal IQ composite
        + Curriculum endpoint

Backward-compatible routes:
  /reset  /step  /state  → Tier 2 (original OpenEnv)
  /cascade/*            → Tiers 4-6 (original LexDomino)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lexarena_models import LexArenaConfig, LegalIQScore, Tier3Action, Tier1Sample, Tier1Output
from lexarena_scorer import compute_legal_iq
from cuad_loader import load_cuad_dataset
from tier1_grader import grade_sample
from tier3_environment import Tier3MappingEnv, load_tier3_scenarios, grade_tier3_batch
from cascade_environment import LexDominoCrisisEnv
from cascade_models import CascadeAction, CascadeActionType
from environment import ContractReviewEnv
from models import Action

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_VERSION = "3.0.0"
_PORT = int(os.getenv("PORT", "7860"))

app = FastAPI(
    title="LexArena — Complete Legal Intelligence Benchmark",
    description=(
        "The world's first 6-tier legal AI benchmark. "
        "Clause reading → risk classification → dependency mapping → "
        "systemic crisis management. Single HF Space, all tiers, one port."
    ),
    version=_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session state (per-process; adequate for HF Space single-worker usage)
# ---------------------------------------------------------------------------

# Tier 1
_t1_samples: List[Tier1Sample] = []
_t1_idx: int = 0

# Tier 2 (OpenEnv clause review)
_t2_env: Optional[ContractReviewEnv] = None
_t2_task_id: str = ""

# Tier 3
_t3_env: Optional[Tier3MappingEnv] = None
_t3_scenarios: List[dict] = []
_t3_idx: int = 0
_t3_results: List = []

# Tiers 4-6 (LexDomino crisis)
_cascade_env: Optional[LexDominoCrisisEnv] = None

# Legal IQ leaderboard (in-memory)
_iq_scores: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Health + Root Spec
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Lightweight health check for Docker/HF Space."""
    return {"status": "ok", "service": "LexArena", "version": _VERSION}


@app.get("/", tags=["Meta"])
def root_spec():
    """Return the full LexArena benchmark specification."""
    return {
        "benchmark": "LexArena",
        "version": _VERSION,
        "description": (
            "The first benchmark to test the complete legal intelligence stack: "
            "READ → CLASSIFY → CONNECT → DECIDE → SURVIVE"
        ),
        "tiers": {
            "T1": {"name": "Clause Reading", "weight": 0.15, "reset": "/tier1/reset", "step": "/tier1/step"},
            "T2": {"name": "Risk Classification", "weight": 0.15, "reset": "/reset", "step": "/step"},
            "T3": {"name": "Dependency Mapping", "weight": 0.20, "reset": "/tier3/reset", "step": "/tier3/step"},
            "T4": {"name": "Crisis Easy", "weight": 0.125, "reset": "/cascade/reset", "step": "/cascade/step"},
            "T5": {"name": "Crisis Medium", "weight": 0.175, "reset": "/cascade/reset", "step": "/cascade/step"},
            "T6": {"name": "Crisis Hard", "weight": 0.20, "reset": "/cascade/reset", "step": "/cascade/step"},
        },
        "legal_iq_formula": "0.15·T1 + 0.15·T2 + 0.20·T3 + 0.50·(0.25·T4 + 0.35·T5 + 0.40·T6)",
        "scoring": "Pure deterministic math — zero LLM judges",
        "probes": "/probes",
        "curriculum": "/curriculum",
        "legal_iq": "/legal_iq",
        "leaderboard": "/legal_iq/leaderboard",
        "docs": "/docs",
    }


@app.get("/curriculum", tags=["Meta"])
def curriculum():
    """
    Return the recommended task order for an agent entering LexArena.

    Designed to maximise learning signal: start with reading (T1),
    progress through classification (T2), then structural reasoning (T3),
    then crisis management (T4 → T5 → T6).
    """
    return {
        "description": (
            "Complete tiers in this order for maximum learning signal. "
            "Each tier builds on skills from the previous one."
        ),
        "curriculum": [
            {
                "step": 1, "tier": "T1", "task": "tier1_clause_reading",
                "goal": "Learn precise legal language extraction",
                "reset": "/tier1/reset", "step_ep": "/tier1/step",
                "pass_threshold": 0.50,
                "tip": "Extract the verbatim sentence. Do not paraphrase. If absent: 'No related clause.'"
            },
            {
                "step": 2, "tier": "T2a", "task": "task_1_easy",
                "goal": "Learn clause taxonomy (15 types)",
                "reset": "/reset", "step_ep": "/step",
                "pass_threshold": 0.60,
                "tip": "Start with the template. Focus on clause_type accuracy."
            },
            {
                "step": 3, "tier": "T2b", "task": "task_2_medium",
                "goal": "Add risk assessment and issue flagging",
                "reset": "/reset", "step_ep": "/step",
                "pass_threshold": 0.55,
                "tip": "Read for both type AND risk. 'missing_liability_cap' is the most common missed flag."
            },
            {
                "step": 4, "tier": "T2c", "task": "task_3_hard",
                "goal": "Full contract review with cross-clause reasoning",
                "reset": "/reset", "step_ep": "/step",
                "pass_threshold": 0.50,
                "tip": "Check for conflicting clauses. Look at the full contract before analysing each clause."
            },
            {
                "step": 5, "tier": "T3", "task": "tier3_dependency_mapping",
                "goal": "Map hidden dependency edges between contracts",
                "reset": "/tier3/reset", "step_ep": "/tier3/step",
                "pass_threshold": 0.40,
                "tip": (
                    "Focus on edge types: cascade_trigger, mutual_exclusion, condition_precedent. "
                    "Ask: does invoking clause X change the legal validity of clause Y?"
                )
            },
            {
                "step": 6, "tier": "T4", "task": "task_4_cascade_easy",
                "goal": "Survive a 15-day single-contract crisis",
                "reset": "/cascade/reset", "step_ep": "/cascade/step",
                "pass_threshold": 0.65,
                "tip": (
                    "Always investigate and cross_reference FIRST. "
                    "Act on the soonest deadline. Never pay without checking legal position."
                )
            },
            {
                "step": 7, "tier": "T5", "task": "task_5_cascade_medium",
                "goal": "Multi-contract crisis with hidden dependencies",
                "reset": "/cascade/reset", "step_ep": "/cascade/step",
                "pass_threshold": 0.55,
                "tip": (
                    "Check debt covenants before paying any penalty. "
                    "Assess counterparty risk early — aggressive profiles escalate if ignored."
                )
            },
            {
                "step": 8, "tier": "T6", "task": "task_6_cascade_hard",
                "goal": "30-day systemic cascade with compound shocks",
                "reset": "/cascade/reset", "step_ep": "/cascade/step",
                "pass_threshold": 0.45,
                "tip": (
                    "Day 1: file insurance BEFORE invoking Force Majeure. "
                    "Day 14-16: expect a second shock. Reserve cash. "
                    "Never cave to unverified aggressive demands — check legal position first."
                )
            },
        ],
        "legal_iq_endpoint": "/legal_iq",
    }


# ---------------------------------------------------------------------------
# Tier 1 — Clause Reading (CUAD)
# ---------------------------------------------------------------------------

class Tier1ResetRequest(BaseModel):
    max_samples: int = 15
    priority_only: bool = True


class Tier1StepRequest(BaseModel):
    sample_id: str
    extracted_text: str


@app.post("/tier1/reset", tags=["Tier 1 — Clause Reading"])
def tier1_reset(req: Tier1ResetRequest):
    """Load CUAD samples and return the first one."""
    global _t1_samples, _t1_idx
    _t1_samples = load_cuad_dataset(
        max_samples=req.max_samples,
        priority_only=req.priority_only,
    )
    _t1_idx = 0
    sample = _t1_samples[0] if _t1_samples else None
    return {
        "total_samples": len(_t1_samples),
        "current_index": 0,
        "current_sample": sample.model_dump() if sample else None,
        "instructions": (
            "Extract the verbatim sentence that answers the question. "
            "Do NOT paraphrase. If no clause applies, respond: 'No related clause.'"
        ),
    }


@app.post("/tier1/step", tags=["Tier 1 — Clause Reading"])
def tier1_step(req: Tier1StepRequest):
    """Submit an extraction and receive score + corrective feedback."""
    global _t1_idx
    if not _t1_samples:
        raise HTTPException(400, "Call /tier1/reset first.")
    if _t1_idx >= len(_t1_samples):
        raise HTTPException(400, "All samples exhausted. Call /tier1/reset.")

    sample = _t1_samples[_t1_idx]
    output = Tier1Output(
        sample_id=req.sample_id,
        extracted_text=req.extracted_text,
        is_no_clause="no related clause" in req.extracted_text.lower(),
    )
    result = grade_sample(sample, output)

    # Build corrective feedback for the model
    feedback = _build_t1_feedback(sample, req.extracted_text, result)

    _t1_idx += 1
    next_sample = _t1_samples[_t1_idx].model_dump() if _t1_idx < len(_t1_samples) else None

    return {
        "sample_result": result.model_dump(),
        "corrective_feedback": feedback,
        "samples_remaining": len(_t1_samples) - _t1_idx,
        "next_sample": next_sample,
        "done": next_sample is None,
    }


def _build_t1_feedback(sample: Tier1Sample, extracted: str, result) -> str:
    """Generate corrective feedback for a Tier 1 extraction step."""
    f2 = result.f2_score
    gt = sample.ground_truth[0] if sample.ground_truth else ""

    if result.is_no_clause_correct:
        return "Correct. No applicable clause exists in this contract section."

    if result.false_laziness:
        return (
            f"Incorrect: you responded 'No related clause' but one exists. "
            f"The correct answer was: \"{gt[:200]}\". "
            f"Legal principle: missing a material clause (false negative) is a critical error "
            f"in contract review — it means the risk goes unnoticed."
        )

    if f2 >= 0.85:
        return "Correct extraction. Strong precision and recall."

    if f2 >= 0.50:
        return (
            f"Partially correct (F2={f2:.2f}). Ground truth: \"{gt[:200]}\". "
            f"Tip: extract the exact sentence, not a summary. "
            f"F2 is recall-weighted — missing words costs more than extra words."
        )

    return (
        f"Incorrect (F2={f2:.2f}). Ground truth: \"{gt[:200]}\". "
        f"You extracted: \"{extracted[:200]}\". "
        f"Rule: Copy the verbatim sentence from the contract text. Do not rephrase."
    )


@app.get("/tier1/sample", tags=["Tier 1 — Clause Reading"])
def tier1_current():
    if not _t1_samples or _t1_idx >= len(_t1_samples):
        return {"sample": None, "done": True}
    return {"sample": _t1_samples[_t1_idx].model_dump(), "index": _t1_idx, "total": len(_t1_samples)}


# ---------------------------------------------------------------------------
# Tier 2 — Clause Review (backward-compatible OpenEnv routes)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str


@app.post("/reset", tags=["Tier 2 — Clause Review"])
def tier2_reset(req: ResetRequest):
    """Reset the clause review environment (backward-compatible with OpenEnv)."""
    global _t2_env, _t2_task_id
    _t2_env = ContractReviewEnv()
    _t2_task_id = req.task_id
    obs = _t2_env.reset(req.task_id)
    return obs.model_dump()


@app.post("/step", tags=["Tier 2 — Clause Review"])
def tier2_step(action_dict: Dict[str, Any]):
    """Submit a clause review action (backward-compatible with OpenEnv)."""
    if _t2_env is None:
        raise HTTPException(400, "Call /reset first.")
    action = Action(**action_dict)
    obs, reward, done, info = _t2_env.step(action)
    obs_dict = obs.model_dump()

    # Build corrective feedback from ground-truth data in info
    obs_dict["corrective_feedback"] = _build_t2_feedback(
        action_dict, info, reward.model_dump()
    )

    return {
        "observation": obs_dict,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


def _build_t2_feedback(action_dict: Dict[str, Any], info: Dict[str, Any], reward_dict: Dict[str, Any]) -> str:
    """Generate corrective feedback for a Tier 2 clause review step."""
    gt = info.get("current_clause_ground_truth", {})
    if not gt:
        return ""

    action_type = action_dict.get("action_type", "")
    score = reward_dict.get("score", 0.0)

    if action_type == "classify":
        submitted = action_dict.get("clause_type", "?")
        correct = gt.get("clause_type", "?")
        if submitted == correct:
            return f"Correct. This is a '{correct}' clause."
        return (
            f"Incorrect classification. You submitted '{submitted}', "
            f"but this is a '{correct}' clause. "
            f"Key indicator: look for the operative obligation language, not the subject matter."
        )

    if action_type == "rate_severity":
        submitted = action_dict.get("risk_level", "?")
        correct = gt.get("risk_level", "?")
        if submitted == correct:
            return f"Correct risk level: '{correct}'."
        return (
            f"Incorrect risk level. You submitted '{submitted}', correct is '{correct}'. "
            f"Risk escalates when liability is uncapped, obligations are one-sided, "
            f"or the clause conflicts with another in the same contract."
        )

    if action_type == "flag":
        submitted_flags = set(action_dict.get("flags", []))
        correct_flags = set(gt.get("flags", []))
        missed = correct_flags - submitted_flags
        extra = submitted_flags - correct_flags
        parts = []
        if missed:
            parts.append(f"Missed flag(s): {', '.join(missed)}.")
        if extra:
            parts.append(f"False positive flag(s): {', '.join(extra)}.")
        if not parts:
            return "Correct flags identified."
        return " ".join(parts)

    if action_type == "suggest":
        submitted = action_dict.get("suggested_action", "?")
        correct = gt.get("suggested_action", "?")
        if submitted == correct:
            return f"Correct suggested action: '{correct}'."
        return (
            f"Suggested action mismatch. You submitted '{submitted}', "
            f"expected '{correct}'. Higher-risk clauses require escalation "
            f"or rejection rather than simple flagging."
        )

    return f"Step score: {score:.3f}."



@app.get("/state", tags=["Tier 2 — Clause Review"])
def tier2_state():
    """Return current environment state."""
    if _t2_env is None:
        raise HTTPException(400, "No active session. Call /reset first.")
    return _t2_env.state().model_dump()


# ---------------------------------------------------------------------------
# Tier 3 — Dependency Mapping
# ---------------------------------------------------------------------------

class Tier3ResetRequest(BaseModel):
    scenario_index: int = 0
    time_budget: int = 10


@app.post("/tier3/reset", tags=["Tier 3 — Dependency Mapping"])
def tier3_reset(req: Tier3ResetRequest):
    """Load a dependency mapping scenario and return contract summaries."""
    global _t3_env, _t3_scenarios, _t3_idx, _t3_results
    _t3_scenarios = load_tier3_scenarios()
    _t3_results = []
    if not _t3_scenarios:
        raise HTTPException(404, "No scenarios found.")
    idx = req.scenario_index % len(_t3_scenarios)
    _t3_idx = idx
    _t3_env = Tier3MappingEnv(time_budget=req.time_budget)
    obs = _t3_env.reset(_t3_scenarios[idx])
    obs_dict = obs.model_dump()
    obs_dict["instructions"] = (
        "You are given multiple contracts. Your task: identify ALL hidden dependency edges "
        "between clauses across different contracts. Submit a JSON list of dependencies. "
        "Edge types: cascade_trigger, mutual_exclusion, condition_precedent, supersession, temporal_gate."
    )
    return obs_dict


@app.post("/tier3/step", tags=["Tier 3 — Dependency Mapping"])
def tier3_step(action: Tier3Action):
    """Submit a dependency graph and receive precision/recall feedback."""
    if _t3_env is None:
        raise HTTPException(400, "Call /tier3/reset first.")
    obs, result = _t3_env.step(action)
    resp = {"observation": obs.model_dump(), "result": None, "corrective_feedback": ""}
    if result:
        _t3_results.append(result)
        resp["result"] = result.model_dump()
        resp["corrective_feedback"] = _build_t3_feedback(result)
    return resp


def _build_t3_feedback(result) -> str:
    """Corrective feedback for a Tier 3 dependency mapping submission."""
    r = result.recall
    p = result.precision
    missed = getattr(result, "missed_edges", [])
    false_pos = getattr(result, "false_positive_edges", [])

    lines = [f"Recall={r:.2f}, Precision={p:.2f}."]
    if missed:
        edge = missed[0]
        lines.append(
            f"Missed edge: '{edge.get('source_clause','?')}' → '{edge.get('target_clause','?')}' "
            f"({edge.get('edge_type','?')}). "
            f"Tip: check if invoking any clause changes the validity of another."
        )
    if false_pos:
        lines.append(
            f"{len(false_pos)} false positive(s): edges you identified that do not exist "
            f"in ground truth. Only submit edges where you can cite specific clause language."
        )
    if r >= 0.9 and p >= 0.9:
        lines = ["Excellent. All critical dependency edges found with high precision."]
    return " ".join(lines)


@app.get("/tier3/score", tags=["Tier 3 — Dependency Mapping"])
def tier3_score():
    if not _t3_results:
        return {"score": None, "message": "No completed scenarios yet."}
    return grade_tier3_batch(_t3_results).model_dump()


# ---------------------------------------------------------------------------
# Tiers 4-6 — LexDomino Crisis (backward-compatible /cascade/* routes)
# ---------------------------------------------------------------------------

class CascadeResetRequest(BaseModel):
    task_id: str
    scenario_index: int = 0


@app.post("/cascade/reset", tags=["Tiers 4-6 — Crisis Management"])
def cascade_reset(req: CascadeResetRequest):
    """Reset a LexDomino crisis scenario."""
    global _cascade_env
    _cascade_env = LexDominoCrisisEnv()
    obs = _cascade_env.reset(req.task_id, req.scenario_index)
    obs_dict = obs.model_dump()
    obs_dict["corrective_feedback"] = (
        "Episode started. Review your inbox first. Identify all deadlines. "
        "Cross-reference contracts before taking financial actions."
    )
    return obs_dict


@app.post("/cascade/step", tags=["Tiers 4-6 — Crisis Management"])
def cascade_step(action: CascadeAction):
    """Submit a crisis management action."""
    if _cascade_env is None:
        raise HTTPException(400, "Call /cascade/reset first.")
    obs, reward, done, info = _cascade_env.step(action)
    obs_dict = obs.model_dump()
    # Build corrective feedback from cascade info
    obs_dict["corrective_feedback"] = _build_cascade_feedback(action, info, reward.model_dump())
    return {
        "observation": obs_dict,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


def _build_cascade_feedback(action: CascadeAction, info: Dict[str, Any], reward_dict: Dict[str, Any]) -> str:
    """Generate corrective feedback for a crisis management step."""
    score = reward_dict.get("score", 0.0)
    reason = reward_dict.get("reason", "")
    action_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)

    bankruptcy = info.get("bankruptcy", False)
    if bankruptcy:
        return (
            "BANKRUPTCY. The company ran out of cash or violated its debt covenant. "
            "Critical rule: always protect cash first. Draw credit facility before "
            "paying any penalty. Never pay a demand without verifying its legal validity."
        )

    if score > 0.1:
        return f"Good action ({action_type}). {reason}" if reason else f"Good action ({action_type})."

    if action_type in ("invoke_force_majeure",):
        return (
            f"Low reward for '{action_type}'. "
            f"Check: did you file an insurance claim FIRST? "
            f"Invoking Force Majeure before filing insurance may void your coverage (mutual exclusion clause)."
        )

    if action_type == "pay_penalty":
        return (
            f"Low reward for '{action_type}'. "
            f"Check: is this penalty legally enforceable? "
            f"Use 'assess_counterparty_risk' and 'cross_reference_contracts' before paying any demand. "
            f"Aggressive counterparties often claim penalties that are time-barred or superseded."
        )

    if action_type == "advance_day" and score <= 0:
        return (
            "Advancing the day with active unaddressed deadlines loses time. "
            "Review 'active_deadlines' in the observation and act on the most urgent one."
        )

    return f"Action '{action_type}' scored {score:.3f}. {reason}" if reason else f"Action '{action_type}' scored {score:.3f}."



@app.get("/cascade/state", tags=["Tiers 4-6 — Crisis Management"])
def cascade_state():
    if _cascade_env is None:
        raise HTTPException(400, "No active crisis session.")
    return _cascade_env.state().model_dump()


@app.get("/cascade/actions", tags=["Tiers 4-6 — Crisis Management"])
def cascade_actions():
    return {"available_actions": [a.value for a in CascadeActionType]}


@app.get("/cascade/deadlines", tags=["Tiers 4-6 — Crisis Management"])
def cascade_deadlines():
    if _cascade_env is None:
        raise HTTPException(400, "No active session.")
    state = _cascade_env.state()
    return {"deadlines": [d.model_dump() for d in _cascade_env.deadlines]}


@app.get("/cascade/metrics", tags=["Tiers 4-6 — Crisis Management"])
def cascade_metrics():
    if _cascade_env is None:
        raise HTTPException(400, "No active session.")
    state = _cascade_env.state()
    return {
        "cash_balance": _cascade_env.financial_state.cash_balance,
        "bankruptcy": _cascade_env.bankruptcy,
        "current_day": _cascade_env.current_day,
        "step_number": _cascade_env.step_number,
    }


# ---------------------------------------------------------------------------
# Legal IQ — Composite Score
# ---------------------------------------------------------------------------

class LegalIQRequest(BaseModel):
    model_name: str = "agent"
    t1_score: float = 0.0
    t2_score: float = 0.0
    t3_score: float = 0.0
    t4_score: float = 0.0
    t5_score: float = 0.0
    t6_score: float = 0.0


@app.post("/legal_iq", tags=["Legal IQ"])
def compute_iq(req: LegalIQRequest):
    """Compute the Legal IQ composite score from tier scores."""
    score = compute_legal_iq(
        t1_score=req.t1_score, t2_score=req.t2_score, t3_score=req.t3_score,
        t4_score=req.t4_score, t5_score=req.t5_score, t6_score=req.t6_score,
    )
    _iq_scores[req.model_name] = score.model_dump()
    return score.model_dump()


@app.get("/legal_iq/leaderboard", tags=["Legal IQ"])
def leaderboard():
    """Return all scored models ranked by Legal IQ."""
    ranked = sorted(_iq_scores.items(), key=lambda x: -x[1]["legal_iq"])
    return {
        "leaderboard": [
            {"rank": i + 1, "model": m, **{k: v for k, v in d.items()}}
            for i, (m, d) in enumerate(ranked)
        ]
    }


@app.get("/legal_iq/weights", tags=["Legal IQ"])
def weights():
    return {
        "formula": "Legal_IQ = 0.15·T1 + 0.15·T2 + 0.20·T3 + 0.50·(0.25·T4 + 0.35·T5 + 0.40·T6)",
        "rationale": "Crisis management (50%) is weighted highest — impossible to fake with legal fine-tuning.",
        "tier_weights": {"T1": "15%", "T2": "15%", "T3": "20%", "T4-6": "50%"},
        "crisis_sub_weights": {"T4": "12.5%", "T5": "17.5%", "T6": "20%"},
        "scoring": "Pure math. Zero LLM judges.",
    }


# ---------------------------------------------------------------------------
# Adversarial Probes
# ---------------------------------------------------------------------------

@app.get("/probes", tags=["Adversarial Probes"])
def list_probes():
    """List all 10 adversarial probe scenarios and their failure modes."""
    probe_dir = os.path.join("data", "probes")
    if not os.path.exists(probe_dir):
        return {"probes": [], "total": 0}
    probes = []
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(probe_dir, fname), encoding="utf-8") as fp:
            d = json.load(fp)
        probes.append({
            "probe_id": d.get("probe_id"),
            "failure_mode": d.get("failure_mode"),
            "description": d.get("description"),
        })
    return {"probes": probes, "total": len(probes)}


@app.get("/probes/{probe_id}", tags=["Adversarial Probes"])
def get_probe(probe_id: str):
    """Return the full scenario data for one adversarial probe."""
    path = os.path.join("data", "probes", f"{probe_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Probe not found: {probe_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/probes/run", tags=["Adversarial Probes"])
def run_probe(body: Dict[str, Any]):
    """
    Run a single adversarial probe in-process.
    Body: {"probe_id": "probe_fm_void", "strategy": "deadline_first"}
    """
    from probe_runner import ProbeRunner
    probe_id = body.get("probe_id")
    strategy = body.get("strategy", "deadline_first")
    if not probe_id:
        raise HTTPException(400, "probe_id is required.")
    runner = ProbeRunner()
    result = runner.run_probe(probe_id, strategy=strategy, verbose=False)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_PORT)
