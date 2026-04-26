"""
LexDomino — Cascade Server
FastAPI server exposing the crisis simulator environment.
Runs alongside the existing clause-review server routes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cascade_environment import LexDominoCrisisEnv
from cascade_models import CascadeAction, CascadeActionType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lexdomino.cascade_server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LexDomino — Systemic Legal Cascade Simulator",
    description=(
        "OpenEnv-compliant API for the LexDomino crisis management environment. "
        "Agent must navigate multi-contract legal cascades, meet deadlines, and keep the company solvent."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (single-session server)
_env = LexDominoCrisisEnv()


# ---------------------------------------------------------------------------
# Request / response models (thin wrappers for API ergonomics)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str
    scenario_index: int = 0


class StepRequest(BaseModel):
    action_type: str
    contract_id: Optional[str] = None
    clause_id: Optional[str] = None
    counterparty_id: Optional[str] = None
    amount: Optional[float] = None
    justification: Optional[str] = None
    proposed_terms: Optional[str] = None
    query: Optional[str] = None
    contract_ids: Optional[list] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "environment": "LexDomino Cascade Simulator", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Start a new crisis episode."""
    try:
        obs = _env.reset(req.task_id, req.scenario_index)
        return obs.model_dump()
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Reset error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one action in the crisis simulation."""
    try:
        action_type = CascadeActionType(req.action_type)
    except ValueError:
        valid = [a.value for a in CascadeActionType]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action_type '{req.action_type}'. Valid: {valid}",
        )
    action = CascadeAction(
        action_type=action_type,
        contract_id=req.contract_id,
        clause_id=req.clause_id,
        counterparty_id=req.counterparty_id,
        amount=req.amount,
        justification=req.justification,
        proposed_terms=req.proposed_terms,
        query=req.query,
        contract_ids=req.contract_ids,
    )
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        logger.exception("Step error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return full internal environment state."""
    try:
        s = _env.state()
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/actions")
def list_actions():
    """Return all valid action types with descriptions."""
    return {
        "actions": [
            {"action_type": a.value, "category": _action_category(a)}
            for a in CascadeActionType
        ]
    }


@app.get("/contracts")
def list_contracts():
    """Return contracts loaded in the current episode."""
    if not _env.contracts:
        return {"contracts": []}
    return {
        "contracts": [
            {
                "contract_id": c.contract_id,
                "type": c.contract_type,
                "parties": c.parties,
                "clauses": [
                    {"clause_id": cl.clause_id, "type": cl.clause_type, "text": cl.text}
                    for cl in c.clauses
                ],
            }
            for c in _env.contracts
        ]
    }


@app.get("/deadlines")
def list_deadlines():
    """Return all deadlines in the current episode."""
    if not _env.deadlines:
        return {"deadlines": []}
    return {
        "deadlines": [
            {
                "deadline_id": d.deadline_id,
                "description": d.description,
                "contract_id": d.contract_id,
                "due_day": d.due_day,
                "days_remaining": d.due_day - _env.current_day,
                "penalty_if_missed": d.penalty_if_missed,
                "status": d.status.value,
                "consequence": d.consequence_description,
            }
            for d in sorted(_env.deadlines, key=lambda x: x.due_day)
        ]
    }


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics summary."""
    fs = _env.financial_state
    return {
        "cash_balance": fs.cash_balance if fs else 0.0,
        "covenant_violated": fs.covenant_violated if fs else False,
        "insurance_active": fs.insurance_coverage_active if fs else False,
        "current_day": _env.current_day,
        "step_number": _env.step_number,
        "done": _env.done,
        "bankruptcy": _env.bankruptcy,
    }


def _action_category(action: CascadeActionType) -> str:
    legal = {
        CascadeActionType.INVOKE_FORCE_MAJEURE, CascadeActionType.SEND_BREACH_NOTICE,
        CascadeActionType.FILE_INSURANCE_CLAIM, CascadeActionType.TERMINATE_CONTRACT,
        CascadeActionType.INVOKE_INDEMNIFICATION, CascadeActionType.REQUEST_WAIVER,
    }
    financial = {
        CascadeActionType.PAY_PENALTY, CascadeActionType.NEGOTIATE_PAYMENT_PLAN,
        CascadeActionType.DRAW_CREDIT_FACILITY, CascadeActionType.ACCELERATE_RECEIVABLE,
    }
    investigation = {
        CascadeActionType.CROSS_REFERENCE_CONTRACTS, CascadeActionType.ANALYZE_FINANCIAL_IMPACT,
        CascadeActionType.REVIEW_DEADLINE_STATUS, CascadeActionType.ASSESS_COUNTERPARTY_RISK,
    }
    communication = {
        CascadeActionType.SEND_FORMAL_NOTICE, CascadeActionType.REQUEST_INFORMATION,
        CascadeActionType.PROPOSE_AMENDMENT,
    }
    if action in legal:
        return "legal"
    if action in financial:
        return "financial"
    if action in investigation:
        return "investigation"
    if action in communication:
        return "communication"
    return "control"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
