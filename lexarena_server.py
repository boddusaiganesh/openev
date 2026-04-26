"""
LexArena — Unified 6-Tier FastAPI Server
==========================================
Single entry point for the HF Space. Serves:
  - Tier 2 (clause review): /reset  /step  /state
  - Tier 1 (extraction):    /tier1/reset  /tier1/step
  - Tier 3 (dependency):    /tier3/reset  /tier3/step
  - Legal IQ composite:     /legal_iq  /legal_iq/leaderboard
  - Health/info:            /  /health  /info  /curriculum
  - Adversarial probes:     /probes  /probes/run  (scaffold)

Notes:
  - Backward-compatible with original /reset /step /state endpoints
  - asyncio.Lock per tier to allow concurrent tier usage
  - Tier 4/5/6 (LexDomino cascade) endpoints are defined but return 501
    until the cascade environment is implemented
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment import ContractReviewEnv
from models import (
    Action,
    Reward,
    StepResponse,
    Tier1Action,
    Tier1ExtractionSample,
    Tier1Observation,
    Tier3Action,
    Tier3Observation,
    Tier3ScenarioData,
)
from tasks import list_task_ids, TASK_REGISTRY
from graders import grade_tier1_extraction, grade_tier3_dependency
from lexarena_scorer import compute_legal_iq, score_from_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lexarena.server")

DATA_DIR = os.getenv("DATA_DIR", "data")


# ===========================================================================
# App lifecycle
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("LexArena 6-Tier Server — Starting")
    logger.info(f"Tier 2 tasks: {list_task_ids()}")

    # Tier 2 environment
    app.state.t2_lock = asyncio.Lock()
    app.state.t2_env  = ContractReviewEnv(data_dir=DATA_DIR)

    # Validate all Tier 2 tasks on startup
    test_env = ContractReviewEnv(data_dir=DATA_DIR)
    for tid in list_task_ids():
        try:
            obs = test_env.reset(tid)
            logger.info(f"  T2 {tid}: OK ({obs.total_clauses} clauses)")
        except Exception as exc:
            logger.error(f"  T2 {tid}: FAILED ({exc})")
            raise RuntimeError(f"Task {tid} failed validation: {exc}")

    # Tier 1 state (stateless, per-request; keep last sample for /state equiv)
    app.state.t1_lock        = asyncio.Lock()
    app.state.t1_current:    Optional[Tier1ExtractionSample] = None
    app.state.t1_done:       bool = False

    # Tier 3 state (stateless, per-request)
    app.state.t3_lock        = asyncio.Lock()
    app.state.t3_current:    Optional[Tier3ScenarioData] = None
    app.state.t3_done:       bool = False

    # In-memory leaderboard
    app.state.leaderboard: List[Dict[str, Any]] = []

    logger.info("All tasks validated. LexArena server ready.")
    logger.info("=" * 60)

    try:
        yield
    finally:
        logger.info("LexArena Server — Shutting down")


app = FastAPI(
    title="LexArena — Complete Legal Intelligence Benchmark",
    description=(
        "6-tier legal AI benchmark: clause extraction → risk classification → "
        "dependency mapping → crisis management. "
        "Composite Legal IQ score. Zero LLM-as-judge."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response     = await call_next(request)
        elapsed_ms   = (time.time() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(
            "%s %s -> 500 (%.1fms) %s",
            request.method, request.url.path, elapsed_ms, exc,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ===========================================================================
# Health / Info
# ===========================================================================

@app.get("/")
@app.get("/health")
async def health():
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "lexarena",
            "version": "3.0.0",
            "spec": "openenv",
            "tiers": [1, 2, 3, 4, 5, 6],
            "t2_tasks": list_task_ids(),
            "legal_iq_formula": (
                "0.15·T1 + 0.15·T2 + 0.20·T3 + "
                "0.50·(0.25·T4 + 0.35·T5 + 0.40·T6)"
            ),
        },
        status_code=200,
    )


@app.get("/info")
async def info():
    tasks_info = []
    for tid, tc in TASK_REGISTRY.items():
        tasks_info.append({
            "task_id":         tc.task_id,
            "name":            tc.name,
            "difficulty":      tc.difficulty.value,
            "description":     tc.description,
            "max_steps":       tc.max_steps,
            "required_actions": [a.value for a in tc.required_action_types],
            "grader_weights":  tc.grader_weights,
            "tier":            getattr(tc, "task_tier", 2),
        })
    return JSONResponse(content={
        "environment": "lexarena",
        "version": "3.0.0",
        "tiers": 6,
        "t2_tasks": tasks_info,
        "endpoints": {
            "health":     "GET /",
            "info":       "GET /info",
            "curriculum": "GET /curriculum",
            "t2_reset":   "POST /reset",
            "t2_step":    "POST /step",
            "t2_state":   "GET /state",
            "t1_reset":   "POST /tier1/reset",
            "t1_step":    "POST /tier1/step",
            "t3_reset":   "POST /tier3/reset",
            "t3_step":    "POST /tier3/step",
            "legal_iq":   "POST /legal_iq",
            "leaderboard": "GET /legal_iq/leaderboard",
            "probes":     "GET /probes",
            "probe_run":  "POST /probes/run",
        },
    }, status_code=200)


@app.get("/curriculum")
async def curriculum():
    """Describe the recommended curriculum progression."""
    return JSONResponse(content={
        "progression": [
            {"tier": 1, "task": "tier1_clause_reading",    "endpoint": "/tier1/reset",   "difficulty": "extraction"},
            {"tier": 2, "task": "task_1_easy",             "endpoint": "/reset",          "difficulty": "easy"},
            {"tier": 2, "task": "task_2_medium",           "endpoint": "/reset",          "difficulty": "medium"},
            {"tier": 2, "task": "task_3_hard",             "endpoint": "/reset",          "difficulty": "hard"},
            {"tier": 3, "task": "tier3_dependency_mapping", "endpoint": "/tier3/reset",   "difficulty": "novel"},
            {"tier": 4, "task": "task_4_cascade_easy",     "endpoint": "/cascade/reset",  "difficulty": "cascade_easy"},
            {"tier": 5, "task": "task_5_cascade_medium",   "endpoint": "/cascade/reset",  "difficulty": "cascade_medium"},
            {"tier": 6, "task": "task_6_cascade_hard",     "endpoint": "/cascade/reset",  "difficulty": "cascade_hard"},
        ],
        "tip": (
            "Start with Tier 1 and 2. Only advance to Tier 3 once your model "
            "achieves >0.50 on task_2_medium. Tiers 4-6 require a capable "
            "strategic reasoner (generally >7B parameters)."
        ),
    })


# ===========================================================================
# Tier 2 — Clause Review (backward-compatible)
# ===========================================================================

@app.post("/reset")
async def t2_reset(request: Optional[Dict[str, Any]] = None):
    env  = app.state.t2_env
    lock = app.state.t2_lock
    try:
        request    = request or {}
        task_id    = request.get("task_id") or (list_task_ids()[0] if list_task_ids() else "task_1_easy")
        sc_idx     = request.get("scenario_index")
        logger.info("T2 RESET: task_id=%s", task_id)
        async with lock:
            obs = env.reset(task_id, scenario_index=sc_idx)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("T2 RESET FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def t2_step(action_data: Optional[Dict[str, Any]] = None):
    env  = app.state.t2_env
    lock = app.state.t2_lock
    action_data = action_data or {}

    if not action_data:
        async with lock:
            if env.task_config is None:
                obs = env._build_empty_observation()
            else:
                obs = env._build_observation("No action provided.")
        return StepResponse(
            observation=obs,
            reward=Reward(score=0.0, message="No action provided."),
            done=env.done,
            info=env._build_info(),
        ).model_dump()

    if env.task_config is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    if "action_type" not in action_data:
        raise HTTPException(status_code=400, detail="Missing action_type.")

    try:
        action = Action(**action_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {e}")

    try:
        async with lock:
            obs, reward, done, step_info = env.step(action)
        if done and env.grader_result:
            logger.info(
                "T2 EPISODE DONE: task=%s score=%.4f",
                env.task_config.task_id if env.task_config else "?",
                float(env.grader_result.score),
            )
        return StepResponse(observation=obs, reward=reward, done=done, info=step_info).model_dump()
    except Exception as e:
        logger.error("T2 STEP FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def t2_state():
    env  = app.state.t2_env
    lock = app.state.t2_lock
    try:
        async with lock:
            env_state = env.state()
        return env_state.model_dump()
    except Exception as e:
        logger.error("T2 STATE FAILED: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# Tier 1 — Clause Extraction
# ===========================================================================

def _load_tier1_samples() -> List[Dict[str, Any]]:
    """Load Tier 1 extraction samples from data/tier1_extraction/."""
    path = os.path.join(DATA_DIR, "tier1_extraction", "samples.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/tier1/reset")
async def tier1_reset(request: Optional[Dict[str, Any]] = None):
    request = request or {}
    samples = _load_tier1_samples()
    if not samples:
        raise HTTPException(
            status_code=503,
            detail="Tier 1 samples not loaded. Ensure data/tier1_extraction/samples.json exists.",
        )
    idx = int(request.get("sample_index", 0)) % len(samples)
    sample_dict = samples[idx]
    sample = Tier1ExtractionSample(**sample_dict)

    async with app.state.t1_lock:
        app.state.t1_current = sample
        app.state.t1_done    = False

    obs = Tier1Observation(
        sample_id         = sample.sample_id,
        contract_name     = sample.contract_name,
        question_category = sample.question_category,
        question          = sample.question,
        context           = sample.context,
        has_answer        = sample.has_answer,
    )
    logger.info("T1 RESET: sample_id=%s category=%s", sample.sample_id, sample.question_category)
    return obs.model_dump()


@app.post("/tier1/step")
async def tier1_step(action_data: Optional[Dict[str, Any]] = None):
    action_data = action_data or {}
    async with app.state.t1_lock:
        sample = app.state.t1_current
        done   = app.state.t1_done

    if sample is None:
        raise HTTPException(status_code=400, detail="No active Tier 1 episode. Call /tier1/reset first.")
    if done:
        return {"done": True, "message": "Tier 1 episode already complete."}

    extracted = action_data.get("extracted_text", "")
    if not extracted:
        raise HTTPException(status_code=400, detail="extracted_text is required.")

    result = grade_tier1_extraction(sample, extracted)

    async with app.state.t1_lock:
        app.state.t1_done = True

    logger.info("T1 STEP: sample=%s score=%.4f", sample.sample_id, float(result.score))
    return {
        "done":       True,
        "score":      float(result.score),
        "breakdown":  result.breakdown,
        "message":    result.message,
        "corrective_feedback": result.message,
    }


# ===========================================================================
# Tier 3 — Dependency Graph Mapping
# ===========================================================================

def _load_tier3_scenarios() -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, "tier3_dependency", "scenarios.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/tier3/reset")
async def tier3_reset(request: Optional[Dict[str, Any]] = None):
    request   = request or {}
    scenarios = _load_tier3_scenarios()
    if not scenarios:
        raise HTTPException(
            status_code=503,
            detail="Tier 3 scenarios not loaded. Ensure data/tier3_dependency/scenarios.json exists.",
        )
    idx  = int(request.get("scenario_index", 0)) % len(scenarios)
    s    = scenarios[idx]
    task_meta = s.get("task_meta", {})
    max_steps = task_meta.get("max_steps", 10)

    async with app.state.t3_lock:
        app.state.t3_current = s
        app.state.t3_done    = False

    obs = Tier3Observation(
        scenario_id     = s.get("scenario_id", f"scenario_{idx}"),
        contracts       = s.get("contracts", []),
        steps_remaining = max_steps,
    )
    logger.info("T3 RESET: scenario_id=%s", obs.scenario_id)
    return obs.model_dump()


@app.post("/tier3/step")
async def tier3_step(action_data: Optional[Dict[str, Any]] = None):
    action_data = action_data or {}
    async with app.state.t3_lock:
        scenario = app.state.t3_current
        done     = app.state.t3_done

    if scenario is None:
        raise HTTPException(status_code=400, detail="No active Tier 3 episode. Call /tier3/reset first.")
    if done:
        return {"done": True, "message": "Tier 3 episode already complete."}

    predicted_edges   = action_data.get("dependencies", [])
    ground_truth_edges = scenario.get("ground_truth_edges", [])

    result = grade_tier3_dependency(predicted_edges, ground_truth_edges)

    async with app.state.t3_lock:
        app.state.t3_done = True

    logger.info(
        "T3 STEP: scenario=%s score=%.4f",
        scenario.get("scenario_id", "?"),
        float(result.score),
    )
    return {
        "done":              True,
        "score":             float(result.score),
        "breakdown":         result.breakdown,
        "message":           result.message,
        "corrective_feedback": result.message,
    }


# ===========================================================================
# Tiers 4–6 — Cascade (scaffold — awaiting cascade_environment.py)
# ===========================================================================

@app.post("/cascade/reset")
async def cascade_reset(request: Optional[Dict[str, Any]] = None):
    raise HTTPException(
        status_code=501,
        detail=(
            "Tiers 4-6 (LexDomino cascade) are not yet implemented. "
            "Complete Tiers 1-3 first. Set TIERS=1,2,3 to skip cascade."
        ),
    )


@app.post("/cascade/step")
async def cascade_step(action_data: Optional[Dict[str, Any]] = None):
    raise HTTPException(status_code=501, detail="Cascade environment not yet implemented.")


@app.get("/cascade/state")
async def cascade_state():
    raise HTTPException(status_code=501, detail="Cascade environment not yet implemented.")


@app.get("/cascade/actions")
async def cascade_actions():
    raise HTTPException(status_code=501, detail="Cascade environment not yet implemented.")


@app.get("/cascade/deadlines")
async def cascade_deadlines():
    raise HTTPException(status_code=501, detail="Cascade environment not yet implemented.")


@app.get("/cascade/metrics")
async def cascade_metrics():
    raise HTTPException(status_code=501, detail="Cascade environment not yet implemented.")


# ===========================================================================
# Legal IQ Composite Score
# ===========================================================================

@app.post("/legal_iq")
async def legal_iq_compute(payload: Optional[Dict[str, Any]] = None):
    """
    Compute Legal IQ from per-tier scores.

    POST body: {"results": [...run_task result dicts...], "model_name": "..."}
    OR:        {"t1": 0.7, "t2": 0.6, "t3": 0.4, "t4": 0.0, "t5": 0.0, "t6": 0.0}
    """
    payload    = payload or {}
    model_name = payload.get("model_name", "")

    if "results" in payload:
        iq = score_from_results(payload["results"], model_name=model_name)
    else:
        iq = compute_legal_iq(
            t1_score   = float(payload.get("t1", 0.0)),
            t2_score   = float(payload.get("t2", 0.0)),
            t3_score   = float(payload.get("t3", 0.0)),
            t4_score   = float(payload.get("t4", 0.0)),
            t5_score   = float(payload.get("t5", 0.0)),
            t6_score   = float(payload.get("t6", 0.0)),
            model_name = model_name,
        )

    # Auto-append to leaderboard if model_name provided
    if model_name:
        entry = iq.model_dump()
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        app.state.leaderboard.append(entry)
        # Keep only top 100
        app.state.leaderboard = sorted(
            app.state.leaderboard, key=lambda x: x["legal_iq"], reverse=True
        )[:100]

    logger.info("LEGAL_IQ: model=%s iq=%.4f label=%s", model_name, iq.legal_iq, iq.label)
    return iq.model_dump()


@app.get("/legal_iq/leaderboard")
async def legal_iq_leaderboard():
    return JSONResponse(content={
        "leaderboard": app.state.leaderboard,
        "count": len(app.state.leaderboard),
    })


# ===========================================================================
# Adversarial Probes (scaffold)
# ===========================================================================

PROBE_DEFINITIONS = [
    {"id": "probe_fm_void",            "description": "Invoking FM voids insurance (mutual exclusion trap)"},
    {"id": "probe_sycophancy",         "description": "Aggressive counterparty demand that is legally invalid"},
    {"id": "probe_covenant_blindness", "description": "Missing that penalty payment breaches debt covenant"},
    {"id": "probe_cross_default",      "description": "Judgment in Contract A triggers cross-default in B"},
    {"id": "probe_deadline_stack",     "description": "Three deadlines in 48 hours requiring prioritisation"},
    {"id": "probe_key_person_chain",   "description": "3-hop dependency: bonus → departure → license loss"},
    {"id": "probe_false_urgency",      "description": "Counterparty creates fake urgency for non-binding demand"},
    {"id": "probe_supersession",       "description": "Later clause overrides earlier clause — agent must find it"},
    {"id": "probe_compound_shock",     "description": "Primary crisis + secondary shock 10 days later"},
    {"id": "probe_lazy_reader",        "description": "Scenario where 'No related clause' is always wrong"},
]


@app.get("/probes")
async def probes_list():
    return JSONResponse(content={"probes": PROBE_DEFINITIONS, "count": len(PROBE_DEFINITIONS)})


@app.post("/probes/run")
async def probe_run(payload: Optional[Dict[str, Any]] = None):
    payload  = payload or {}
    probe_id = payload.get("probe_id", "")
    probe    = next((p for p in PROBE_DEFINITIONS if p["id"] == probe_id), None)
    if not probe:
        raise HTTPException(status_code=404, detail=f"Probe '{probe_id}' not found.")
    raise HTTPException(
        status_code=501,
        detail=f"Probe '{probe_id}' is defined but not yet implemented. Implement cascade first.",
    )


# ===========================================================================
# Misc utility
# ===========================================================================

@app.options("/")
async def options_root():
    return JSONResponse(content={"ok": True}, status_code=200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
