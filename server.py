"""
Phase 6 — Production Server
Startup validation, graceful error handling, HF Spaces compatible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment import ContractReviewEnv
from models import Action, Reward, StepResponse
from tasks import list_task_ids, TASK_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("contract-review-env")

app = FastAPI(
    title="Contract Clause Review Environment",
    description="OpenEnv environment for real-world contract clause review.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Contract Clause Review Environment — Starting")
    logger.info(f"Tasks: {list_task_ids()}")

    app.state.env_lock = asyncio.Lock()
    app.state.env = ContractReviewEnv()

    test_env = ContractReviewEnv()
    for tid in list_task_ids():
        try:
            obs = test_env.reset(tid)
            logger.info(f"  {tid}: OK ({obs.total_clauses} clauses)")
        except Exception as exc:
            logger.error(f"  {tid}: FAILED ({exc})")
            raise RuntimeError(f"Task {tid} failed validation: {exc}")

    logger.info("All tasks validated. Server ready.")
    logger.info("=" * 60)

    try:
        yield
    finally:
        logger.info("Contract Clause Review Environment — Shutting down")


app.router.lifespan_context = lifespan


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    method = request.method
    path = request.url.path
    try:
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"{method} {path} -> {response.status_code} ({elapsed_ms:.1f}ms)")
        return response
    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"{method} {path} -> 500 ({elapsed_ms:.1f}ms) ERROR: {exc}")
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error."}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


@app.get("/")
async def health():
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "contract-clause-review",
            "version": "1.0.0",
            "tasks": list_task_ids(),
            "spec": "openenv",
        },
        status_code=200,
    )


@app.get("/info")
async def info():
    tasks_info = []
    for tid, tc in TASK_REGISTRY.items():
        tasks_info.append(
            {
                "task_id": tc.task_id,
                "name": tc.name,
                "difficulty": tc.difficulty.value,
                "description": tc.description,
                "max_steps": tc.max_steps,
                "required_actions": [a.value for a in tc.required_action_types],
                "grader_weights": tc.grader_weights,
            }
        )
    return JSONResponse(
        content={
            "environment": "contract-clause-review",
            "version": "1.0.0",
            "tasks": tasks_info,
            "endpoints": {
                "health": "GET /",
                "info": "GET /info",
                "reset": "POST /reset",
                "step": "POST /step",
                "state": "GET /state",
            },
        },
        status_code=200,
    )


@app.post("/reset")
async def reset(request: Dict[str, Any] | None = None):
    env = app.state.env
    lock = app.state.env_lock
    try:
        request = request or {}
        task_id = request.get("task_id")
        if not task_id:
            task_id = list_task_ids()[0] if list_task_ids() else "task_1_easy"

        scenario_index = request.get("scenario_index")
        logger.info(f"RESET: task_id={task_id}")
        async with lock:
            observation = env.reset(task_id, scenario_index=scenario_index)
        logger.info(
            f"RESET OK: clauses={observation.total_clauses} max_steps={observation.max_steps}"
        )
        return observation.model_dump()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"RESET FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action_data: Dict[str, Any] | None = None):
    env = app.state.env
    lock = app.state.env_lock
    action_data = action_data or {}

    # Backward compatibility for earliest phase tests that POST /step with empty body.
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
        raise HTTPException(
            status_code=400, detail="No active episode. Call POST /reset first."
        )

    if "action_type" not in action_data:
        raise HTTPException(
            status_code=400, detail="Invalid action payload: missing action_type"
        )

    try:
        action = Action(**action_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {e}")

    try:
        async with lock:
            observation, reward, done, step_info = env.step(action)
        logger.info(
            f"STEP: {action.action_type.value} -> reward={reward.score:+.3f} done={done} "
            f"clause={observation.clause_index}/{observation.total_clauses}"
        )
        if done and env.grader_result:
            task_id = env.task_config.task_id if env.task_config else "none"
            scenario_index = getattr(env, "_active_scenario_index", None)
            scenario_file = getattr(env, "_active_scenario_file", "")
            grader_score = float(env.grader_result.score)
            logger.info(
                "EPISODE DONE: task_id=%s scenario_index=%s scenario_file=%s grader_score=%.4f in_open_interval=%s",
                task_id,
                scenario_index,
                scenario_file,
                grader_score,
                0.0 < grader_score < 1.0,
            )
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=step_info,
        ).model_dump()
    except Exception as e:
        logger.error(f"STEP FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    env = app.state.env
    lock = app.state.env_lock
    try:
        async with lock:
            env_state = env.state()
        return env_state.model_dump()
    except Exception as e:
        logger.error(f"STATE FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/")
async def options_root():
    return JSONResponse(content={"ok": True}, status_code=200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
