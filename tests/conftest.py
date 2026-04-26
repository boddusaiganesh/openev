"""
Shared pytest fixtures for the Contract Clause Review test suite.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ContractReviewEnv
from models import (
    Action,
    ActionType,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import get_task_config, list_task_ids


@pytest.fixture
def env():
    """Fresh environment instance."""
    return ContractReviewEnv()


@pytest.fixture
def env_task1(env):
    """Environment reset to task_1_easy."""
    env.reset("task_1_easy")
    return env


@pytest.fixture
def env_task2(env):
    """Environment reset to task_2_medium."""
    env.reset("task_2_medium")
    return env


@pytest.fixture
def env_task3(env):
    """Environment reset to task_3_hard."""
    env.reset("task_3_hard")
    return env


@pytest.fixture(params=["task_1_easy", "task_2_medium", "task_3_hard"])
def env_all_tasks(request, env):
    """Environment reset to each task (parametrized)."""
    env.reset(request.param)
    return env


@pytest.fixture
def all_task_ids():
    """List of all valid task IDs."""
    return list_task_ids()


def run_optimal_trajectory(env: ContractReviewEnv, task_id: str) -> float:
    """Run a perfect trajectory using ground truth. Returns grader score."""
    obs = env.reset(task_id)
    cfg = env.task_config
    required = cfg.required_action_types

    for i in range(obs.total_clauses):
        if env.done:
            break
        gt = env.scenario.clauses[i]

        for at in required:
            if env.done:
                break
            if at == ActionType.CLASSIFY:
                env.step(Action(action_type=at, clause_type=gt.clause_type))
            elif at == ActionType.RATE_SEVERITY:
                env.step(Action(action_type=at, risk_level=gt.risk_level))
            elif at == ActionType.FLAG:
                env.step(Action(action_type=at, flags=gt.issues))
            elif at == ActionType.SUGGEST:
                env.step(Action(action_type=at, suggested_action=gt.recommended_action))
            elif at == ActionType.REASON:
                env.step(Action(action_type=at, reasoning=" ".join(gt.reasoning_keywords)))

        if not env.done and i < obs.total_clauses - 1:
            env.step(Action(action_type=ActionType.NEXT_CLAUSE))

    if not env.done:
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

    return env.grader_result.score if env.grader_result else 0.0
