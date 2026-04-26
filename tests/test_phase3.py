"""
Phase 3 — Validation Tests
Run: python -m pytest tests/test_phase3.py -v
"""

import subprocess
import sys
import time
import socket

import httpx
import pytest

from models import (
    ActionType,
    ClauseActionRecord,
    Difficulty,
    EpisodeMeta,
    GraderResult,
    RiskLevel,
    SuggestedActionType,
)
from tasks import TASK_REGISTRY, TASK_INSTRUCTIONS, get_task_config
from graders import grade_episode


class TestTaskRegistry:
    def test_three_tasks_registered(self):
        assert len(TASK_REGISTRY) == 3

    def test_easy_task(self):
        task = TASK_REGISTRY["task_1_easy"]
        assert task.difficulty == Difficulty.EASY
        assert task.max_steps == 10

    def test_medium_task(self):
        task = TASK_REGISTRY["task_2_medium"]
        assert task.difficulty == Difficulty.MEDIUM
        assert task.max_steps == 20

    def test_hard_task(self):
        task = TASK_REGISTRY["task_3_hard"]
        assert task.difficulty == Difficulty.HARD
        assert task.max_steps == 40

    def test_task_instructions_exist(self):
        assert "task_1_easy" in TASK_INSTRUCTIONS
        assert "task_2_medium" in TASK_INSTRUCTIONS
        assert "task_3_hard" in TASK_INSTRUCTIONS

    def test_get_task_config(self):
        config = get_task_config("task_1_easy")
        assert config is not None
        assert config.task_id == "task_1_easy"


class TestGraderTask1:
    def test_grade_perfect_task_1(self):
        from models import ClauseGroundTruth

        gt = [
            ClauseGroundTruth(
                text="test",
                clause_type="confidentiality",
                risk_level=RiskLevel.LOW,
                issues=[],
                recommended_action=SuggestedActionType.ACCEPT_AS_IS,
                reasoning_keywords=["confidential"],
            )
        ]

        records = [
            ClauseActionRecord(
                clause_index=0,
                classify_action="confidentiality",
                action_count=1,
            )
        ]

        meta = EpisodeMeta(
            total_steps=4,
            max_steps=10,
            clauses_reviewed=1,
            total_clauses=1,
            completed_normally=True,
        )

        task_config = TASK_REGISTRY["task_1_easy"]
        result = grade_episode("task_1_easy", records, gt, task_config, meta)

        assert result.score > 0.8

    def test_grade_wrong_task_1(self):
        from models import ClauseGroundTruth

        gt = [
            ClauseGroundTruth(
                text="test",
                clause_type="confidentiality",
                risk_level=RiskLevel.LOW,
                issues=[],
                recommended_action=SuggestedActionType.ACCEPT_AS_IS,
                reasoning_keywords=["confidential"],
            )
        ]

        records = [
            ClauseActionRecord(
                clause_index=0,
                classify_action="termination",
                action_count=1,
            )
        ]

        meta = EpisodeMeta(
            total_steps=4,
            max_steps=10,
            clauses_reviewed=1,
            total_clauses=1,
            completed_normally=True,
        )

        task_config = TASK_REGISTRY["task_1_easy"]
        result = grade_episode("task_1_easy", records, gt, task_config, meta)

        assert result.score < 0.5


class TestGraderTask2:
    def test_grade_perfect_task_2(self):
        from models import ClauseGroundTruth

        gt = [
            ClauseGroundTruth(
                text="test",
                clause_type="indemnification",
                risk_level=RiskLevel.MEDIUM,
                issues=["one_sided_obligation"],
                recommended_action=SuggestedActionType.REQUEST_MODIFICATION,
                reasoning_keywords=["indemnify"],
            )
        ]

        records = [
            ClauseActionRecord(
                clause_index=0,
                classify_action="indemnification",
                risk_action=RiskLevel.MEDIUM,
                flag_action=["one_sided_obligation"],
                action_count=3,
            )
        ]

        meta = EpisodeMeta(
            total_steps=6,
            max_steps=20,
            clauses_reviewed=1,
            total_clauses=1,
            completed_normally=True,
        )

        task_config = TASK_REGISTRY["task_2_medium"]
        result = grade_episode("task_2_medium", records, gt, task_config, meta)

        assert result.score > 0.8


class TestGraderTask3:
    def test_grade_perfect_task_3(self):
        from models import ClauseGroundTruth

        gt = [
            ClauseGroundTruth(
                text="test",
                clause_type="indemnification",
                risk_level=RiskLevel.HIGH,
                issues=["one_sided_obligation", "overly_broad_scope"],
                recommended_action=SuggestedActionType.REQUEST_MODIFICATION,
                reasoning_keywords=["broad", "indemnify"],
            )
        ]

        records = [
            ClauseActionRecord(
                clause_index=0,
                classify_action="indemnification",
                risk_action=RiskLevel.HIGH,
                flag_action=["one_sided_obligation", "overly_broad_scope"],
                suggest_action=SuggestedActionType.REQUEST_MODIFICATION,
                reason_action="This clause is overly broad and creates one-sided obligations.",
                action_count=5,
            )
        ]

        meta = EpisodeMeta(
            total_steps=10,
            max_steps=40,
            clauses_reviewed=1,
            total_clauses=1,
            completed_normally=True,
        )

        task_config = TASK_REGISTRY["task_3_hard"]
        result = grade_episode("task_3_hard", records, gt, task_config, meta)

        assert result.score > 0.8


class TestGraderPenalties:
    def test_invalid_action_penalty(self):
        from models import ClauseGroundTruth

        gt = [
            ClauseGroundTruth(
                text="test",
                clause_type="confidentiality",
                risk_level=RiskLevel.LOW,
                issues=[],
                recommended_action=SuggestedActionType.ACCEPT_AS_IS,
                reasoning_keywords=["confidential"],
            )
        ]

        records = [
            ClauseActionRecord(
                clause_index=0,
                classify_action="confidentiality",
                action_count=1,
            )
        ]

        meta = EpisodeMeta(
            total_steps=4,
            max_steps=10,
            total_invalid_actions=2,
            clauses_reviewed=1,
            total_clauses=1,
            completed_normally=True,
        )

        task_config = TASK_REGISTRY["task_1_easy"]
        result = grade_episode("task_1_easy", records, gt, task_config, meta)

        assert "invalid_actions" in result.penalties
        assert result.penalties["invalid_actions"] < 0


class TestFullEpisodeWithGrader:
    def test_complete_episode_task_1(self):
        from environment import ContractReviewEnv
        from models import Action

        env = ContractReviewEnv()
        env.reset("task_1_easy")

        done = False
        while not done:
            obs, reward, done, info = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
            )
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        obs, reward, done, info = env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert done is True
        assert "grader_result" in info
        assert info["grader_result"]["score"] > 0

    def test_complete_episode_task_2(self):
        from environment import ContractReviewEnv
        from models import Action

        env = ContractReviewEnv()
        env.reset("task_2_medium")

        done = False
        while not done:
            obs, reward, done, info = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="indemnification")
            )
            if not done:
                obs, reward, done, info = env.step(
                    Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.MEDIUM)
                )
            if not done:
                obs, reward, done, info = env.step(
                    Action(action_type=ActionType.FLAG, flags=["one_sided_obligation"])
                )
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        obs, reward, done, info = env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert done is True
        assert "grader_result" in info


class TestStrictGraderScoreBounds:
    def test_all_tasks_emit_strict_open_interval_scores(self):
        from environment import ContractReviewEnv
        from models import Action

        env = ContractReviewEnv()
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            env.reset(tid)
            _, _, _, info = env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            score = info.get("grader_score")
            assert score is not None
            assert 0.0 < score < 1.0


class TestServerWithGrader:
    @pytest.fixture(scope="class")
    def server_process(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        base_url = f"http://127.0.0.1:{port}"
        ready = False
        for _ in range(40):
            try:
                r = httpx.get(f"{base_url}/", timeout=1.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.25)

        if not ready:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("Server failed to start in time")

        yield {"process": proc, "base_url": base_url}

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def test_grader_in_step_response(self, server_process):
        base_url = server_process["base_url"]
        httpx.post(f"{base_url}/reset", json={"task_id": "task_1_easy"})

        for _ in range(3):
            r = httpx.post(f"{base_url}/step", json={"action_type": "classify", "clause_type": "confidentiality"})
            r = httpx.post(f"{base_url}/step", json={"action_type": "next_clause"})

        r = httpx.post(f"{base_url}/step", json={"action_type": "complete_review"})

        assert r.status_code == 200
        data = r.json()
        assert "info" in data
        assert "grader_result" in data["info"]

    def test_state_includes_grader(self, server_process):
        base_url = server_process["base_url"]
        httpx.post(f"{base_url}/reset", json={"task_id": "task_1_easy"})

        for _ in range(3):
            r = httpx.post(f"{base_url}/step", json={"action_type": "classify", "clause_type": "confidentiality"})
            r = httpx.post(f"{base_url}/step", json={"action_type": "next_clause"})

        r = httpx.post(f"{base_url}/step", json={"action_type": "complete_review"})

        r = httpx.get(f"{base_url}/state")
        assert r.status_code == 200
        data = r.json()
        assert "grader_result" in data
