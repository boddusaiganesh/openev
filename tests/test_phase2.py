"""
Phase 2 — Validation Tests
Run: python -m pytest tests/test_phase2.py -v
"""

import subprocess
import sys

import httpx
import pytest

from models import (
    ACTION_TYPES,
    Action,
    ActionType,
    CLAUSE_TAXONOMY,
    ClauseGroundTruth,
    ContractMeta,
    Difficulty,
    ISSUE_FLAGS,
    Observation,
    Reward,
    RiskLevel,
    ScenarioData,
    GraderResult,
    SuggestedActionType,
)
from environment import ContractReviewEnv, TASK_REGISTRY


class TestActionModel:
    def test_valid_classify(self):
        a = Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        assert a.action_type == ActionType.CLASSIFY
        assert a.clause_type == "confidentiality"

    def test_valid_rate_severity(self):
        a = Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.HIGH)
        assert a.risk_level == RiskLevel.HIGH

    def test_valid_flag(self):
        a = Action(action_type=ActionType.FLAG, flags=["vague_language", "one_sided_obligation"])
        assert len(a.flags) == 2

    def test_valid_suggest(self):
        a = Action(
            action_type=ActionType.SUGGEST,
            suggested_action=SuggestedActionType.REQUEST_MODIFICATION,
        )
        assert a.suggested_action == SuggestedActionType.REQUEST_MODIFICATION

    def test_valid_reason(self):
        a = Action(action_type=ActionType.REASON, reasoning="This clause is too broad.")
        assert "broad" in a.reasoning

    def test_valid_next_clause(self):
        a = Action(action_type=ActionType.NEXT_CLAUSE)
        assert a.action_type == ActionType.NEXT_CLAUSE

    def test_valid_complete_review(self):
        a = Action(action_type=ActionType.COMPLETE_REVIEW)
        assert a.action_type == ActionType.COMPLETE_REVIEW

    def test_invalid_clause_type_rejected(self):
        with pytest.raises(Exception):
            Action(action_type=ActionType.CLASSIFY, clause_type="not_a_real_type")

    def test_invalid_flag_rejected(self):
        with pytest.raises(Exception):
            Action(action_type=ActionType.FLAG, flags=["fake_flag"])

    def test_serialization_roundtrip(self):
        a = Action(action_type=ActionType.CLASSIFY, clause_type="termination")
        d = a.model_dump()
        a2 = Action(**d)
        assert a == a2

    def test_all_taxonomy_values_accepted(self):
        for ct in CLAUSE_TAXONOMY:
            a = Action(action_type=ActionType.CLASSIFY, clause_type=ct)
            assert a.clause_type == ct

    def test_all_issue_flags_accepted(self):
        a = Action(action_type=ActionType.FLAG, flags=ISSUE_FLAGS)
        assert len(a.flags) == len(ISSUE_FLAGS)

    def test_empty_flags_accepted(self):
        a = Action(action_type=ActionType.FLAG, flags=[])
        assert a.flags == []


class TestRewardModel:
    def test_default_reward(self):
        r = Reward()
        assert r.score == 0.0
        assert r.breakdown == {}
        assert r.message == ""

    def test_positive_reward(self):
        r = Reward(score=0.15, breakdown={"accuracy": 0.15}, message="Correct")
        assert r.score == 0.15

    def test_negative_reward(self):
        r = Reward(score=-0.05, breakdown={"penalty": -0.05}, message="Wrong")
        assert r.score == -0.05

    def test_boundary_rewards(self):
        r_max = Reward(score=1.0)
        r_min = Reward(score=-1.0)
        assert r_max.score == 1.0
        assert r_min.score == -1.0

    def test_out_of_range_rejected(self):
        with pytest.raises(Exception):
            Reward(score=1.5)
        with pytest.raises(Exception):
            Reward(score=-1.5)

    def test_serialization_roundtrip(self):
        r = Reward(score=0.1, breakdown={"a": 0.05, "b": 0.05}, message="ok")
        d = r.model_dump()
        r2 = Reward(**d)
        assert r == r2


class TestGraderResultModel:
    def test_score_must_be_strictly_between_zero_and_one(self):
        with pytest.raises(Exception):
            GraderResult(score=0.0)
        with pytest.raises(Exception):
            GraderResult(score=1.0)

    def test_score_inside_open_interval_is_valid(self):
        g = GraderResult(score=0.5)
        assert g.score == pytest.approx(0.5)


class TestObservationModel:
    def test_minimal_observation(self):
        obs = Observation(
            task_id="task_1_easy",
            step_number=0,
            max_steps=10,
            clause_text="Some clause text.",
            clause_index=0,
            total_clauses=3,
            contract_type="NDA",
            parties=["A", "B"],
            jurisdiction="DE",
            instructions="Classify.",
            available_actions=["classify"],
        )
        assert obs.done is False
        assert obs.accumulated_score == 0.0


class TestClauseGroundTruth:
    def test_valid_ground_truth(self):
        gt = ClauseGroundTruth(
            text="Some clause.",
            clause_type="confidentiality",
            risk_level=RiskLevel.LOW,
            issues=[],
            recommended_action=SuggestedActionType.ACCEPT_AS_IS,
            reasoning_keywords=["confidential"],
        )
        assert gt.clause_type == "confidentiality"
        assert gt.risk_level == RiskLevel.LOW


class TestScenarioData:
    def test_valid_scenario(self):
        meta = ContractMeta(
            contract_type="NDA",
            parties=["A", "B"],
            jurisdiction="DE",
        )
        gt = ClauseGroundTruth(
            text="Test.",
            clause_type="termination",
            risk_level=RiskLevel.MEDIUM,
            issues=["vague_language"],
            recommended_action=SuggestedActionType.REQUEST_MODIFICATION,
            reasoning_keywords=["test"],
        )
        scenario = ScenarioData(contract_meta=meta, clauses=[gt])
        assert scenario.contract_meta.contract_type == "NDA"
        assert len(scenario.clauses) == 1


class TestEnvironmentReset:
    def test_reset_task_1(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")
        assert obs.task_id == "task_1_easy"
        assert obs.step_number == 0

    def test_reset_task_2(self):
        env = ContractReviewEnv()
        obs = env.reset("task_2_medium")
        assert obs.task_id == "task_2_medium"

    def test_reset_task_3(self):
        env = ContractReviewEnv()
        obs = env.reset("task_3_hard")
        assert obs.task_id == "task_3_hard"

    def test_reset_unknown_task_raises(self):
        env = ContractReviewEnv()
        with pytest.raises(ValueError):
            env.reset("invalid_task")

    def test_reset_clears_state(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        env.reset("task_1_easy")
        assert env.step_number == 0
        assert env.accumulated_score == 0.0

    def test_reset_produces_clause_text(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")
        assert len(obs.clause_text) > 10

    def test_reset_has_instructions(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")
        assert len(obs.instructions) > 10

    def test_reset_has_available_actions(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")
        assert len(obs.available_actions) > 0


class TestEnvironmentStep:
    def test_classify_correct(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        )
        assert reward.score > 0

    def test_classify_incorrect(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="termination")
        )
        assert reward.score < 0

    def test_rate_severity_correct(self):
        env = ContractReviewEnv()
        env.reset("task_2_medium")
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="indemnification")
        )
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.MEDIUM)
        )
        assert reward.score != 0

    def test_flag_correct(self):
        env = ContractReviewEnv()
        env.reset("task_2_medium")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="indemnification"))
        env.step(Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.MEDIUM))
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.FLAG, flags=["one_sided_obligation", "overly_broad_scope"])
        )
        assert reward.score != 0

    def test_next_clause_advances(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        initial_index = env.clause_index
        obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))
        assert env.clause_index > initial_index

    def test_complete_review_ends_episode(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        obs, reward, done, info = env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        assert done is True

    def test_step_after_done_returns_zero(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        obs, reward, done, info = env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        assert reward.score == 0.0

    def test_max_steps_terminates(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        for _ in range(15):
            obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))
        assert done is True

    def test_missing_clause_type_penalized(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        obs, reward, done, info = env.step(Action(action_type=ActionType.CLASSIFY))
        assert reward.score < 0

    def test_missing_risk_level_penalized(self):
        env = ContractReviewEnv()
        env.reset("task_2_medium")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="indemnification"))
        obs, reward, done, info = env.step(Action(action_type=ActionType.RATE_SEVERITY))
        assert reward.score < 0


class TestDegenerateBehavior:
    def test_repeated_classify_penalized(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        obs, reward, done, info = env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        assert "degenerate_penalty" in reward.breakdown or reward.score < 0


class TestEnvironmentState:
    def test_state_after_reset(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        state = env.state()
        assert state.task_id == "task_1_easy"
        assert state.step_number == 0

    def test_state_after_step(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        state = env.state()
        assert state.step_number == 1


class TestClauseRecords:
    def test_records_initialized(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        assert len(env.clause_records) > 0

    def test_classify_updates_record(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        assert env.clause_records[0].classify_action == "confidentiality"


class TestFullEpisodeFlow:
    def test_optimal_task_1(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        done = False
        while not done:
            obs, reward, done, info = env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))
        assert env.accumulated_score > 0

    def test_optimal_task_2(self):
        env = ContractReviewEnv()
        env.reset("task_2_medium")
        done = False
        while not done:
            obs, reward, done, info = env.step(Action(action_type=ActionType.CLASSIFY, clause_type="indemnification"))
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.RATE_SEVERITY, risk_level=RiskLevel.MEDIUM))
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.FLAG, flags=[]))
            if not done:
                obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))


class TestTaskRegistry:
    def test_three_tasks_registered(self):
        assert len(TASK_REGISTRY) == 3

    def test_easy_task(self):
        assert TASK_REGISTRY["task_1_easy"].difficulty == Difficulty.EASY

    def test_medium_task(self):
        assert TASK_REGISTRY["task_2_medium"].difficulty == Difficulty.MEDIUM

    def test_hard_task(self):
        assert TASK_REGISTRY["task_3_hard"].difficulty == Difficulty.HARD

    def test_difficulty_escalation_steps(self):
        easy = TASK_REGISTRY["task_1_easy"].max_steps
        medium = TASK_REGISTRY["task_2_medium"].max_steps
        hard = TASK_REGISTRY["task_3_hard"].max_steps
        assert medium > easy
        assert hard > medium


class TestServerStub:
    @pytest.fixture(scope="class")
    def server_process(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "7861"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        import time
        time.sleep(3)
        yield proc
        proc.terminate()
        proc.wait()

    def test_health(self, server_process):
        response = httpx.get("http://127.0.0.1:7861/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_reset_task_1(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_1_easy"

    def test_reset_invalid_task(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "invalid"})
        assert response.status_code == 400

    def test_step_classify(self, server_process):
        httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        response = httpx.post("http://127.0.0.1:7861/step", json={"action_type": "classify", "clause_type": "confidentiality"})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data

    def test_step_without_reset(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/step", json={"action_type": "classify", "clause_type": "confidentiality"})
        assert response.status_code in (200, 400)

    def test_state_endpoint(self, server_process):
        httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        response = httpx.get("http://127.0.0.1:7861/state")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
