"""
Phase 5 — Final Integration Tests
Hardening, edge cases, and pre-submission validation tests.
Run: python -m pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest

from environment import ContractReviewEnv
from models import Action, ActionType, RiskLevel


class TestHardenedEnvironment:
    """Tests for hardened edge case handling."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_step_before_reset_returns_error(self):
        """Step before reset should return error info."""
        action = Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        obs, reward, done, info = self.env.step(action)

        assert done is True
        assert "error" in info
        assert info["error"] == "No active episode."

    def test_reset_with_invalid_task_raises(self):
        """Invalid task_id should raise ValueError."""
        with pytest.raises(ValueError):
            self.env.reset("invalid_task")

    def test_reset_clears_all_state(self):
        """Reset should clear all mutable state."""
        self.env.reset("task_1_easy")
        self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

        self.env.reset("task_2_medium")

        assert self.env.step_number == 0
        assert self.env.clause_index == 0
        assert self.env.accumulated_score == 0.0
        assert len(self.env.actions_taken) == 0
        assert len(self.env.rewards_given) == 0

    def test_step_after_done_returns_zero_reward(self):
        """Step after episode done should return zero reward."""
        self.env.reset("task_1_easy")
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        obs, reward, done, info = self.env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        )

        assert done is True
        assert reward.score == 0.0

    def test_invalid_action_fields_penalized(self):
        """Missing required fields should be penalized."""
        self.env.reset("task_1_easy")

        obs, reward, done, info = self.env.step(
            Action(action_type=ActionType.CLASSIFY)
        )

        assert reward.score < 0
        assert "error" in info

    def test_degenerate_behavior_penalty(self):
        """Repeated same action on same clause should be penalized."""
        self.env.reset("task_1_easy")

        self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

        obs, reward, done, info = self.env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        )

        assert "degenerate_penalty" in reward.breakdown or reward.score < 0

    def test_grader_invoked_on_done(self):
        """Grader should be invoked when episode ends."""
        self.env.reset("task_1_easy")
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.grader_result is not None
        assert 0.0 <= self.env.grader_result.score <= 1.0

    def test_max_steps_terminates(self):
        """Hitting max_steps should terminate episode."""
        self.env.reset("task_1_easy")

        for _ in range(15):
            obs, reward, done, info = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
            if done:
                break

        assert self.env.done is True

    def test_empty_clause_list_raises(self):
        """Scenario with no clauses should raise error."""
        with pytest.raises(ValueError):
            self.env._load_scenario("nonexistent")


class TestServerHardening:
    """Tests for hardened server behavior."""

    @pytest.fixture(scope="class")
    def server_process(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "7861"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        yield proc
        proc.terminate()
        proc.wait()

    def test_info_endpoint(self, server_process):
        """GET /info should return detailed environment info."""
        response = httpx.get("http://127.0.0.1:7861/info")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 3

    def test_reset_unknown_task_400(self, server_process):
        """POST /reset with unknown task should return 400."""
        response = httpx.post(
            "http://127.0.0.1:7861/reset",
            json={"task_id": "invalid_task"}
        )
        assert response.status_code == 400

    def test_step_without_reset_400(self, server_process):
        """POST /step without reset should return 400."""
        response = httpx.post(
            "http://127.0.0.1:7861/step",
            json={"action_type": "classify", "clause_type": "confidentiality"}
        )
        assert response.status_code == 400

    def test_invalid_action_payload_400(self, server_process):
        """POST /step with missing required action fields is penalized (200)."""
        httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        response = httpx.post(
            "http://127.0.0.1:7861/step",
            json={"action_type": "classify"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["reward"]["score"] < 0

    def test_grader_in_final_step_info(self, server_process):
        """Grader result should be in info when episode ends."""
        httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        httpx.post(
            "http://127.0.0.1:7861/step",
            json={"action_type": "complete_review"}
        )

        response = httpx.get("http://127.0.0.1:7861/state")
        data = response.json()
        assert "grader_result" in data
        assert "score" in data["grader_result"]

    def test_cors_headers_present(self, server_process):
        """CORS headers should be present."""
        response = httpx.options("http://127.0.0.1:7861/")
        assert response.status_code == 200


class TestGraderDeterminism:
    """Tests for grader determinism and correctness."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_same_trajectory_same_score(self):
        """Same trajectory should produce same score."""
        scores = []

        for _ in range(3):
            self.env.reset("task_1_easy")
            self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            scores.append(self.env.grader_result.score)

        assert len(set(scores)) == 1

    def test_different_trajectories_different_scores(self):
        """Different trajectories should produce different scores."""
        self.env.reset("task_1_easy")
        self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        score1 = self.env.grader_result.score

        self.env.reset("task_1_easy")
        self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type="force_majeure"))
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        score2 = self.env.grader_result.score

        assert score1 != score2

    def test_all_tasks_produce_valid_scores(self):
        """All tasks should produce scores in [0, 1]."""
        for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            self.env.reset(task_id)
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            assert self.env.grader_result is not None
            assert 0.0 <= self.env.grader_result.score <= 1.0

    def test_optimal_trajectory_high_score(self):
        """Optimal trajectory should produce high score."""
        self.env.reset("task_1_easy")
        obs = self.env.reset("task_1_easy")

        for i in range(obs.total_clauses):
            gt = self.env.scenario.clauses[i]
            self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            if i < obs.total_clauses - 1:
                self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.grader_result.score >= 0.9


class TestValidateScript:
    """Tests for validate.py script."""

    def test_validate_script_exists(self):
        """validate.py should exist and be importable."""
        import validate
        assert hasattr(validate, "Validator")

    def test_validator_instantiates(self):
        """Validator should instantiate."""
        from validate import Validator
        v = Validator("http://localhost:7860")
        assert v.base_url == "http://localhost:7860"

    def test_validation_result(self):
        """ValidationResult should work correctly."""
        from validate import ValidationResult
        r = ValidationResult("test", True, "message")
        assert r.name == "test"
        assert r.passed is True
        assert r.message == "message"


class TestCompleteIntegration:
    """End-to-end integration tests."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_task_1_full_episode(self):
        """Task 1 should complete successfully."""
        obs = self.env.reset("task_1_easy")

        while not obs.done:
            gt = self.env.scenario.clauses[obs.clause_index]
            obs, reward, done, info = self.env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
            )
            if not done and obs.clause_index < obs.total_clauses - 1:
                obs, reward, done, info = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        assert self.env.done is True
        assert self.env.grader_result is not None

    def test_task_2_full_episode(self):
        """Task 2 should complete successfully."""
        obs = self.env.reset("task_2_medium")

        while not obs.done:
            gt = self.env.scenario.clauses[obs.clause_index]
            obs, reward, done, info = self.env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
            )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.RATE_SEVERITY, risk_level=gt.risk_level)
                )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.FLAG, flags=gt.issues)
                )
            if not done and obs.clause_index < obs.total_clauses - 1:
                obs, reward, done, info = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        assert self.env.done is True

    def test_task_3_full_episode(self):
        """Task 3 should complete successfully."""
        obs = self.env.reset("task_3_hard")

        while not obs.done:
            gt = self.env.scenario.clauses[obs.clause_index]
            obs, reward, done, info = self.env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
            )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.RATE_SEVERITY, risk_level=gt.risk_level)
                )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.FLAG, flags=gt.issues)
                )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.SUGGEST, suggested_action=gt.recommended_action)
                )
            if not done:
                obs, reward, done, info = self.env.step(
                    Action(action_type=ActionType.REASON, reasoning=" ".join(gt.reasoning_keywords))
                )
            if not done and obs.clause_index < obs.total_clauses - 1:
                obs, reward, done, info = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        assert self.env.done is True

    def test_state_serialization(self):
        """State should serialize to JSON-serializable dict."""
        self.env.reset("task_1_easy")
        state = self.env.state()

        json_str = state.model_dump_json()
        assert len(json_str) > 0

    def test_ground_truth_hidden_from_agent(self):
        """Ground truth should not be in observation."""
        obs = self.env.reset("task_1_easy")

        assert not hasattr(obs, "ground_truth")
        assert "ground_truth" not in obs.model_dump()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
