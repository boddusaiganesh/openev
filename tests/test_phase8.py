"""
Phase 8 — Final Integration & Deployment Readiness Tests
========================================================
Run: python -m pytest tests/test_phase8.py -v

These tests verify the ENTIRE system works as one integrated unit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import pytest

from models import (
    Action,
    ActionType,
    ClauseActionRecord,
    EnvironmentState,
    GraderResult,
    Observation,
    Reward,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import TASK_REGISTRY, get_task_config, list_task_ids
from environment import ContractReviewEnv
from tests.conftest import run_optimal_trajectory


class TestFullEpisodeFlow:
    """Simulate exactly what inference.py does, without the LLM."""

    def test_task1_optimal_flow(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")

        assert obs.task_id == "task_1_easy"
        assert obs.step_number == 0
        assert obs.clause_index == 0
        assert obs.done is False
        assert obs.total_clauses >= 3
        assert len(obs.clause_text) > 20
        assert len(obs.instructions) > 10

        steps_taken = 0
        for i in range(obs.total_clauses):
            gt = env.scenario.clauses[i]

            obs_r, reward, done, info = env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
            )
            steps_taken += 1
            assert reward.score > 0, "Correct classify should reward positively"
            assert "type_accuracy" in reward.breakdown

            if i < obs.total_clauses - 1:
                obs_r, reward, done, info = env.step(
                    Action(action_type=ActionType.NEXT_CLAUSE)
                )
                steps_taken += 1
                assert obs_r.clause_index == i + 1

        obs_r, reward, done, info = env.step(
            Action(action_type=ActionType.COMPLETE_REVIEW)
        )
        steps_taken += 1

        assert done is True
        assert env.grader_result is not None
        assert env.grader_result.score >= 0.90

        state = env.state()
        assert state.done is True
        assert state.step_number == steps_taken

    def test_task2_optimal_flow(self):
        env = ContractReviewEnv()
        obs = env.reset("task_2_medium")

        for i in range(obs.total_clauses):
            if env.done:
                break
            gt = env.scenario.clauses[i]

            env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            env.step(Action(action_type=ActionType.RATE_SEVERITY, risk_level=gt.risk_level))
            env.step(Action(action_type=ActionType.FLAG, flags=gt.issues))

            if not env.done and i < obs.total_clauses - 1:
                env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        if not env.done:
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert env.done
        assert env.grader_result is not None
        assert env.grader_result.score >= 0.75
        assert "type_accuracy" in env.grader_result.breakdown

    def test_task3_optimal_flow(self):
        env = ContractReviewEnv()
        obs = env.reset("task_3_hard")

        for i in range(obs.total_clauses):
            if env.done:
                break
            gt = env.scenario.clauses[i]

            env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            if env.done: break
            env.step(Action(action_type=ActionType.RATE_SEVERITY, risk_level=gt.risk_level))
            if env.done: break
            env.step(Action(action_type=ActionType.FLAG, flags=gt.issues))
            if env.done: break
            env.step(Action(action_type=ActionType.SUGGEST, suggested_action=gt.recommended_action))
            if env.done: break
            env.step(Action(action_type=ActionType.REASON, reasoning=" ".join(gt.reasoning_keywords)))
            if env.done: break

            if i < obs.total_clauses - 1:
                env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        if not env.done:
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert env.done
        assert env.grader_result is not None
        assert env.grader_result.score >= 0.50


class TestCrossEpisodeIsolation:
    """Verify episodes are fully isolated from each other."""

    def test_sequential_tasks_no_leakage(self):
        env = ContractReviewEnv()

        score1 = run_optimal_trajectory(env, "task_1_easy")
        assert score1 > 0.5

        obs = env.reset("task_2_medium")
        assert obs.step_number == 0
        assert obs.clause_index == 0
        assert obs.accumulated_score == 0.0
        assert obs.task_id == "task_2_medium"
        assert env.grader_result is None

        score2 = run_optimal_trajectory(env, "task_2_medium")
        assert score2 > 0.5

    def test_reset_mid_episode_cleans_everything(self):
        env = ContractReviewEnv()
        env.reset("task_3_hard")

        for _ in range(5):
            env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

        assert env.step_number > 0
        assert env.accumulated_score != 0.0

        obs = env.reset("task_1_easy")
        assert obs.step_number == 0
        assert obs.accumulated_score == 0.0
        assert obs.clause_index == 0
        assert obs.task_id == "task_1_easy"
        assert len(env.actions_taken) == 0
        assert len(env.rewards_given) == 0
        assert env.grader_result is None

    def test_ten_sequential_episodes(self):
        env = ContractReviewEnv()
        tasks = list_task_ids() * 4

        for i, tid in enumerate(tasks[:10]):
            obs = env.reset(tid)
            assert obs.step_number == 0, f"Episode {i} leaked step_number"
            assert obs.accumulated_score == 0.0, f"Episode {i} leaked score"

            gt = env.scenario.clauses[0]
            env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            assert env.done
            assert env.grader_result is not None


class TestGraderScoreDistribution:
    """Verify grader produces meaningful score distributions."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_perfect_vs_empty_separation(self):
        for tid in list_task_ids():
            perfect = run_optimal_trajectory(self.env, tid)

            self.env.reset(tid)
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            empty = self.env.grader_result.score

            gap = perfect - empty
            assert gap >= 0.30, f"{tid}: gap={gap:.3f} too small"

    def test_grader_not_constant_across_inputs(self):
        for tid in list_task_ids():
            scores = set()
            scores.add(round(run_optimal_trajectory(self.env, tid), 4))

            self.env.reset(tid)
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            scores.add(round(self.env.grader_result.score, 4))

            self.env.reset(tid)
            gt = self.env.scenario.clauses[0]
            self.env.step(Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type))
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            scores.add(round(self.env.grader_result.score, 4))

            assert len(scores) >= 2, f"{tid}: only {len(scores)} unique scores"

    def test_all_scores_in_valid_range(self):
        for tid in list_task_ids():
            s = run_optimal_trajectory(self.env, tid)
            assert 0.0 <= s <= 1.0

            self.env.reset(tid)
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            assert 0.0 <= self.env.grader_result.score <= 1.0


class TestObservationContract:
    """Verify observation objects always have all required fields."""

    def test_observation_fields_after_reset(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")

        required = [
            "task_id", "step_number", "max_steps", "clause_text",
            "clause_index", "total_clauses", "contract_type", "parties",
            "jurisdiction", "instructions", "available_actions",
            "accumulated_score", "done",
        ]
        obs_dict = obs.model_dump()
        for field in required:
            assert field in obs_dict, f"Missing field: {field}"

    def test_observation_fields_after_step(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        )

        assert obs.step_number == 1
        assert isinstance(reward.score, float)
        assert isinstance(reward.breakdown, dict)
        assert isinstance(reward.message, str)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_info_dict_has_required_keys(self):
        env = ContractReviewEnv()
        env.reset("task_1_easy")
        _, _, _, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
        )
        assert "step_number" in info
        assert "clause_index" in info
        assert "accumulated_score" in info


class TestStateSerialization:
    """state() must produce JSON-serializable output."""

    @pytest.mark.parametrize("task_id", ["task_1_easy", "task_2_medium", "task_3_hard"])
    def test_state_roundtrip(self, task_id):
        env = ContractReviewEnv()
        env.reset(task_id)
        env.step(Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"))

        state = env.state()
        d = state.model_dump()
        j = json.dumps(d)
        loaded = json.loads(j)

        assert loaded["task_id"] == task_id
        assert loaded["step_number"] == 1
        assert isinstance(loaded["ground_truth"], list)


class TestInferenceCompatibility:
    """Verify inference.py components work with the environment."""

    def test_direct_adapter_reset_step_state(self):
        from inference import DirectEnvAdapter
        adapter = DirectEnvAdapter()

        obs = adapter.reset("task_1_easy")
        assert obs["task_id"] == "task_1_easy"
        assert obs["step_number"] == 0

        result = adapter.step({
            "action_type": "classify",
            "clause_type": "confidentiality",
        })
        assert "observation" in result
        assert "reward" in result
        assert "done" in result

        state = adapter.state()
        assert state["task_id"] == "task_1_easy"

    def test_parse_valid_json_response(self):
        from inference import parse_llm_response

        response = json.dumps({
            "clause_type": "confidentiality",
            "risk_level": "low",
            "flags": ["market_standard"],
            "suggested_action": "accept_as_is",
            "reasoning": "Standard NDA clause.",
        })

        result = parse_llm_response(response)
        assert result["clause_type"] == "confidentiality"
        assert result["risk_level"] == "low"

    def test_build_action_dict_all_types(self):
        from inference import build_action_dict

        analysis = {
            "clause_type": "termination",
            "risk_level": "medium",
            "flags": ["missing_time_limit"],
            "suggested_action": "request_modification",
            "reasoning": "Missing notice period.",
        }

        for act_type in ["classify", "rate_severity", "flag", "suggest", "reason"]:
            d = build_action_dict(act_type, analysis)
            assert d["action_type"] == act_type


class TestFileIntegrity:
    """Verify all required files exist and are well-formed."""

    REQUIRED_FILES = [
        "inference.py", "server.py", "environment.py", "models.py",
        "tasks.py", "graders.py", "rewards.py", "Dockerfile",
        "requirements.txt", "openenv.yaml", "README.md",
        "data/manifest.json",
    ]

    @pytest.mark.parametrize("filepath", REQUIRED_FILES)
    def test_file_exists(self, filepath):
        assert os.path.exists(filepath), f"Missing: {filepath}"

    def test_manifest_has_all_tasks(self):
        with open("data/manifest.json") as f:
            manifest = json.load(f)
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            assert tid in manifest, f"Missing task in manifest: {tid}"

    def test_all_scenario_files_exist(self):
        with open("data/manifest.json") as f:
            manifest = json.load(f)
        for tid, entry in manifest.items():
            for sf in entry.get("scenario_files", []):
                path = os.path.join("data", sf)
                assert os.path.exists(path), f"Missing scenario: {path}"

    def test_requirements_has_core_deps(self):
        with open("requirements.txt") as f:
            content = f.read()
        for dep in ["fastapi", "uvicorn", "pydantic", "openai"]:
            assert dep in content, f"Missing dep: {dep}"

    def test_no_hardcoded_api_keys(self):
        pattern = re.compile(r'(hf_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{30,})')
        for fp in self.REQUIRED_FILES:
            if fp.endswith(".py") and os.path.exists(fp):
                with open(fp) as f:
                    matches = pattern.findall(f.read())
                assert not matches, f"API key found in {fp}"


class TestDataQuality:
    """Verify scenario data is well-formed and realistic."""

    def test_all_clause_types_are_valid(self):
        env = ContractReviewEnv()
        for tid in list_task_ids():
            env.reset(tid)
            for clause in env.scenario.clauses:
                assert clause.clause_type in CLAUSE_TAXONOMY

    def test_all_risk_levels_valid(self):
        env = ContractReviewEnv()
        for tid in list_task_ids():
            env.reset(tid)
            for clause in env.scenario.clauses:
                assert clause.risk_level in list(RiskLevel)

    def test_clause_text_is_substantive(self):
        env = ContractReviewEnv()
        for tid in list_task_ids():
            env.reset(tid)
            for clause in env.scenario.clauses:
                assert len(clause.text) >= 50

    def test_difficulty_escalation(self):
        env = ContractReviewEnv()
        counts = {}
        for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            env.reset(tid)
            counts[tid] = len(env.scenario.clauses)

        assert counts["task_1_easy"] <= counts["task_2_medium"]
        assert counts["task_2_medium"] <= counts["task_3_hard"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
