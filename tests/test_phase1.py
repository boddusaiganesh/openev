"""
Phase 1 — Validation tests.
Run: python -m pytest tests/test_phase1.py -v
"""

import subprocess
import sys

import httpx
import pytest

from domain_config import (
    ACTION_TYPES,
    CLAUSE_TAXONOMY,
    DOMAIN,
    DOMAIN_CONFIRMATION,
    EPISODE_CONSTRAINTS,
    ISSUE_FLAGS,
    RESOURCE_CONSTRAINTS,
    RISK_LEVELS,
    RISK_REGISTER,
    RUBRIC_ALIGNMENT,
    SUGGESTED_ACTIONS,
    TASK_DIFFICULTY_MATRIX,
    validate_phase1,
)


class TestDomainLock:
    def test_domain_is_locked(self):
        assert DOMAIN_CONFIRMATION["domain_locked"] is True

    def test_domain_has_name(self):
        assert DOMAIN["name"] == "contract-clause-review"

    def test_domain_has_description(self):
        assert len(DOMAIN["description"]) > 100

    def test_domain_has_version(self):
        assert DOMAIN["version"] == "1.0.0"

    def test_domain_has_tags(self):
        assert "openenv" in DOMAIN["tags"]


class TestTaskDifficulty:
    def test_three_tasks_defined(self):
        assert len(TASK_DIFFICULTY_MATRIX) >= 3

    def test_easy_task_exists(self):
        assert "task_1_easy" in TASK_DIFFICULTY_MATRIX
        assert TASK_DIFFICULTY_MATRIX["task_1_easy"]["difficulty"] == "easy"

    def test_medium_task_exists(self):
        assert "task_2_medium" in TASK_DIFFICULTY_MATRIX
        assert TASK_DIFFICULTY_MATRIX["task_2_medium"]["difficulty"] == "medium"

    def test_hard_task_exists(self):
        assert "task_3_hard" in TASK_DIFFICULTY_MATRIX
        assert TASK_DIFFICULTY_MATRIX["task_3_hard"]["difficulty"] == "hard"

    def test_difficulty_escalation_in_clauses(self):
        easy_max = TASK_DIFFICULTY_MATRIX["task_1_easy"]["num_clauses_range"][1]
        medium_min = TASK_DIFFICULTY_MATRIX["task_2_medium"]["num_clauses_range"][0]
        hard_min = TASK_DIFFICULTY_MATRIX["task_3_hard"]["num_clauses_range"][0]
        assert medium_min >= easy_max
        assert hard_min > medium_min

    def test_difficulty_escalation_in_steps(self):
        easy = TASK_DIFFICULTY_MATRIX["task_1_easy"]["max_steps"]
        medium = TASK_DIFFICULTY_MATRIX["task_2_medium"]["max_steps"]
        hard = TASK_DIFFICULTY_MATRIX["task_3_hard"]["max_steps"]
        assert medium > easy
        assert hard > medium

    def test_difficulty_escalation_in_actions_per_clause(self):
        easy = len(TASK_DIFFICULTY_MATRIX["task_1_easy"]["actions_required_per_clause"])
        medium = len(TASK_DIFFICULTY_MATRIX["task_2_medium"]["actions_required_per_clause"])
        hard = len(TASK_DIFFICULTY_MATRIX["task_3_hard"]["actions_required_per_clause"])
        assert medium > easy
        assert hard > medium

    def test_frontier_scores_decrease_with_difficulty(self):
        easy_high = TASK_DIFFICULTY_MATRIX["task_1_easy"]["expected_frontier_score"][1]
        medium_high = TASK_DIFFICULTY_MATRIX["task_2_medium"]["expected_frontier_score"][1]
        hard_high = TASK_DIFFICULTY_MATRIX["task_3_hard"]["expected_frontier_score"][1]
        assert easy_high >= medium_high
        assert medium_high >= hard_high

    def test_all_tasks_have_required_fields(self):
        required = [
            "task_id", "name", "difficulty", "description",
            "num_clauses_range", "max_steps", "actions_required_per_clause",
            "ambiguity_level", "cross_clause_dependencies",
            "expected_frontier_score", "expected_weak_score",
            "expected_random_score",
        ]
        for task_id, task in TASK_DIFFICULTY_MATRIX.items():
            for field in required:
                assert field in task, f"{task_id} missing field: {field}"


class TestTaxonomies:
    def test_clause_taxonomy_size(self):
        assert len(CLAUSE_TAXONOMY) >= 10

    def test_clause_taxonomy_unique(self):
        assert len(CLAUSE_TAXONOMY) == len(set(CLAUSE_TAXONOMY))

    def test_risk_levels(self):
        assert RISK_LEVELS == ["low", "medium", "high", "critical"]

    def test_issue_flags_size(self):
        assert len(ISSUE_FLAGS) >= 10

    def test_issue_flags_unique(self):
        assert len(ISSUE_FLAGS) == len(set(ISSUE_FLAGS))

    def test_suggested_actions_size(self):
        assert len(SUGGESTED_ACTIONS) >= 4

    def test_action_types_size(self):
        assert len(ACTION_TYPES) >= 5

    def test_action_types_include_core(self):
        assert "classify" in ACTION_TYPES
        assert "rate_severity" in ACTION_TYPES
        assert "flag" in ACTION_TYPES
        assert "suggest" in ACTION_TYPES
        assert "complete_review" in ACTION_TYPES


class TestResourceConstraints:
    def test_vcpu_limit(self):
        assert RESOURCE_CONSTRAINTS["max_vcpu"] <= 2

    def test_memory_limit(self):
        assert RESOURCE_CONSTRAINTS["max_memory_gb"] <= 8

    def test_runtime_limit(self):
        assert RESOURCE_CONSTRAINTS["max_inference_runtime_minutes"] <= 20

    def test_port(self):
        assert RESOURCE_CONSTRAINTS["server_port"] == 7860

    def test_no_gpu(self):
        assert RESOURCE_CONSTRAINTS["no_gpu_required"] is True

    def test_estimated_runtime_under_limit(self):
        total_seconds = EPISODE_CONSTRAINTS["total_with_overhead_seconds"]
        limit_seconds = RESOURCE_CONSTRAINTS["max_inference_runtime_minutes"] * 60
        assert total_seconds < limit_seconds


class TestRubricAlignment:
    def test_five_criteria_covered(self):
        criteria = [k for k in RUBRIC_ALIGNMENT if k != "total_projected_range"]
        assert len(criteria) == 5

    def test_weights_sum_to_one(self):
        criteria = [k for k in RUBRIC_ALIGNMENT if k != "total_projected_range"]
        total = sum(RUBRIC_ALIGNMENT[c]["weight"] for c in criteria)
        assert abs(total - 1.0) < 0.001

    def test_all_criteria_have_justification(self):
        criteria = [k for k in RUBRIC_ALIGNMENT if k != "total_projected_range"]
        for c in criteria:
            assert len(RUBRIC_ALIGNMENT[c]["justification"]) > 50

    def test_projected_scores_reasonable(self):
        criteria = [k for k in RUBRIC_ALIGNMENT if k != "total_projected_range"]
        for c in criteria:
            low, high = RUBRIC_ALIGNMENT[c]["projected_score_range"]
            assert 0 <= low <= high
            assert high <= 30


class TestRiskRegister:
    def test_minimum_risks_identified(self):
        assert len(RISK_REGISTER) >= 5

    def test_all_risks_have_required_fields(self):
        for risk in RISK_REGISTER:
            assert "risk" in risk
            assert "impact" in risk
            assert "likelihood" in risk
            assert "mitigation" in risk

    def test_critical_risks_have_mitigation(self):
        for risk in RISK_REGISTER:
            if risk["impact"] == "critical":
                assert len(risk["mitigation"]) > 20


class TestValidateFunction:
    def test_validate_phase1_passes(self):
        result = validate_phase1()
        assert result is True


class TestDockerfileExists:
    def test_dockerfile_parseable(self):
        with open("Dockerfile", "r") as f:
            content = f.read()
        assert "FROM python" in content
        assert "7860" in content
        assert "uvicorn" in content


class TestOpenenvYamlExists:
    def test_yaml_parseable(self):
        import yaml
        with open("openenv.yaml", "r") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "contract-clause-review"
        assert len(data["tasks"]) >= 3
        assert data["server"]["port"] == 7860


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

    def test_health_endpoint(self, server_process):
        response = httpx.get("http://127.0.0.1:7861/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["environment"] == "contract-clause-review"

    def test_reset_endpoint(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset")
        assert response.status_code == 200

    def test_step_endpoint(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/step")
        assert response.status_code == 200

    def test_state_endpoint(self, server_process):
        response = httpx.get("http://127.0.0.1:7861/state")
        assert response.status_code == 200


class TestManifestExists:
    def test_manifest_parseable(self):
        import json
        with open("data/manifest.json", "r") as f:
            data = json.load(f)
        assert "task_1_easy" in data
        assert "task_2_medium" in data
        assert "task_3_hard" in data


class TestRequirementsExists:
    def test_requirements_has_core_deps(self):
        with open("requirements.txt", "r") as f:
            content = f.read()
        assert "fastapi" in content
        assert "uvicorn" in content
        assert "pydantic" in content
        assert "openai" in content
