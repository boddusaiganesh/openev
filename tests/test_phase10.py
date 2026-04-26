"""
Phase 10 — Deployment, Multi-Model Eval & Final Validation Tests
Run: python -m pytest tests/test_phase10.py -v
"""

from __future__ import annotations

import os
import pytest

from environment import ContractReviewEnv
from models import Action, ActionType
from tasks import list_task_ids


class TestDeployScript:
    """Test deploy.py functions."""

    def test_deploy_files_defined(self):
        from deploy import DEPLOY_FILES
        assert len(DEPLOY_FILES) > 10
        assert "inference.py" in DEPLOY_FILES
        assert "server.py" in DEPLOY_FILES
        assert "Dockerfile" in DEPLOY_FILES

    def test_deploy_check_local(self):
        from deploy import run_pre_deploy_checks
        result = run_pre_deploy_checks()
        assert isinstance(result, bool)


class TestMonitorScript:
    """Test monitor.py functions."""

    def test_check_health_function_exists(self):
        from monitor import check_health
        assert callable(check_health)

    def test_health_check_returns_dict(self):
        from monitor import check_health
        result = check_health("http://127.0.0.1:7860", timeout=5.0)
        assert isinstance(result, dict)
        assert "healthy" in result
        assert "checks" in result


class TestMultiModelEval:
    """Test multi_model_eval.py functions."""

    def test_default_models_defined(self):
        from multi_model_eval import DEFAULT_MODELS
        assert len(DEFAULT_MODELS) >= 1
        assert isinstance(DEFAULT_MODELS[0], str)

    def test_evaluate_model_function_exists(self):
        from multi_model_eval import evaluate_model
        assert callable(evaluate_model)


class TestFinalIntegration:
    """Final integration tests."""

    def test_all_tasks_work_end_to_end(self):
        """Verify all 3 tasks can complete a full episode."""
        env = ContractReviewEnv()
        for tid in list_task_ids():
            obs = env.reset(tid)
            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            assert env.grader_result is not None
            assert 0.0 <= env.grader_result.score <= 1.0

    def test_environment_deterministic(self):
        """Same action sequence should give same score."""
        for tid in list_task_ids():
            scores = []
            for _ in range(2):
                env = ContractReviewEnv()
                env.reset(tid)
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
                scores.append(env.grader_result.score)
            assert scores[0] == scores[1], f"Non-deterministic for {tid}"

    def test_manifest_accessible(self):
        import json
        with open("data/manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest) >= 3

    def test_readme_has_frontmatter(self):
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("---")

    def test_changelog_exists(self):
        assert os.path.exists("CHANGELOG.md")

    def test_license_exists(self):
        assert os.path.exists("LICENSE")


class TestGitHubWorkflow:
    """Test GitHub workflow."""

    def test_workflow_has_deploy_job(self):
        with open(".github/workflows/validate.yml") as f:
            content = f.read()
        assert "docker:" in content
        assert "validate:" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
