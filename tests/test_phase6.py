"""
Phase 6 — Deployment, Submission Pipeline, End-to-End Tests
Run: python -m pytest tests/test_phase6.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import httpx
import pytest
import yaml

from models import (
    Action,
    ActionType,
    ClauseActionRecord,
    GraderResult,
    Observation,
    Reward,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import TASK_REGISTRY, get_task_config
from environment import ContractReviewEnv


class TestReadmeFrontmatter:
    def test_readme_has_yaml_frontmatter(self):
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("---"), "README must start with YAML frontmatter ---"
        parts = content.split("---", 2)
        assert len(parts) >= 3, "README must have opening and closing ---"

    def test_frontmatter_has_title(self):
        fm = self._load_frontmatter()
        assert "title" in fm
        assert len(fm["title"]) > 0

    def test_frontmatter_has_sdk_docker(self):
        fm = self._load_frontmatter()
        assert fm.get("sdk") == "docker"

    def test_frontmatter_has_app_port(self):
        fm = self._load_frontmatter()
        assert fm.get("app_port") == 7860

    def test_frontmatter_has_openenv_tag(self):
        fm = self._load_frontmatter()
        tags = fm.get("tags", [])
        assert "openenv" in tags

    def test_frontmatter_has_emoji(self):
        fm = self._load_frontmatter()
        assert "emoji" in fm

    def _load_frontmatter(self) -> Dict[str, Any]:
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        parts = content.split("---", 2)
        return yaml.safe_load(parts[1])


class TestDockerfileProduction:
    def test_has_user_creation(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "useradd" in content

    def test_has_healthcheck(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "HEALTHCHECK" in content

    def test_has_curl_installed(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "curl" in content

    def test_has_expose(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "EXPOSE 7860" in content

    def test_has_cmd(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "CMD" in content


class TestServerProduction:
    def test_server_starts_without_error(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "7866"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        try:
            r = httpx.get("http://127.0.0.1:7866/", timeout=5.0)
            assert r.status_code == 200
            data = r.json()
            assert data.get("status") == "ok"
        finally:
            proc.terminate()
            proc.wait()

    def test_startup_validates_tasks(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "7867"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        try:
            r = httpx.get("http://127.0.0.1:7867/info", timeout=5.0)
            assert r.status_code == 200
            data = r.json()
            assert "tasks" in data
            assert len(data["tasks"]) >= 3
        finally:
            proc.terminate()
            proc.wait()


class TestInferenceDualMode:
    def test_direct_env_adapter(self):
        from inference import DirectEnvAdapter
        adapter = DirectEnvAdapter()
        obs = adapter.reset("task_1_easy")
        assert "task_id" in obs
        assert "clause_text" in obs

    def test_http_env_adapter_mock(self):
        from inference import HttpEnvAdapter
        adapter = HttpEnvAdapter("http://localhost:7860")
        assert adapter.base_url == "http://localhost:7860"

    def test_build_action_dict(self):
        from inference import build_action_dict
        analysis = {"clause_type": "confidentiality", "risk_level": "low"}
        action = build_action_dict("classify", analysis)
        assert action["action_type"] == "classify"
        assert action["clause_type"] == "confidentiality"

    def test_env_mode_env_var(self):
        from inference import ENV_MODE
        assert ENV_MODE in ["direct", "http"]


class TestRunSubmissionScript:
    def test_script_exists(self):
        assert os.path.exists("run_submission.py")

    def test_script_is_executable(self):
        result = subprocess.run(
            [sys.executable, "run_submission.py", "--help"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0


class TestGitignore:
    def test_gitignore_exists(self):
        assert os.path.exists(".gitignore")

    def test_gitignore_has_pycache(self):
        with open(".gitignore") as f:
            content = f.read()
        assert "__pycache__" in content

    def test_gitignore_has_venv(self):
        with open(".gitignore") as f:
            content = f.read()
        assert "venv" in content

    def test_gitignore_has_env(self):
        with open(".gitignore") as f:
            content = f.read()
        assert ".env" in content


class TestEndToEnd:
    def test_full_task_execution(self):
        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")
        assert obs.task_id == "task_1_easy"
        assert obs.total_clauses > 0

        gt = env.scenario.clauses[0]
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
        )

        obs, reward, done, info = env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        obs, reward, done, info = env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert done is True
        assert env.grader_result is not None
        assert 0.0 <= env.grader_result.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
