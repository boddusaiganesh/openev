"""
Phase 5 — Pre-Submission Validation Script
Runs all checks required before submission.

Usage:
    python validate.py                   # Validate against local server
    python validate.py --url URL         # Validate against deployed server
    python validate.py --docker          # Build and test Docker
    python validate.py --all             # Run everything
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import py_compile
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import yaml


DEFAULT_URL = "http://127.0.0.1:7860"
TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]
REQUIRED_FILES = [
    "inference.py",
    "server.py",
    "environment.py",
    "models.py",
    "tasks.py",
    "graders.py",
    "rewards.py",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "README.md",
    "data/manifest.json",
]
README_REQUIRED_SECTIONS = [
    "Action Space",
    "Observation Space",
    "task_1_easy",
    "task_2_medium",
    "task_3_hard",
]


class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class Validator:
    def __init__(self, base_url: str = DEFAULT_URL):
        self.base_url = base_url
        self.results: List[ValidationResult] = []
        self.client = httpx.Client(timeout=30.0)
        self._managed_server: Optional[subprocess.Popen] = None

    def run_all_local(self):
        """Run all local (non-server) checks."""
        start_idx = len(self.results)
        print("\n" + "=" * 70)
        print("PRE-SUBMISSION VALIDATION — Local Checks")
        print("=" * 70)

        self._check_required_files()
        self._check_openenv_yaml()
        self._check_readme()
        self._check_dockerfile()
        self._check_requirements()
        self._check_manifest()
        self._check_scenario_files()
        self._check_models_import()
        self._check_environment_import()
        self._check_grader_import()
        self._check_inference_import()

        self._print_results("Local Checks", start_idx)

    def run_all_server(self):
        """Run all server checks (server must be running)."""
        start_idx = len(self.results)
        print("\n" + "=" * 70)
        print(f"PRE-SUBMISSION VALIDATION — Server Checks ({self.base_url})")
        print("=" * 70)

        managed = self._ensure_local_server_running()

        self._check_health()
        self._check_reset_all_tasks()
        self._check_step_actions()
        self._check_state_endpoint()
        self._check_grader_scores()
        self._check_grader_determinism()
        self._check_grader_not_constant()
        self._check_full_episode()
        self._check_reset_clears_state()
        self._check_invalid_action_handling()
        self._check_step_after_done()
        self._check_cors_headers()

        self._print_results("Server Checks", start_idx)

        if managed:
            self._stop_managed_server()

    def run_environment_tests(self):
        """Run environment unit tests directly (no server)."""
        start_idx = len(self.results)
        print("\n" + "=" * 70)
        print("PRE-SUBMISSION VALIDATION — Environment Tests")
        print("=" * 70)

        self._check_env_reset_all_tasks()
        self._check_env_optimal_trajectories()
        self._check_env_wrong_trajectories()
        self._check_env_grader_range()
        self._check_env_state_serialization()
        self._check_env_no_cross_episode_leakage()

        self._print_results("Environment Tests", start_idx)

    def run_docker_check(self):
        """Build and test Docker image."""
        start_idx = len(self.results)
        print("\n" + "=" * 70)
        print("PRE-SUBMISSION VALIDATION — Docker")
        print("=" * 70)

        self._check_docker_build()
        self._print_results("Docker", start_idx)

    def _parse_base_url_host_port(self) -> Tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        if parsed.port is not None:
            return host, parsed.port
        return host, 443 if parsed.scheme == "https" else 80

    def _is_server_reachable(self, timeout: float = 1.5) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/", timeout=timeout)
            return r.status_code < 500
        except Exception:
            return False

    def _ensure_local_server_running(self) -> bool:
        """Auto-start local uvicorn server for server checks when needed.

        Returns True when this validator started and now manages the server.
        """
        host, port = self._parse_base_url_host_port()
        if host not in ("127.0.0.1", "localhost"):
            return False

        if self._is_server_reachable():
            return False

        try:
            self._managed_server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "server:app",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.results.append(ValidationResult("Server auto-start", False, str(e)))
            return False

        deadline = time.time() + 20
        while time.time() < deadline:
            if self._is_server_reachable(timeout=1.0):
                self.results.append(
                    ValidationResult(
                        "Server auto-start",
                        True,
                        f"Started local server at {self.base_url}",
                    )
                )
                return True
            time.sleep(0.5)

        self.results.append(
            ValidationResult(
                "Server auto-start",
                False,
                f"Could not start server at {self.base_url}",
            )
        )
        self._stop_managed_server()
        return False

    def _stop_managed_server(self) -> None:
        proc = self._managed_server
        self._managed_server = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _check_required_files(self):
        for filepath in REQUIRED_FILES:
            exists = os.path.exists(filepath)
            self.results.append(
                ValidationResult(
                    f"File exists: {filepath}",
                    exists,
                    "" if exists else f"Missing: {filepath}",
                )
            )

    def _check_openenv_yaml(self):
        try:
            with open("openenv.yaml") as f:
                data = yaml.safe_load(f)

            self.results.append(ValidationResult("openenv.yaml: parseable", True))

            name_ok = data.get("name") == "contract-clause-review"
            self.results.append(ValidationResult("openenv.yaml: name correct", name_ok))

            tasks = data.get("tasks", [])
            tasks_ok = len(tasks) >= 3
            self.results.append(
                ValidationResult(
                    f"openenv.yaml: {len(tasks)} tasks defined (need ≥3)",
                    tasks_ok,
                )
            )

            task_ids = [t.get("id") for t in tasks]
            for tid in TASK_IDS:
                self.results.append(
                    ValidationResult(
                        f"openenv.yaml: task {tid} defined",
                        tid in task_ids,
                    )
                )

            port_ok = data.get("server", {}).get("port") == 7860
            self.results.append(ValidationResult("openenv.yaml: port is 7860", port_ok))

            tags = data.get("tags", [])
            self.results.append(
                ValidationResult(
                    "openenv.yaml: 'openenv' tag present",
                    "openenv" in tags,
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult("openenv.yaml: parseable", False, str(e))
            )

    def _check_readme(self):
        try:
            with open("README.md", "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            self.results.append(
                ValidationResult("README.md: exists and readable", True)
            )

            self.results.append(
                ValidationResult(
                    f"README.md: length ({len(content)} chars)",
                    len(content) > 1000,
                    "Should be >1000 chars" if len(content) <= 1000 else "",
                )
            )

            for section in README_REQUIRED_SECTIONS:
                found = section in content
                self.results.append(
                    ValidationResult(
                        f"README.md: contains '{section}'",
                        found,
                    )
                )

        except Exception as e:
            self.results.append(ValidationResult("README.md: readable", False, str(e)))

    def _check_dockerfile(self):
        try:
            with open("Dockerfile") as f:
                content = f.read()

            self.results.append(ValidationResult("Dockerfile: exists", True))

            checks = [
                ("FROM", "FROM" in content),
                ("EXPOSE 7860", "7860" in content),
                ("CMD/ENTRYPOINT", "CMD" in content or "ENTRYPOINT" in content),
            ]
            for name, ok in checks:
                self.results.append(ValidationResult(f"Dockerfile: has {name}", ok))

        except Exception as e:
            self.results.append(ValidationResult("Dockerfile: exists", False, str(e)))

    def _check_requirements(self):
        try:
            with open("requirements.txt") as f:
                content = f.read()

            required_deps = ["fastapi", "uvicorn", "pydantic", "openai"]
            for dep in required_deps:
                self.results.append(
                    ValidationResult(
                        f"requirements.txt: has {dep}",
                        dep in content,
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult("requirements.txt: readable", False, str(e))
            )

    def _check_manifest(self):
        try:
            with open("data/manifest.json") as f:
                data = json.load(f)

            for tid in TASK_IDS:
                self.results.append(
                    ValidationResult(
                        f"manifest.json: has {tid}",
                        tid in data,
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult("manifest.json: parseable", False, str(e))
            )

    def _check_scenario_files(self):
        try:
            with open("data/manifest.json") as f:
                manifest = json.load(f)

            for tid, entry in manifest.items():
                for sf in entry.get("scenario_files", []):
                    path = os.path.join("data", sf)
                    exists = os.path.exists(path)
                    self.results.append(
                        ValidationResult(
                            f"Scenario file: {path}",
                            exists,
                        )
                    )

                    if exists:
                        with open(path) as f2:
                            raw = json.load(f2)
                        n_clauses = len(raw.get("clauses", []))
                        self.results.append(
                            ValidationResult(
                                f"  {path}: {n_clauses} clauses",
                                n_clauses >= 3,
                            )
                        )

        except Exception as e:
            self.results.append(
                ValidationResult("Scenario files: readable", False, str(e))
            )

    def _check_models_import(self):
        try:
            import models

            self.results.append(
                ValidationResult("models.py: imports successfully", True)
            )

            required_classes = [
                "Action",
                "Observation",
                "Reward",
                "EnvironmentState",
                "GraderResult",
                "ClauseGroundTruth",
                "ScenarioData",
            ]
            for cls_name in required_classes:
                has = hasattr(models, cls_name)
                self.results.append(ValidationResult(f"models.py: has {cls_name}", has))

        except Exception as e:
            self.results.append(ValidationResult("models.py: imports", False, str(e)))

    def _check_environment_import(self):
        try:
            from environment import ContractReviewEnv

            env = ContractReviewEnv()
            self.results.append(
                ValidationResult("environment.py: ContractReviewEnv instantiates", True)
            )

            for method in ["reset", "step", "state"]:
                has = hasattr(env, method) and callable(getattr(env, method))
                self.results.append(
                    ValidationResult(f"environment.py: has {method}()", has)
                )

        except Exception as e:
            self.results.append(
                ValidationResult("environment.py: imports", False, str(e))
            )

    def _check_grader_import(self):
        try:
            from graders import grade_episode

            self.results.append(
                ValidationResult("graders.py: grade_episode imports", True)
            )
        except Exception as e:
            self.results.append(ValidationResult("graders.py: imports", False, str(e)))

    def _check_inference_import(self):
        try:
            py_compile.compile("inference.py", doraise=True)
            self.results.append(
                ValidationResult("inference.py: syntactically valid", True)
            )
        except Exception as e:
            self.results.append(
                ValidationResult("inference.py: syntax check", False, str(e))
            )

    def _check_health(self):
        try:
            r = self.client.get(f"{self.base_url}/")
            ok = r.status_code == 200
            data = r.json()
            has_status = data.get("status") == "ok"
            self.results.append(ValidationResult("GET /: returns 200", ok))
            self.results.append(ValidationResult("GET /: status=ok", has_status))
        except Exception as e:
            self.results.append(ValidationResult("GET /: reachable", False, str(e)))

    def _check_reset_all_tasks(self):
        for tid in TASK_IDS:
            try:
                r = self.client.post(
                    f"{self.base_url}/reset",
                    json={"task_id": tid},
                )
                ok = r.status_code == 200
                data = r.json()
                has_clause = len(data.get("clause_text", "")) > 10
                self.results.append(
                    ValidationResult(f"POST /reset {tid}: returns 200", ok)
                )
                self.results.append(
                    ValidationResult(f"POST /reset {tid}: has clause_text", has_clause)
                )
            except Exception as e:
                self.results.append(
                    ValidationResult(f"POST /reset {tid}", False, str(e))
                )

    def _check_step_actions(self):
        try:
            self.client.post(
                f"{self.base_url}/reset",
                json={"task_id": "task_1_easy"},
            )

            r = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify", "clause_type": "confidentiality"},
            )
            self.results.append(
                ValidationResult(
                    "POST /step classify: returns 200",
                    r.status_code == 200,
                )
            )

            data = r.json()
            has_obs = "observation" in data
            has_reward = "reward" in data
            has_done = "done" in data
            has_info = "info" in data
            self.results.append(
                ValidationResult("POST /step: has observation", has_obs)
            )
            self.results.append(ValidationResult("POST /step: has reward", has_reward))
            self.results.append(ValidationResult("POST /step: has done", has_done))
            self.results.append(ValidationResult("POST /step: has info", has_info))

            score = data.get("reward", {}).get("score", None)
            self.results.append(
                ValidationResult(
                    "POST /step: reward.score is float",
                    isinstance(score, (int, float)),
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("POST /step", False, str(e)))

    def _check_state_endpoint(self):
        try:
            self.client.post(
                f"{self.base_url}/reset",
                json={"task_id": "task_1_easy"},
            )
            r = self.client.get(f"{self.base_url}/state")
            ok = r.status_code == 200
            data = r.json()

            self.results.append(ValidationResult("GET /state: returns 200", ok))

            required_fields = [
                "task_id",
                "step_number",
                "clause_index",
                "done",
                "ground_truth",
            ]
            for field in required_fields:
                self.results.append(
                    ValidationResult(
                        f"GET /state: has {field}",
                        field in data,
                    )
                )

        except Exception as e:
            self.results.append(ValidationResult("GET /state", False, str(e)))

    def _check_grader_scores(self):
        for tid in TASK_IDS:
            try:
                self.client.post(
                    f"{self.base_url}/reset",
                    json={"task_id": tid},
                )
                r = self.client.post(
                    f"{self.base_url}/step",
                    json={"action_type": "complete_review"},
                )
                data = r.json()
                score = data.get("info", {}).get("grader_score", None)

                self.results.append(
                    ValidationResult(
                        f"Grader {tid}: score present",
                        score is not None,
                    )
                )
                if score is not None:
                    in_range = 0.0 < score < 1.0
                    self.results.append(
                        ValidationResult(
                            f"Grader {tid}: score in (0,1) (got {score:.4f})",
                            in_range,
                        )
                    )

            except Exception as e:
                self.results.append(ValidationResult(f"Grader {tid}", False, str(e)))

    def _check_grader_determinism(self):
        tid = "task_1_easy"
        try:
            scores = []
            for _ in range(3):
                self.client.post(
                    f"{self.base_url}/reset",
                    json={"task_id": tid},
                )
                r = self.client.post(
                    f"{self.base_url}/step",
                    json={"action_type": "classify", "clause_type": "confidentiality"},
                )
                r = self.client.post(
                    f"{self.base_url}/step",
                    json={"action_type": "complete_review"},
                )
                data = r.json()
                score = data.get("info", {}).get("grader_score", -1)
                scores.append(score)

            all_same = len(set(scores)) == 1
            self.results.append(
                ValidationResult(
                    f"Grader determinism: 3 runs same score ({scores})",
                    all_same,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Grader determinism", False, str(e)))

    def _check_grader_not_constant(self):
        tid = "task_1_easy"
        try:
            self.client.post(f"{self.base_url}/reset", json={"task_id": tid})
            self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify", "clause_type": "confidentiality"},
            )
            r1 = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "complete_review"},
            )
            score1 = r1.json().get("info", {}).get("grader_score", -1)

            self.client.post(f"{self.base_url}/reset", json={"task_id": tid})
            self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify", "clause_type": "force_majeure"},
            )
            r2 = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "complete_review"},
            )
            score2 = r2.json().get("info", {}).get("grader_score", -1)

            self.client.post(f"{self.base_url}/reset", json={"task_id": tid})
            r3 = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "complete_review"},
            )
            score3 = r3.json().get("info", {}).get("grader_score", -1)

            unique_scores = len({score1, score2, score3})
            self.results.append(
                ValidationResult(
                    f"Grader not constant: {unique_scores} unique scores",
                    unique_scores >= 2,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Grader not constant", False, str(e)))

    def _check_full_episode(self):
        tid = "task_1_easy"
        try:
            r = self.client.post(f"{self.base_url}/reset", json={"task_id": tid})
            obs = r.json()
            total = obs.get("total_clauses", 0)

            state_r = self.client.get(f"{self.base_url}/state")
            state = state_r.json()
            gts = state.get("ground_truth", [])

            for i in range(total):
                gt_type = gts[i]["clause_type"] if i < len(gts) else "confidentiality"
                self.client.post(
                    f"{self.base_url}/step",
                    json={"action_type": "classify", "clause_type": gt_type},
                )
                if i < total - 1:
                    self.client.post(
                        f"{self.base_url}/step",
                        json={"action_type": "next_clause"},
                    )

            r = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "complete_review"},
            )
            data = r.json()
            done = data.get("done", False)
            score = data.get("info", {}).get("grader_score", 0)

            self.results.append(
                ValidationResult(f"Full episode {tid}: completes", done)
            )
            self.results.append(
                ValidationResult(
                    f"Full episode {tid}: score={score:.4f} (>0.9 expected)",
                    score >= 0.9,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult(f"Full episode {tid}", False, str(e)))

    def _check_reset_clears_state(self):
        try:
            self.client.post(f"{self.base_url}/reset", json={"task_id": "task_1_easy"})
            self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify", "clause_type": "confidentiality"},
            )

            r = self.client.post(
                f"{self.base_url}/reset",
                json={"task_id": "task_2_medium"},
            )
            data = r.json()
            task_id = data.get("task_id", "")
            self.results.append(
                ValidationResult(
                    f"Reset clears state: new task_id is task_2_medium",
                    task_id == "task_2_medium",
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Reset clears state", False, str(e)))

    def _check_invalid_action_handling(self):
        try:
            self.client.post(f"{self.base_url}/reset", json={"task_id": "task_1_easy"})
            r = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify"},
            )
            ok = r.status_code in (200, 400)
            self.results.append(
                ValidationResult("Invalid action handling: returns 200 or 400", ok)
            )

            data = (
                r.json()
                if r.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            has_error = (
                "error" in data
                or "detail" in data
                or "message" in data.get("reward", {})
            )
            self.results.append(
                ValidationResult(
                    "Invalid action handling: has error feedback", has_error
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult("Invalid action handling", False, str(e))
            )

    def _check_step_after_done(self):
        try:
            self.client.post(f"{self.base_url}/reset", json={"task_id": "task_1_easy"})
            self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "complete_review"},
            )
            r = self.client.post(
                f"{self.base_url}/step",
                json={"action_type": "classify", "clause_type": "confidentiality"},
            )
            data = r.json()
            done = data.get("done", True)
            self.results.append(
                ValidationResult("Step after done: returns done=true", done)
            )

        except Exception as e:
            self.results.append(ValidationResult("Step after done", False, str(e)))

    def _check_cors_headers(self):
        try:
            r = self.client.options(f"{self.base_url}/")
            headers = dict(r.headers)
            cors_ok = "access-control-allow-origin" in headers
            preflight_handled = r.status_code in (200, 204, 405)
            self.results.append(
                ValidationResult(
                    "CORS: headers present (or preflight handled)",
                    cors_ok or preflight_handled,
                )
            )
        except Exception as e:
            self.results.append(ValidationResult("CORS check", False, str(e)))

    def _check_env_reset_all_tasks(self):
        try:
            from environment import ContractReviewEnv

            env = ContractReviewEnv()
            for tid in TASK_IDS:
                obs = env.reset(tid)
                self.results.append(
                    ValidationResult(
                        f"env.reset({tid}): returns observation",
                        obs is not None,
                    )
                )
        except Exception as e:
            self.results.append(ValidationResult("Environment reset", False, str(e)))

    def _check_env_optimal_trajectories(self):
        try:
            from environment import ContractReviewEnv
            from models import Action, ActionType

            env = ContractReviewEnv()
            tid = "task_1_easy"
            obs = env.reset(tid)

            for i in range(obs.total_clauses):
                gt = env.scenario.clauses[i]
                env.step(
                    Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
                )
                if i < obs.total_clauses - 1:
                    env.step(Action(action_type=ActionType.NEXT_CLAUSE))

            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            score = env.grader_result.score if env.grader_result else 0
            self.results.append(
                ValidationResult(
                    f"Optimal trajectory: score >= 0.9",
                    score >= 0.9,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Optimal trajectory", False, str(e)))

    def _check_env_wrong_trajectories(self):
        try:
            from environment import ContractReviewEnv
            from models import Action, ActionType

            env = ContractReviewEnv()
            tid = "task_1_easy"
            obs = env.reset(tid)

            for i in range(obs.total_clauses):
                env.step(
                    Action(action_type=ActionType.CLASSIFY, clause_type="force_majeure")
                )
                if i < obs.total_clauses - 1:
                    env.step(Action(action_type=ActionType.NEXT_CLAUSE))

            env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            score = env.grader_result.score if env.grader_result else 1
            self.results.append(
                ValidationResult(
                    f"Wrong trajectory: score < 0.5",
                    score < 0.5,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Wrong trajectory", False, str(e)))

    def _check_env_grader_range(self):
        try:
            from environment import ContractReviewEnv
            from models import Action, ActionType

            env = ContractReviewEnv()
            scores = []

            for tid in TASK_IDS:
                obs = env.reset(tid)
                gt = env.scenario.clauses[0]
                env.step(
                    Action(action_type=ActionType.CLASSIFY, clause_type=gt.clause_type)
                )
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
                score = env.grader_result.score if env.grader_result else 0
                scores.append(score)

            all_in_range = all(0.0 < s < 1.0 for s in scores)
            self.results.append(
                ValidationResult(
                    f"Grader range: all scores in (0,1)",
                    all_in_range,
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("Grader range", False, str(e)))

    def _check_env_state_serialization(self):
        try:
            from environment import ContractReviewEnv

            env = ContractReviewEnv()
            env.reset("task_1_easy")
            state = env.state()

            self.results.append(
                ValidationResult(
                    "State serialization: produces dict",
                    isinstance(state.model_dump(), dict),
                )
            )

        except Exception as e:
            self.results.append(ValidationResult("State serialization", False, str(e)))

    def _check_env_no_cross_episode_leakage(self):
        try:
            from environment import ContractReviewEnv
            from models import Action, ActionType

            env = ContractReviewEnv()

            env.reset("task_1_easy")
            env.step(
                Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality")
            )

            env.reset("task_2_medium")
            state = env.state()

            self.results.append(
                ValidationResult(
                    "No cross-episode leakage: step_number=0",
                    state.step_number == 0,
                )
            )

        except Exception as e:
            self.results.append(
                ValidationResult("Cross-episode leakage check", False, str(e))
            )

    def _check_docker_build(self):
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "contract-review-test", "."],
                capture_output=True,
                text=True,
                timeout=300,
            )
            message = ""
            if result.returncode != 0:
                raw = (result.stderr or result.stdout or "").strip()
                if raw:
                    message = raw[-1000:]
            self.results.append(
                ValidationResult(
                    "Docker build: success",
                    result.returncode == 0,
                    message,
                )
            )

            if result.returncode == 0:
                subprocess.run(
                    ["docker", "rmi", "contract-review-test"],
                    capture_output=True,
                )

        except FileNotFoundError:
            self.results.append(
                ValidationResult("Docker: not available", False, "Docker not installed")
            )
        except subprocess.TimeoutExpired:
            self.results.append(
                ValidationResult("Docker build: timeout", False, "Build took too long")
            )
        except Exception as e:
            self.results.append(ValidationResult("Docker build", False, str(e)))

    def _print_results(self, category: str, start_idx: int = 0):
        section_results = self.results[start_idx:]
        passed = sum(1 for r in section_results if r.passed)
        total = len(section_results)
        print(f"\n{category}: {passed}/{total} passed")

        for r in section_results:
            status = "PASS" if r.passed else "FAIL"
            symbol = "✓" if r.passed else "✗"
            print(f"  [{symbol}] {status}: {r.name}")
            if r.message:
                print(f"      {r.message}")

        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pre-submission validation")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server URL")
    parser.add_argument("--local", action="store_true", help="Run local checks")
    parser.add_argument("--server", action="store_true", help="Run server checks")
    parser.add_argument("--env", action="store_true", help="Run environment tests")
    parser.add_argument("--docker", action="store_true", help="Run Docker check")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()

    v = Validator(args.url)

    if args.local or args.all:
        v.run_all_local()

    if args.env or args.all:
        v.run_environment_tests()

    if args.server or args.all:
        v.run_all_server()

    if args.docker or args.all:
        v.run_docker_check()

    if not (args.local or args.server or args.env or args.docker or args.all):
        print("No checks selected. Use --help for options.")
        sys.exit(1)

    passed = sum(1 for r in v.results if r.passed)
    total = len(v.results)
    if passed == total:
        print(f"\nALL CHECKS PASSED ({passed}/{total})")
        sys.exit(0)
    else:
        print(f"\nSOME CHECKS FAILED ({passed}/{total})")
        sys.exit(1)


if __name__ == "__main__":
    main()
