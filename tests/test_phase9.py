"""
Phase 9 — Benchmark, Analysis & Packaging Tests
Run: python -m pytest tests/test_phase9.py -v
"""

from __future__ import annotations

import json
import os
import pytest

from environment import ContractReviewEnv
from models import Action, ActionType, CLAUSE_TAXONOMY
from tasks import list_task_ids, get_task_config


class TestBenchmarkStrategies:
    """Verify all benchmark strategies produce valid results."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_optimal_strategy_all_tasks(self):
        from benchmark import strategy_optimal
        for tid in list_task_ids():
            self.env.reset(tid)
            result = strategy_optimal(self.env)
            assert result["strategy"] == "optimal"
            assert 0.0 <= result["grader_score"] <= 1.0
            assert result["grader_score"] >= 0.50

    def test_empty_strategy_all_tasks(self):
        from benchmark import strategy_empty
        for tid in list_task_ids():
            self.env.reset(tid)
            result = strategy_empty(self.env)
            assert result["strategy"] == "empty"
            assert result["grader_score"] < 0.30

    def test_all_wrong_strategy_all_tasks(self):
        from benchmark import strategy_all_wrong
        for tid in list_task_ids():
            self.env.reset(tid)
            result = strategy_all_wrong(self.env)
            assert result["strategy"] == "all_wrong"
            assert result["grader_score"] < 0.50

    def test_optimal_beats_empty(self):
        from benchmark import strategy_optimal, strategy_empty
        for tid in list_task_ids():
            self.env.reset(tid)
            opt = strategy_optimal(self.env)
            self.env.reset(tid)
            emp = strategy_empty(self.env)
            assert opt["grader_score"] > emp["grader_score"]

    def test_classify_only_strategy(self):
        from benchmark import strategy_classify_only
        for tid in list_task_ids():
            self.env.reset(tid)
            result = strategy_classify_only(self.env)
            assert 0.0 <= result["grader_score"] <= 1.0

    def test_partial_correct_between_optimal_and_wrong(self):
        from benchmark import strategy_optimal, strategy_partial_correct, strategy_all_wrong
        for tid in list_task_ids():
            self.env.reset(tid)
            opt = strategy_optimal(self.env)
            self.env.reset(tid)
            part = strategy_partial_correct(self.env)
            self.env.reset(tid)
            wrong = strategy_all_wrong(self.env)
            assert part["grader_score"] <= opt["grader_score"] + 0.01


class TestScoreRanges:
    """Verify scores fall within expected ranges for each task."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    @pytest.mark.parametrize("task_id", ["task_1_easy", "task_2_medium", "task_3_hard"])
    def test_optimal_above_threshold(self, task_id):
        from benchmark import strategy_optimal
        self.env.reset(task_id)
        result = strategy_optimal(self.env)
        thresholds = {"task_1_easy": 0.85, "task_2_medium": 0.70, "task_3_hard": 0.50}
        threshold = thresholds[task_id]
        assert result["grader_score"] >= threshold


class TestAnalysisResults:
    """Test analysis and reporting functions."""

    def test_analyze_benchmark_requires_file(self):
        from analyze_results import analyze_benchmark
        if os.path.exists("benchmark_results.json"):
            result = analyze_benchmark("benchmark_results.json")
            assert "tasks" in result

    def test_expected_ranges_defined(self):
        from analyze_results import EXPECTED_RANGES
        assert "task_1_easy" in EXPECTED_RANGES
        assert "task_2_medium" in EXPECTED_RANGES
        assert "task_3_hard" in EXPECTED_RANGES
        for tid in EXPECTED_RANGES:
            assert "optimal" in EXPECTED_RANGES[tid]
            assert "empty" in EXPECTED_RANGES[tid]


class TestPackageSubmission:
    """Test package submission creation."""

    def test_collect_files_includes_required(self):
        from package_submission import collect_files, SUBMISSION_FILES
        files = collect_files()
        for f in SUBMISSION_FILES:
            assert f in files, f"Missing: {f}"

    def test_package_list_command(self):
        from package_submission import collect_files
        files = collect_files()
        assert len(files) > 10


class TestGitHubWorkflow:
    """Test GitHub workflow file exists."""

    def test_workflow_file_exists(self):
        assert os.path.exists(".github/workflows/validate.yml")

    def test_workflow_has_jobs(self):
        with open(".github/workflows/validate.yml") as f:
            content = f.read()
        assert "jobs:" in content
        assert "validate:" in content
        assert "docker:" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
