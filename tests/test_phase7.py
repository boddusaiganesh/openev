"""
Phase 7 — Calibration & Stress Tests
Run: python -m pytest tests/test_phase7.py -v
"""

from __future__ import annotations

import pytest

from calibrate import (
    run_calibration,
    run_perfect_trajectory,
    run_good_trajectory,
    run_partial_trajectory,
    run_poor_trajectory,
    run_empty_trajectory,
    CalibrationResult,
)
from stress_test import (
    run_all_stress_tests,
    test_random_agent,
    test_rapid_resets,
    test_mid_episode_reset,
    test_degenerate_spam,
    test_boundary_max_steps,
    test_all_action_combos,
    test_grader_determinism,
    test_step_after_done,
    test_state_serialization_every_step,
    test_accumulated_score_consistency,
    StressResult,
)


class TestCalibration:
    """Test grader calibration across quality tiers."""

    def test_perfect_trajectory_returns_valid_score(self):
        from environment import ContractReviewEnv
        env = ContractReviewEnv()
        score = run_perfect_trajectory(env, "task_1_easy")
        assert 0.0 <= score <= 1.0
        assert score >= 0.9

    def test_good_trajectory_returns_valid_score(self):
        from environment import ContractReviewEnv
        env = ContractReviewEnv()
        score = run_good_trajectory(env, "task_1_easy")
        assert 0.0 <= score <= 1.0

    def test_partial_trajectory_returns_valid_score(self):
        from environment import ContractReviewEnv
        env = ContractReviewEnv()
        score = run_partial_trajectory(env, "task_1_easy")
        assert 0.0 <= score <= 1.0

    def test_poor_trajectory_returns_valid_score(self):
        from environment import ContractReviewEnv
        env = ContractReviewEnv()
        score = run_poor_trajectory(env, "task_1_easy")
        assert 0.0 <= score <= 1.0

    def test_empty_trajectory_returns_valid_score(self):
        from environment import ContractReviewEnv
        env = ContractReviewEnv()
        score = run_empty_trajectory(env, "task_1_easy")
        assert 0.0 <= score <= 1.0

    def test_calibration_monotonicity(self):
        results = run_calibration(verbose=False)
        for cr in results:
            assert cr.perfect >= cr.good >= cr.partial >= cr.poor >= cr.empty

    def test_calibration_spread(self):
        results = run_calibration(verbose=False)
        for cr in results:
            assert cr.spread > 0.0

    def test_calibration_all_tasks(self):
        results = run_calibration(verbose=False)
        assert len(results) == 3
        task_ids = [cr.task_id for cr in results]
        assert "task_1_easy" in task_ids
        assert "task_2_medium" in task_ids
        assert "task_3_hard" in task_ids


class TestStressTests:
    """Test stress scenarios."""

    def test_random_agent_no_crash(self):
        result = test_random_agent(verbose=False)
        assert result.passed

    def test_rapid_resets_no_leakage(self):
        result = test_rapid_resets(verbose=False)
        assert result.passed

    def test_mid_episode_reset_clean(self):
        result = test_mid_episode_reset(verbose=False)
        assert result.passed

    def test_degenerate_spam_escalates(self):
        result = test_degenerate_spam(verbose=False)
        assert result.passed

    def test_boundary_max_steps_terminates(self):
        result = test_boundary_max_steps(verbose=False)
        assert result.passed

    def test_all_action_combos_no_crash(self):
        result = test_all_action_combos(verbose=False)
        assert result.passed

    def test_grader_determinism(self):
        result = test_grader_determinism(verbose=False)
        assert result.passed

    def test_step_after_done_returns_zero(self):
        result = test_step_after_done(verbose=False)
        assert result.passed

    def test_state_serialization_every_step(self):
        result = test_state_serialization_every_step(verbose=False)
        assert result.passed

    def test_accumulated_score_consistency(self):
        result = test_accumulated_score_consistency(verbose=False)
        assert result.passed


class TestSubmissionReport:
    """Test submission report generation."""

    def test_check_files(self):
        from submission_report import check_files
        results = check_files()
        assert isinstance(results, dict)
        assert all(isinstance(v, bool) for v in results.values())

    def test_check_scenario_files(self):
        from submission_report import check_scenario_files
        results = check_scenario_files()
        assert isinstance(results, dict)

    def test_check_no_secrets(self):
        from submission_report import check_no_secrets
        result = check_no_secrets()
        assert isinstance(result, bool)

    def test_generate_report_structure(self):
        from submission_report import (
            check_files, check_scenario_files, check_no_secrets, generate_report
        )
        cal_results = run_calibration(verbose=False)
        stress_results = run_all_stress_tests(verbose=False)
        file_checks = check_files()
        scenario_checks = check_scenario_files()
        no_secrets = check_no_secrets()

        report = generate_report(
            cal_results, stress_results, file_checks, scenario_checks, no_secrets
        )

        assert "overall" in report
        assert "sections" in report
        assert "calibration" in report["sections"]
        assert "stress_tests" in report["sections"]
        assert "files" in report["sections"]
        assert "scenarios" in report["sections"]
        assert "security" in report["sections"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
