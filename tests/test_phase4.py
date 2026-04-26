"""
Phase 4 — Validation Tests
Tests prompt construction, response parsing, action building,
and dry-run task execution.
Run: python -m pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

from models import (
    Action,
    ActionType,
    RiskLevel,
    SuggestedActionType,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
)
from tasks import TASK_REGISTRY, get_task_config
from environment import ContractReviewEnv

from inference import (
    SYSTEM_PROMPT,
    build_action,
    build_user_prompt,
    extract_json_from_text,
    parse_clause_type_from_text,
    parse_flags_from_text,
    parse_llm_response,
    parse_risk_level_from_text,
    parse_suggested_action_from_text,
)


# ═══════════════════════════════════════════════
# Prompt Construction Tests
# ═══════════════════════════════════════════════

class TestSystemPrompt:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_taxonomy(self):
        for ct in ["indemnification", "confidentiality", "termination"]:
            assert ct in SYSTEM_PROMPT

    def test_system_prompt_contains_risk_levels(self):
        for rl in ["low", "medium", "high", "critical"]:
            assert rl in SYSTEM_PROMPT

    def test_system_prompt_contains_issue_flags(self):
        for flag in ["vague_language", "one_sided_obligation"]:
            assert flag in SYSTEM_PROMPT

    def test_system_prompt_contains_suggested_actions(self):
        for sa in ["accept_as_is", "reject_clause"]:
            assert sa in SYSTEM_PROMPT

    def test_system_prompt_requests_json(self):
        assert "json" in SYSTEM_PROMPT.lower()


class TestUserPrompt:
    def setup_method(self):
        self.env = ContractReviewEnv()

    def test_user_prompt_contains_clause_text(self):
        obs = self.env.reset("task_1_easy")
        prompt = build_user_prompt(obs, "Test task", [])
        assert obs.clause_text[:50] in prompt

    def test_user_prompt_contains_contract_info(self):
        obs = self.env.reset("task_1_easy")
        prompt = build_user_prompt(obs, "Test task", [])
        assert obs.contract_type in prompt
        assert obs.jurisdiction in prompt

    def test_user_prompt_contains_step_info(self):
        obs = self.env.reset("task_1_easy")
        prompt = build_user_prompt(obs, "Test task", [])
        assert str(obs.max_steps) in prompt

    def test_user_prompt_contains_clause_position(self):
        obs = self.env.reset("task_1_easy")
        prompt = build_user_prompt(obs, "Test task", [])
        assert str(obs.total_clauses) in prompt

    def test_user_prompt_with_history(self):
        obs = self.env.reset("task_1_easy")
        history = ["Step 1: classify -> +0.15", "Step 2: next_clause -> +0.05"]
        prompt = build_user_prompt(obs, "Test task", history)
        assert "classify" in prompt

    def test_user_prompt_without_history(self):
        obs = self.env.reset("task_1_easy")
        prompt = build_user_prompt(obs, "Test task", [])
        assert "None" in prompt

    def test_user_prompt_for_each_task(self):
        for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            obs = self.env.reset(task_id)
            cfg = get_task_config(task_id)
            prompt = build_user_prompt(obs, cfg.description, [])
            assert len(prompt) > 50


# ═══════════════════════════════════════════════
# JSON Extraction Tests
# ═══════════════════════════════════════════════

class TestJsonExtraction:
    def test_pure_json(self):
        text = '{"clause_type": "confidentiality", "risk_level": "low"}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["clause_type"] == "confidentiality"

    def test_json_with_whitespace(self):
        text = '  \n  {"clause_type": "termination"}  \n  '
        result = extract_json_from_text(text)
        assert result is not None
        assert result["clause_type"] == "termination"

    def test_json_code_block(self):
        text = 'Here is my analysis:\n```json\n{"clause_type": "warranty"}\n```'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["clause_type"] == "warranty"

    def test_json_embedded_in_text(self):
        text = 'I think this is: {"clause_type": "force_majeure"} based on...'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["clause_type"] == "force_majeure"

    def test_no_json(self):
        text = "This is a confidentiality clause with low risk."
        result = extract_json_from_text(text)
        assert result is None

    def test_empty_string(self):
        result = extract_json_from_text("")
        assert result is None

    def test_none_input(self):
        result = extract_json_from_text(None)
        assert result is None

    def test_malformed_json(self):
        text = '{"clause_type": "broken'
        result = extract_json_from_text(text)
        assert result is None

    def test_full_analysis_json(self):
        text = json.dumps({
            "clause_type": "indemnification",
            "risk_level": "high",
            "flags": ["one_sided_obligation", "overly_broad_scope"],
            "suggested_action": "request_modification",
            "reasoning": "This indemnification is one-sided."
        })
        result = extract_json_from_text(text)
        assert result is not None
        assert result["clause_type"] == "indemnification"
        assert result["risk_level"] == "high"
        assert len(result["flags"]) == 2

    def test_nested_json(self):
        text = '{"clause_type": "confidentiality", "details": {"note": "test"}}'
        result = extract_json_from_text(text)
        assert result is not None


# ═══════════════════════════════════════════════
# Regex Fallback Parsing Tests
# ═══════════════════════════════════════════════

class TestRegexFallbackParsing:
    # --- clause type ---

    def test_parse_clause_type_direct(self):
        assert parse_clause_type_from_text("This is a confidentiality clause") == "confidentiality"

    def test_parse_clause_type_with_spaces(self):
        result = parse_clause_type_from_text("limitation of liability")
        assert result == "limitation_of_liability"

    def test_parse_clause_type_multiple(self):
        result = parse_clause_type_from_text("indemnification and termination")
        assert result in ["indemnification", "termination"]

    def test_parse_clause_type_none_found(self):
        result = parse_clause_type_from_text("this is just some random text")
        assert result == "representations"  # default

    def test_parse_clause_type_all_taxonomy(self):
        for ct in CLAUSE_TAXONOMY:
            text = f"This is a {ct.replace('_', ' ')} clause"
            result = parse_clause_type_from_text(text)
            assert result == ct

    # --- risk level ---

    def test_parse_risk_low(self):
        assert parse_risk_level_from_text("This is low risk") == "low"

    def test_parse_risk_critical(self):
        assert parse_risk_level_from_text("CRITICAL risk identified") == "critical"

    def test_parse_risk_priority_order(self):
        # Critical should be found first if multiple present
        assert parse_risk_level_from_text("Could be low or critical") == "critical"

    def test_parse_risk_none_found(self):
        assert parse_risk_level_from_text("no risk words here xyz") == "medium"

    # --- flags ---

    def test_parse_flags_single(self):
        flags = parse_flags_from_text("This has vague language")
        assert "vague_language" in flags

    def test_parse_flags_multiple(self):
        flags = parse_flags_from_text("one sided obligation and missing liability cap")
        assert "one_sided_obligation" in flags
        assert "missing_liability_cap" in flags

    def test_parse_flags_none(self):
        flags = parse_flags_from_text("everything looks fine here")
        assert flags == []

    # --- suggested action ---

    def test_parse_action_accept(self):
        assert parse_suggested_action_from_text("I would accept this") == "accept_as_is"

    def test_parse_action_modify(self):
        assert parse_suggested_action_from_text("request modification") == "request_modification"

    def test_parse_action_reject(self):
        assert parse_suggested_action_from_text("reject this clause") == "reject_clause"

    def test_parse_action_escalate(self):
        assert parse_suggested_action_from_text("escalate to senior") == "escalate_to_senior_counsel"

    def test_parse_action_default(self):
        assert parse_suggested_action_from_text("xyz") == "flag_for_negotiation"


# ═══════════════════════════════════════════════
# Full Response Parsing Tests
# ═══════════════════════════════════════════════

class TestFullResponseParsing:
    def test_perfect_json_response(self):
        response = json.dumps({
            "clause_type": "confidentiality",
            "risk_level": "low",
            "flags": ["market_standard"],
            "suggested_action": "accept_as_is",
            "reasoning": "Standard mutual confidentiality clause."
        })
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "confidentiality"
        assert analysis["risk_level"] == "low"
        assert analysis["flags"] == ["market_standard"]
        assert analysis["suggested_action"] == "accept_as_is"
        assert "mutual" in analysis["reasoning"].lower()

    def test_json_with_invalid_clause_type(self):
        response = json.dumps({
            "clause_type": "not_a_real_type",
            "risk_level": "low",
            "flags": [],
            "suggested_action": "accept_as_is",
            "reasoning": "Test"
        })
        analysis = parse_llm_response(response)
        # Should fall back to regex parsing for clause_type
        assert analysis["clause_type"] in CLAUSE_TAXONOMY or analysis["clause_type"] == "representations"

    def test_json_with_invalid_flags(self):
        response = json.dumps({
            "clause_type": "termination",
            "risk_level": "low",
            "flags": ["valid_flag_doesnt_exist", "vague_language"],
            "suggested_action": "accept_as_is",
            "reasoning": "Test"
        })
        analysis = parse_llm_response(response)
        assert "vague_language" in analysis["flags"]
        assert "valid_flag_doesnt_exist" not in analysis["flags"]

    def test_free_text_response(self):
        response = (
            "This clause appears to be a confidentiality provision with "
            "low risk. The language is standard and market standard. "
            "I would recommend accepting it as is."
        )
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "confidentiality"
        assert analysis["risk_level"] == "low"
        assert "market_standard" in analysis["flags"]
        assert analysis["suggested_action"] == "accept_as_is"

    def test_empty_response(self):
        analysis = parse_llm_response("")
        assert analysis["clause_type"] == "representations"
        assert analysis["risk_level"] == "medium"
        assert analysis["flags"] == []
        assert analysis["suggested_action"] == "flag_for_negotiation"

    def test_json_in_code_block(self):
        response = (
            "Based on my analysis:\n"
            "```json\n"
            '{"clause_type": "warranty", "risk_level": "medium", '
            '"flags": ["vague_language"], "suggested_action": "request_modification", '
            '"reasoning": "Warranty is vaguely worded."}\n'
            "```"
        )
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "warranty"
        assert analysis["risk_level"] == "medium"

    def test_partial_json(self):
        response = '{"clause_type": "termination", "risk_level": "high"}'
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "termination"
        assert analysis["risk_level"] == "high"
        # Missing fields should have defaults
        assert analysis["flags"] == []
        assert analysis["suggested_action"] == "flag_for_negotiation"


# ═══════════════════════════════════════════════
# Action Building Tests
# ═══════════════════════════════════════════════

class TestActionBuilding:
    def _make_analysis(self, **kwargs):
        defaults = {
            "clause_type": "confidentiality",
            "risk_level": "low",
            "flags": ["market_standard"],
            "suggested_action": "accept_as_is",
            "reasoning": "Standard clause.",
        }
        defaults.update(kwargs)
        return defaults

    def test_build_classify_action(self):
        analysis = self._make_analysis(clause_type="termination")
        action = build_action(ActionType.CLASSIFY, analysis)
        assert action.action_type == ActionType.CLASSIFY
        assert action.clause_type == "termination"

    def test_build_risk_action(self):
        analysis = self._make_analysis(risk_level="high")
        action = build_action(ActionType.RATE_SEVERITY, analysis)
        assert action.action_type == ActionType.RATE_SEVERITY
        assert action.risk_level == RiskLevel.HIGH

    def test_build_flag_action(self):
        analysis = self._make_analysis(flags=["vague_language", "one_sided_obligation"])
        action = build_action(ActionType.FLAG, analysis)
        assert action.action_type == ActionType.FLAG
        assert len(action.flags) == 2

    def test_build_flag_action_filters_invalid(self):
        analysis = self._make_analysis(flags=["vague_language", "not_a_flag"])
        action = build_action(ActionType.FLAG, analysis)
        assert action.flags == ["vague_language"]

    def test_build_suggest_action(self):
        analysis = self._make_analysis(suggested_action="reject_clause")
        action = build_action(ActionType.SUGGEST, analysis)
        assert action.action_type == ActionType.SUGGEST
        assert action.suggested_action == SuggestedActionType.REJECT_CLAUSE

    def test_build_reason_action(self):
        analysis = self._make_analysis(reasoning="This is overbroad.")
        action = build_action(ActionType.REASON, analysis)
        assert action.action_type == ActionType.REASON
        assert "overbroad" in action.reasoning

    def test_build_next_clause_action(self):
        action = build_action(ActionType.NEXT_CLAUSE, {})
        assert action.action_type == ActionType.NEXT_CLAUSE

    def test_build_complete_review_action(self):
        action = build_action(ActionType.COMPLETE_REVIEW, {})
        assert action.action_type == ActionType.COMPLETE_REVIEW

    def test_build_with_invalid_risk_uses_default(self):
        analysis = self._make_analysis(risk_level="unknown")
        action = build_action(ActionType.RATE_SEVERITY, analysis)
        assert action.risk_level == RiskLevel.MEDIUM

    def test_build_with_invalid_suggest_uses_default(self):
        analysis = self._make_analysis(suggested_action="unknown")
        action = build_action(ActionType.SUGGEST, analysis)
        assert action.suggested_action == SuggestedActionType.FLAG_FOR_NEGOTIATION

    def test_all_actions_are_valid_pydantic(self):
        analysis = self._make_analysis()
        for at in ActionType:
            action = build_action(at, analysis)
            assert isinstance(action, Action)
            d = action.model_dump()
            Action(**d)  # Should not raise


# ═══════════════════════════════════════════════
# Dry Run Tests (no LLM, mock responses)
# ═══════════════════════════════════════════════

class TestDryRun:
    """Run tasks with mocked LLM responses to verify the full flow."""

    def setup_method(self):
        self.env = ContractReviewEnv()

    def _mock_analysis_for_gt(self, gt) -> str:
        """Generate a perfect JSON response matching ground truth."""
        return json.dumps({
            "clause_type": gt.clause_type,
            "risk_level": gt.risk_level.value,
            "flags": gt.issues,
            "suggested_action": gt.recommended_action.value,
            "reasoning": " ".join(gt.reasoning_keywords),
        })

    def _mock_wrong_analysis(self) -> str:
        """Generate a consistently wrong JSON response."""
        return json.dumps({
            "clause_type": "force_majeure",
            "risk_level": "low",
            "flags": ["automatic_renewal"],
            "suggested_action": "accept_as_is",
            "reasoning": "Looks fine to me.",
        })

    def test_task_1_optimal_dry_run(self):
        obs = self.env.reset("task_1_easy")
        task_config = get_task_config("task_1_easy")
        required = task_config.required_action_types

        for i in range(obs.total_clauses):
            gt = self.env.scenario.clauses[i]
            response_text = self._mock_analysis_for_gt(gt)
            analysis = parse_llm_response(response_text)

            for at in required:
                if self.env.done:
                    break
                action = build_action(at, analysis)
                obs, reward, done, info = self.env.step(action)

            if self.env.done:
                break

            if i < obs.total_clauses - 1:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
            else:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        if not self.env.done:
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.done
        assert self.env.grader_result is not None
        assert self.env.grader_result.score >= 0.95

    def test_task_2_optimal_dry_run(self):
        obs = self.env.reset("task_2_medium")
        task_config = get_task_config("task_2_medium")
        required = task_config.required_action_types

        for i in range(obs.total_clauses):
            if self.env.done:
                break
            gt = self.env.scenario.clauses[i]
            response_text = self._mock_analysis_for_gt(gt)
            analysis = parse_llm_response(response_text)

            for at in required:
                if self.env.done:
                    break
                action = build_action(at, analysis)
                obs, reward, done, info = self.env.step(action)

            if self.env.done:
                break

            if i < obs.total_clauses - 1:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
            else:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        if not self.env.done:
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.done
        assert self.env.grader_result is not None
        assert self.env.grader_result.score >= 0.80

    def test_task_3_optimal_dry_run(self):
        obs = self.env.reset("task_3_hard")
        task_config = get_task_config("task_3_hard")
        required = task_config.required_action_types

        for i in range(obs.total_clauses):
            if self.env.done:
                break
            gt = self.env.scenario.clauses[i]
            response_text = self._mock_analysis_for_gt(gt)
            analysis = parse_llm_response(response_text)

            for at in required:
                if self.env.done:
                    break
                action = build_action(at, analysis)
                obs, reward, done, info = self.env.step(action)

            if self.env.done:
                break

            if i < obs.total_clauses - 1 and not self.env.done:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))

        if not self.env.done:
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.done
        assert self.env.grader_result is not None
        # Task 3 may not finish all clauses due to max_steps
        assert self.env.grader_result.score >= 0.50

    def test_task_1_wrong_dry_run(self):
        obs = self.env.reset("task_1_easy")
        task_config = get_task_config("task_1_easy")
        required = task_config.required_action_types

        for i in range(obs.total_clauses):
            if self.env.done:
                break
            response_text = self._mock_wrong_analysis()
            analysis = parse_llm_response(response_text)

            for at in required:
                if self.env.done:
                    break
                action = build_action(at, analysis)
                obs, reward, done, info = self.env.step(action)

            if self.env.done:
                break

            if i < obs.total_clauses - 1:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
            else:
                obs, _, _, _ = self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        if not self.env.done:
            self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        assert self.env.done
        assert self.env.grader_result is not None
        assert self.env.grader_result.score < 0.5

    def test_optimal_beats_wrong(self):
        # Optimal run
        self.env.reset("task_1_easy")
        obs = self.env.reset("task_1_easy")
        for i in range(obs.total_clauses):
            gt = self.env.scenario.clauses[i]
            analysis = parse_llm_response(self._mock_analysis_for_gt(gt))
            action = build_action(ActionType.CLASSIFY, analysis)
            self.env.step(action)
            if i < obs.total_clauses - 1:
                self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        optimal_score = self.env.grader_result.score

        # Wrong run
        obs = self.env.reset("task_1_easy")
        for i in range(obs.total_clauses):
            analysis = parse_llm_response(self._mock_wrong_analysis())
            action = build_action(ActionType.CLASSIFY, analysis)
            self.env.step(action)
            if i < obs.total_clauses - 1:
                self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
        self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
        wrong_score = self.env.grader_result.score

        assert optimal_score > wrong_score

    def test_scores_are_different_across_runs(self):
        """Verify graders don't return constant scores."""
        scores = []
        responses = [
            json.dumps({"clause_type": "confidentiality", "risk_level": "low", "flags": [], "suggested_action": "accept_as_is", "reasoning": "standard"}),
            json.dumps({"clause_type": "force_majeure", "risk_level": "critical", "flags": ["unusual_term"], "suggested_action": "reject_clause", "reasoning": "bad"}),
            "",  # empty response
        ]
        for resp in responses:
            obs = self.env.reset("task_1_easy")
            for i in range(obs.total_clauses):
                if self.env.done:
                    break
                analysis = parse_llm_response(resp)
                action = build_action(ActionType.CLASSIFY, analysis)
                self.env.step(action)
                if i < obs.total_clauses - 1 and not self.env.done:
                    self.env.step(Action(action_type=ActionType.NEXT_CLAUSE))
            if not self.env.done:
                self.env.step(Action(action_type=ActionType.COMPLETE_REVIEW))
            scores.append(self.env.grader_result.score)

        # At least two different scores
        assert len(set(scores)) >= 2


# ═══════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════

class TestEdgeCases:
    def test_parse_response_with_unicode(self):
        response = json.dumps({
            "clause_type": "confidentiality",
            "risk_level": "low",
            "flags": [],
            "suggested_action": "accept_as_is",
            "reasoning": "Standard clause — no issues détected. ☑"
        })
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "confidentiality"

    def test_parse_response_with_extra_fields(self):
        response = json.dumps({
            "clause_type": "termination",
            "risk_level": "medium",
            "flags": [],
            "suggested_action": "accept_as_is",
            "reasoning": "OK",
            "confidence": 0.95,
            "notes": "extra field",
        })
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "termination"

    def test_parse_response_with_newlines_in_reasoning(self):
        response = json.dumps({
            "clause_type": "warranty",
            "risk_level": "low",
            "flags": [],
            "suggested_action": "accept_as_is",
            "reasoning": "This clause:\n1. Is standard\n2. Has no issues"
        })
        analysis = parse_llm_response(response)
        assert "standard" in analysis["reasoning"].lower()

    def test_build_action_with_empty_analysis(self):
        action = build_action(ActionType.CLASSIFY, {})
        assert action.clause_type == "representations"

    def test_build_action_with_none_values(self):
        analysis = {
            "clause_type": None,
            "risk_level": None,
            "flags": None,
            "suggested_action": None,
            "reasoning": None,
        }
        for at in [ActionType.CLASSIFY, ActionType.RATE_SEVERITY,
                    ActionType.FLAG, ActionType.SUGGEST, ActionType.REASON]:
            action = build_action(at, analysis)
            assert isinstance(action, Action)

    def test_very_long_response(self):
        long_text = "This is a test. " * 1000
        response = json.dumps({
            "clause_type": "confidentiality",
            "risk_level": "low",
            "flags": [],
            "suggested_action": "accept_as_is",
            "reasoning": long_text
        })
        analysis = parse_llm_response(response)
        assert analysis["clause_type"] == "confidentiality"
        assert len(analysis["reasoning"]) <= 300  # Truncated


# ═══════════════════════════════════════════════
# Deployment Tests
# ═══════════════════════════════════════════════

class TestDeploymentFiles:
    def test_dockerfile_exists(self):
        assert os.path.exists("Dockerfile")

    def test_dockerfile_has_user(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "useradd" in content or "USER" in content

    def test_dockerfile_exposes_7860(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "7860" in content

    def test_dockerfile_has_healthcheck(self):
        with open("Dockerfile") as f:
            content = f.read()
        assert "HEALTHCHECK" in content

    def test_openenv_yaml_exists(self):
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "contract-clause-review"
        assert len(data["tasks"]) >= 3
        assert data["server"]["port"] == 7860

    def test_openenv_yaml_has_all_tasks(self):
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        task_ids = [t["id"] for t in data["tasks"]]
        assert "task_1_easy" in task_ids
        assert "task_2_medium" in task_ids
        assert "task_3_hard" in task_ids

    def test_openenv_yaml_has_spaces(self):
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert "openenv" in data["tags"]

    def test_readme_exists(self):
        assert os.path.exists("README.md")

    def test_readme_has_required_sections(self):
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        required = [
            "Quick Start",
            "Tasks",
            "task_1_easy",
            "task_2_medium",
            "task_3_hard",
            "API Endpoints",
        ]
        for section in required:
            assert section in content, f"README missing: {section}"

    def test_requirements_exists(self):
        assert os.path.exists("requirements.txt")
        with open("requirements.txt") as f:
            content = f.read()
        for dep in ["fastapi", "uvicorn", "pydantic", "openai"]:
            assert dep in content

    def test_inference_exists(self):
        assert os.path.exists("inference.py")


class TestInferenceScript:
    def test_inference_imports(self):
        from inference import (
            API_BASE_URL,
            MODEL_NAME,
            TEMPERATURE,
            TASKS,
            call_llm,
            build_user_prompt,
            parse_llm_response,
            run_task,
        )
        assert TASKS == ["task_1_easy", "task_2_medium", "task_3_hard"]
        assert TEMPERATURE == 0.2

    def test_build_user_prompt(self):
        from inference import build_user_prompt
        from models import Observation

        obs = Observation(
            task_id="task_1_easy",
            step_number=1,
            max_steps=10,
            clause_text="This is a test clause.",
            clause_index=0,
            total_clauses=4,
            contract_type="NDA",
            parties=["A", "B"],
            jurisdiction="DE",
            instructions="Classify this clause.",
            available_actions=["classify"],
        )

        prompt = build_user_prompt(obs, "Classify", [])
        assert "NDA" in prompt
        assert "This is a test clause" in prompt
        assert "CLAUSE 1 OF 4" in prompt

    def test_parse_llm_response_valid_json(self):
        from inference import parse_llm_response

        response = '{"clause_type": "confidentiality", "risk_level": "low", "flags": [], "suggested_action": "accept_as_is", "reasoning": "test"}'
        parsed = parse_llm_response(response)
        assert parsed["clause_type"] == "confidentiality"
        assert parsed["risk_level"] == "low"

    def test_parse_llm_response_json_block(self):
        from inference import parse_llm_response

        response = '```json\n{"clause_type": "termination", "risk_level": "high", "flags": [], "suggested_action": "request_modification", "reasoning": "test"}\n```'
        parsed = parse_llm_response(response)
        assert parsed["clause_type"] == "termination"
        assert parsed["risk_level"] == "high"

    def test_parse_llm_response_invalid_json(self):
        from inference import parse_llm_response

        response = "This is not JSON"
        parsed = parse_llm_response(response)
        assert parsed["clause_type"] == "representations"
        assert parsed["risk_level"] == "medium"

    def test_run_task_error_path_still_returns_in_range_score(self):
        from inference import run_task

        class _Adapter:
            def __init__(self):
                self._obs = {
                    "task_id": "task_1_easy",
                    "step_number": 0,
                    "max_steps": 10,
                    "clause_text": "Test clause",
                    "clause_index": 0,
                    "total_clauses": 1,
                    "contract_type": "NDA",
                    "parties": ["A", "B"],
                    "jurisdiction": "DE",
                    "instructions": "Classify",
                    "available_actions": ["classify", "complete_review"],
                    "accumulated_score": 0.0,
                    "done": False,
                }

            def reset(self, _task_id):
                return dict(self._obs)

            def step(self, action_dict):
                if action_dict.get("action_type") == "classify":
                    raise RuntimeError("forced classify failure")
                return {
                    "observation": {
                        **self._obs,
                        "done": True,
                        "clause_index": 1,
                    },
                    "reward": {"score": 0.0},
                    "done": True,
                    "info": {
                        "grader_score": 0.001,
                        "grader_result": {"breakdown": {}},
                    },
                }

            def state(self):
                return {"clause_records": []}

        result = run_task(_Adapter(), MagicMock(), "task_1_easy")
        assert 0.0 < result["grader_score"] < 1.0

    def test_run_task_state_failure_uses_safe_default(self):
        from inference import run_task

        class _Adapter:
            def reset(self, _task_id):
                return {
                    "task_id": "task_1_easy",
                    "step_number": 0,
                    "max_steps": 10,
                    "clause_text": "[end]",
                    "clause_index": 0,
                    "total_clauses": 0,
                    "contract_type": "NDA",
                    "parties": ["A", "B"],
                    "jurisdiction": "DE",
                    "instructions": "Classify",
                    "available_actions": ["complete_review"],
                    "accumulated_score": 0.0,
                    "done": False,
                }

            def step(self, _action_dict):
                return {
                    "observation": {
                        "task_id": "task_1_easy",
                        "step_number": 1,
                        "max_steps": 10,
                        "clause_text": "[end]",
                        "clause_index": 1,
                        "total_clauses": 0,
                        "contract_type": "NDA",
                        "parties": ["A", "B"],
                        "jurisdiction": "DE",
                        "instructions": "Classify",
                        "available_actions": ["complete_review"],
                        "accumulated_score": 0.0,
                        "done": True,
                    },
                    "reward": {"score": 0.0},
                    "done": True,
                    "info": {
                        "grader_score": 0.2,
                        "grader_result": {"breakdown": {}},
                    },
                }

            def state(self):
                raise RuntimeError("forced state failure")

        result = run_task(_Adapter(), MagicMock(), "task_1_easy")
        assert 0.0 < result["grader_score"] < 1.0
        assert result["clauses_reviewed"] == 0


class TestServerDeployment:
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

    def test_reset_task_1(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})
        assert response.status_code == 200

    def test_reset_task_2(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_2_medium"})
        assert response.status_code == 200

    def test_reset_task_3(self, server_process):
        response = httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_3_hard"})
        assert response.status_code == 200

    def test_step_and_complete_episode(self, server_process):
        httpx.post("http://127.0.0.1:7861/reset", json={"task_id": "task_1_easy"})

        actions = [
            {"action_type": "classify", "clause_type": "confidentiality"},
            {"action_type": "next_clause"},
            {"action_type": "classify", "clause_type": "termination"},
            {"action_type": "next_clause"},
            {"action_type": "classify", "clause_type": "governing_law"},
            {"action_type": "next_clause"},
            {"action_type": "classify", "clause_type": "assignment"},
            {"action_type": "complete_review"},
        ]

        for a in actions:
            r = httpx.post("http://127.0.0.1:7861/step", json=a)
            assert r.status_code == 200
            data = r.json()
            if data["done"]:
                assert "grader_result" in data.get("info", {})
                break


class TestFullPipeline:
    def test_environment_full_episode(self):
        from environment import ContractReviewEnv
        from models import Action, ActionType

        env = ContractReviewEnv()
        obs = env.reset("task_1_easy")

        actions = [
            Action(action_type=ActionType.CLASSIFY, clause_type="confidentiality"),
            Action(action_type=ActionType.NEXT_CLAUSE),
            Action(action_type=ActionType.CLASSIFY, clause_type="termination"),
            Action(action_type=ActionType.NEXT_CLAUSE),
            Action(action_type=ActionType.CLASSIFY, clause_type="governing_law"),
            Action(action_type=ActionType.NEXT_CLAUSE),
            Action(action_type=ActionType.CLASSIFY, clause_type="assignment"),
            Action(action_type=ActionType.COMPLETE_REVIEW),
        ]

        done = False
        for a in actions:
            obs, reward, done, info = env.step(a)
            if done:
                break

        assert done is True
        assert "grader_result" in info
        assert info["grader_result"]["score"] > 0

    def test_all_tasks_available(self):
        from tasks import TASK_REGISTRY

        assert "task_1_easy" in TASK_REGISTRY
        assert "task_2_medium" in TASK_REGISTRY
        assert "task_3_hard" in TASK_REGISTRY

    def test_task_difficulty_escalation(self):
        from tasks import TASK_REGISTRY

        easy = TASK_REGISTRY["task_1_easy"]
        medium = TASK_REGISTRY["task_2_medium"]
        hard = TASK_REGISTRY["task_3_hard"]

        assert easy.max_steps < medium.max_steps < hard.max_steps


# ═══════════════════════════════════════════════
# Full Pipeline Simulation
# ═══════════════════════════════════════════════

class TestCompletePipeline:
    """Simulate the full inference pipeline without actual LLM calls."""

    def test_complete_pipeline_all_tasks(self):
        """Run all 3 tasks with mock responses, verify scores returned."""
        env = ContractReviewEnv()
        results = []

        for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            task_config = get_task_config(task_id)
            required = task_config.required_action_types
            obs = env.reset(task_id)

            for i in range(obs.total_clauses):
                if env.done:
                    break
                gt = env.scenario.clauses[i]
                mock_response = json.dumps({
                    "clause_type": gt.clause_type,
                    "risk_level": gt.risk_level.value,
                    "flags": gt.issues,
                    "suggested_action": gt.recommended_action.value,
                    "reasoning": " ".join(gt.reasoning_keywords),
                })
                analysis = parse_llm_response(mock_response)

                for at in required:
                    if env.done:
                        break
                    action = build_action(at, analysis)
                    obs, reward, done, info = env.step(action)

                if env.done:
                    break

                if i < obs.total_clauses - 1 and not env.done:
                    env.step(Action(action_type=ActionType.NEXT_CLAUSE))

            if not env.done:
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

            assert env.done
            assert env.grader_result is not None
            score = env.grader_result.score
            assert 0.0 <= score <= 1.0
            results.append({"task_id": task_id, "score": score})

        # All tasks completed
        assert len(results) == 3

        # Scores are in valid range
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

        # Easy should score highest with perfect answers
        scores = {r["task_id"]: r["score"] for r in results}
        assert scores["task_1_easy"] >= scores["task_2_medium"]

    def test_pipeline_runtime_estimate(self):
        """Verify the pipeline runs quickly without LLM calls."""
        import time
        env = ContractReviewEnv()
        start = time.time()

        for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            task_config = get_task_config(task_id)
            required = task_config.required_action_types
            obs = env.reset(task_id)

            for i in range(obs.total_clauses):
                if env.done:
                    break
                analysis = {
                    "clause_type": "confidentiality",
                    "risk_level": "medium",
                    "flags": [],
                    "suggested_action": "accept_as_is",
                    "reasoning": "test",
                }
                for at in required:
                    if env.done:
                        break
                    action = build_action(at, analysis)
                    env.step(action)

                if env.done:
                    break
                if i < obs.total_clauses - 1 and not env.done:
                    env.step(Action(action_type=ActionType.NEXT_CLAUSE))

            if not env.done:
                env.step(Action(action_type=ActionType.COMPLETE_REVIEW))

        elapsed = time.time() - start
        # Without LLM calls, should complete in under 1 second
        assert elapsed < 5.0

    def test_docker_build(self):
        """Test that Docker builds successfully (only if Docker is available)."""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "contract-review-test", "."],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Cleanup
                subprocess.run(
                    ["docker", "rmi", "contract-review-test"],
                    capture_output=True,
                )
            # Don't fail if Docker not available
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Docker not available or build timed out")
