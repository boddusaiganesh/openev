"""
LexArena — Hardened Core Environment Logic (Tier 2)
====================================================
Implements the OpenEnv-compliant ContractReviewEnv with all bug fixes.

Key fixes over v1:
  1. _detect_degenerate() now only fires when the record ALREADY HAS the
     action stored — normal classify→rate_severity→flag sequences on the
     same clause are NOT degenerate.
  2. corrective_feedback is propagated through Observation every step.
  3. Episode termination is protected: cannot be faked via extra steps.
  4. State is fully resettable with no cross-episode leakage.
  5. All global state is instance-scoped (no class-level statics).

Design (hackathon rubric §4, §5, §8):
  What does the agent observe? → Observation (clause text + metadata + feedback)
  What actions can it take?    → classify|rate_severity|flag|suggest|reason|
                                  next_clause|complete_review
  What ends an episode?        → complete_review OR max_steps reached OR
                                  all clauses exhausted
  How is reward computed?      → rewards.py (multiple independent components)
  Anti-abuse?                  → degenerate penalty, invalid penalty, timeout
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    ClauseActionRecord,
    ClauseGroundTruth,
    ContractMeta,
    EnvironmentState,
    EpisodeMeta,
    GraderResult,
    Observation,
    Reward,
    ScenarioData,
    TaskConfig,
)
from tasks import (
    TASK_REGISTRY,
    get_task_config,
    get_task_instruction,
    list_task_ids,
)
from rewards import (
    compute_classify_reward,
    compute_completion_reward,
    compute_degenerate_penalty,
    compute_flag_reward,
    compute_invalid_action_penalty,
    compute_no_clause_penalty,
    compute_reason_reward,
    compute_risk_reward,
    compute_suggest_reward,
    compute_progress_reward,
)
from graders import grade_episode


logger = logging.getLogger("lexarena.environment")


class ContractReviewEnv:
    """
    OpenEnv-compliant environment for Contract Clause Review (Tier 2).

    Hardened with:
    - Defensive coding for all edge cases
    - Corrective feedback at every step
    - Proper degenerate detection (doesn't penalise normal action sequences)
    - No cross-episode state leakage
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.manifest: Dict = self._load_manifest()

        # Episode state — all reset in reset()
        self.task_config:            Optional[TaskConfig]    = None
        self.scenario:               Optional[ScenarioData]  = None
        self.step_number:            int                     = 0
        self.clause_index:           int                     = 0
        self.actions_taken:          List[Action]            = []
        self.rewards_given:          List[Reward]            = []
        self.clause_records:         List[ClauseActionRecord] = []
        self.accumulated_score:      float                   = 0.0
        self.done:                   bool                    = False
        self.episode_start_time:     Optional[str]           = None
        self.grader_result:          Optional[GraderResult]  = None
        self._last_corrective:       str                     = ""

        # Action-tracking for degenerate detection (per-clause, per-action-type)
        self._last_action_type:      Optional[ActionType]   = None
        self._last_clause_index:     int                    = -1
        self._consecutive_repeats:   int                    = 0

        # Counters
        self._invalid_action_count:  int  = 0
        self._redundant_action_count: int = 0

        # Audit flag
        self._score_audit_verbose: bool = os.getenv(
            "SCORE_AUDIT_VERBOSE", "false"
        ).lower() in ("1", "true", "yes")

        # Scenario tracking
        self._active_scenario_index: int  = 0
        self._active_scenario_file:  str  = ""

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str, scenario_index: Optional[int] = None) -> Observation:
        """Load scenario, wipe all state, return first observation."""
        self.task_config = get_task_config(task_id)

        task_entry     = self.manifest.get(task_id)
        if not task_entry:
            raise ValueError(f"Unknown task_id in manifest: {task_id}")
        scenario_files = task_entry.get("scenario_files", [])
        if not scenario_files:
            raise ValueError(f"No scenario files for task: {task_id}")

        if scenario_index is None:
            scenario_rel_path      = scenario_files[0]
            resolved_scenario_index = 0
        else:
            if scenario_index < 0 or scenario_index >= len(scenario_files):
                raise ValueError(
                    f"Invalid scenario_index {scenario_index} for task {task_id}. "
                    f"Expected 0..{len(scenario_files)-1}."
                )
            scenario_rel_path      = scenario_files[scenario_index]
            resolved_scenario_index = scenario_index

        self.scenario               = self._load_scenario(task_id, scenario_rel_path)
        self._active_scenario_index = resolved_scenario_index
        self._active_scenario_file  = scenario_rel_path

        # Wipe all state (no cross-episode leakage)
        self.step_number         = 0
        self.clause_index        = 0
        self.actions_taken       = []
        self.rewards_given       = []
        self.accumulated_score   = 0.0
        self.done                = False
        self.episode_start_time  = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.grader_result       = None
        self._last_corrective    = ""

        self._last_action_type     = None
        self._last_clause_index    = -1
        self._consecutive_repeats  = 0
        self._invalid_action_count = 0
        self._redundant_action_count = 0

        self.clause_records = [
            ClauseActionRecord(clause_index=i)
            for i in range(len(self.scenario.clauses))
        ]

        return self._build_observation("Episode started. Review the first clause.")

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Validate action, mutate state, compute reward, check termination."""
        if self.task_config is None or self.scenario is None:
            return (
                self._build_empty_observation(),
                Reward(score=0.0, message="No active episode. Call reset() first."),
                True,
                {"error": "No active episode."},
            )

        if self.done:
            obs = self._build_observation("Episode already finished.")
            return (
                obs,
                Reward(score=0.0, message="Episode done."),
                True,
                self._build_info(),
            )

        self.step_number += 1
        self.actions_taken.append(action)

        # --- Field validation (anti-abuse §8) ---
        validation_error = self._validate_action_fields(action)
        if validation_error:
            self._invalid_action_count += 1
            reward, _  = compute_invalid_action_penalty(validation_error)
            self.rewards_given.append(reward)
            self.accumulated_score += reward.score
            self._last_corrective   = reward.corrective_feedback
            self._check_termination()
            if self.done and self.grader_result is None:
                self.grader_result = self._run_grader()
            obs  = self._build_observation(validation_error)
            info = self._build_info()
            info["error"] = validation_error
            return obs, reward, self.done, info

        # --- Degenerate detection (anti-abuse §8) ---
        # FIX: only penalise when the RECORD already has the same action stored.
        # Normal sequences (classify → rate_severity → flag) on the same clause
        # do NOT trigger this penalty.
        deg_penalty = self._detect_degenerate(action)

        reward, feedback = self._process_action(action)

        if deg_penalty != 0.0:
            reward.score = max(reward.score + deg_penalty, -1.0)
            reward.breakdown["degenerate_penalty"] = deg_penalty

        reward.score = max(-1.0, min(1.0, reward.score))

        self.rewards_given.append(reward)
        self.accumulated_score += reward.score
        self._last_corrective   = reward.corrective_feedback

        self._last_action_type  = action.action_type
        self._last_clause_index = self.clause_index

        self._check_termination()

        if self.done and self.grader_result is None:
            self.grader_result = self._run_grader()

        obs = self._build_observation(feedback)
        return obs, reward, self.done, self._build_info()

    def state(self) -> EnvironmentState:
        """Serializable snapshot of internal state."""
        return EnvironmentState(
            task_id   = self.task_config.task_id if self.task_config else "none",
            difficulty = self.task_config.difficulty.value if self.task_config else "",
            step_number        = self.step_number,
            max_steps          = self.task_config.max_steps if self.task_config else 0,
            clause_index       = self.clause_index,
            total_clauses      = len(self.scenario.clauses) if self.scenario else 0,
            actions_taken      = [a.model_dump() for a in self.actions_taken],
            rewards_given      = [r.model_dump() for r in self.rewards_given],
            clause_records     = [cr.model_dump() for cr in self.clause_records],
            ground_truth       = (
                [gt.model_dump() for gt in self.scenario.clauses] if self.scenario else []
            ),
            accumulated_score  = round(self.accumulated_score, 4),
            done               = self.done,
            episode_start_time = self.episode_start_time,
            grader_result      = (
                self.grader_result.model_dump() if self.grader_result else None
            ),
        )

    # -----------------------------------------------------------------------
    # Private — action processing
    # -----------------------------------------------------------------------

    def _validate_action_fields(self, action: Action) -> Optional[str]:
        at = action.action_type
        if at == ActionType.CLASSIFY      and not action.clause_type:
            return "classify requires clause_type."
        if at == ActionType.RATE_SEVERITY and not action.risk_level:
            return "rate_severity requires risk_level."
        if at == ActionType.FLAG          and action.flags is None:
            return "flag requires flags list (may be empty)."
        if at == ActionType.SUGGEST       and not action.suggested_action:
            return "suggest requires suggested_action."
        if at == ActionType.REASON        and not action.reasoning:
            return "reason requires reasoning text."
        return None

    def _detect_degenerate(self, action: Action) -> float:
        """
        Return a penalty only when the SAME action_type is repeated on the
        SAME clause AND the record already has that action stored.

        Normal sequences like classify→rate_severity→flag on the same clause
        are NOT degenerate. Only exact repetition after recording is penalised.
        """
        at = action.action_type
        if at in (ActionType.NEXT_CLAUSE, ActionType.COMPLETE_REVIEW):
            self._consecutive_repeats = 0
            return 0.0

        if self.clause_index >= len(self.clause_records):
            return 0.0

        record = self.clause_records[self.clause_index]

        # Check whether this action_type is already stored in the record
        already_done = False
        if at == ActionType.CLASSIFY      and record.classify_action  is not None:
            already_done = True
        elif at == ActionType.RATE_SEVERITY and record.risk_action    is not None:
            already_done = True
        elif at == ActionType.FLAG          and record.flag_action     is not None:
            already_done = True
        elif at == ActionType.SUGGEST       and record.suggest_action  is not None:
            already_done = True
        elif at == ActionType.REASON        and record.reason_action   is not None:
            already_done = True

        if already_done and at == self._last_action_type and self.clause_index == self._last_clause_index:
            self._consecutive_repeats  += 1
            self._redundant_action_count += 1
            return compute_degenerate_penalty(self._consecutive_repeats)
        else:
            self._consecutive_repeats = 0
            return 0.0

    def _process_action(self, action: Action) -> Tuple[Reward, str]:
        at = action.action_type

        if at == ActionType.NEXT_CLAUSE:
            return self._handle_next_clause()
        if at == ActionType.COMPLETE_REVIEW:
            return self._handle_complete_review()

        if not self.scenario or self.clause_index >= len(self.scenario.clauses):
            return compute_no_clause_penalty()

        record = self.clause_records[self.clause_index]
        gt     = self.scenario.clauses[self.clause_index]

        reward_fns = {
            ActionType.CLASSIFY:      compute_classify_reward,
            ActionType.RATE_SEVERITY: compute_risk_reward,
            ActionType.FLAG:          compute_flag_reward,
            ActionType.SUGGEST:       compute_suggest_reward,
            ActionType.REASON:        compute_reason_reward,
        }

        reward_fn = reward_fns.get(at)
        if reward_fn is None:
            return Reward(score=0.0, message="Unknown action."), "Unknown action."

        reward, msg = reward_fn(action, record, gt)

        is_redundant = any(k.startswith("redundant_") for k in reward.breakdown)
        if is_redundant:
            self._redundant_action_count += 1
        else:
            self._update_record(action, record)

        return reward, msg

    def _update_record(self, action: Action, record: ClauseActionRecord):
        at = action.action_type
        if at == ActionType.CLASSIFY:
            record.classify_action = action.clause_type
            record.action_count   += 1
        elif at == ActionType.RATE_SEVERITY:
            record.risk_action    = action.risk_level
            record.action_count  += 1
        elif at == ActionType.FLAG:
            record.flag_action    = action.flags or []
            record.action_count  += 1
        elif at == ActionType.SUGGEST:
            record.suggest_action = action.suggested_action
            record.action_count  += 1
        elif at == ActionType.REASON:
            record.reason_action  = action.reasoning
            record.action_count  += 1

    def _handle_next_clause(self) -> Tuple[Reward, str]:
        if not self.scenario:
            return compute_no_clause_penalty()

        is_last = self.clause_index >= len(self.scenario.clauses) - 1

        if is_last:
            return (
                Reward(
                    score=-0.01,
                    breakdown={"progress": -0.01},
                    message="You are already on the final clause. Use complete_review to finish.",
                    corrective_feedback=(
                        "You are already on the final clause. "
                        "Use complete_review to submit your final assessment."
                    ),
                ),
                "Already on the final clause. Use complete_review to finish.",
            )

        record = (
            self.clause_records[self.clause_index]
            if self.clause_index < len(self.clause_records)
            else ClauseActionRecord(clause_index=0)
        )
        required_actions = self.task_config.required_action_types if self.task_config else []
        reward, msg = compute_progress_reward(
            record=record,
            required_actions=required_actions,
            clause_index=self.clause_index,
        )
        self.clause_index += 1
        return reward, msg

    def _handle_complete_review(self) -> Tuple[Reward, str]:
        self.done = True
        total_clauses = len(self.scenario.clauses) if self.scenario else 0
        max_steps     = self.task_config.max_steps  if self.task_config else 10
        reward, msg   = compute_completion_reward(
            total_clauses,
            clauses_reviewed=sum(1 for r in self.clause_records if r.action_count > 0),
            step_number=self.step_number,
            max_steps=max_steps,
        )
        return reward, msg

    def _check_termination(self):
        if self.done:
            return
        if self.task_config and self.step_number >= self.task_config.max_steps:
            self.done = True
        if self.scenario and self.clause_index >= len(self.scenario.clauses):
            self.done = True

    # -----------------------------------------------------------------------
    # Private — grader
    # -----------------------------------------------------------------------

    def _run_grader(self) -> GraderResult:
        if not self.scenario or not self.task_config:
            return GraderResult(score=0.001, message="No scenario loaded.")

        episode_meta = EpisodeMeta(
            total_steps            = self.step_number,
            max_steps              = self.task_config.max_steps,
            total_invalid_actions  = self._invalid_action_count,
            total_redundant_actions = self._redundant_action_count,
            clauses_reviewed       = sum(1 for r in self.clause_records if r.action_count > 0),
            total_clauses          = len(self.scenario.clauses),
            completed_normally     = any(
                a.action_type == ActionType.COMPLETE_REVIEW for a in self.actions_taken
            ),
        )
        try:
            result = grade_episode(
                task_id       = self.task_config.task_id,
                clause_records = self.clause_records,
                ground_truth   = self.scenario.clauses,
                task_config    = self.task_config,
                episode_meta   = episode_meta,
            )
        except Exception as exc:
            return GraderResult(
                score=0.001,
                message=f"Grader failed safely: {type(exc).__name__}: {exc}",
            )

        raw_score = float(result.score)
        if not math.isfinite(raw_score):
            logger.error("INVALID GRADER SCORE: non-finite — task=%s", self.task_config.task_id)
            raw_score = 0.001

        safe_score = max(0.001, min(0.999, raw_score))

        if self._score_audit_verbose:
            logger.info(
                "GRADER: task=%s scenario=%s raw=%.4f safe=%.4f steps=%d reviewed=%d/%d",
                self.task_config.task_id,
                self._active_scenario_file,
                raw_score, safe_score,
                episode_meta.total_steps,
                episode_meta.clauses_reviewed,
                episode_meta.total_clauses,
            )

        if not (0.0 < raw_score < 1.0):
            logger.error(
                "INVALID GRADER SCORE: task=%s raw=%.4f breakdown=%s",
                self.task_config.task_id, raw_score, result.breakdown,
            )

        if safe_score != raw_score:
            result.score = safe_score
        return result

    # -----------------------------------------------------------------------
    # Private — observation builders
    # -----------------------------------------------------------------------

    def _build_observation(self, feedback: str) -> Observation:
        clauses = self.scenario.clauses if self.scenario else []
        total   = len(clauses)

        clause_text = (
            clauses[self.clause_index].text
            if 0 <= self.clause_index < total
            else "[End of Contract — no more clauses]"
        )

        meta = (
            self.scenario.contract_meta
            if self.scenario
            else ContractMeta(contract_type="Unknown", parties=[], jurisdiction="Unknown")
        )

        instructions = get_task_instruction(
            self.task_config.task_id if self.task_config else ""
        )

        return Observation(
            task_id              = self.task_config.task_id if self.task_config else "none",
            step_number          = self.step_number,
            max_steps            = self.task_config.max_steps if self.task_config else 10,
            clause_text          = clause_text,
            clause_index         = self.clause_index,
            total_clauses        = total,
            contract_type        = meta.contract_type,
            parties              = meta.parties,
            jurisdiction         = meta.jurisdiction,
            instructions         = instructions,
            available_actions    = [at.value for at in ActionType],
            last_action_feedback = feedback,
            corrective_feedback  = self._last_corrective,
            accumulated_score    = round(self.accumulated_score, 4),
            done                 = self.done,
        )

    def _build_empty_observation(self) -> Observation:
        return Observation(
            task_id           = "none",
            step_number       = 0,
            max_steps         = 1,
            clause_text       = "No active episode.",
            clause_index      = 0,
            total_clauses     = 0,
            contract_type     = "None",
            parties           = [],
            jurisdiction      = "None",
            instructions      = "Call reset() to start an episode.",
            available_actions = [],
            accumulated_score = 0.0,
            done              = True,
        )

    def _build_info(self) -> Dict[str, Any]:
        clauses = self.scenario.clauses if self.scenario else []
        info: Dict[str, Any] = {
            "step_number":       self.step_number,
            "clause_index":      self.clause_index,
            "accumulated_score": round(self.accumulated_score, 4),
        }
        if 0 <= self.clause_index < len(clauses):
            info["current_clause_ground_truth"] = clauses[self.clause_index].model_dump()
        if self.done:
            info["final_accumulated_reward"] = round(self.accumulated_score, 4)
            info["clauses_reviewed"] = sum(
                1 for r in self.clause_records if r.action_count > 0
            )
            info["total_clauses"] = len(clauses)
            if self.grader_result:
                grader_score = float(self.grader_result.score)
                info["grader_result"]       = self.grader_result.model_dump()
                info["grader_score"]        = grader_score
                info["grader_score_valid"]  = 0.0 < grader_score < 1.0
                info["grader_task_id"]      = (
                    self.task_config.task_id if self.task_config else "none"
                )
                info["grader_scenario_index"] = self._active_scenario_index
                info["grader_scenario_file"]  = self._active_scenario_file
        return info

    # -----------------------------------------------------------------------
    # Private — data loading
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> Dict:
        path = os.path.join(self.data_dir, "manifest.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _load_scenario(
        self, task_id: str, scenario_rel_path: Optional[str] = None
    ) -> ScenarioData:
        # Backward compat: if scenario_rel_path not given, raise ValueError (tested)
        if scenario_rel_path is None:
            raise ValueError(
                f"No scenario_rel_path specified for task '{task_id}'. "
                "Call reset() to load a valid scenario."
            )
        full_path = os.path.join(self.data_dir, scenario_rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Scenario file not found: {full_path}")
        with open(full_path, "r") as f:
            raw = json.load(f)
        meta    = ContractMeta(**raw["contract_meta"])
        clauses = [ClauseGroundTruth(**c) for c in raw["clauses"]]
        if not clauses:
            raise ValueError(f"Scenario has no clauses: {full_path}")
        return ScenarioData(contract_meta=meta, clauses=clauses)
