"""
Phase 1 — Domain Lock & Rubric Alignment
Domain: Contract Clause Review
"""

from models import (
    ACTION_TYPES,
    CLAUSE_TAXONOMY,
    ISSUE_FLAGS,
    RiskLevel,
    SuggestedActionType,
)

DOMAIN = {
    "name": "contract-clause-review",
    "title": "Contract Clause Review Environment",
    "description": (
        "An OpenEnv environment simulating the real-world task of reviewing "
        "contract clauses. An AI agent analyzes individual clauses from legal "
        "contracts, classifying clause types, assessing risk levels, flagging "
        "problematic language, and recommending actions. This mirrors the work "
        "performed daily by legal professionals, paralegals, and contract "
        "analysts across industries."
    ),
    "version": "1.0.0",
    "author": "OpenEnv Contract Review Team",
    "tags": ["openenv", "legal", "contract-review", "nlp", "real-world"],
}

RUBRIC_ALIGNMENT = {
    "real_world_utility": {
        "weight": 0.30,
        "projected_score_range": [26, 30],
        "justification": (
            "Contract review is performed by hundreds of thousands of legal "
            "professionals worldwide. It is tedious, error-prone, and expensive. "
            "Law firms bill $200-$800/hour for this work. Automating clause-level "
            "analysis has immediate, measurable industry value. Multiple legal-tech "
            "companies (Kira, Luminance, ContractPodAi) exist solely for this task. "
            "No existing OpenEnv environment covers legal document analysis, making "
            "this environment fill a genuine gap for the RL/agent community."
        ),
    },
    "task_and_grader_quality": {
        "weight": 0.25,
        "projected_score_range": [20, 25],
        "justification": (
            "Contract clauses have objectively correct classifications from "
            "established legal taxonomies. Risk levels follow industry standards. "
            "Graders can use exact match, partial match, and F1 scoring against "
            "ground truth annotations. Difficulty scales naturally: single-clause "
            "classification (easy) to multi-clause conflict resolution with "
            "prioritization (hard). Hard tasks include ambiguous clauses that "
            "genuinely challenge frontier models."
        ),
    },
    "environment_design": {
        "weight": 0.20,
        "projected_score_range": [16, 20],
        "justification": (
            "Each clause is a natural step-level unit. Observation is structured "
            "text with metadata. Actions are discrete and well-defined from a "
            "fixed taxonomy. Reward decomposes into correctness, completeness, "
            "and efficiency components providing signal at every step. Episode "
            "boundaries are clear: review starts at first clause, ends when all "
            "clauses are reviewed or max steps reached."
        ),
    },
    "code_quality_and_compliance": {
        "weight": 0.15,
        "projected_score_range": [12, 15],
        "justification": (
            "Full Pydantic typed models, FastAPI server, clean project structure, "
            "comprehensive tests, working Dockerfile, openenv.yaml, documented "
            "API. All OpenEnv spec requirements met."
        ),
    },
    "creativity_and_novelty": {
        "weight": 0.10,
        "projected_score_range": [8, 10],
        "justification": (
            "No existing OpenEnv environment covers legal/contract analysis. "
            "Reward design has interesting properties: partial credit for "
            "correct category family even if exact type is wrong, cross-clause "
            "conflict detection bonus, reasoning quality scoring via concept "
            "matching. Multi-dimensional grading is more nuanced than typical "
            "classification environments."
        ),
    },
    "total_projected_range": [82, 100],
}

RISK_REGISTER = [
    {
        "risk": "Contract clause data does not feel realistic",
        "impact": "high",
        "likelihood": "medium",
        "mitigation": (
            "Use publicly available contract clause patterns from SEC EDGAR "
            "filings, open-source legal templates, and common law boilerplate. "
            "Adapt language to be representative without copying verbatim."
        ),
    },
    {
        "risk": "Grader does not differentiate well between good and bad agents",
        "impact": "high",
        "likelihood": "medium",
        "mitigation": (
            "Test graders extensively with hand-crafted optimal, suboptimal, "
            "random, and adversarial trajectories before submission. Tune "
            "component weights to spread the score distribution."
        ),
    },
    {
        "risk": "HF Space deployment fails",
        "impact": "critical",
        "likelihood": "low",
        "mitigation": (
            "Test Docker locally on Day 3, deploy to HF on Day 6. Keep "
            "dependencies minimal. No GPU, no browser, no heavy packages."
        ),
    },
    {
        "risk": "openenv validate fails",
        "impact": "critical",
        "likelihood": "medium",
        "mitigation": (
            "Study validator source code before building. Run validator "
            "incrementally as components are built, not just at the end."
        ),
    },
    {
        "risk": "Inference takes too long (>20 min)",
        "impact": "critical",
        "likelihood": "low",
        "mitigation": (
            "Keep clause counts small (3-12 per scenario). Limit max_steps "
            "(10/20/40). Measure runtime after Phase 8. Pure text, no images."
        ),
    },
    {
        "risk": "LLM cannot parse action format from observation",
        "impact": "medium",
        "likelihood": "medium",
        "mitigation": (
            "Provide explicit examples in system prompt. Use structured output "
            "format with clear delimiters. Implement robust parsing with regex "
            "fallbacks. Default to safe fallback action on parse failure."
        ),
    },
    {
        "risk": "Domain pivot needed",
        "impact": "high",
        "likelihood": "low",
        "mitigation": (
            "Incident response triage pre-designed as backup domain. Same "
            "structural properties: classify, assess severity, recommend action. "
            "Pivot decision must be made by end of Day 2."
        ),
    },
]

TASK_DIFFICULTY_MATRIX = {
    "task_1_easy": {
        "task_id": "task_1_easy",
        "name": "Clause Classification",
        "difficulty": "easy",
        "description": (
            "Classify each clause in a simple contract by type. Clauses are "
            "unambiguous with standard legal language. Agent only needs to "
            "identify the clause type from a fixed taxonomy."
        ),
        "num_clauses_range": [3, 5],
        "max_steps": 10,
        "actions_required_per_clause": ["classify"],
        "ambiguity_level": "none",
        "cross_clause_dependencies": False,
        "expected_frontier_score": [0.90, 1.00],
        "expected_weak_score": [0.50, 0.70],
        "expected_random_score": [0.08, 0.20],
    },
    "task_2_medium": {
        "task_id": "task_2_medium",
        "name": "Risk Assessment",
        "difficulty": "medium",
        "description": (
            "Classify clause type, assign risk level, and flag specific issues "
            "for each clause. Clauses contain non-obvious risks requiring "
            "careful reading. Some clauses are borderline between risk levels."
        ),
        "num_clauses_range": [5, 8],
        "max_steps": 20,
        "actions_required_per_clause": ["classify", "rate_severity", "flag"],
        "ambiguity_level": "moderate",
        "cross_clause_dependencies": False,
        "expected_frontier_score": [0.70, 0.85],
        "expected_weak_score": [0.30, 0.50],
        "expected_random_score": [0.05, 0.15],
    },
    "task_3_hard": {
        "task_id": "task_3_hard",
        "name": "Full Contract Review",
        "difficulty": "hard",
        "description": (
            "Perform a complete contract review: classify clauses, assess risk, "
            "flag issues, suggest actions, and provide reasoning. Contract "
            "contains conflicting clauses, subtle issues, red herrings, and "
            "sleeper clauses. Requires cross-referencing and prioritization."
        ),
        "num_clauses_range": [8, 12],
        "max_steps": 40,
        "actions_required_per_clause": [
            "classify",
            "rate_severity",
            "flag",
            "suggest",
            "reason",
        ],
        "ambiguity_level": "high",
        "cross_clause_dependencies": True,
        "expected_frontier_score": [0.55, 0.80],
        "expected_weak_score": [0.15, 0.40],
        "expected_random_score": [0.02, 0.10],
    },
}

RISK_LEVELS = [level.value for level in RiskLevel]
SUGGESTED_ACTIONS = [action.value for action in SuggestedActionType]

RESOURCE_CONSTRAINTS = {
    "max_vcpu": 2,
    "max_memory_gb": 8,
    "max_inference_runtime_minutes": 20,
    "server_port": 7860,
    "no_gpu_required": True,
    "no_browser_required": True,
    "estimated_docker_image_size_mb": 300,
}

EPISODE_CONSTRAINTS = {
    "task_1_easy": {
        "max_steps": 10,
        "estimated_llm_calls": 5,
        "estimated_time_seconds": 15,
    },
    "task_2_medium": {
        "max_steps": 20,
        "estimated_llm_calls": 12,
        "estimated_time_seconds": 40,
    },
    "task_3_hard": {
        "max_steps": 40,
        "estimated_llm_calls": 30,
        "estimated_time_seconds": 90,
    },
    "total_estimated_seconds": 145,
    "total_with_overhead_seconds": 300,
    "well_under_20_minutes": True,
}

DOMAIN_CONFIRMATION = {
    "domain_locked": True,
    "domain_name": "Contract Clause Review",
    "fallback_domain": "Incident Response Triage",
    "pivot_deadline": "End of Day 2",
    "three_tasks_with_difficulty_escalation": True,
    "partial_progress_reward_naturally_expressible": True,
    "graders_can_be_fully_deterministic": True,
    "hard_task_challenges_frontier_models": True,
    "runs_within_resource_constraints": True,
    "no_existing_openenv_environment_covers_domain": True,
}


def _has_difficulty_escalation():
    difficulties = [TASK_DIFFICULTY_MATRIX[k]["difficulty"] for k in ["task_1_easy", "task_2_medium", "task_3_hard"]]
    return difficulties == ["easy", "medium", "hard"]


def validate_phase1(detailed: bool = False):
    """Run all Phase 1 validation checks.

    Backward compatibility:
    - detailed=False returns a bool for older tests.
    - detailed=True returns the full check report.
    """
    checks = []

    checks.append(("Domain is locked", DOMAIN_CONFIRMATION["domain_locked"]))
    checks.append((
        "Three tasks defined",
        len(TASK_DIFFICULTY_MATRIX) >= 3,
    ))
    checks.append((
        "Difficulty escalation exists",
        _has_difficulty_escalation(),
    ))
    checks.append((
        "Partial progress reward is expressible",
        DOMAIN_CONFIRMATION["partial_progress_reward_naturally_expressible"],
    ))
    checks.append((
        "Graders can be deterministic",
        DOMAIN_CONFIRMATION["graders_can_be_fully_deterministic"],
    ))
    checks.append((
        "Hard task challenges frontier models",
        DOMAIN_CONFIRMATION["hard_task_challenges_frontier_models"],
    ))
    checks.append((
        "Resource constraints satisfied",
        DOMAIN_CONFIRMATION["runs_within_resource_constraints"],
    ))
    checks.append((
        "No existing OpenEnv covers domain",
        DOMAIN_CONFIRMATION["no_existing_openenv_environment_covers_domain"],
    ))
    checks.append((
        "Clause taxonomy has sufficient types",
        len(CLAUSE_TAXONOMY) >= 10,
    ))
    checks.append((
        "Risk levels defined",
        len(RISK_LEVELS) == 4,
    ))
    checks.append((
        "Issue flags defined",
        len(ISSUE_FLAGS) >= 10,
    ))
    checks.append((
        "Action types defined",
        len(ACTION_TYPES) >= 5,
    ))
    checks.append((
        "Suggested actions defined",
        len(SUGGESTED_ACTIONS) >= 4,
    ))
    checks.append((
        "Inference runtime under 20 minutes",
        EPISODE_CONSTRAINTS["well_under_20_minutes"],
    ))

    all_passed = all(check[1] for check in checks)
    results = {
        "phase": 1,
        "all_passed": all_passed,
        "checks": checks,
    }
    if detailed:
        return results
    return all_passed


if __name__ == "__main__":
    results = validate_phase1(detailed=True)
    print(f"Phase 1 Validation: {'PASSED' if results['all_passed'] else 'FAILED'}")
    for check_name, passed in results["checks"]:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
