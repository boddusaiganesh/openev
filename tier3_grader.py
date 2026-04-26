"""
tier3_grader.py — Public API wrapper for Tier 3 dependency mapping grader.

Re-exports the grader functions and result types from tier3_environment so
that the plan's tier3_grader.py is a first-class importable module.
"""

from tier3_environment import (
    grade_tier3_batch,
    Tier3SampleResult,
    Tier3Score,
    Tier3MappingEnv,
    load_tier3_scenarios,
)

# Backward-compatible aliases used in older references
Tier3MappingResult = Tier3SampleResult
Tier3BatchScore = Tier3Score

__all__ = [
    "grade_tier3_batch",
    "Tier3SampleResult",
    "Tier3Score",
    "Tier3MappingResult",   # alias
    "Tier3BatchScore",       # alias
    "Tier3MappingEnv",
    "load_tier3_scenarios",
]
