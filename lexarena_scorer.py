"""
LexArena — Composite Scorer
Computes the Legal IQ score from all 6 tier results.
Pure math — fully deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from lexarena_models import (
    LegalIQScore, TierScore, TierName, ProbeResult, ProbeOutcome
)


# ---------------------------------------------------------------------------
# Weights (from LexArena plan)
# ---------------------------------------------------------------------------

TIER_WEIGHTS = {
    TierName.TIER1_READING:        0.15,
    TierName.TIER2_CLASSIFICATION: 0.15,
    TierName.TIER3_DEPENDENCY:     0.20,
}
# Tiers 4-6 combined weight = 0.50
CRISIS_WEIGHT = 0.50
CRISIS_SUB_WEIGHTS = {
    TierName.TIER4_CASCADE_EASY:   0.25,
    TierName.TIER5_CASCADE_MEDIUM: 0.35,
    TierName.TIER6_CASCADE_HARD:   0.40,
}


def compute_legal_iq(
    t1_score: float = 0.0,
    t2_score: float = 0.0,
    t3_score: float = 0.0,
    t4_score: float = 0.0,
    t5_score: float = 0.0,
    t6_score: float = 0.0,
    probe_results: Optional[List[ProbeResult]] = None,
) -> LegalIQScore:
    """
    Compute the composite Legal IQ score.

    All tier scores are in [0, 1].
    Missing tiers default to 0.0 but their weight is still applied —
    this penalises agents that skip tiers.
    """
    probe_results = probe_results or []

    # Individual tier contributions
    t1_contrib = t1_score * TIER_WEIGHTS[TierName.TIER1_READING]
    t2_contrib = t2_score * TIER_WEIGHTS[TierName.TIER2_CLASSIFICATION]
    t3_contrib = t3_score * TIER_WEIGHTS[TierName.TIER3_DEPENDENCY]

    # Crisis sub-scores
    t4_contrib = t4_score * CRISIS_SUB_WEIGHTS[TierName.TIER4_CASCADE_EASY]
    t5_contrib = t5_score * CRISIS_SUB_WEIGHTS[TierName.TIER5_CASCADE_MEDIUM]
    t6_contrib = t6_score * CRISIS_SUB_WEIGHTS[TierName.TIER6_CASCADE_HARD]
    crisis_combined = t4_contrib + t5_contrib + t6_contrib  # [0, 1] within crisis block

    # Apply crisis block weight
    crisis_contrib = crisis_combined * CRISIS_WEIGHT

    # Total Legal IQ
    legal_iq = round(t1_contrib + t2_contrib + t3_contrib + crisis_contrib, 4)
    legal_iq = max(0.0, min(1.0, legal_iq))

    # Build tier score list
    tier_scores = [
        TierScore(
            tier=TierName.TIER1_READING,
            raw_score=round(t1_score, 4),
            weight=TIER_WEIGHTS[TierName.TIER1_READING],
            weighted_contribution=round(t1_contrib, 4),
        ),
        TierScore(
            tier=TierName.TIER2_CLASSIFICATION,
            raw_score=round(t2_score, 4),
            weight=TIER_WEIGHTS[TierName.TIER2_CLASSIFICATION],
            weighted_contribution=round(t2_contrib, 4),
        ),
        TierScore(
            tier=TierName.TIER3_DEPENDENCY,
            raw_score=round(t3_score, 4),
            weight=TIER_WEIGHTS[TierName.TIER3_DEPENDENCY],
            weighted_contribution=round(t3_contrib, 4),
        ),
        TierScore(
            tier=TierName.TIER4_CASCADE_EASY,
            raw_score=round(t4_score, 4),
            weight=CRISIS_SUB_WEIGHTS[TierName.TIER4_CASCADE_EASY] * CRISIS_WEIGHT,
            weighted_contribution=round(t4_contrib * CRISIS_WEIGHT, 4),
        ),
        TierScore(
            tier=TierName.TIER5_CASCADE_MEDIUM,
            raw_score=round(t5_score, 4),
            weight=CRISIS_SUB_WEIGHTS[TierName.TIER5_CASCADE_MEDIUM] * CRISIS_WEIGHT,
            weighted_contribution=round(t5_contrib * CRISIS_WEIGHT, 4),
        ),
        TierScore(
            tier=TierName.TIER6_CASCADE_HARD,
            raw_score=round(t6_score, 4),
            weight=CRISIS_SUB_WEIGHTS[TierName.TIER6_CASCADE_HARD] * CRISIS_WEIGHT,
            weighted_contribution=round(t6_contrib * CRISIS_WEIGHT, 4),
        ),
    ]

    # Probe summary
    passed = sum(1 for p in probe_results if p.outcome == ProbeOutcome.PASSED)
    failed = sum(1 for p in probe_results if p.outcome == ProbeOutcome.FAILED)
    triggered = [p.failure_mode.value for p in probe_results if p.outcome == ProbeOutcome.FAILED]

    return LegalIQScore(
        tier_scores=tier_scores,
        probe_results=probe_results,
        legal_iq=legal_iq,
        t1_reading=round(t1_score, 4),
        t2_classification=round(t2_score, 4),
        t3_dependency=round(t3_score, 4),
        t4_crisis_easy=round(t4_score, 4),
        t5_crisis_medium=round(t5_score, 4),
        t6_crisis_hard=round(t6_score, 4),
        t4_t5_t6_combined=round(crisis_combined, 4),
        probes_passed=passed,
        probes_failed=failed,
        probes_total=len(probe_results),
        failure_modes_triggered=triggered,
        label=LegalIQScore.compute_label(legal_iq),
    )


def print_legal_iq(score: LegalIQScore) -> None:
    """Pretty-print the Legal IQ score breakdown."""
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  L E X A R E N A   —   Legal IQ Score")
    print(bar)
    print(f"  {score.label}")
    print(f"  Legal IQ:  {score.legal_iq:.4f}  ({score.legal_iq*100:.1f}/100)")
    print(bar)
    print(f"  {'Tier':<30} {'Raw':>6}  {'Weight':>7}  {'Contrib':>8}")
    print(f"  {'-'*55}")
    for ts in score.tier_scores:
        print(f"  {ts.tier.value:<30} {ts.raw_score:>6.4f}  {ts.weight:>7.1%}  {ts.weighted_contribution:>8.4f}")
    print(f"  {'-'*55}")
    print(f"  {'LEGAL IQ TOTAL':<30} {'':>6}  {'100.0%':>7}  {score.legal_iq:>8.4f}")

    if score.probes_total > 0:
        print(f"\n  Adversarial Probes: {score.probes_passed}/{score.probes_total} passed")
        if score.failure_modes_triggered:
            print(f"  Failure Modes Triggered: {', '.join(score.failure_modes_triggered)}")
    print(f"{bar}\n")


def compare_scores(scores: Dict[str, LegalIQScore]) -> None:
    """Print a leaderboard comparing multiple models."""
    ranked = sorted(scores.items(), key=lambda x: -x[1].legal_iq)
    bar = "=" * 90
    print(f"\n{bar}")
    print(f"  L E X A R E N A   L E A D E R B O A R D")
    print(bar)
    header = f"  {'#':<3} {'Model':<28} {'LegalIQ':>8} {'T1':>6} {'T2':>6} {'T3':>6} {'T4-6':>6} {'Label':<25}"
    print(header)
    print(f"  {'-'*83}")
    for rank, (model, s) in enumerate(ranked, 1):
        print(
            f"  {rank:<3} {model:<28} {s.legal_iq:>8.4f} "
            f"{s.t1_reading:>6.3f} {s.t2_classification:>6.3f} "
            f"{s.t3_dependency:>6.3f} {s.t4_t5_t6_combined:>6.3f} "
            f"{s.label:<25}"
        )
    print(f"{bar}\n")
