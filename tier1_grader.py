"""
LexArena — Tier 1 Grader
F2 + Jaccard + Laziness scoring for clause extraction.
Pure math — no LLM judge.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from lexarena_models import (
    Tier1Sample, Tier1Output, Tier1SampleResult, Tier1Score
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r'[.,;:/]')


def _normalise(text: str) -> str:
    text = _PUNCT.sub("", text.lower())
    text = text.replace("/", " ")
    return text.strip()


def _tokenset(text: str) -> set:
    return set(_normalise(text).split())


def jaccard(gt: str, pred: str) -> float:
    """Word-level Jaccard similarity between ground truth and prediction."""
    g = _tokenset(gt)
    p = _tokenset(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def is_no_clause_response(text: str) -> bool:
    return "no related clause" in text.strip(" \n`").lower()


def check_inclusion(output: str, labels: List[str]) -> bool:
    """True if all label substrings appear verbatim in the output."""
    clean_out = output.strip(" \n`")
    return all(lbl.strip(" \n`") in clean_out for lbl in labels)


# ---------------------------------------------------------------------------
# Per-sample grader
# ---------------------------------------------------------------------------

def grade_sample(
    sample: Tier1Sample,
    output: Tier1Output,
) -> Tier1SampleResult:
    """Grade one Tier 1 extraction sample."""
    pred_text = output.extracted_text
    is_no_clause = is_no_clause_response(pred_text)

    gt = sample.ground_truth
    has_answer = sample.has_answer  # ground truth is non-empty

    tp = tn = fp = fn = False
    jac = 0.0
    is_lazy = False

    if not has_answer:
        # No relevant clause exists
        if is_no_clause:
            tn = True
        else:
            fp = True
    else:
        # A relevant clause exists
        if is_no_clause:
            fn = True
            is_lazy = True
        else:
            if check_inclusion(pred_text, gt):
                tp = True
            else:
                fn = True
        # Jaccard only for positive samples where model gave a non-null response
        if not is_no_clause:
            jac = jaccard(" ".join(gt), pred_text.strip(" \n`"))

    return Tier1SampleResult(
        sample_id=sample.sample_id,
        question_category=sample.question_category,
        is_true_positive=tp,
        is_true_negative=tn,
        is_false_positive=fp,
        is_false_negative=fn,
        jaccard_score=round(jac, 4),
        is_lazy=is_lazy,
    )


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------

def aggregate_tier1(
    results: List[Tier1SampleResult],
    total_positive_samples: int,
) -> Tier1Score:
    """Compute aggregate Tier 1 score from per-sample results."""
    tp = sum(1 for r in results if r.is_true_positive)
    tn = sum(1 for r in results if r.is_true_negative)
    fp = sum(1 for r in results if r.is_false_positive)
    fn = sum(1 for r in results if r.is_false_negative)
    lazy_cnt = sum(1 for r in results if r.is_lazy)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # F2: weights recall twice as much as precision
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    jac_scores = [r.jaccard_score for r in results if r.jaccard_score > 0]
    jac_mean = sum(jac_scores) / len(jac_scores) if jac_scores else 0.0

    n = len(results)
    laziness_rate = lazy_cnt / n if n > 0 else 0.0
    false_no_clause_rate = lazy_cnt / total_positive_samples if total_positive_samples > 0 else 0.0

    # Composite Tier 1 score
    tier_score = (
        0.60 * f2
        + 0.25 * jac_mean
        + 0.15 * (1.0 - laziness_rate)
    )
    tier_score = max(0.0, min(1.0, round(tier_score, 4)))

    return Tier1Score(
        total_samples=n,
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        jaccard_mean=round(jac_mean, 4),
        laziness_rate=round(laziness_rate, 4),
        false_no_clause_rate=round(false_no_clause_rate, 4),
        tier_score=tier_score,
    )


def grade_tier1(
    samples: List[Tier1Sample],
    outputs: List[Tier1Output],
) -> Tuple[List[Tier1SampleResult], Tier1Score]:
    """Grade all Tier 1 samples and return per-sample + aggregate results."""
    if len(samples) != len(outputs):
        raise ValueError(
            f"Mismatch: {len(samples)} samples vs {len(outputs)} outputs"
        )
    per_sample = [grade_sample(s, o) for s, o in zip(samples, outputs)]
    total_positive = sum(1 for s in samples if s.has_answer)
    score = aggregate_tier1(per_sample, total_positive)
    return per_sample, score


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

def category_breakdown(results: List[Tier1SampleResult]) -> dict:
    """Return per-category F2 and Jaccard summary."""
    cats: dict = {}
    for r in results:
        cat = r.question_category
        cats.setdefault(cat, {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "jac": []})
        if r.is_true_positive:
            cats[cat]["tp"] += 1
        elif r.is_true_negative:
            cats[cat]["tn"] += 1
        elif r.is_false_positive:
            cats[cat]["fp"] += 1
        else:
            cats[cat]["fn"] += 1
        if r.jaccard_score > 0:
            cats[cat]["jac"].append(r.jaccard_score)

    summary = {}
    for cat, m in cats.items():
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        r_ = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f2 = (5 * p * r_) / (4 * p + r_) if (4 * p + r_) > 0 else 0.0
        jac_mean = sum(m["jac"]) / len(m["jac"]) if m["jac"] else 0.0
        summary[cat] = {
            "f2": round(f2, 3),
            "jaccard_mean": round(jac_mean, 3),
            "count": m["tp"] + m["tn"] + m["fp"] + m["fn"],
        }

    return dict(sorted(summary.items(), key=lambda x: -x[1]["f2"]))
