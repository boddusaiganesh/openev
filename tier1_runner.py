"""
LexArena — Tier 1 Runner
Runs clause extraction inference for Tier 1.
Supports OpenAI (proprietary) and HuggingFace (open-source) backends.
Also includes a deterministic baseline for testing.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Optional

from cuad_loader import load_cuad_dataset
from lexarena_models import Tier1Sample, Tier1Output
from tier1_grader import grade_tier1, category_breakdown

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


# ---------------------------------------------------------------------------
# System prompt (identical to ContractEval)
# ---------------------------------------------------------------------------

TIER1_SYSTEM_PROMPT = """You are an assistant with strong legal knowledge, supporting senior lawyers by preparing reference materials.
Given a Context and a Question, extract and return only the sentence(s) from the Context that directly address or relate to the Question.
Do not rephrase or summarize in any way — respond with exact sentences from the Context relevant to the Question. If a relevant sentence contains unrelated elements such as page numbers or whitespace, include them exactly as they appear.
If no part of the Context is relevant to the Question, respond with: "No related clause."
"""

TIER1_USER_TEMPLATE = """Context:
```
{context}
```
Question:
```
Does this contract contain a {question} clause? If so, extract the exact relevant sentence(s).
```
"""


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def _openai_extract(
    sample: Tier1Sample,
    client,
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    """Call OpenAI Responses API for Tier 1 extraction."""
    user_prompt = TIER1_USER_TEMPLATE.format(
        context=sample.context[:8000],  # Truncate for token safety
        question=sample.question_category,
    )
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": TIER1_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return resp.choices[0].message.content or "No related clause."
    except Exception as e:
        return f"[ERROR: {e}]"


def _hf_pipeline_extract(sample: Tier1Sample, pipe) -> str:
    """Run HuggingFace text-generation pipeline for Tier 1 extraction."""
    user_prompt = TIER1_USER_TEMPLATE.format(
        context=sample.context[:6000],
        question=sample.question_category,
    )
    conversation = [
        {"role": "system", "content": TIER1_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        out = pipe(conversation, do_sample=False, max_new_tokens=500, temperature=0.)
        return out[0]["generated_text"][-1]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Baseline strategies (deterministic — for benchmarking without LLM)
# ---------------------------------------------------------------------------

def _baseline_always_extract_first_sentence(sample: Tier1Sample) -> str:
    """Baseline: always extract the first sentence of the context."""
    sentences = [s.strip() for s in sample.context.split(".") if s.strip()]
    return sentences[0] + "." if sentences else "No related clause."


def _baseline_always_no_clause(_: Tier1Sample) -> str:
    """Worst-case baseline: always say no related clause."""
    return "No related clause."


def _baseline_full_context(sample: Tier1Sample) -> str:
    """Upper-bound baseline: return full context (maximises recall, hurts Jaccard)."""
    return sample.context.strip()


BASELINES: dict = {
    "first_sentence": _baseline_always_extract_first_sentence,
    "always_no_clause": _baseline_always_no_clause,
    "full_context": _baseline_full_context,
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class Tier1Runner:
    """Runs Tier 1 (clause extraction) for a given inference function."""

    def __init__(
        self,
        infer_fn: Callable[[Tier1Sample], str],
        cache_dir: str = "./cache_dir",
        max_samples: int = 200,
        priority_only: bool = True,
    ):
        self.infer_fn = infer_fn
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.priority_only = priority_only

    def run(
        self,
        samples: Optional[List[Tier1Sample]] = None,
        verbose: bool = True,
    ):
        """Run Tier 1 on samples and return (outputs, per_sample_results, score, breakdown)."""
        if samples is None:
            samples = load_cuad_dataset(
                cache_dir=self.cache_dir,
                max_samples=self.max_samples,
                priority_only=self.priority_only,
            )

        outputs: List[Tier1Output] = []
        t0 = time.time()

        for i, sample in enumerate(samples):
            raw = self.infer_fn(sample)
            out = Tier1Output(
                sample_id=sample.sample_id,
                extracted_text=raw,
                is_no_clause="no related clause" in raw.strip().lower(),
            )
            outputs.append(out)
            if verbose and (i + 1) % 25 == 0:
                print(f"  [T1] {i+1}/{len(samples)} samples processed "
                      f"({time.time()-t0:.1f}s)")

        per_sample, score = grade_tier1(samples, outputs)
        breakdown = category_breakdown(per_sample)

        if verbose:
            print(f"\n[Tier 1 Results] {len(samples)} samples in {time.time()-t0:.1f}s")
            print(f"  F2: {score.f2:.4f} | Jaccard: {score.jaccard_mean:.4f} "
                  f"| Laziness: {score.laziness_rate:.2%} | T1 Score: {score.tier_score:.4f}")

        return outputs, per_sample, score, breakdown


def run_tier1_openai(
    model_id: str = "gpt-4o-mini",
    max_samples: int = 100,
    cache_dir: str = "./cache_dir",
    priority_only: bool = True,
    verbose: bool = True,
):
    """Run Tier 1 with OpenAI model."""
    if not _openai_available:
        raise ImportError("openai package not installed.")
    client = OpenAI()
    runner = Tier1Runner(
        infer_fn=lambda s: _openai_extract(s, client, model_id),
        cache_dir=cache_dir,
        max_samples=max_samples,
        priority_only=priority_only,
    )
    return runner.run(verbose=verbose)


def run_tier1_hf(
    model_name: str,
    max_samples: int = 100,
    cache_dir: str = "./cache_dir",
    priority_only: bool = True,
    verbose: bool = True,
):
    """Run Tier 1 with a HuggingFace model."""
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"cache_dir": cache_dir},
        device_map="auto",
    )
    runner = Tier1Runner(
        infer_fn=lambda s: _hf_pipeline_extract(s, pipe),
        cache_dir=cache_dir,
        max_samples=max_samples,
        priority_only=priority_only,
    )
    return runner.run(verbose=verbose)


def run_tier1_baseline(
    strategy: str = "first_sentence",
    max_samples: int = 200,
    cache_dir: str = "./cache_dir",
    priority_only: bool = True,
    verbose: bool = True,
):
    """Run Tier 1 with a deterministic baseline strategy."""
    if strategy not in BASELINES:
        raise ValueError(f"Unknown baseline: {strategy}. Choose: {list(BASELINES)}")
    fn = BASELINES[strategy]
    runner = Tier1Runner(
        infer_fn=fn,
        cache_dir=cache_dir,
        max_samples=max_samples,
        priority_only=priority_only,
    )
    return runner.run(verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="baseline", choices=["baseline", "openai", "hf"])
    parser.add_argument("--strategy", default="first_sentence")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=15)
    args = parser.parse_args()

    if args.backend == "baseline":
        _, _, score, breakdown = run_tier1_baseline(
            strategy=args.strategy, max_samples=args.max_samples
        )
    elif args.backend == "openai":
        _, _, score, breakdown = run_tier1_openai(
            model_id=args.model, max_samples=args.max_samples
        )
    else:
        _, _, score, breakdown = run_tier1_hf(
            model_name=args.model, max_samples=args.max_samples
        )

    print("\nTop categories by F2:")
    for cat, m in list(breakdown.items())[:5]:
        print(f"  {cat:<35} F2={m['f2']:.3f} Jac={m['jaccard_mean']:.3f}")
