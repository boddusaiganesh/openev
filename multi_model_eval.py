"""
Phase 10 — Multi-Model Evaluation Runner
=======================================
Runs inference across multiple LLM models and produces a comparative
leaderboard table.

Usage:
    python multi_model_eval.py
    python multi_model_eval.py --models model1,model2
    python multi_model_eval.py --quick
    python multi_model_eval.py --output leaderboard.json

Required env vars:
    API_BASE_URL   — LLM API endpoint
    HF_TOKEN       — API key
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

from openai import OpenAI

from environment import ContractReviewEnv
from inference import (
    DirectEnvAdapter,
    build_user_prompt,
    call_llm,
    parse_llm_response,
    build_action_dict,
    SYSTEM_PROMPT,
)
from models import Action, ActionType
from tasks import get_task_config, list_task_ids


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TEMPERATURE = 0.2
MAX_TOKENS = 600

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]


def evaluate_model(
    model_name: str,
    task_ids: List[str],
    api_base: str = API_BASE_URL,
    api_key: str = "",
) -> Dict[str, Any]:
    """Run one model across specified tasks, return structured results."""
    client = OpenAI(base_url=api_base, api_key=api_key)
    adapter = DirectEnvAdapter()
    model_results = {
        "model": model_name,
        "api_base": api_base,
        "tasks": {},
        "total_time": 0.0,
        "total_llm_calls": 0,
    }

    overall_start = time.time()

    for task_id in task_ids:
        task_start = time.time()
        cfg = get_task_config(task_id)
        required = [a.value for a in cfg.required_action_types]
        task_desc = cfg.description

        obs_data = adapter.reset(task_id)
        history: List[str] = []
        llm_calls = 0
        steps = 0
        done = False
        result = None

        print(f"    {task_id}: ", end="", flush=True)

        while not done:
            if obs_data.get("clause_index", 0) >= obs_data.get("total_clauses", 0):
                result = adapter.step({"action_type": "complete_review"})
                obs_data = result["observation"]
                done = result["done"]
                steps += 1
                break

            prompt = build_user_prompt(obs_data, task_desc, history)

            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"[LLM ERROR: {exc}] ", end="")
                response_text = ""

            llm_calls += 1
            analysis = parse_llm_response(response_text)

            for act_type in required:
                if done:
                    break
                action_dict = build_action_dict(act_type, analysis)
                result = adapter.step(action_dict)
                obs_data = result["observation"]
                done = result["done"]
                steps += 1
                history.append(
                    f"Step {obs_data['step_number']}: {act_type} -> "
                    f"{result['reward']['score']:+.3f}"
                )

            if done:
                break

            ci = obs_data.get("clause_index", 0)
            tc = obs_data.get("total_clauses", 0)
            if ci < tc - 1:
                result = adapter.step({"action_type": "next_clause"})
                obs_data = result["observation"]
                done = result["done"]
                steps += 1
            else:
                result = adapter.step({"action_type": "complete_review"})
                obs_data = result["observation"]
                done = result["done"]
                steps += 1

        task_elapsed = time.time() - task_start
        final_info = result.get("info", {}) if result else {}
        grader_score = final_info.get("grader_score", 0.0)

        print(f"score={grader_score:.4f} steps={steps} llm={llm_calls} "
              f"time={task_elapsed:.1f}s")

        model_results["tasks"][task_id] = {
            "grader_score": round(grader_score, 4),
            "breakdown": final_info.get("grader_result", {}).get("breakdown", {}),
            "steps": steps,
            "llm_calls": llm_calls,
            "elapsed": round(task_elapsed, 2),
        }
        model_results["total_llm_calls"] += llm_calls

    model_results["total_time"] = round(time.time() - overall_start, 2)

    scores = [t["grader_score"] for t in model_results["tasks"].values()]
    model_results["avg_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    return model_results


def print_leaderboard(all_results: List[Dict[str, Any]]):
    """Print a comparative leaderboard table."""
    print("\n" + "=" * 80)
    print("MULTI-MODEL LEADERBOARD")
    print("=" * 80)

    sorted_results = sorted(all_results, key=lambda x: x["avg_score"], reverse=True)
    task_ids = list_task_ids()

    header = f"{'#':<3} {'Model':<45} {'Avg':>6}"
    for tid in task_ids:
        short = tid.replace("task_", "T").replace("_easy", "E").replace("_medium", "M").replace("_hard", "H")
        header += f" {short:>6}"
    header += f" {'LLM':>5} {'Time':>7}"
    print(header)
    print("-" * 80)

    for rank, mr in enumerate(sorted_results, 1):
        line = f"{rank:<3} {mr['model']:<45} {mr['avg_score']:>6.4f}"
        for tid in task_ids:
            score = mr["tasks"].get(tid, {}).get("grader_score", 0.0)
            line += f" {score:>6.4f}"
        line += f" {mr['total_llm_calls']:>5} {mr['total_time']:>6.1f}s"
        print(line)

    print("=" * 80)


def generate_readme_leaderboard(all_results: List[Dict[str, Any]]) -> str:
    """Generate markdown leaderboard for README."""
    sorted_results = sorted(all_results, key=lambda x: x["avg_score"], reverse=True)
    task_ids = list_task_ids()

    lines = [
        "## LLM Baseline Leaderboard",
        "",
        "| # | Model | Avg Score | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | LLM Calls | Time |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for rank, mr in enumerate(sorted_results, 1):
        scores = [mr["tasks"].get(tid, {}).get("grader_score", 0.0) for tid in task_ids]
        lines.append(
            f"| {rank} | `{mr['model']}` | {mr['avg_score']:.4f} | "
            f"{scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | "
            f"{mr['total_llm_calls']} | {mr['total_time']:.0f}s |"
        )

    lines.append("")
    lines.append("Run `python multi_model_eval.py` to reproduce.")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model evaluation")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument("--quick", action="store_true", help="Task 1 only")
    parser.add_argument("--output", default="leaderboard.json", help="Output file")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    task_ids = ["task_1_easy"] if args.quick else list_task_ids()

    print("=" * 80)
    print("MULTI-MODEL EVALUATION")
    print("=" * 80)
    print(f"API: {API_BASE_URL}")
    print(f"Models: {models}")
    print(f"Tasks: {task_ids}")

    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        print(f"\n{'─' * 80}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 80}")

        try:
            result = evaluate_model(model_name, task_ids, API_BASE_URL, API_KEY)
            all_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_results.append({
                "model": model_name, "avg_score": 0.0,
                "tasks": {}, "total_time": 0.0, "total_llm_calls": 0,
            })

    print_leaderboard(all_results)

    with open(args.output, "w") as f:
        json.dump({"leaderboard": all_results}, f, indent=2)
    print(f"\nResults saved to {args.output}")

    if args.update_readme:
        md = generate_readme_leaderboard(all_results)
        with open("README.md") as f:
            content = f.read()
        marker = "## LLM Baseline Leaderboard"
        if marker in content:
            idx = content.index(marker)
            rest = content[idx + len(marker):]
            next_h = rest.find("\n## ")
            if next_h > 0:
                content = content[:idx] + md + "\n" + rest[next_h + 1:]
            else:
                content = content[:idx] + md
        else:
            content += "\n\n" + md
        with open("README.md", "w") as f:
            f.write(content)
        print("README.md updated with leaderboard.")


if __name__ == "__main__":
    main()
