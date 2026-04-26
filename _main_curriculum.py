

def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all tier results."""
    print("\n" + "=" * 72)
    print(f"  LexArena Results -- Model: {MODEL_NAME}")
    print("=" * 72)
    print(f"  {'Task':<35} {'Score':>8} {'Steps':>7} {'LLM Calls':>10}")
    print("-" * 72)
    for r in results:
        task = r.get("task_id", "?")
        score = r.get("grader_score", 0.0)
        steps = r.get("total_steps", 0)
        calls = r.get("llm_calls", 0)
        err = " [ERROR]" if r.get("error") else ""
        print(f"  {task:<35} {score:>8.4f} {steps:>7} {calls:>10}{err}")
    print("=" * 72)


def _run_tier2_curriculum(
    client: OpenAI,
    env_url: str,
    env_mode: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run Tier 2 tasks (task_1_easy -> task_2_medium -> task_3_hard).

    Returns the mean grader score and list of per-task results.
    """
    tier2_results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for task_id in TIER2_TASKS:
        adapter = HttpEnvAdapter(env_url) if env_mode == "http" else DirectEnvAdapter()
        try:
            result = run_task(adapter, client, task_id)
            tier2_results.append(result)
            scores.append(result.get("grader_score", 0.0))
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 2 task {task_id} failed: {exc}\n")
            tier2_results.append({"task_id": task_id, "grader_score": 0.001, "error": str(exc)})
            scores.append(0.001)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, tier2_results


def main() -> None:
    """Full 6-tier LexArena curriculum runner.

    Progresses: T1 (reading) -> T2 (classification) -> T3 (dependency) ->
                T4 (crisis easy) -> T5 (crisis medium) -> T6 (crisis hard).

    Each tier uses a dedicated system prompt. Corrective feedback from the
    environment is fed into subsequent prompts to help the model improve.

    Emits [START], [STEP], and [END] log lines per OpenEnv convention.
    Saves full results to baseline_results.json.
    """
    # --- Validate required environment variables ---
    missing = [v for v in ("API_BASE_URL", "API_KEY", "MODEL_NAME")
               if not os.getenv(v, "").strip()]
    if missing:
        for var in missing:
            sys.stderr.write(f"ERROR: {var} environment variable not set.\n")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_mode = ENV_MODE
    env_url = ENV_URL

    # Verify HTTP server is reachable
    if env_mode == "http":
        try:
            r = httpx.get(f"{env_url}/health", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:
            sys.stderr.write(f"ERROR: Cannot reach LexArena server at {env_url}: {exc}\n")
            sys.exit(1)

    overall_start = time.time()
    all_results: List[Dict[str, Any]] = []
    tier_scores: Dict[str, float] = {}

    print(f"[LEXARENA] model={MODEL_NAME} mode={env_mode} tiers={ACTIVE_TIERS}")

    # ---------------------------------------------------------------- Tier 1
    if 1 in ACTIVE_TIERS:
        print("\n[TIER 1] Clause Reading (CUAD verbatim extraction)")
        try:
            t1_scores: List[float] = []
            if env_mode == "http":
                reset_r = httpx.post(
                    f"{env_url}/tier1/reset",
                    json={"max_samples": 15, "priority_only": True},
                    timeout=30.0,
                )
                reset_r.raise_for_status()
                state = reset_r.json()
                total_t1 = state.get("total_samples", 0)

                for _ in range(total_t1):
                    sample = state.get("current_sample") or {}
                    if not sample:
                        break
                    user_prompt = (
                        f"Contract: {sample.get('contract_name', 'Unknown')}\n"
                        f"Category: {sample.get('question_category', '')}\n\n"
                        f"Excerpt:\n{sample.get('context', '')}\n\n"
                        f"Question: Is there a {sample.get('question_category', '')} clause?"
                    )
                    response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_T1)
                    step_r = httpx.post(
                        f"{env_url}/tier1/step",
                        json={"sample_id": sample.get("sample_id", ""), "extracted_text": response.strip()},
                        timeout=30.0,
                    )
                    step_r.raise_for_status()
                    state = step_r.json()
                    f2 = state.get("sample_result", {}).get("f2_score", 0.0)
                    t1_scores.append(f2)
                    print(f"  [STEP] T1 sample={sample.get('sample_id','')} f2={f2:.4f}")
                    if state.get("done"):
                        break
            else:
                print("  [INFO] Tier 1 requires http mode. Skipping (score=0.0).")

            t1_score = sum(t1_scores) / len(t1_scores) if t1_scores else 0.0
            tier_scores["t1"] = t1_score
            all_results.append({"task_id": "tier1_clause_reading", "tier": 1,
                                 "grader_score": t1_score, "total_steps": len(t1_scores), "llm_calls": len(t1_scores)})
            print(f"  [END] T1 score={t1_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 1 failed: {exc}\n")
            tier_scores["t1"] = 0.001
            all_results.append({"task_id": "tier1_clause_reading", "tier": 1, "grader_score": 0.001, "error": str(exc)})

    # ---------------------------------------------------------------- Tier 2
    if 2 in ACTIVE_TIERS:
        print("\n[TIER 2] Clause Review (Easy -> Medium -> Hard)")
        t2_score, t2_results = _run_tier2_curriculum(client, env_url, env_mode)
        tier_scores["t2"] = t2_score
        all_results.extend(t2_results)
        print(f"  [END] T2 mean_score={t2_score:.4f}")

    # ---------------------------------------------------------------- Tier 3
    if 3 in ACTIVE_TIERS:
        print("\n[TIER 3] Dependency Graph Mapping")
        try:
            t3_score = 0.0
            if env_mode == "http":
                reset_r = httpx.post(f"{env_url}/tier3/reset", json={"scenario_index": 0}, timeout=30.0)
                reset_r.raise_for_status()
                obs = reset_r.json()
                contracts_summary = json.dumps(obs.get("contracts", []), indent=2)
                user_prompt = (
                    f"Analyse these contracts and identify ALL dependency edges.\n\n"
                    f"Contracts:\n{contracts_summary}\n\n"
                    f"Output a JSON array of dependency edges."
                )
                response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_T3)
                try:
                    raw_deps = json.loads(response.strip())
                    if not isinstance(raw_deps, list):
                        raw_deps = []
                except (json.JSONDecodeError, ValueError):
                    raw_deps = []

                step_r = httpx.post(f"{env_url}/tier3/step", json={"dependencies": raw_deps}, timeout=30.0)
                step_r.raise_for_status()
                step_data = step_r.json()
                result = step_data.get("result") or {}
                t3_score = result.get("combined_score", 0.0)
                print(f"  [STEP] T3 recall={result.get('recall',0):.3f} precision={result.get('precision',0):.3f}")
            else:
                print("  [INFO] Tier 3 requires http mode. Skipping (score=0.0).")

            tier_scores["t3"] = t3_score
            all_results.append({"task_id": "tier3_dependency_mapping", "tier": 3,
                                 "grader_score": t3_score, "total_steps": 1, "llm_calls": 1})
            print(f"  [END] T3 score={t3_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier 3 failed: {exc}\n")
            tier_scores["t3"] = 0.001
            all_results.append({"task_id": "tier3_dependency_mapping", "tier": 3, "grader_score": 0.001, "error": str(exc)})

    # ------------------------------------------------------------ Tiers 4-6
    for tier, task_id, label in [
        (4, "task_4_cascade_easy",   "Crisis Easy"),
        (5, "task_5_cascade_medium", "Crisis Medium"),
        (6, "task_6_cascade_hard",   "Crisis Hard"),
    ]:
        if tier not in ACTIVE_TIERS:
            continue
        print(f"\n[TIER {tier}] {label}")
        try:
            tier_score = 0.0
            step_count = 0
            llm_count = 0
            if env_mode == "http":
                reset_r = httpx.post(
                    f"{env_url}/cascade/reset",
                    json={"task_id": task_id, "scenario_index": 0},
                    timeout=30.0,
                )
                reset_r.raise_for_status()
                obs = reset_r.json()
                done = obs.get("done", False)
                crisis_feedback = obs.get("corrective_feedback", "")

                while not done:
                    obs_text = json.dumps({
                        k: obs[k] for k in (
                            "current_day", "cash_balance", "covenant_min_cash",
                            "active_deadlines", "inbox_messages", "discovered_edges",
                            "contracts_summary", "available_actions",
                        ) if k in obs
                    }, indent=2)
                    user_prompt = (
                        f"CORRECTIVE FEEDBACK FROM LAST ACTION:\n{crisis_feedback}\n\n"
                        f"CURRENT SITUATION (Day {obs.get('current_day', '?')}):\n{obs_text}\n\n"
                        f"Choose your next action. Respond with JSON."
                    )
                    response = call_llm(client, user_prompt, system_prompt=SYSTEM_PROMPT_CRISIS)
                    llm_count += 1
                    try:
                        action_dict = json.loads(response.strip())
                    except (json.JSONDecodeError, ValueError):
                        action_dict = {"action_type": "advance_day", "justification": "parse error fallback"}

                    step_r = httpx.post(f"{env_url}/cascade/step", json=action_dict, timeout=30.0)
                    step_r.raise_for_status()
                    step_data = step_r.json()
                    obs = step_data.get("observation", {})
                    done = step_data.get("done", False)
                    reward = step_data.get("reward", {}).get("score", 0.0)
                    crisis_feedback = obs.get("corrective_feedback", "")
                    step_count += 1
                    print(f"  [STEP] T{tier} day={obs.get('current_day','?')} step={step_count} reward={reward:.3f}")

                # Retrieve final grader score
                state_r = httpx.get(f"{env_url}/cascade/state", timeout=10.0)
                state_r.raise_for_status()
                state = state_r.json()
                grader = state.get("grader_result") or {}
                tier_score = grader.get("score", 0.001)
            else:
                print(f"  [INFO] Tier {tier} requires http mode. Skipping (score=0.0).")

            tier_scores[f"t{tier}"] = tier_score
            all_results.append({"task_id": task_id, "tier": tier, "grader_score": tier_score,
                                 "total_steps": step_count, "llm_calls": llm_count})
            print(f"  [END] T{tier} score={tier_score:.4f}")
        except Exception as exc:
            sys.stderr.write(f"  [WARN] Tier {tier} failed: {exc}\n")
            tier_scores[f"t{tier}"] = 0.001
            all_results.append({"task_id": task_id, "tier": tier, "grader_score": 0.001, "error": str(exc)})

    # ----------------------------------------------------- Legal IQ Composite
    t1 = tier_scores.get("t1", 0.0)
    t2 = tier_scores.get("t2", 0.0)
    t3 = tier_scores.get("t3", 0.0)
    t4 = tier_scores.get("t4", 0.0)
    t5 = tier_scores.get("t5", 0.0)
    t6 = tier_scores.get("t6", 0.0)
    legal_iq = round(0.15*t1 + 0.15*t2 + 0.20*t3 + 0.50*(0.25*t4 + 0.35*t5 + 0.40*t6), 4)
    label = ""

    if env_mode == "http":
        try:
            iq_r = httpx.post(
                f"{env_url}/legal_iq",
                json={"model_name": MODEL_NAME, "t1_score": t1, "t2_score": t2,
                      "t3_score": t3, "t4_score": t4, "t5_score": t5, "t6_score": t6},
                timeout=10.0,
            )
            iq_r.raise_for_status()
            iq_data = iq_r.json()
            legal_iq = iq_data.get("legal_iq", legal_iq)
            label = iq_data.get("label", "")
        except Exception:
            pass  # Keep locally computed score

    # ------------------------------------------------------ Print + Save
    overall_elapsed = time.time() - overall_start
    total_llm_calls = sum(int(r.get("llm_calls", 0)) for r in all_results)

    print_results_table(all_results)
    print(f"\n  Legal IQ : {legal_iq:.4f}  ({label})")
    print(f"  T1={t1:.3f}  T2={t2:.3f}  T3={t3:.3f}  T4={t4:.3f}  T5={t5:.3f}  T6={t6:.3f}")
    print(f"  Total runtime: {overall_elapsed:.1f}s\n")

    if total_llm_calls <= 0:
        sys.stderr.write(
            "ERROR: No LLM calls were made. Check API_BASE_URL/API_KEY setup.\n"
        )
        sys.exit(1)

    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "mode": env_mode,
        "temperature": TEMPERATURE,
        "active_tiers": ACTIVE_TIERS,
        "tier_scores": {
            "t1_reading": t1, "t2_classification": t2, "t3_dependency": t3,
            "t4_crisis_easy": t4, "t5_crisis_medium": t5, "t6_crisis_hard": t6,
        },
        "legal_iq": legal_iq,
        "label": label,
        "results": all_results,
        "total_llm_calls": total_llm_calls,
        "total_runtime_seconds": round(overall_elapsed, 2),
    }
    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("[LEXARENA] Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
