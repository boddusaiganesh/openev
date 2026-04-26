---
title: LexArena — Legal Intelligence Benchmark
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - legal
  - contract-review
  - benchmark
  - real-world
pinned: false
---

# LexArena — Complete Legal Intelligence Benchmark

> *The first benchmark that tests an AI agent's complete legal intelligence stack — from reading a single clause to surviving a 30-day systemic corporate crisis.*

An **OpenEnv-compliant** benchmark environment simulating the real-world task of contract clause review and crisis management. All tiers run from a single Hugging Face Space on port 7860.

---

## Motivation

Contract review is performed by hundreds of thousands of legal professionals worldwide. Law firms bill $200–$800/hour for clause-level analysis. Existing AI benchmarks test only one layer of legal intelligence in isolation. **LexArena closes all three gaps simultaneously:**

| Benchmark | What it tests | What it misses |
|---|---|---|
| ContractEval | Can the model *read* a clause? | Can it *act* on what it read? |
| CUAD / LexGLUE | Can the model *classify* legal text? | Can it *reason across* multiple documents? |
| LexArena | **READ → CLASSIFY → CONNECT → DECIDE → SURVIVE** | Nothing |

---

## Quick Start

### Local (no Docker)

```bash
pip install -r requirements.txt

# Start the unified LexArena server (all 6 tiers, port 7860)
uvicorn lexarena_server:app --host 0.0.0.0 --port 7860

# In another terminal — verify it's running
curl http://localhost:7860/health

# See all endpoints and tier specs
curl http://localhost:7860/
```

### Docker

```bash
docker build -t lexarena .
docker run -p 7860:7860 lexarena

# Verify
curl http://localhost:7860/health
```

### Run Baseline Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENAI_API_KEY="your-api-key"

# Run the full 6-tier curriculum
python inference.py

# Run only specific tiers (e.g. just clause review)
TIERS=2 python inference.py

# Run adversarial probe suite (no API key needed)
python probe_runner.py --strategy deadline_first
```


---

## Tasks

LexArena has **8 tasks** across **6 tiers**, ranging from easy to frontier difficulty.

### Tier 1 — Clause Reading (Easy)

| Property | Value |
|---|---|
| Task ID | `tier1_clause_reading` |
| Difficulty | Easy |
| Dataset | CUAD (15 priority categories) |
| Expected Frontier Score | 0.70–0.90 |
| Grading | F2-weighted token overlap (recall-weighted) |

Extract the verbatim sentence from a contract that answers a given legal question. Tests precise reading comprehension. No paraphrasing — exact extraction only.

---

### Tier 2a — Clause Classification (Easy)

| Property | Value |
|---|---|
| Task ID | `task_1_easy` |
| Difficulty | Easy |
| Clauses | 3–5 |
| Max Steps | 10 |
| Required Actions | `classify` |
| Expected Frontier Score | 0.80–0.95 |

Classify each clause in a simple contract from the 15-type taxonomy. Unambiguous language, standard forms.

---

### Tier 2b — Risk Assessment (Medium)

| Property | Value |
|---|---|
| Task ID | `task_2_medium` |
| Difficulty | Medium |
| Clauses | 5–8 |
| Max Steps | 20 |
| Required Actions | `classify`, `rate_severity`, `flag` |
| Expected Frontier Score | 0.60–0.80 |

Classify clause type, assign risk level (low/medium/high/critical), and flag specific issues. Non-obvious risks require careful reading.

---

### Tier 2c — Full Contract Review (Hard)

| Property | Value |
|---|---|
| Task ID | `task_3_hard` |
| Difficulty | Hard |
| Clauses | 8–12 |
| Max Steps | 40 |
| Required Actions | `classify`, `rate_severity`, `flag`, `suggest`, `reason` |
| Expected Frontier Score | 0.45–0.70 |

Full contract review with conflicting clauses, sleeper clauses, red herrings, and subtle cross-clause reasoning.

---

### Tier 3 — Dependency Mapping (Hard)

| Property | Value |
|---|---|
| Task ID | `tier3_dependency_mapping` |
| Difficulty | Hard |
| Contracts | 3–5 |
| Expected Frontier Score | 0.35–0.60 |
| Grading | Precision × Recall on dependency graph |

Map hidden cross-document dependency edges between clauses. Edge types: `cascade_trigger`, `mutual_exclusion`, `condition_precedent`, `supersession`, `temporal_gate`.

---

### Tier 4 — Crisis Response: Easy

| Property | Value |
|---|---|
| Task ID | `task_4_cascade_easy` |
| Difficulty | Easy |
| Contracts | 2–3 |
| Crisis Duration | 15 days |
| Expected Frontier Score | 0.65–0.85 |
| Grading | Cash preservation + deadline compliance |

Single-cascade crisis. Manage a trade disruption affecting 2–3 linked contracts.

---

### Tier 5 — Crisis Response: Medium

| Property | Value |
|---|---|
| Task ID | `task_5_cascade_medium` |
| Difficulty | Medium |
| Contracts | 4–5 |
| Crisis Duration | 20 days |
| Expected Frontier Score | 0.40–0.65 |

Multi-document crisis with hidden dependency edges. Counterparty risk and hidden covenant violations.

---

### Tier 6 — Crisis Response: Hard (Frontier)

| Property | Value |
|---|---|
| Task ID | `task_6_cascade_hard` |
| Difficulty | Frontier |
| Contracts | 5–6 |
| Crisis Duration | 30 days |
| Expected Frontier Score | 0.20–0.50 |

Full systemic cascade. Compound shocks on Day 14–16. Force Majeure/insurance ordering constraints. Aggressive counterparties. Debt covenant risk.

---

## API Reference

All tiers accessible from port `7860`.

### Core OpenEnv Routes (Tier 2 — backward-compatible)

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | `{"task_id": "task_1_easy"}` |
| `/step` | POST | Action JSON |
| `/state` | GET | Current environment state |

### Tier 1 Routes

| Endpoint | Method | Description |
|---|---|---|
| `/tier1/reset` | POST | `{"max_samples": 15, "priority_only": true}` |
| `/tier1/step` | POST | `{"sample_id": "...", "extracted_text": "..."}` |
| `/tier1/sample` | GET | Current sample |

### Tier 3 Routes

| Endpoint | Method | Description |
|---|---|---|
| `/tier3/reset` | POST | `{"scenario_index": 0}` |
| `/tier3/step` | POST | `{"dependencies": [...]}` |
| `/tier3/score` | GET | Aggregated precision/recall |

### Tiers 4-6 Routes

| Endpoint | Method | Description |
|---|---|---|
| `/cascade/reset` | POST | `{"task_id": "task_4_cascade_easy", "scenario_index": 0}` |
| `/cascade/step` | POST | Crisis action JSON |
| `/cascade/state` | GET | Current crisis state |
| `/cascade/actions` | GET | All valid action types |
| `/cascade/metrics` | GET | Cash balance, bankruptcy flag |
| `/cascade/deadlines` | GET | Active deadline list |

### Meta Routes

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Full benchmark specification |
| `/health` | GET | `{"status": "ok"}` |
| `/curriculum` | GET | Recommended tier progression with tips |
| `/legal_iq` | POST | Compute composite Legal IQ score |
| `/legal_iq/leaderboard` | GET | All-time leaderboard |
| `/legal_iq/weights` | GET | Formula and tier weights |
| `/probes` | GET | List all 10 adversarial probes |
| `/probes/{probe_id}` | GET | Single probe scenario |
| `/probes/run` | POST | Run probe in-process |

---

## Observation Space

Each `step()` call returns:

```json
{
  "task_id": "task_2_medium",
  "step_number": 4,
  "max_steps": 20,
  "clause_text": "Vendor may auto-renew annually unless terminated 60 days prior.",
  "clause_index": 1,
  "total_clauses": 6,
  "contract_type": "SaaS Agreement",
  "parties": ["Customer", "Vendor"],
  "jurisdiction": "Delaware",
  "instructions": "Review each clause...",
  "available_actions": ["classify", "rate_severity", "flag", "suggest", "reason", "next_clause", "complete_review"],
  "last_action_feedback": "Correct risk level: high.",
  "accumulated_score": 0.41,
  "corrective_feedback": "Incorrect classification. You submitted 'payment_terms', but this is a 'confidentiality' clause...",
  "done": false
}
```

---

## Action Space

### Tier 2 Actions

```json
{"action_type": "classify",      "clause_type": "indemnification"}
{"action_type": "rate_severity", "risk_level": "high"}
{"action_type": "flag",          "flags": ["one_sided_obligation", "missing_liability_cap"]}
{"action_type": "suggest",       "suggested_action": "request_modification"}
{"action_type": "reason",        "reasoning": "Unlimited indemnification without a cap is non-standard."}
{"action_type": "next_clause"}
{"action_type": "complete_review"}
```

### Tier 4-6 Crisis Actions (19 total)

| Category | Actions |
|---|---|
| **Investigation** | `cross_reference_contracts`, `review_deadline_status`, `assess_counterparty_risk`, `analyze_financial_impact` |
| **Legal** | `invoke_force_majeure`, `file_insurance_claim`, `send_formal_notice`, `request_waiver`, `terminate_contract` |
| **Financial** | `pay_penalty`, `negotiate_payment_plan`, `draw_credit_facility`, `accelerate_receivable` |
| **Communication** | `send_breach_notice`, `request_information`, `propose_amendment`, `invoke_indemnification` |
| **Control** | `advance_day`, `complete_crisis` |

---

## Reward Function

### Step Rewards (Tier 2)

All step scores are strictly in **(0.001, 0.999)** per OpenEnv specification.

| Action | Correct | Partial | Wrong |
|---|---|---|---|
| `classify` | 0.15 | 0.05 (same family) | 0.001 |
| `rate_severity` | 0.15 | 0.05 (1 level off) | 0.001 |
| `flag` | F2-based up to 0.10 | proportional | 0.001 |
| `suggest` | 0.10 | 0.04 (acceptable alt) | 0.001 |
| `reason` | 0.05 | 0.02 (partial) | 0.001 |
| `next_clause` | 0.05 (complete) | proportional | — |
| `complete_review` | 0.10 (+ 0.05 efficiency) | 0.03 | 0.001 |

Negative raw rewards are clamped to `0.001` — a signal of failure without silent zero-padding.

### Trajectory Grader Score (Tier 2)

```
Score = 0.40 × type_accuracy
      + 0.25 × risk_accuracy
      + 0.20 × flag_f2
      + 0.15 × suggestion_accuracy
      - penalty(skipped_clauses, redundant_actions)
```

### Crisis Trajectory Score (Tiers 4-6)

```
Score = 0.75 × (cash_final / cash_initial)
      + 0.15 × (deadlines_met / deadlines_total)
      + 0.10 × (rights_preserved / rights_total)
      - cascade_depth_penalty (2% per extra link)
```

Bankruptcy → score clamps to `0.001`. All math deterministic, no LLM-as-judge.

### Legal IQ Composite

```
Legal_IQ = 0.15·T1 + 0.15·T2 + 0.20·T3 + 0.50·(0.25·T4 + 0.35·T5 + 0.40·T6)
```

---

## Baseline Scores

Measured with `Qwen2.5-72B-Instruct` via HuggingFace Inference API.

| Task | Difficulty | Expected Score Range |
|---|---|---|
| `tier1_clause_reading` | Easy | 0.60–0.80 |
| `task_1_easy` | Easy | 0.70–0.90 |
| `task_2_medium` | Medium | 0.50–0.75 |
| `task_3_hard` | Hard | 0.35–0.65 |
| `tier3_dependency_mapping` | Hard | 0.30–0.55 |
| `task_4_cascade_easy` | Easy | 0.60–0.80 |
| `task_5_cascade_medium` | Medium | 0.40–0.65 |
| `task_6_cascade_hard` | Frontier | 0.20–0.50 |
| **Legal IQ (all tiers)** | Composite | **0.45–0.70** |

Scores above `0.80 Legal IQ` indicate expert-level legal reasoning.

---

## Clause Taxonomy (15 types)

`indemnification` · `limitation_of_liability` · `termination` · `confidentiality` · `non_compete` · `force_majeure` · `assignment` · `governing_law` · `warranty` · `intellectual_property` · `payment_terms` · `representations` · `dispute_resolution` · `data_protection` · `insurance`

## Risk Levels

`low` · `medium` · `high` · `critical`

## Issue Flags (13 types)

`vague_language` · `missing_liability_cap` · `one_sided_obligation` · `unusual_term` · `market_standard` · `overly_broad_scope` · `missing_time_limit` · `ambiguous_definition` · `conflicting_with_other_clause` · `missing_carve_out` · `automatic_renewal` · `unreasonable_penalty` · `silent_on_key_issue`

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| Typed `Observation`, `Action`, `Reward` Pydantic models | ✅ |
| `step(action) → (observation, reward, done, info)` | ✅ |
| `reset() → initial observation` | ✅ |
| `state() → current state` | ✅ |
| `openenv.yaml` with metadata | ✅ |
| Score strictly in `(0.001, 0.999)` | ✅ clamped at model level |
| ≥ 3 tasks with programmatic graders | ✅ 8 tasks |
| Easy → Medium → Hard range | ✅ |
| Meaningful step-level reward signal | ✅ partial credit per action |
| Real-world task simulation | ✅ contract clause review + crisis CRO |
| Dockerfile + `docker build` + `docker run` | ✅ |
| HF Space tagged `openenv` | ✅ |
| Baseline inference script | ✅ `inference.py` |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) | Required |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `OPENAI_API_KEY` or `HF_TOKEN` | API key | Required |
| `ENV_MODE` | `direct` or `http` | `direct` |
| `ENV_URL` | Server URL (http mode) | `http://127.0.0.1:7860` |
| `TIERS` | Comma-separated tiers to run | `1,2,3,4,5,6` |
| `DEBUG` | Verbose logging | `false` |


---

## Project Structure

```
lexarena/
├── lexarena_server.py      # Unified entry point (all 6 tiers, port 7860)
├── inference.py            # 6-tier curriculum agent (baseline script)
├── probe_runner.py         # Adversarial probe suite (10 probes)
├── openenv.yaml            # OpenEnv spec compliance metadata
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
│
├── models.py               # Tier 2 Pydantic models (Observation, Action, Reward)
├── environment.py          # Tier 2 ContractReviewEnv
├── rewards.py              # Tier 2 step-level reward functions
├── graders.py              # Tier 2 trajectory graders
├── tasks.py                # Task registry and instructions
│
├── cascade_models.py       # Tier 4-6 Pydantic models
├── cascade_environment.py  # LexDominoCrisisEnv
├── cascade_rewards.py      # Crisis step rewards
├── cascade_graders.py      # Crisis trajectory grader
│
├── cuad_loader.py          # Tier 1 CUAD dataset loader
├── tier1_grader.py         # Tier 1 F2-weighted grader
├── tier3_environment.py    # Tier 3 dependency mapping env + grader
├── lexarena_models.py      # Shared Tier 1/3 models
├── lexarena_scorer.py      # Legal IQ composite formula
│
└── data/
    ├── manifest.json
    ├── task_1_easy/        ← 3 scenarios
    ├── task_2_medium/      ← 3 scenarios
    ├── task_3_hard/        ← 3 scenarios
    ├── task_4_cascade_easy/    ← 2 scenarios
    ├── task_5_cascade_medium/  ← 2 scenarios
    ├── task_6_cascade_hard/    ← 2 scenarios
    ├── graph_mapping/      ← Tier 3 dependency scenarios
    └── probes/             ← 10 adversarial probe JSONs
```

---

## License

MIT
