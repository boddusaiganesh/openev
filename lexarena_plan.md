# LexArena: The Complete Legal Intelligence Benchmark

> *The first benchmark in the world that tests an AI agent's complete legal intelligence stack — from reading a single clause to keeping a company alive through a 30-day systemic crisis.*

---

## The Core Thesis

Every legal AI benchmark today tests **one layer** of intelligence in isolation:

| Benchmark | What it tests | What it misses |
|---|---|---|
| **ContractEval** | Can the model *read* a clause? | Can it *act* on what it read? |
| **CUAD / LexGLUE** | Can the model *classify* legal text? | Can it *reason across* multiple documents? |
| **LexDomino (current)** | Can the agent *survive a crisis*? | Can it actually *read* the contracts it is acting on? |

**LexArena closes all three gaps simultaneously.**

It is the first benchmark to test the **complete chain of legal intelligence**:

```
READ → CLASSIFY → CONNECT → DECIDE → SURVIVE
```

If an agent cannot read clause language precisely, it will make wrong decisions.
If it can read but cannot find cross-document dependencies, it will miss the cascade.
If it finds the dependencies but cannot act strategically under time pressure, it dies.
**LexArena measures all of this with a single composite score.**

---

## The Architecture — 6 Tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         L E X A R E N A                             │
│                  Complete Legal Intelligence Stack                   │
│                                                                     │
│  TIER 1         TIER 2         TIER 3         TIERS 4-6             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  CLAUSE  │  │   RISK   │  │  GRAPH   │  │    SYSTEMIC      │   │
│  │ READING  │─►│ ANALYSIS │─►│ MAPPING  │─►│    CASCADE       │   │
│  │          │  │          │  │          │  │  CRISIS MGMT     │   │
│  │ContractEv│  │OpenEnv   │  │ NEW TIER │  │ LexDomino 4/5/6  │   │
│  │  al CUAD │  │tasks 1-3 │  │          │  │                  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│       15%           15%           20%              50%              │
│                                                                     │
│          ──────────────────────────────────────────                 │
│                     Legal IQ Composite Score                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tier 1 — Clause Reading Comprehension
**Adapted from ContractEval / CUAD**

**What it tests**: Can the agent extract the *exact, verbatim* sentence that answers a legal question?

**Data**: CUAD dataset — 510 real commercial contracts, 41 question types (exclusivity, indemnification, termination, force majeure, governing law, etc.)

**Task**: Given `{contract clause}` + `{question}`, extract the exact relevant sentence.
If none: respond `"No related clause."`

**Scoring** (pure math, no LLM judge):
```
T1_score = 0.60 × F2_score          # recall-weighted (missing clause = worse than over-extracting)
         + 0.25 × Jaccard_mean      # partial credit for near-correct extractions
         + 0.15 × (1 - laziness)    # penalise false "No related clause" refusals
```

**Why F2, not F1**: In legal risk identification, a missed clause (FN) is far more dangerous than a spurious extraction (FP). F2 penalises recall failures 2× more than precision failures.

**Cognitive dimension tested**: *Precise legal language comprehension*

---

## Tier 2 — Risk Classification
**Current OpenEnv tasks 1/2/3**

**What it tests**: Given a clause, can the agent correctly classify its type, risk level, and recommend the correct action?

**Task progression**:
- Task 1 (Easy): Classify clause type from taxonomy
- Task 2 (Medium): Classify type + rate risk + flag issues
- Task 3 (Hard): Full review — type, risk, flags, recommendation, cross-clause reasoning

**Scoring**: Weighted label accuracy (existing grader)

**The upgrade**: Replace hand-written clause text in tasks 1-3 with **real CUAD clause language** — making Tiers 1 and 2 coherent (same underlying contract text, tested at different levels of depth).

**Cognitive dimension tested**: *Legal risk assessment*

---

## Tier 3 — Dependency Graph Mapping *(NEW)*
**The missing middle layer**

This is the most novel tier — it tests whether an agent can understand the **architecture of a contract portfolio** before a crisis hits.

**What it tests**: Given 3-5 contracts, can the agent **proactively map all dependency edges** (without being asked about a specific crisis)?

**Task**: The agent is given a set of contracts and a time budget. It must output a dependency graph:
```json
{
  "dependencies": [
    {
      "source": "MSA-001 Clause 4.2",
      "target": "LOAN-001 Covenant 3",
      "type": "cascade_trigger",
      "reasoning": "Penalty payment drops cash below covenant threshold"
    }
  ]
}
```

**Scoring**:
```
T3_score = precision × recall × F1 against ground-truth edge list
         + bonus for discovering edge type correctly (not just existence)
         + bonus for correctly ordering edges by severity
```

**Ground truth**: The `dependency_edges` arrays in LexDomino scenario JSONs. The agent is scored against those known edges.

**Why this matters**: Current benchmarks test either reading (Tier 1) or acting (Tiers 4-6). No benchmark tests whether an agent can understand the *structural relationships* between documents — the exact skill that determines whether an agent will see a cascade coming.

**Cognitive dimension tested**: *Multi-document structural reasoning*

---

## Tiers 4–6 — Systemic Crisis Management
**LexDomino (existing, enhanced)**

### Tier 4: Single Cascade (`task_4_cascade_easy`)
- 2-3 contracts, 10 days, cooperative counterparties
- 0-1 hidden dependency edges
- Expected score range: 0.75–0.90

### Tier 5: Multi-Document Crisis (`task_5_cascade_medium`)
- 4-5 contracts, 20 days, mixed counterparties
- 2-3 hidden dependency edges, secondary effects
- Expected score range: 0.45–0.70

### Tier 6: Full Systemic Cascade (`task_6_cascade_hard`)
- 5-6 contracts, 30 days, adversarial counterparties
- Compound shocks (Day 1 + Day 14-16)
- 4-5 hidden dependencies
- Expected score range: 0.20–0.50

**Scoring** (pure math):
```
T4-6_score = 0.75 × (cash_final / cash_initial)
           + 0.15 × (deadlines_met / deadlines_total)
           + 0.10 × (dependencies_discovered / dependencies_total)
           - cascade_depth_penalty
```

**The upgrade from Tier 3 integration**: Agents that scored well in Tier 3 (proactive dependency mapping) should score measurably better in Tiers 4-6. This creates a **testable hypothesis** for the academic paper:
> *"Proactive structural reasoning (Tier 3) is the strongest predictor of crisis survival (Tiers 4-6), more than clause reading accuracy (Tier 1) or risk classification (Tier 2)."*

**Cognitive dimensions tested**: *Temporal reasoning, deadline management, anti-sycophancy, multi-objective optimization, long-horizon planning*

---

## The Composite Score — "Legal IQ"

```
Legal_IQ = 0.15 × T1_Reading_Score
          + 0.15 × T2_Classification_Score
          + 0.20 × T3_Dependency_Score
          + 0.50 × T4_T5_T6_Crisis_Score

where T4_T5_T6_Crisis_Score = (0.25 × T4 + 0.35 × T5 + 0.40 × T6)
```

**Why 50% weight on crisis management?**
Because any model can be fine-tuned to extract clauses. Strategic decision-making under time pressure is the hard, generalizable capability — the one that actually matters in the real world.

**Score interpretation**:
| Score | Meaning |
|---|---|
| 0.85 – 1.00 | **Expert CRO level** — reads precisely, maps structure, navigates crises |
| 0.70 – 0.84 | **Senior Lawyer level** — strong comprehension, handles moderate crises |
| 0.50 – 0.69 | **Junior Associate level** — reads well, misses cascade dependencies |
| 0.30 – 0.49 | **Paralegal level** — basic comprehension, struggles under pressure |
| 0.00 – 0.29 | **Fails legal practice bar** — systematic blind spots |

---

## The Data Strategy — CUAD + LexDomino as One Coherent Dataset

The most important architectural decision: **Tier 1 and Tiers 4-6 share the same underlying contract text.**

### How it works:

**Step 1**: Take a real CUAD contract (e.g., a Master Service Agreement with force majeure, indemnification, and SLA clauses).

**Step 2**: Use that exact contract text as the foundation for a LexDomino scenario — same clauses, same legal language.

**Step 3**: Create CUAD-style QA pairs for Tier 1 testing on those exact clauses.

**Step 4**: Create the dependency graph and crisis scenario for Tiers 3-6 using those same clauses.

**Result**: An agent is tested on the *same contract* at four different levels of depth:
- Can it read a clause? (Tier 1)
- Can it classify its risk? (Tier 2)
- Can it find its hidden connections? (Tier 3)
- Can it act on that knowledge under crisis? (Tiers 4-6)

**This is completely novel** — no existing benchmark tests depth of understanding this way.

---

## 5 Cognitive Dimensions (Mapped to Tiers)

| Dimension | Tier | What Failure Looks Like |
|---|---|---|
| **Precise Language Comprehension** | 1 | Extracts wrong sentence, paraphrases instead of quoting |
| **Legal Risk Assessment** | 2 | Misclassifies penalty clause as limitation of liability |
| **Structural Graph Reasoning** | 3 | Misses that FM clause in Contract A voids insurance in Contract B |
| **Temporal Deadline Management** | 4-6 | Ignores 48-hour notice window, pays $500K penalty |
| **Anti-Sycophancy Under Pressure** | 4-6 | Pays aggressive counterparty demand without checking legal position |

---

## 10 Adversarial Probe Scenarios

Beyond the regular difficulty ladder, LexArena includes **10 adversarial probes** designed to catch specific known LLM failure modes:

| Probe | Failure Mode Targeted |
|---|---|
| `probe_fm_void` | Invoking FM when it voids insurance (mutual exclusion trap) |
| `probe_lazy_reader` | Scenario where "No related clause" is always wrong |
| `probe_sycophancy` | Aggressive counterparty demand that is legally invalid |
| `probe_covenant_blindness` | Missing that penalty payment will breach debt covenant |
| `probe_deadline_stack` | Three deadlines in 48 hours requiring prioritisation |
| `probe_key_person_chain` | 3-hop dependency (bonus → departure → license loss → bankruptcy) |
| `probe_false_urgency` | Counterparty creates fake urgency for a non-binding demand |
| `probe_supersession` | Later clause overrides earlier clause — agent must find it |
| `probe_compound_shock` | Primary crisis + secondary shock 10 days later |
| `probe_cross_default` | Judgment in Contract A triggers cross-default in Contract B |

---

## Academic Novelty Claims

**Claim 1: First benchmark testing complete legal intelligence chain**
No existing benchmark (CUAD, LexGLUE, ContractEval, MAUD, ContractBench) tests comprehension + classification + structural reasoning + strategic crisis management in a single coherent framework.

**Claim 2: Objective, fully deterministic scoring**
Every metric is a mathematical function. No LLM-as-judge anywhere. Two researchers running LexArena on the same model will produce bit-identical scores.

**Claim 3: Tests anti-sycophancy in legal context**
No existing benchmark explicitly tests whether an LLM will resist an aggressive but legally invalid counterparty demand. LexArena's adversarial probes directly measure this.

**Claim 4: Testable hypothesis on skill hierarchy**
The architecture allows testing: *"Does Tier 3 structural reasoning predict Tier 4-6 crisis performance better than Tier 1 reading accuracy?"* This is a genuine research question with a clear empirical answer.

**Claim 5: Grounded in real-world corporate crises**
LexDomino scenarios are structurally modelled on real events (Wirecard, SVB, Enron supply-chain failures). This is not a synthetic toy environment.

---

## System Architecture (Technical)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LexArena Engine                              │
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  Tier 1 Runner  │   │  Tier 2 Runner  │   │  Tier 3 Runner  │  │
│  │  (ContractEval  │   │  (OpenEnv Env)  │   │  (Graph Grader) │  │
│  │   adapted)      │   │                 │   │                 │  │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘  │
│           │                     │                     │            │
│           └─────────────────────┴─────────────────────┘            │
│                                 │                                   │
│                    ┌────────────▼──────────────┐                   │
│                    │    LexArena Compositor     │                   │
│                    │  Aggregates T1+T2+T3 into  │                   │
│                    │  Legal_IQ pre-score        │                   │
│                    └────────────┬──────────────┘                   │
│                                 │                                   │
│                    ┌────────────▼──────────────┐                   │
│                    │   LexDomino Crisis Env     │                   │
│                    │   Tiers 4 / 5 / 6          │                   │
│                    │   (cascade_environment.py) │                   │
│                    └────────────┬──────────────┘                   │
│                                 │                                   │
│                    ┌────────────▼──────────────┐                   │
│                    │    Final Legal IQ Score    │                   │
│                    │    + Per-Tier Breakdown    │                   │
│                    │    + Failure Mode Report   │                   │
│                    └───────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## New Files Required (Implementation Plan)

### Phase 1 — CUAD Integration (2 weeks)
| File | Purpose |
|---|---|
| `cuad_loader.py` | Download CUAD, convert to LexArena Tier 1 format |
| `tier1_runner.py` | ContractEval-adapted extraction evaluator |
| `tier1_grader.py` | F2 + Jaccard + laziness scoring |
| `data/cuad_scenarios/` | CUAD clauses reformatted as LexArena Tier 1 samples |

### Phase 2 — Tier 3 (Dependency Mapping) (2 weeks)
| File | Purpose |
|---|---|
| `tier3_environment.py` | New environment: agent maps dependency graph, scored vs. ground truth |
| `tier3_grader.py` | Precision/recall against known edge list |
| `data/graph_mapping/` | Graph mapping scenarios derived from LexDomino scenario edges |

### Phase 3 — LexArena Compositor (1 week)
| File | Purpose |
|---|---|
| `lexarena_runner.py` | Orchestrates Tiers 1→2→3→4→5→6 in sequence |
| `lexarena_scorer.py` | Computes Legal IQ composite score |
| `lexarena_report.py` | Generates per-tier breakdown + failure mode report |
| `lexarena_server.py` | Unified API: single `/run` endpoint, runs full 6-tier suite |

### Phase 4 — Adversarial Probes (1 week)
| File | Purpose |
|---|---|
| `data/probes/probe_*.json` | 10 adversarial probe scenarios |
| `probe_runner.py` | Runs all 10 probes, produces failure mode heatmap |

### Phase 5 — Paper & Leaderboard (2 weeks)
| File | Purpose |
|---|---|
| `lexarena_benchmark.py` | Full deterministic benchmark: 4 strategies × all tiers |
| `leaderboard/` | Static leaderboard HTML with per-model tier breakdown |
| `paper/` | LaTeX paper draft with results |

---

## Expected Results (Hypothesis)

Based on the current LexDomino benchmark numbers and ContractEval's published results:

| Model Class | T1 Reading | T2 Classify | T3 Graph | T4-6 Crisis | **Legal IQ** |
|---|---|---|---|---|---|
| GPT-4o / Claude 3.5 | 0.72 | 0.78 | 0.45 | 0.55 | **0.58** |
| GPT-4o-mini / Haiku | 0.61 | 0.65 | 0.28 | 0.38 | **0.44** |
| Gemma 3 12B | 0.55 | 0.58 | 0.20 | 0.30 | **0.37** |
| Qwen3 14B (thinking) | 0.65 | 0.70 | 0.35 | 0.42 | **0.48** |
| Fine-tuned Legal LLM | 0.82 | 0.85 | 0.25 | 0.25 | **0.46** |

**Key insight the paper will prove**: A fine-tuned legal LLM scores highest on Tiers 1-2 but lowest on Tiers 4-6. General reasoning models dominate crisis management. **The benchmark proves that legal fine-tuning does not transfer to strategic legal reasoning** — a genuine, publishable finding.

---

## The One-Liner for the Paper Abstract

> *"LexArena is the first benchmark to evaluate legal AI across the complete intelligence stack — from verbatim clause extraction to 30-day corporate crisis survival — using a six-tier framework with a fully deterministic composite score, revealing that legal language fine-tuning fails to transfer to strategic crisis reasoning."*

---

## Open Questions for Review

> [!IMPORTANT]
> **Q1**: Should Tier 3 (Dependency Mapping) be a standalone pre-episode phase, or should it be embedded as the first N steps of the LexDomino crisis episode itself? Embedded is more realistic but harder to score cleanly.

> [!IMPORTANT]
> **Q2**: Should the Legal IQ score be a single number or a radar chart (5 dimensions)? A single number is easier to publish; a radar chart is more diagnostic.

> [!WARNING]
> **Q3**: CUAD data is Creative Commons licensed for research use. If this is intended for commercial deployment (not just research), we need to check licensing compatibility.

> [!NOTE]
> **Q4**: Should the 10 adversarial probes count toward the Legal IQ score, or be a separate "stress test" battery reported independently?

> [!NOTE]
> **Q5**: What is the target venue? NeurIPS/ICML (machine learning), ACL (NLP), or ICAIL (legal AI)? The framing of the contribution differs significantly by venue.
