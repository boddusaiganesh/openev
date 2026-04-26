"""
Full audit script — runs all checks and reports issues.
"""
import sys, os, json, glob, traceback

issues = []
warnings = []

def ok(msg): print(f"  OK   {msg}")
def warn(msg): print(f"  WARN {msg}"); warnings.append(msg)
def fail(msg): print(f"  FAIL {msg}"); issues.append(msg)

# ===========================================================================
# 1. Imports
# ===========================================================================
print("\n=== 1. Module Imports ===")
import_targets = [
    "models", "tasks", "environment", "rewards", "graders",
    "cascade_models", "cascade_environment", "cascade_rewards", "cascade_graders",
    "lexarena_models", "lexarena_scorer",
    "cuad_loader", "tier1_grader", "tier3_environment", "tier3_grader",
    "probe_runner",
]
for mod in import_targets:
    try:
        __import__(mod)
        ok(mod)
    except Exception as e:
        fail(f"Import {mod}: {e}")

# ===========================================================================
# 2. Server import + route check
# ===========================================================================
print("\n=== 2. Server Routes ===")
try:
    import lexarena_server
    routes = {r.path for r in lexarena_server.app.routes}
    required = [
        "/", "/health", "/curriculum",
        "/tier1/reset", "/tier1/step", "/tier1/sample",
        "/reset", "/step", "/state",
        "/tier3/reset", "/tier3/step", "/tier3/score",
        "/cascade/reset", "/cascade/step", "/cascade/state",
        "/cascade/actions", "/cascade/deadlines", "/cascade/metrics",
        "/legal_iq", "/legal_iq/leaderboard", "/legal_iq/weights",
        "/probes", "/probes/{probe_id}", "/probes/run",
    ]
    for r in required:
        if r in routes:
            ok(f"Route {r}")
        else:
            fail(f"Missing route: {r}")
except Exception as e:
    fail(f"lexarena_server import: {e}")
    traceback.print_exc()

# ===========================================================================
# 3. Score clamping
# ===========================================================================
print("\n=== 3. Score Clamping (must be strictly 0.001-0.999) ===")
try:
    from models import Reward, GraderResult
    from cascade_models import CascadeReward, CascadeGraderResult

    for cls, name, raw_val, expected in [
        (Reward,              "Reward(-1.0)",          -1.0,  0.001),
        (Reward,              "Reward(0.0)",            0.0,  0.001),
        (Reward,              "Reward(0.5)",            0.5,  0.5),
        (Reward,              "Reward(1.0)",            1.0,  0.999),
        (CascadeReward,       "CascadeReward(-1.0)",   -1.0,  0.001),
        (CascadeReward,       "CascadeReward(0.0)",     0.0,  0.001),
        (CascadeReward,       "CascadeReward(1.0)",     1.0,  0.999),
        (GraderResult,        "GraderResult(0.0)",      0.0,  0.001),
        (GraderResult,        "GraderResult(1.0)",      1.0,  0.999),
        (CascadeGraderResult, "CascadeGraderResult(0)", 0.0,  0.001),
        (CascadeGraderResult, "CascadeGraderResult(1)", 1.0,  0.999),
    ]:
        got = cls(score=raw_val).score
        if abs(got - expected) < 1e-9:
            ok(f"{name} -> {got}")
        else:
            fail(f"{name} -> {got} (expected {expected})")
except Exception as e:
    fail(f"Score clamping test: {e}")
    traceback.print_exc()

# ===========================================================================
# 4. Data files
# ===========================================================================
print("\n=== 4. Data Files ===")
try:
    with open("data/manifest.json") as f:
        manifest = json.load(f)
    for task_id, cfg in manifest.items():
        for rel in cfg.get("scenario_files", []):
            path = os.path.join("data", rel)
            if os.path.exists(path):
                ok(f"{task_id}/{rel}")
            else:
                fail(f"Missing: {path}")
except Exception as e:
    fail(f"manifest.json: {e}")

# Probes
probes = sorted(glob.glob("data/probes/*.json"))
if len(probes) >= 10:
    ok(f"{len(probes)} probe files found")
else:
    fail(f"Expected >= 10 probes, found {len(probes)}")

for p in probes:
    with open(p) as f:
        d = json.load(f)
    required_fields = ["probe_id", "failure_mode", "description", "initial_cash", "deadlines"]
    missing = [k for k in required_fields if k not in d]
    if missing:
        fail(f"{os.path.basename(p)} missing fields: {missing}")
    else:
        ok(f"{os.path.basename(p)} schema OK")

# Graph mapping
gm = glob.glob("data/graph_mapping/*.json")
if gm:
    ok(f"{len(gm)} graph_mapping scenario(s)")
else:
    warn("No data/graph_mapping/ scenarios (Tier 3 will use built-in)")

# ===========================================================================
# 5. openenv.yaml
# ===========================================================================
print("\n=== 5. openenv.yaml ===")
try:
    import yaml
    with open("openenv.yaml", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    required_keys = ["name", "version", "description", "tasks"]
    for k in required_keys:
        if k in spec:
            ok(f"openenv.yaml has '{k}'")
        else:
            fail(f"openenv.yaml missing '{k}'")
    tasks = spec.get("tasks", [])
    if len(tasks) >= 3:
        ok(f"{len(tasks)} tasks defined")
    else:
        fail(f"Need >= 3 tasks, got {len(tasks)}")
    for t in tasks:
        for k in ["id", "tier", "weight"]:
            if k not in t:
                fail(f"Task '{t.get('id','?')}' missing field '{k}'")
except Exception as e:
    fail(f"openenv.yaml: {e}")

# ===========================================================================
# 6. README.md
# ===========================================================================
print("\n=== 6. README.md ===")
try:
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
    checks = {
        "HF Space tags (openenv)": "tags:" in readme and "openenv" in readme,
        "sdk: docker": "sdk: docker" in readme,
        "app_port: 7860": "app_port: 7860" in readme,
        "Motivation section": "Motivation" in readme or "motivation" in readme,
        "Action Space section": "Action Space" in readme or "action_type" in readme,
        "Observation Space section": "Observation Space" in readme or "observation" in readme.lower(),
        "Reward Function section": "Reward Function" in readme or "reward" in readme.lower(),
        "Baseline Scores section": "Baseline Score" in readme or "Expected Score" in readme,
        "Setup instructions": "docker build" in readme,
        "At least 3 task IDs": readme.count("task_") >= 3,
        "Docker run": "docker run" in readme,
    }
    for check, passed in checks.items():
        if passed:
            ok(f"README: {check}")
        else:
            fail(f"README missing: {check}")
except Exception as e:
    fail(f"README.md: {e}")

# ===========================================================================
# 7. Dockerfile
# ===========================================================================
print("\n=== 7. Dockerfile ===")
try:
    with open("Dockerfile") as f:
        df = f.read()
    df_checks = {
        "FROM python": "FROM python" in df,
        "EXPOSE 7860": "7860" in df,
        "lexarena_server": "lexarena_server" in df,
        "CMD or ENTRYPOINT": "CMD" in df or "ENTRYPOINT" in df,
        "COPY requirements": "requirements" in df,
        "pip install": "pip install" in df,
    }
    for check, passed in df_checks.items():
        if passed:
            ok(f"Dockerfile: {check}")
        else:
            fail(f"Dockerfile missing: {check}")
except Exception as e:
    fail(f"Dockerfile: {e}")

# ===========================================================================
# 8. requirements.txt
# ===========================================================================
print("\n=== 8. requirements.txt ===")
try:
    with open("requirements.txt") as f:
        reqs = f.read()
    required_pkgs = ["fastapi", "uvicorn", "pydantic", "httpx", "openai", "datasets", "numpy"]
    for pkg in required_pkgs:
        if pkg in reqs:
            ok(f"requirements: {pkg}")
        else:
            fail(f"requirements.txt missing: {pkg}")
except Exception as e:
    fail(f"requirements.txt: {e}")

# ===========================================================================
# 9. Legal IQ formula
# ===========================================================================
print("\n=== 9. Legal IQ Formula ===")
try:
    from lexarena_scorer import compute_legal_iq
    r = compute_legal_iq(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    expected = 0.15 + 0.15 + 0.20 + 0.50 * (0.25 + 0.35 + 0.40)
    if abs(r.legal_iq - expected) < 1e-9:
        ok(f"Legal IQ(1,1,1,1,1,1) = {r.legal_iq}")
    else:
        fail(f"Legal IQ wrong: got {r.legal_iq}, expected {expected}")
    r0 = compute_legal_iq(0,0,0,0,0,0)
    if r0.legal_iq == 0.0:
        ok(f"Legal IQ(0,0,0,0,0,0) = 0.0")
    else:
        warn(f"Legal IQ(all 0) = {r0.legal_iq} (expected 0.0)")
except Exception as e:
    fail(f"Legal IQ formula: {e}")

# ===========================================================================
# 10. inference.py structure
# ===========================================================================
print("\n=== 10. inference.py Structure ===")
try:
    with open("inference.py", encoding="utf-8") as f:
        inf_src = f.read()
    checks = {
        "SYSTEM_PROMPT_T1 defined": "SYSTEM_PROMPT_T1" in inf_src,
        "SYSTEM_PROMPT_T2 defined": "SYSTEM_PROMPT_T2" in inf_src,
        "SYSTEM_PROMPT_T3 defined": "SYSTEM_PROMPT_T3" in inf_src,
        "SYSTEM_PROMPT_CRISIS defined": "SYSTEM_PROMPT_CRISIS" in inf_src,
        "get_system_prompt function": "def get_system_prompt" in inf_src,
        "call_llm function": "def call_llm" in inf_src,
        "main function": "def main" in inf_src,
        "ACTIVE_TIERS": "ACTIVE_TIERS" in inf_src,
        "curriculum runner (T1-T6)": "TIER 1" in inf_src and "TIER 2" in inf_src,
        "API_BASE_URL env var": "API_BASE_URL" in inf_src,
        "API_KEY env var": "API_KEY" in inf_src,
        "baseline_results.json output": "baseline_results.json" in inf_src,
    }
    for check, passed in checks.items():
        if passed:
            ok(f"inference.py: {check}")
        else:
            fail(f"inference.py missing: {check}")
except Exception as e:
    fail(f"inference.py: {e}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print(f"  WARNINGS : {len(warnings)}")
print(f"  FAILURES : {len(issues)}")
print("=" * 60)
if issues:
    print("\nFAILURES:")
    for i in issues:
        print(f"  - {i}")
if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(f"  - {w}")
if not issues:
    print("\n  All checks passed!")
