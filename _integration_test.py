import httpx, json

base = "http://127.0.0.1:7860"

# --- Tier 2: corrective_feedback after step ---
r = httpx.post(f"{base}/reset", json={"task_id": "task_1_easy"}, timeout=5)
obs = r.json()
action = {"action_type": "classify", "clause_type": "payment_terms"}
r2 = httpx.post(f"{base}/step", json=action, timeout=5)
resp = r2.json()
obs2 = resp.get("observation", {})
feedback = obs2.get("corrective_feedback", "NOT PRESENT")
print("Tier 2 corrective_feedback:", repr(feedback[:100]))

# --- Cascade: corrective_feedback after step ---
r = httpx.post(f"{base}/cascade/reset",
               json={"task_id": "task_4_cascade_easy", "scenario_index": 0},
               timeout=5)
obs_c = r.json()
action_c = {"action_type": "cross_reference_contracts", "justification": "test"}
r3 = httpx.post(f"{base}/cascade/step", json=action_c, timeout=5)
resp3 = r3.json()
obs3 = resp3.get("observation", {})
cf3 = obs3.get("corrective_feedback", "NOT PRESENT")
print("Cascade corrective_feedback:", repr(cf3[:100]))

# --- Legal IQ ---
r4 = httpx.post(f"{base}/legal_iq", json={
    "model_name": "test_model",
    "t1_score": 0.72, "t2_score": 0.65, "t3_score": 0.58,
    "t4_score": 0.80, "t5_score": 0.70, "t6_score": 0.55
}, timeout=5)
iq = r4.json()
legal_iq_val = iq.get("legal_iq", "?")
label = iq.get("label", "?")
print(f"Legal IQ: {legal_iq_val} -> {label}")

# --- Curriculum ---
r5 = httpx.get(f"{base}/curriculum", timeout=5)
curr = r5.json()
steps = curr.get("curriculum", [])
print(f"Curriculum steps: {len(steps)}")
for s in steps:
    print(f"  Step {s['step']}: {s.get('tier','?')} - {s.get('goal','?')}")

print("\nAll integration checks complete.")
