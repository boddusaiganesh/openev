"""Build leaderboard/index.html from latest benchmark results."""
import json, os, glob

os.makedirs("leaderboard", exist_ok=True)

files = sorted(glob.glob("artifacts/lexarena_full_benchmark_*.json"))
results = json.load(open(files[-1]))
ranked = sorted(results.items(), key=lambda x: -x[1]["legal_iq"])


def clr(s):
    if s >= 0.75: return "#22c55e"
    if s >= 0.50: return "#f59e0b"
    if s >= 0.30: return "#f97316"
    return "#ef4444"


def bar(s):
    c, p = clr(s), int(s * 100)
    return (f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="background:#1e293b;border-radius:3px;width:80px;height:8px;overflow:hidden;">'
            f'<div style="background:{c};width:{p}%;height:100%;border-radius:3px;"></div></div>'
            f'<span style="color:{c};font-size:12px;">{s:.3f}</span></div>')


pal = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]
label_colors = {
    "Expert CRO Level": "#7c3aed",
    "Senior Lawyer Level": "#2563eb",
    "Junior Associate Level": "#0891b2",
    "Paralegal Level": "#d97706",
    "Fails Legal Practice Bar": "#dc2626",
}
medals = {1: "🥇", 2: "🥈", 3: "🥉"}

rows = ""
for rank, (m, d) in enumerate(ranked, 1):
    iq = d["legal_iq"]
    lbl = d["label"]
    bc = label_colors.get(lbl, "#64748b")
    ic = clr(iq)
    med = medals.get(rank, f"#{rank}")
    rows += (f'<tr style="border-bottom:1px solid #1e293b;">'
             f'<td style="padding:14px 12px;text-align:center;font-size:18px;">{med}</td>'
             f'<td style="padding:14px 12px;font-weight:600;color:#f1f5f9;">{m}</td>'
             f'<td style="padding:14px 12px;"><span style="background:{bc}22;color:{bc};'
             f'padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">{lbl}</span></td>'
             f'<td style="padding:14px 12px;text-align:center;font-size:24px;font-weight:900;color:{ic};">{iq:.4f}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t1_reading"])}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t2_classification"])}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t3_dependency"])}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t4_crisis_easy"])}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t5_crisis_medium"])}</td>'
             f'<td style="padding:14px 12px;">{bar(d["t6_crisis_hard"])}</td>'
             f'</tr>\n')

datasets = json.dumps([
    {"label": m,
     "data": [round(d["t1_reading"]*100,1), round(d["t2_classification"]*100,1),
               round(d["t3_dependency"]*100,1), round(d["t4_crisis_easy"]*100,1),
               round(d["t6_crisis_hard"]*100,1)],
     "borderColor": pal[i % 4],
     "backgroundColor": pal[i % 4] + "22",
     "pointBackgroundColor": pal[i % 4]}
    for i, (m, d) in enumerate(ranked)
])
labels_json = '["Clause Reading","Risk Classify","Dep. Mapping","Crisis Easy","Crisis Hard"]'

probes = [
    ("probe_fm_void", "FM invocation voids insurance — mutual exclusion trap"),
    ("probe_sycophancy", "Paying legally invalid aggressive demand"),
    ("probe_covenant_blindness", "Missing debt covenant cascade trigger"),
    ("probe_cross_default", "Judgment A triggers loan acceleration in B"),
    ("probe_deadline_stack", "Three deadlines in 48 hours — must prioritise"),
    ("probe_key_person_chain", "3-hop: bonus miss → departure → license → bankruptcy"),
    ("probe_false_urgency", "Non-binding LOI disguised as formal notice"),
    ("probe_supersession", "Later addendum overrides original penalty clause"),
    ("probe_compound_shock", "Two simultaneous crisis events on Day 1"),
    ("probe_lazy_reader", "All deadlines are explicit — agent must actually read"),
]
probe_html = "\n".join(
    f'<div class="probe"><div class="pt">{p}</div>{d}</div>'
    for p, d in probes
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LexArena — Legal Intelligence Leaderboard</title>
<meta name="description" content="LexArena: The first AI benchmark testing the complete legal intelligence stack from clause reading to 30-day corporate crisis survival.">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Inter,system-ui,sans-serif;background:#030712;color:#e2e8f0;min-height:100vh}}
.hero{{background:linear-gradient(135deg,#0f0c29,#302b63 50%,#24243e);padding:70px 40px 60px;text-align:center;border-bottom:1px solid #1e293b;position:relative;overflow:hidden}}
.hero::before{{content:"";position:absolute;inset:0;background:radial-gradient(ellipse at 50% 0%,#6366f133,transparent 70%)}}
.badge{{display:inline-block;background:linear-gradient(90deg,#6366f1,#a855f7);color:#fff;padding:6px 20px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;position:relative}}
h1{{font-size:56px;font-weight:900;background:linear-gradient(135deg,#e2e8f0,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px;position:relative}}
.subtitle{{color:#94a3b8;font-size:16px;margin-bottom:6px;position:relative}}
.tagline{{color:#6366f1;font-size:14px;letter-spacing:2px;position:relative}}
.container{{max-width:1320px;margin:0 auto;padding:40px 24px}}
.sec{{font-size:18px;font-weight:800;color:#f1f5f9;margin:40px 0 20px;display:flex;align-items:center;gap:12px}}
.sec::after{{content:"";flex:1;height:1px;background:linear-gradient(90deg,#334155,transparent)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:40px}}
.stat{{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;text-align:center;transition:all .2s;cursor:default}}
.stat:hover{{border-color:#6366f1;transform:translateY(-2px)}}
.stat .v{{font-size:30px;font-weight:900;margin-bottom:4px}}
.stat .l{{color:#64748b;font-size:12px}}
.table-wrap{{background:#0f172a;border:1px solid #1e293b;border-radius:14px;overflow-x:auto;margin-bottom:40px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{background:#0a0f1e;color:#64748b;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:1px;padding:14px 12px;text-align:left;border-bottom:1px solid #1e293b;white-space:nowrap}}
tbody tr{{transition:background .15s}}
tbody tr:hover{{background:#1e293b33}}
.chart-wrap{{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:30px;max-width:560px;margin:0 auto 40px}}
.formula{{background:#0a0f1e;border:1px solid #334155;border-radius:12px;padding:24px;font-family:"Courier New",monospace;font-size:13px;line-height:2.4;color:#94a3b8;margin-bottom:40px}}
.formula .a{{color:#818cf8;font-weight:700}}
.probes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:40px}}
.probe{{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;font-size:13px;color:#94a3b8;transition:all .2s}}
.probe:hover{{border-color:#f59e0b66;transform:translateY(-1px)}}
.probe .pt{{color:#f1f5f9;font-weight:600;margin-bottom:5px;font-family:"Courier New",monospace;font-size:12px;color:#f59e0b}}
.tiers{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-bottom:40px}}
.tier{{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;transition:all .2s}}
.tier:hover{{border-color:#6366f155;transform:translateY(-2px)}}
.tier-header{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.tier-badge{{color:#fff;border-radius:7px;padding:3px 12px;font-weight:800;font-size:13px}}
.tier-name{{font-size:15px;font-weight:700;color:#f1f5f9}}
.tier-weight{{margin-left:auto;border-radius:12px;padding:2px 10px;font-size:11px;font-weight:600}}
.tier-desc{{color:#64748b;font-size:12px;line-height:1.6}}
footer{{text-align:center;padding:40px;color:#334155;font-size:12px;border-top:1px solid #1e293b}}
</style>
</head>
<body>
<div class="hero">
  <div class="badge">&#9654; Open Benchmark v1.0</div>
  <h1>LexArena</h1>
  <p class="subtitle">The Complete Legal Intelligence Benchmark</p>
  <p class="tagline">READ &rarr; CLASSIFY &rarr; CONNECT &rarr; DECIDE &rarr; SURVIVE</p>
</div>
<div class="container">

<div class="sec">Overview</div>
<div class="stats">
  <div class="stat"><div class="v" style="color:#6366f1;">6</div><div class="l">Evaluation Tiers</div></div>
  <div class="stat"><div class="v" style="color:#22c55e;">{ranked[0][1]['legal_iq']:.4f}</div><div class="l">Best Legal IQ</div></div>
  <div class="stat"><div class="v" style="color:#f59e0b;">10</div><div class="l">Adversarial Probes</div></div>
  <div class="stat"><div class="v" style="color:#ef4444;">15</div><div class="l">Scenario Files</div></div>
  <div class="stat"><div class="v" style="color:#a855f7;">5</div><div class="l">Cognitive Dimensions</div></div>
  <div class="stat"><div class="v" style="color:#0ea5e9;">0</div><div class="l">LLM Judges Used</div></div>
</div>

<div class="sec">Leaderboard</div>
<div class="table-wrap"><table>
<thead><tr>
  <th style="width:50px">Rank</th><th>Strategy / Model</th><th>Label</th>
  <th style="text-align:center">Legal IQ &#9660;</th>
  <th>T1 Reading</th><th>T2 Classify</th><th>T3 Dep.</th>
  <th>T4 Easy</th><th>T5 Med.</th><th>T6 Hard</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>

<div class="sec">Cognitive Dimension Radar</div>
<div class="chart-wrap"><canvas id="rc" width="500" height="500"></canvas></div>

<div class="sec">Tier Architecture</div>
<div class="tiers">
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#6366f1;">T1</div><div class="tier-name">Clause Reading</div><div class="tier-weight" style="background:#6366f122;color:#6366f1;">15%</div></div><div class="tier-desc">Extract the exact verbatim sentence answering a legal question.<br><b style="color:#94a3b8;">Metric:</b> F2 + Jaccard + Laziness Rate | <b style="color:#94a3b8;">Data:</b> CUAD (510 contracts)</div></div>
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#22c55e;">T2</div><div class="tier-name">Risk Classification</div><div class="tier-weight" style="background:#22c55e22;color:#22c55e;">15%</div></div><div class="tier-desc">Classify clause type, risk level, flags and recommended action.<br><b style="color:#94a3b8;">Metric:</b> Weighted label accuracy | <b style="color:#94a3b8;">Data:</b> OpenEnv tasks 1-3</div></div>
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#f59e0b;">T3</div><div class="tier-name">Dependency Mapping</div><div class="tier-weight" style="background:#f59e0b22;color:#f59e0b;">20%</div></div><div class="tier-desc">Proactively map all hidden cross-contract dependency edges before a crisis.<br><b style="color:#94a3b8;">Metric:</b> Precision / Recall / F1 vs. ground truth | Novel tier</div></div>
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#f97316;">T4</div><div class="tier-name">Crisis Easy</div><div class="tier-weight" style="background:#f9731622;color:#f97316;">12.5%</div></div><div class="tier-desc">Single cascade, 15 days, cooperative counterparties, 0-1 hidden edges.</div></div>
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#ef4444;">T5</div><div class="tier-name">Crisis Medium</div><div class="tier-weight" style="background:#ef444422;color:#ef4444;">17.5%</div></div><div class="tier-desc">Multi-contract crisis, 20 days, mixed counterparties, 2-3 hidden dependencies.</div></div>
  <div class="tier"><div class="tier-header"><div class="tier-badge" style="background:#dc2626;">T6</div><div class="tier-name">Crisis Hard</div><div class="tier-weight" style="background:#dc262622;color:#dc2626;">20%</div></div><div class="tier-desc">Full systemic cascade, 30 days, adversarial counterparties, compound shocks, 4-5 hidden edges.</div></div>
</div>

<div class="sec">Scoring Formula</div>
<div class="formula">
<span class="a">Legal_IQ</span> = <span style="color:#22c55e">0.15</span> &times; T1_Reading<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#22c55e">0.15</span> &times; T2_Classification<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#f59e0b">0.20</span> &times; T3_Dependency<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#ef4444">0.50</span> &times; (0.25&times;T4 + 0.35&times;T5 + 0.40&times;T6)<br><br>
<span class="a">T1</span> = 0.60 &times; F2 + 0.25 &times; Jaccard + 0.15 &times; (1 &minus; laziness)<br>
<span class="a">T3</span> = 0.50 &times; recall + 0.25 &times; precision + 0.15 &times; edge_type_acc + 0.10 &times; severity_order<br>
<span style="color:#475569">All metrics are pure math. Zero LLM judges anywhere.</span>
</div>

<div class="sec">Adversarial Probe Suite</div>
<div class="probes">{probe_html}</div>

</div>
<footer>LexArena v1.0 &bull; OpenEnv + LexDomino + ContractEval/CUAD &bull; Pure deterministic scoring &bull; No LLM judges</footer>
<script>
new Chart(document.getElementById("rc").getContext("2d"),{{
  type:"radar",
  data:{{labels:{labels_json},datasets:{datasets}}},
  options:{{responsive:true,
    plugins:{{legend:{{labels:{{color:"#94a3b8",font:{{size:12}}}}}}}},
    scales:{{r:{{min:0,max:100,grid:{{color:"#1e293b"}},angleLines:{{color:"#1e293b"}},
      pointLabels:{{color:"#94a3b8",font:{{size:12}}}},
      ticks:{{color:"#475569",backdropColor:"transparent",stepSize:25}}}}}}}}
}});
</script>
</body></html>"""

with open("leaderboard/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print(f"leaderboard/index.html created — {len(html)} chars")
