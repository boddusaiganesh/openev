"""
LexArena — Report Generator
Generates a self-contained HTML report from benchmark results.
No external dependencies required — pure Python + inline CSS/JS.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Score colour helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"   # green
    elif score >= 0.50:
        return "#f59e0b"   # amber
    elif score >= 0.30:
        return "#f97316"   # orange
    else:
        return "#ef4444"   # red


def _label_badge_color(label: str) -> str:
    badges = {
        "Expert CRO Level":          "#7c3aed",
        "Senior Lawyer Level":       "#2563eb",
        "Junior Associate Level":    "#0891b2",
        "Paralegal Level":           "#d97706",
        "Fails Legal Practice Bar":  "#dc2626",
    }
    return badges.get(label, "#6b7280")


def _bar(score: float, width: int = 180) -> str:
    pct = int(score * 100)
    color = _score_color(score)
    return (
        f'<div style="background:#1e293b;border-radius:4px;width:{width}px;height:10px;overflow:hidden;">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;'
        f'transition:width 1s ease;"></div></div>'
        f'<span style="color:{color};font-size:12px;margin-left:6px;">{score:.3f}</span>'
    )


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    results: Dict[str, Any],
    output_path: str = "artifacts/lexarena_report.html",
    run_timestamp: str = "",
) -> str:
    """
    Generate a self-contained HTML report from benchmark results dict.
    Returns path of generated file.
    """
    run_timestamp = run_timestamp or time.strftime("%Y-%m-%d %H:%M:%S")

    # Sort models by Legal IQ descending
    ranked = sorted(results.items(), key=lambda x: -x[1]["legal_iq"])

    # Radar chart data (5 cognitive dimensions)
    radar_labels = ["Clause Reading", "Risk Classification", "Dep. Mapping", "Crisis Easy", "Crisis Hard"]
    radar_datasets = []
    palette = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#0ea5e9"]
    for i, (model, data) in enumerate(ranked):
        radar_datasets.append({
            "label": model,
            "data": [
                round(data["t1_reading"] * 100, 1),
                round(data["t2_classification"] * 100, 1),
                round(data["t3_dependency"] * 100, 1),
                round(data["t4_crisis_easy"] * 100, 1),
                round(data["t6_crisis_hard"] * 100, 1),
            ],
            "borderColor": palette[i % len(palette)],
            "backgroundColor": palette[i % len(palette)] + "22",
            "pointBackgroundColor": palette[i % len(palette)],
        })

    # Build leaderboard rows HTML
    rows_html = ""
    for rank, (model, data) in enumerate(ranked, 1):
        iq = data["legal_iq"]
        label = data["label"]
        badge_color = _label_badge_color(label)
        iq_color = _score_color(iq)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        rows_html += f"""
        <tr class="table-row" data-rank="{rank}">
          <td style="text-align:center;font-size:20px;">{medal}</td>
          <td><span style="font-weight:600;color:#f1f5f9;">{model}</span></td>
          <td>
            <span style="background:{badge_color}22;color:{badge_color};padding:3px 10px;
              border-radius:20px;font-size:12px;font-weight:600;">{label}</span>
          </td>
          <td style="text-align:center;">
            <span style="font-size:22px;font-weight:800;color:{iq_color};">{iq:.4f}</span>
          </td>
          <td>{_bar(data["t1_reading"])}</td>
          <td>{_bar(data["t2_classification"])}</td>
          <td>{_bar(data["t3_dependency"])}</td>
          <td>{_bar(data["t4_crisis_easy"])}</td>
          <td>{_bar(data["t5_crisis_medium"])}</td>
          <td>{_bar(data["t6_crisis_hard"])}</td>
        </tr>"""

    # Build tier explanation cards
    tier_cards = [
        ("T1", "Clause Reading", "15%", "#6366f1",
         "Can the agent extract the exact sentence answering a legal question?<br>"
         "<b>Metric:</b> F2 score (recall-weighted) + Jaccard + Laziness Rate.<br>"
         "<b>Data:</b> CUAD dataset — 510 real contracts, 41 question types."),
        ("T2", "Risk Classification", "15%", "#22c55e",
         "Can the agent classify clause type, risk level, and recommend an action?<br>"
         "<b>Metric:</b> Weighted label accuracy across 3 difficulty tasks.<br>"
         "<b>Data:</b> OpenEnv tasks 1-3."),
        ("T3", "Dependency Mapping", "20%", "#f59e0b",
         "Can the agent proactively map all hidden cross-contract dependency edges?<br>"
         "<b>Metric:</b> Precision/recall/F1 against ground-truth edge list.<br>"
         "<b>Data:</b> LexDomino scenario dependency_edges arrays."),
        ("T4", "Crisis Easy", "12.5%", "#f97316",
         "Can the agent manage a single-contract crisis over 15 days?<br>"
         "<b>Metric:</b> Cash survival ratio + deadlines met + edges discovered."),
        ("T5", "Crisis Medium", "17.5%", "#ef4444",
         "Can the agent handle a multi-contract crisis with hidden dependencies?<br>"
         "<b>Metric:</b> Cash survival ratio — adversarial counterparties."),
        ("T6", "Crisis Hard", "20%", "#dc2626",
         "Can the agent survive 30-day full systemic cascade with compound shocks?<br>"
         "<b>Metric:</b> Cash survival — bankruptcy = 0 score."),
    ]

    cards_html = ""
    for tid, name, weight, color, desc in tier_cards:
        cards_html += f"""
        <div class="tier-card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{color};color:#fff;border-radius:8px;
              padding:4px 12px;font-weight:800;font-size:14px;">{tid}</div>
            <span style="font-size:16px;font-weight:700;color:#f1f5f9;">{name}</span>
            <span style="margin-left:auto;background:{color}22;color:{color};
              border-radius:12px;padding:2px 10px;font-size:12px;">{weight}</span>
          </div>
          <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">{desc}</p>
        </div>"""

    radar_json = json.dumps(radar_datasets)
    labels_json = json.dumps(radar_labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LexArena Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', system-ui, sans-serif;
      background: #030712;
      color: #e2e8f0;
      min-height: 100vh;
    }}

    /* Hero */
    .hero {{
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
      padding: 60px 40px 50px;
      text-align: center;
      border-bottom: 1px solid #1e293b;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 50% 0%, #6366f133 0%, transparent 70%);
    }}
    .hero-badge {{
      display: inline-block;
      background: linear-gradient(90deg, #6366f1, #a855f7);
      color: #fff;
      padding: 6px 20px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      font-size: 52px;
      font-weight: 900;
      background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 12px;
    }}
    .hero p {{
      color: #94a3b8;
      font-size: 16px;
      max-width: 600px;
      margin: 0 auto 8px;
    }}
    .hero .timestamp {{ color: #475569; font-size: 13px; margin-top: 16px; }}

    /* Layout */
    .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 24px; }}
    .section-title {{
      font-size: 20px;
      font-weight: 800;
      color: #f1f5f9;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .section-title::after {{
      content: '';
      flex: 1;
      height: 1px;
      background: linear-gradient(90deg, #334155, transparent);
    }}

    /* Summary cards */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .summary-card {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
      transition: border-color 0.2s;
    }}
    .summary-card:hover {{ border-color: #6366f1; }}
    .summary-card .value {{
      font-size: 32px;
      font-weight: 900;
      margin-bottom: 4px;
    }}
    .summary-card .label {{ color: #64748b; font-size: 13px; }}

    /* Table */
    .table-wrap {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 16px;
      overflow-x: auto;
      margin-bottom: 40px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead th {{
      background: #0a0f1e;
      color: #64748b;
      font-weight: 700;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid #1e293b;
    }}
    .table-row {{ border-bottom: 1px solid #1e293b; transition: background 0.15s; }}
    .table-row:hover {{ background: #1e293b44; }}
    .table-row td {{ padding: 16px; vertical-align: middle; }}
    .table-row:last-child {{ border-bottom: none; }}

    /* Tier cards grid */
    .tiers-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .tier-card {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      transition: border-color 0.2s, transform 0.2s;
    }}
    .tier-card:hover {{ border-color: #6366f155; transform: translateY(-2px); }}

    /* Chart */
    .chart-wrap {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 16px;
      padding: 30px;
      margin-bottom: 40px;
      max-width: 600px;
      margin-left: auto;
      margin-right: auto;
    }}

    /* Scoring formula */
    .formula-box {{
      background: #0a0f1e;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 40px;
      font-family: 'Courier New', monospace;
      font-size: 14px;
      line-height: 2;
      color: #94a3b8;
    }}
    .formula-box .accent {{ color: #818cf8; font-weight: 700; }}

    /* Footer */
    footer {{
      text-align: center;
      padding: 40px;
      color: #334155;
      font-size: 12px;
      border-top: 1px solid #1e293b;
      margin-top: 40px;
    }}
  </style>
</head>
<body>

<div class="hero">
  <div class="hero-badge">Benchmark Report</div>
  <h1>LexArena</h1>
  <p>The Complete Legal Intelligence Benchmark</p>
  <p style="color:#6366f1;font-size:14px;">READ &rarr; CLASSIFY &rarr; CONNECT &rarr; DECIDE &rarr; SURVIVE</p>
  <div class="timestamp">Generated: {run_timestamp}</div>
</div>

<div class="container">

  <!-- Summary Cards -->
  <div class="section-title">Overview</div>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="value" style="color:#6366f1;">{len(ranked)}</div>
      <div class="label">Strategies Evaluated</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color:#22c55e;">{ranked[0][1]['legal_iq']:.4f}</div>
      <div class="label">Best Legal IQ</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color:#f59e0b;">6</div>
      <div class="label">Evaluation Tiers</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color:#ef4444;">10</div>
      <div class="label">Adversarial Probes</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color:#a855f7;">15</div>
      <div class="label">Scenario Files</div>
    </div>
    <div class="summary-card">
      <div class="value" style="color:#0ea5e9;">0</div>
      <div class="label">LLM Judges Used</div>
    </div>
  </div>

  <!-- Leaderboard Table -->
  <div class="section-title">Leaderboard</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:50px;">Rank</th>
          <th>Strategy / Model</th>
          <th>Label</th>
          <th style="text-align:center;">Legal IQ</th>
          <th>T1 Reading</th>
          <th>T2 Classify</th>
          <th>T3 Dep.</th>
          <th>T4 Easy</th>
          <th>T5 Med.</th>
          <th>T6 Hard</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- Radar Chart -->
  <div class="section-title">Cognitive Dimension Profile</div>
  <div class="chart-wrap">
    <canvas id="radarChart" width="500" height="500"></canvas>
  </div>

  <!-- Tier Cards -->
  <div class="section-title">Tier Architecture</div>
  <div class="tiers-grid">
    {cards_html}
  </div>

  <!-- Scoring Formula -->
  <div class="section-title">Legal IQ Scoring Formula</div>
  <div class="formula-box">
    <span class="accent">Legal_IQ</span> = <span style="color:#22c55e;">0.15</span> × T1_Reading_Score<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#22c55e;">0.15</span> × T2_Classification_Score<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#f59e0b;">0.20</span> × T3_Dependency_Score<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#ef4444;">0.50</span> × T4_T5_T6_Crisis_Score<br><br>
    where <span class="accent">T4_T5_T6_Crisis_Score</span> = (0.25×T4 + 0.35×T5 + 0.40×T6)<br><br>
    <span class="accent">T1_score</span> = 0.60×F2 + 0.25×Jaccard_mean + 0.15×(1 − laziness_rate)<br>
    <span class="accent">T3_score</span> = 0.50×recall + 0.25×precision + 0.15×edge_type_accuracy + 0.10×severity_order
  </div>

</div>

<footer>
  LexArena v1.0 &bull; Pure deterministic scoring &bull; No LLM judges &bull;
  Built on OpenEnv + LexDomino + ContractEval/CUAD
</footer>

<script>
const ctx = document.getElementById('radarChart').getContext('2d');
new Chart(ctx, {{
  type: 'radar',
  data: {{
    labels: {labels_json},
    datasets: {radar_json}
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{
        labels: {{ color: '#94a3b8', font: {{ size: 12 }} }}
      }}
    }},
    scales: {{
      r: {{
        min: 0, max: 100,
        grid: {{ color: '#1e293b' }},
        angleLines: {{ color: '#1e293b' }},
        pointLabels: {{ color: '#94a3b8', font: {{ size: 12 }} }},
        ticks: {{ color: '#475569', backdropColor: 'transparent', stepSize: 25 }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[LexArena Report] Saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Generate LexArena HTML report")
    parser.add_argument("--results", default=None, help="Path to benchmark JSON")
    parser.add_argument("--output", default="artifacts/lexarena_report.html")
    args = parser.parse_args()

    if args.results:
        results_path = args.results
    else:
        # Auto-find latest benchmark JSON
        files = sorted(glob.glob("artifacts/lexarena_full_benchmark_*.json"))
        if not files:
            print("No benchmark results found. Run lexarena_benchmark.py first.")
            exit(1)
        results_path = files[-1]
        print(f"Using latest results: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    path = generate_report(results, output_path=args.output)
    print(f"Report ready: {path}")
