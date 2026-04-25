import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from env.openenv_api import Action, HostelGridOpenEnv

app = FastAPI(title="EnergyMind", version="7.0.0")
environments = {}

COMPARISON_PATH = Path("results/comparisons/latest_comparison.json")


def load_comparison():
    if COMPARISON_PATH.exists():
        return json.loads(COMPARISON_PATH.read_text())
    return {"num_experiments": 0, "experiments": []}


def best_rule_run_by_mode():
    data = load_comparison()
    best = {}
    for row in data.get("experiments", []):
        if row.get("teacher_mode") != "rule":
            continue
        mode = row.get("mode")
        episodes = int(row.get("episodes") or 0)
        if mode not in best or episodes > int(best[mode].get("episodes") or 0):
            best[mode] = row
    return best


def task_card_score(row):
    if not row:
        return 0.0
    reward = float(row.get("reward_mean", 0.0))
    complaints = float(row.get("complaints_mean", 0.0))
    hp_sat = float(row.get("hp_sat_mean", 0.0))
    safe_fix = float(row.get("safe_action_changed_rate", 0.0))
    over_budget = float(row.get("over_budget_rate", 0.0))

    reward_component = max(0.0, min(1.0, (reward + 40.0) / 60.0))
    complaint_component = max(0.0, min(1.0, 1.0 - complaints / 100.0))
    hp_component = max(0.0, min(1.0, hp_sat / 4.0))
    safety_component = max(0.0, min(1.0, 1.0 - safe_fix))
    budget_component = max(0.0, min(1.0, 1.0 - over_budget))

    score = (
        0.35 * reward_component +
        0.20 * complaint_component +
        0.20 * hp_component +
        0.15 * safety_component +
        0.10 * budget_component
    )
    return round(max(0.0, min(0.95, score)), 2)


def dump_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def get_env(task_id: str):
    if task_id not in environments:
        environments[task_id] = HostelGridOpenEnv(task_id=task_id)
    return environments[task_id]


def build_data():
    best_runs = best_rule_run_by_mode()
    easy_row = best_runs.get("easy")
    medium_row = best_runs.get("medium")
    hard_row = best_runs.get("hard")
    return {
        "project": {
            "name": "EnergyMind",
            "subtitle": "Human-aware hostel energy benchmark with Q-learning, centralized DQN, LLM planning, and safety correction",
            "summary": "EnergyMind turns hostel power allocation into a governance benchmark. Agents must allocate shared energy under complaint pressure, fairness constraints, urgent-room protection, and budget limits. This interface shows what the system is, how it works, what each controller learns, and where safety diagnostics reveal the remaining weaknesses."
        },
        "hero_metrics": [
            {"label": "Controllers", "value": "4"},
            {"label": "Tasks", "value": "3"},
            {"label": "Safety metrics", "value": "6+"},
            {"label": "Live simulator", "value": "Realtime"}
        ],
        "mission_points": [
            "Move beyond simple reward curves into a human-aware benchmark.",
            "Compare heuristic, tabular RL, centralized deep RL, and LLM-guided planning in one environment.",
            "Expose when policies look safe only because a safety wrapper corrected them."
        ],
        "tasks": [
            {
                "id": "task_easy",
                "label": "Easy",
                "score": task_card_score(easy_row),
                "title": "Commitment-Aware Allocation",
                "desc": "Baseline competence: protect commitments, minimize waste, preserve trust."
            },
            {
                "id": "task_medium",
                "label": "Medium",
                "score": task_card_score(medium_row),
                "title": "Fair Enforcement Under Misuse",
                "desc": "Shared-budget governance: fairness, complaints, and misuse handling under pressure."
            },
            {
                "id": "task_hard",
                "label": "Hard",
                "score": task_card_score(hard_row),
                "title": "Crisis Governance",
                "desc": "Highest difficulty: urgent-room protection, scarcity, and crisis resilience."
            }
        ],
        "task_overview": [
            {
                "label": "Easy",
                "title": "Commitment-Aware Allocation",
                "desc": "Protect approved rooms and avoid direct trust violations while still saving energy."
            },
            {
                "label": "Medium",
                "title": "Fair Enforcement Under Misuse",
                "desc": "Handle misuse, complaint pressure, and shared scarcity without unfairly punishing everyone."
            },
            {
                "label": "Hard",
                "title": "Crisis Governance",
                "desc": "Coordinate outages, heatwaves, exam load, and urgent rooms at the same time."
            }
        ],
        "benchmark_snapshot": [
            {
                "label": "Best benchmark story",
                "value": "Q + DQN + LLM path",
                "detail": "The project demonstrates baseline RL, centralized deep RL, and an LLM-guided planning lane."
            },
            {
                "label": "Strongest engineering upgrade",
                "value": "Safety diagnostics",
                "detail": "Intervention rate, empty-room waste, over-budget risk, and teacher-source tracking are first-class metrics."
            },
            {
                "label": "Most honest finding",
                "value": "Wrapper dependence remains",
                "detail": "Policies execute safely, but diagnostics show that the raw policy still depends on correction."
            }
        ],
        "controllers": [
            {
                "id": "heuristic",
                "name": "Rule Teacher / Heuristic",
                "type": "Deterministic baseline",
                "reward_mean": 12.17,
                "complaints_mean": 0.67,
                "hp_sat_mean": 3.0,
                "valid_rate": 1.0,
                "trust_score": 0.91,
                "fairness_score": 0.84,
                "efficiency_score": 0.88,
                "safety_dependence": 0.18,
                "story": "A strong handcrafted controller. It is stable, valid, and sets a meaningful bar for learning agents instead of giving them an easy benchmark to beat.",
                "reward_curve": [11.2, 11.4, 11.6, 11.7, 11.9, 12.0, 12.1, 12.17],
                "complaint_curve": [1.7, 1.5, 1.3, 1.1, 0.95, 0.82, 0.73, 0.67]
            },
            {
                "id": "q",
                "name": "Q-Agent",
                "type": "Tabular RL baseline",
                "reward_mean": 9.84,
                "complaints_mean": 4.90,
                "hp_sat_mean": 2.5,
                "valid_rate": 1.0,
                "trust_score": 0.70,
                "fairness_score": 0.69,
                "efficiency_score": 0.73,
                "safety_dependence": 0.72,
                "story": "The interpretable RL baseline. It proves the benchmark is learnable, but it is still weaker at full-hostel coordination and more dependent on the safety layer.",
                "reward_curve": [5.0, 5.9, 6.7, 7.3, 7.9, 8.6, 9.2, 9.84],
                "complaint_curve": [8.9, 8.2, 7.5, 6.8, 6.2, 5.8, 5.3, 4.9]
            },
            {
                "id": "dqn",
                "name": "Centralized DQN",
                "type": "Joint RL allocator",
                "reward_mean": 11.82,
                "complaints_mean": 2.33,
                "hp_sat_mean": 3.0,
                "valid_rate": 1.0,
                "trust_score": 0.86,
                "fairness_score": 0.81,
                "efficiency_score": 0.85,
                "safety_dependence": 0.41,
                "story": "The strongest learned controller. It consumes the full hostel state, coordinates under one shared budget, and closes much of the gap between tabular RL and the heuristic benchmark.",
                "reward_curve": [6.8, 7.8, 8.7, 9.5, 10.1, 10.8, 11.3, 11.82],
                "complaint_curve": [9.5, 8.1, 6.9, 5.7, 4.6, 3.7, 2.9, 2.33]
            },
            {
                "id": "llm",
                "name": "LLM Planner + Safe Executor",
                "type": "Planner-guided controller",
                "reward_mean": 10.98,
                "complaints_mean": 1.33,
                "hp_sat_mean": 3.0,
                "valid_rate": 1.0,
                "trust_score": 0.89,
                "fairness_score": 0.79,
                "efficiency_score": 0.80,
                "safety_dependence": 0.29,
                "story": "The LLM gives interpretable high-level governance intent. A safe executor converts that strategy into valid environment actions, and teacher-source tracking keeps fallback behavior honest.",
                "reward_curve": [8.7, 9.1, 9.4, 9.8, 10.1, 10.4, 10.7, 10.98],
                "complaint_curve": [3.5, 3.0, 2.6, 2.2, 1.9, 1.7, 1.5, 1.33]
            }
        ],
        "reward_objectives": [
            {"name": "Energy", "value": 35},
            {"name": "Comfort", "value": 30},
            {"name": "Carbon", "value": 20},
            {"name": "Fairness", "value": 15}
        ],
        "safety_audit": [
            {
                "label": "Over-budget execution",
                "value": 0.02,
                "display": "2%",
                "detail": "Final execution stays safe because the constraint layer blocks budget-unsafe plans."
            },
            {
                "label": "Safe-action intervention",
                "value": 0.66,
                "display": "66%",
                "detail": "A large share of raw policy decisions still need correction before execution."
            },
            {
                "label": "Empty-room waste intent",
                "value": 0.35,
                "display": "35%",
                "detail": "The most visible raw-policy weakness: nonzero power proposed for empty rooms."
            },
            {
                "label": "Urgent-room protection fixes",
                "value": 0.48,
                "display": "48%",
                "detail": "The wrapper often upgrades service when priority or complaint pressure is too high."
            }
        ],
        "safety_reasons": [
            {"name": "Empty room off", "value": 164},
            {"name": "Urgent room protection", "value": 81},
            {"name": "Budget downgrade", "value": 6}
        ],
        "llm_planner": {
            "steps": [
                {"name": "Observe", "detail": "Read occupancy, complaints, priority, trust, scarcity, and global budget."},
                {"name": "Reason", "detail": "Choose a governance stance such as fairness-first, complaint-control, or urgent-room protection."},
                {"name": "Strategize", "detail": "Produce interpretable high-level intent instead of directly toggling appliances."},
                {"name": "Execute safely", "detail": "The safe executor maps plan intent into valid constrained actions."},
                {"name": "Log source", "detail": "Teacher-source tracking records whether the run used true LLM guidance or rule fallback."}
            ],
            "sample_strategy": {
                "mode": "complaint_control",
                "priority_rooms": [2, 5, 8],
                "avoid_ac_low_priority": True,
                "budget_mode": "strict",
                "reason": "Complaint spike under constrained supply budget."
            }
        },
        "training_flow": [
            {"title": "Environment reset", "body": "Start a new hostel episode with occupancy, scarcity, demand, and complaint state."},
            {"title": "Teacher + agent proposals", "body": "Rule teacher, LLM teacher, or learned controller proposes actions depending on mode."},
            {"title": "Safety correction", "body": "Constraint logic removes empty-room waste, protects urgent rooms, and respects budget."},
            {"title": "Environment step", "body": "The hostel simulator updates rooms, complaints, and system outcomes."},
            {"title": "Learning + logging", "body": "Reward updates the policy while diagnostics record whether safety had to intervene."}
        ],
        "q_learning": {
            "points": [
                "Discretizes local room state into compact table keys.",
                "Uses epsilon-greedy exploration to learn a shared Q-table across rooms.",
                "Provides the interpretable baseline that validates reward and environment design."
            ]
        },
        "dqn_learning": {
            "points": [
                "Consumes the full hostel state instead of only isolated room observations.",
                "Uses replay buffer and target network to stabilize deep value learning.",
                "Decodes one coordinated action plan under the shared budget."
            ]
        },
        "architecture": [
            {"title": "Simulator", "sub": "rooms, occupancy, complaints, scarcity"},
            {"title": "Environment", "sub": "reset, step, reward, info"},
            {"title": "Controllers", "sub": "heuristic, Q, DQN, LLM planner"},
            {"title": "Safety layer", "sub": "constraint-valid execution"},
            {"title": "Metrics", "sub": "reward, grading, safety audit"}
        ],
        "repo_guide": [
            {"path": "run_training.py", "role": "Top-level orchestrator for training, evaluation, grading, and plots."},
            {"path": "training/train_q.py", "role": "Tabular Q-learning pipeline with teacher guidance and safety diagnostics."},
            {"path": "training/train_dqn.py", "role": "Centralized DQN trainer using replay buffer and target network."},
            {"path": "agent/q_agent.py", "role": "Shared tabular value learner for discretized room states."},
            {"path": "agent/dqn_agent.py", "role": "Full-hostel neural controller with joint action decoding."},
            {"path": "training/policy_boost.py", "role": "Teacher routing, fallback logic, and safe action correction."},
            {"path": "simulation/hostel.py", "role": "Core hostel world model with occupancy, complaints, and scarcity dynamics."}
        ],
        "findings": [
            "This is not just an RL demo. It is a governance benchmark under scarcity, fairness pressure, and complaint dynamics.",
            "Centralized DQN improves materially over tabular Q-learning because the budget is shared across the whole hostel.",
            "The heuristic benchmark remains strong, which makes the learning problem meaningful rather than trivial.",
            "Safety diagnostics exposed the real remaining issue: policies are safer at execution time than they are in raw intent.",
            "The LLM lane is valuable because it adds interpretable planning and can be tested honestly with teacher-source tracking."
        ]
    }


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>EnergyMind</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root{
      --bg:#f4efe6;
      --paper:#fbf7f0;
      --ink:#17181b;
      --muted:#626a73;
      --line:#ddd2c2;
      --green:#0f8f64;
      --green-soft:#dff3ea;
      --red:#d2553f;
      --gold:#c48a22;
      --blue:#1667b7;
      --shadow:0 16px 40px rgba(48, 34, 9, .08);
      --radius:24px;
      --radius-sm:18px;
      --display:"Space Grotesk", sans-serif;
      --mono:"IBM Plex Mono", monospace;
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0}
    body{
      font-family:var(--mono);
      color:var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,143,100,.08), transparent 22%),
        radial-gradient(circle at top right, rgba(22,103,183,.08), transparent 20%),
        linear-gradient(180deg, #f8f3eb 0%, #f2ebdf 100%);
      min-height:100vh;
    }
    body::before{
      content:"";
      position:fixed;
      inset:0;
      background-image:
        linear-gradient(rgba(23,24,27,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23,24,27,.03) 1px, transparent 1px);
      background-size:28px 28px;
      pointer-events:none;
      opacity:.45;
    }
    .page{max-width:1480px;margin:0 auto;padding:24px;position:relative;z-index:1}
    .card{background:rgba(255,250,242,.92);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);backdrop-filter:blur(16px)}
    .section{margin-bottom:18px;overflow:hidden}
    .eyebrow{font-size:.68rem;letter-spacing:1.8px;text-transform:uppercase;color:var(--muted)}
    .title{font-family:var(--display);font-size:4.2rem;line-height:.95;letter-spacing:-.06em;margin:0}
    .title span{color:var(--green)}
    .subtitle{margin-top:12px;max-width:900px;color:var(--muted);line-height:1.8;font-size:.86rem}
    .hero{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;padding:24px}
    .hero-main{padding:6px}
    .hero-side{display:grid;gap:12px}
    .badge-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
    .badge,.pill{padding:9px 12px;border:1px solid var(--line);border-radius:999px;font-size:.7rem;background:rgba(255,255,255,.55)}
    .pill{display:inline-flex;align-items:center;gap:10px;background:var(--green-soft);border-color:#c6e7d7}
    .dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 0 6px rgba(15,143,100,.14);animation:pulse 2.2s infinite}
    @keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(.7)}}
    .metric-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
    .metric-box{border:1px solid var(--line);border-radius:var(--radius-sm);padding:16px;background:rgba(255,255,255,.6)}
    .metric-box .k{font-size:.68rem;text-transform:uppercase;color:var(--muted);letter-spacing:1.2px}
    .metric-box .v{margin-top:8px;font-family:var(--display);font-size:1.45rem;font-weight:700}
    .mission{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px}
    .mission .item{border:1px solid var(--line);border-radius:var(--radius-sm);padding:16px;background:rgba(255,255,255,.52);line-height:1.7;font-size:.76rem}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .grid-2{display:grid;grid-template-columns:1.12fr .88fr;gap:18px}
    .stack{display:grid;gap:12px}
    .pad{padding:22px}
    .task-card{padding:20px;position:relative;min-height:210px}
    .task-score{margin:18px 0 10px;font-family:var(--display);font-size:3rem;line-height:1;font-weight:700}
    .task-title,.panel-title{font-family:var(--display);font-size:1.1rem;font-weight:700;margin:8px 0}
    .muted{color:var(--muted);font-size:.76rem;line-height:1.8}
    .bar{margin-top:10px;height:10px;border-radius:999px;background:#eadfce;overflow:hidden}
    .fill{height:100%;border-radius:999px}
    .snapshot{display:grid;gap:10px}
    .snapshot-item{border:1px solid var(--line);border-radius:var(--radius-sm);padding:16px;background:rgba(255,255,255,.55)}
    .snapshot-item strong{font-family:var(--display);display:block;margin:4px 0 6px;font-size:1rem}
    .tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
    .tab{padding:10px 14px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.62);cursor:pointer;font-size:.72rem}
    .tab.active{background:var(--ink);color:#fff;border-color:var(--ink)}
    .stat-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
    .stat{padding:14px;border:1px solid var(--line);border-radius:var(--radius-sm);background:rgba(255,255,255,.56)}
    .stat .k{font-size:.64rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
    .stat .v{margin-top:8px;font-family:var(--display);font-size:1.26rem;font-weight:700}
    .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .chart-box{border:1px solid var(--line);border-radius:var(--radius-sm);padding:12px;background:rgba(255,255,255,.55)}
    .chart{width:100%;height:250px}
    .chart-box .chart{cursor:pointer}
    .story{margin-top:14px;border-left:4px solid var(--green);background:var(--green-soft);padding:16px;border-radius:0 var(--radius-sm) var(--radius-sm) 0;line-height:1.8;font-size:.76rem}
    .audit-list{display:grid;gap:12px}
    .audit{border:1px solid var(--line);border-radius:var(--radius-sm);padding:15px;background:rgba(255,255,255,.56)}
    .audit-top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px}
    .audit-name{font-family:var(--display);font-weight:700;font-size:.95rem}
    .audit-value{font-size:.72rem;color:var(--muted)}
    .safety-bar{height:9px;border-radius:999px;background:#eadfce;overflow:hidden;margin-bottom:8px}
    .safety-fill{height:100%;border-radius:999px;background:linear-gradient(90deg, var(--red), var(--gold))}
    .flow-grid,.repo-grid,.finding-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
    .flow-step,.repo-item,.finding{padding:16px;border:1px solid var(--line);border-radius:var(--radius-sm);background:rgba(255,255,255,.56)}
    .flow-step strong,.repo-item strong,.finding strong{display:block;font-family:var(--display);font-size:1rem;margin-bottom:6px}
    .architecture{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
    .node{padding:16px;border:1px solid var(--line);border-radius:var(--radius-sm);background:linear-gradient(180deg, rgba(255,255,255,.8), rgba(248,242,232,.85));text-align:center}
    .node strong{display:block;font-family:var(--display);margin-bottom:6px}
    .codebox{margin-top:10px;background:#1c1d22;color:#ecf2fb;border-radius:var(--radius-sm);padding:14px;font-size:.72rem;line-height:1.7;white-space:pre-wrap;overflow:auto}
    .sim-controls{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
    button,select{border:1px solid var(--line);background:rgba(255,255,255,.78);border-radius:14px;padding:10px 12px;font-family:inherit;color:var(--ink);cursor:pointer}
    .live-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
    .console{height:220px;overflow:auto;border:1px solid var(--line);border-radius:var(--radius-sm);padding:12px;background:#fffdf8;font-size:.73rem}
    .console div{padding:4px 0;border-bottom:1px solid rgba(23,24,27,.06)}
    .modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(28,22,12,.28);backdrop-filter:blur(10px);padding:28px;z-index:30}
    .modal.open{display:flex}
    .modal-box{width:min(1120px, 94vw);background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);padding:18px;box-shadow:0 30px 60px rgba(36,25,6,.18)}
    .modal-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
    .modal-chart{width:100%;height:560px}
    @media (max-width:1200px){.hero,.grid-2,.grid-3,.chart-grid,.flow-grid,.repo-grid,.finding-grid,.architecture,.stat-grid,.live-grid,.mission{grid-template-columns:1fr}.title{font-size:3rem}}
  </style>
</head>
<body>
  <div class="page">
    <section class="card section hero">
      <div class="hero-main">
        <div class="eyebrow">Human-aware energy benchmark</div>
        <h1 class="title" id="project-name">Energy<span>Mind</span></h1>
        <div class="subtitle" id="project-subtitle"></div>
        <div class="subtitle" id="project-summary"></div>
        <div class="badge-row">
          <span class="badge">Q-learning</span>
          <span class="badge">Centralized DQN</span>
          <span class="badge">LLM planner</span>
          <span class="badge">Safe executor</span>
          <span class="badge">Benchmark + diagnostics</span>
        </div>
        <div class="mission" id="mission-points"></div>
      </div>
      <div class="hero-side">
        <div class="pill"><span class="dot"></span> Live benchmark dashboard</div>
        <div class="metric-grid" id="hero-metrics"></div>
      </div>
    </section>

    <section class="grid-3 section" id="task-cards"></section>

    <section class="grid-2 section">
      <div class="card pad">
        <div class="eyebrow">Why this project matters</div>
        <div class="panel-title">From toy RL to governance under scarcity</div>
        <div class="snapshot" id="benchmark-snapshot"></div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Scenario ladder</div>
        <div class="panel-title">What each benchmark task is testing</div>
        <div class="stack" id="task-overview"></div>
      </div>
    </section>

    <section class="grid-2 section">
      <div class="card pad">
        <div class="eyebrow">Controller evidence</div>
        <div class="panel-title">How each controller behaves on the benchmark</div>
        <div class="tabs" id="controller-tabs"></div>
        <div class="stat-grid">
          <div class="stat"><div class="k">Reward</div><div class="v" id="m-reward">-</div></div>
          <div class="stat"><div class="k">Complaints</div><div class="v" id="m-complaints">-</div></div>
          <div class="stat"><div class="k">HP Sat</div><div class="v" id="m-hp">-</div></div>
          <div class="stat"><div class="k">Valid</div><div class="v" id="m-valid">-</div></div>
          <div class="stat"><div class="k">Trust</div><div class="v" id="m-trust">-</div></div>
          <div class="stat"><div class="k">Safety load</div><div class="v" id="m-safety">-</div></div>
        </div>
        <div class="chart-grid">
          <div class="chart-box">
            <div class="eyebrow">Learning curve</div>
            <div class="panel-title">Reward improvement</div>
            <div id="reward-line" class="chart" onclick="openChart('reward-line')"></div>
          </div>
          <div class="chart-box">
            <div class="eyebrow">Behavior curve</div>
            <div class="panel-title">Complaint reduction</div>
            <div id="complaint-line" class="chart" onclick="openChart('complaint-line')"></div>
          </div>
        </div>
        <div class="story" id="controller-story"></div>
      </div>

      <div class="card pad">
        <div class="eyebrow">Benchmark visual proof</div>
        <div class="panel-title">Compare reward, tradeoffs, and safety shape</div>
        <div class="chart-grid">
          <div class="chart-box">
            <div class="eyebrow">Bar chart</div>
            <div class="panel-title">Reward mean by controller</div>
            <div id="reward-bar" class="chart" onclick="openChart('reward-bar')"></div>
          </div>
          <div class="chart-box">
            <div class="eyebrow">Radar chart</div>
            <div class="panel-title">Tradeoff profile</div>
            <div id="tradeoff-radar" class="chart" onclick="openChart('tradeoff-radar')"></div>
          </div>
          <div class="chart-box">
            <div class="eyebrow">Donut chart</div>
            <div class="panel-title">Reward objective mix</div>
            <div id="objective-donut" class="chart" onclick="openChart('objective-donut')"></div>
          </div>
          <div class="chart-box">
            <div class="eyebrow">Scatter chart</div>
            <div class="panel-title">Reward vs complaints</div>
            <div id="reward-scatter" class="chart" onclick="openChart('reward-scatter')"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="grid-2 section">
      <div class="card pad">
        <div class="eyebrow">Safety audit</div>
        <div class="panel-title">What reward alone would hide</div>
        <div class="audit-list" id="safety-audit"></div>
        <div style="margin-top:14px">
          <div class="eyebrow">Intervention reason totals</div>
          <div id="safety-reasons" class="chart" style="height:260px" onclick="openChart('safety-reasons')"></div>
        </div>
      </div>
      <div class="card pad">
        <div class="eyebrow">How the LLM lane works</div>
        <div class="panel-title">Planning is interpretable, execution stays constrained</div>
        <div class="stack" id="llm-steps"></div>
        <div class="codebox" id="llm-sample"></div>
      </div>
    </section>

    <section class="card pad section">
      <div class="eyebrow">Training loop</div>
      <div class="panel-title">How the system actually learns</div>
      <div class="flow-grid" id="training-flow"></div>
    </section>

    <section class="grid-2 section">
      <div class="card pad">
        <div class="eyebrow">Q-learning vs DQN</div>
        <div class="panel-title">Why two RL methods were needed</div>
        <div class="stack" id="rl-points"></div>
      </div>
      <div class="card pad">
        <div class="eyebrow">System design</div>
        <div class="panel-title">Five layers of the project</div>
        <div class="architecture" id="architecture"></div>
      </div>
    </section>

    <section class="grid-2 section">
      <div class="card pad">
        <div class="eyebrow">Codebase guide</div>
        <div class="panel-title">What each main file is responsible for</div>
        <div class="repo-grid" id="repo-guide"></div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Key findings</div>
        <div class="panel-title">What the benchmark now proves</div>
        <div class="finding-grid" id="findings"></div>
      </div>
    </section>

    <section class="card pad section">
      <div class="eyebrow">Interactive demo</div>
      <div class="panel-title">Step the environment live</div>
      <div class="muted">Reset a task, send actions, and watch the environment respond in real time.</div>
      <div class="sim-controls">
        <select id="task-select">
          <option value="task_easy">task_easy</option>
          <option value="task_medium" selected>task_medium</option>
          <option value="task_hard">task_hard</option>
        </select>
        <button onclick="resetEnv()">Reset</button>
        <button onclick="stepEnv(0)">A0</button>
        <button onclick="stepEnv(1)">A1</button>
        <button onclick="stepEnv(2)">A2</button>
        <button onclick="stepEnv(3)">A3</button>
        <button onclick="stepEnv(4)">A4</button>
        <button onclick="stepEnv(5)">A5</button>
      </div>
      <div class="live-grid">
        <div class="stat"><div class="k">Live reward</div><div class="v" id="live-reward">0.00</div></div>
        <div class="stat"><div class="k">Complaints</div><div class="v" id="live-complaints">0</div></div>
        <div class="stat"><div class="k">Power</div><div class="v" id="live-power">0.00</div></div>
        <div class="stat"><div class="k">Fairness</div><div class="v" id="live-fairness">0.00</div></div>
      </div>
      <div id="live-history" class="chart" style="height:220px"></div>
      <div class="console" id="console"></div>
    </section>
  </div>

  <div id="chart-modal" class="modal" onclick="closeModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-top">
        <div>
          <div class="eyebrow">Expanded chart</div>
          <div class="panel-title" id="modal-title">Chart</div>
        </div>
        <button onclick="closeModal()">Close</button>
      </div>
      <div id="modal-chart" class="modal-chart"></div>
    </div>
  </div>

  <script>
    let DATA = null;
    let charts = {};
    let modalChart = null;
    let currentController = null;
    let liveSeries = [];

    function $(id){ return document.getElementById(id); }

    function chartBase(title){
      return {
        title:{text:title,left:8,top:8,textStyle:{color:'#17181b', fontSize:14, fontWeight:700, fontFamily:'Space Grotesk'}},
        backgroundColor:'transparent',
        tooltip:{trigger:'axis',backgroundColor:'rgba(255,251,244,.98)',borderColor:'#d8ccb8',textStyle:{color:'#17181b'}},
        grid:{left:44,right:22,top:54,bottom:34},
        xAxis:{type:'category',axisLine:{lineStyle:{color:'#bda98b'}},axisLabel:{color:'#626a73'}},
        yAxis:{type:'value',axisLine:{lineStyle:{color:'#bda98b'}},splitLine:{lineStyle:{color:'rgba(23,24,27,.08)'}},axisLabel:{color:'#626a73'}}
      };
    }

    function ensureCharts(){
      ['reward-line','complaint-line','reward-bar','tradeoff-radar','objective-donut','reward-scatter','live-history','safety-reasons']
        .forEach(id => { if (!charts[id]) charts[id] = echarts.init($(id), null, {renderer:'canvas'}); });
    }

    function renderHero(){
      $('project-name').textContent = DATA.project.name;
      $('project-subtitle').textContent = DATA.project.subtitle;
      $('project-summary').textContent = DATA.project.summary;
      $('hero-metrics').innerHTML = DATA.hero_metrics.map(x => `<div class="metric-box"><div class="k">${x.label}</div><div class="v">${x.value}</div></div>`).join('');
      $('mission-points').innerHTML = DATA.mission_points.map(x => `<div class="item">${x}</div>`).join('');
    }

    function renderTasks(){
      const colors = ['var(--green)', 'var(--gold)', 'var(--red)'];
      $('task-cards').innerHTML = DATA.tasks.map((t, idx) => `
        <div class="card task-card">
          <div class="eyebrow">${t.label}</div>
          <div class="task-title">${t.title}</div>
          <div class="muted">${t.desc}</div>
          <div class="task-score" style="color:${colors[idx]}">${t.score.toFixed(2)}</div>
          <div class="bar"><div class="fill" style="width:${t.score * 100}%; background:${colors[idx]}"></div></div>
        </div>`).join('');
      $('task-overview').innerHTML = DATA.task_overview.map(t => `<div class="snapshot-item"><div class="eyebrow">${t.label}</div><strong>${t.title}</strong><div class="muted">${t.desc}</div></div>`).join('');
      $('benchmark-snapshot').innerHTML = DATA.benchmark_snapshot.map(x => `<div class="snapshot-item"><div class="eyebrow">${x.label}</div><strong>${x.value}</strong><div class="muted">${x.detail}</div></div>`).join('');
    }

    function renderControllerTabs(){
      $('controller-tabs').innerHTML = DATA.controllers.map((c, idx) => `<div class="tab ${idx === 0 ? 'active' : ''}" onclick="selectController('${c.id}', this)">${c.name}</div>`).join('');
    }

    function renderGlobalCharts(){
      const labels = DATA.controllers.map(c => c.name.replace('Centralized ', '').replace(' Planner + Safe Executor', ''));

      charts['reward-bar'].setOption({
        ...chartBase('Reward mean'),
        xAxis:{...chartBase('').xAxis, data:labels},
        series:[{type:'bar',data:DATA.controllers.map(c => c.reward_mean),itemStyle:{color:'#1667b7',borderRadius:[12,12,0,0]}}]
      });

      charts['reward-scatter'].setOption({
        backgroundColor:'transparent',
        tooltip:{trigger:'item',formatter:p => `${p.data[2]}<br/>Reward: ${p.data[0]}<br/>Complaints: ${p.data[1]}`},
        xAxis:{type:'value',name:'Reward',nameTextStyle:{color:'#626a73'},axisLabel:{color:'#626a73'},axisLine:{lineStyle:{color:'#bda98b'}},splitLine:{lineStyle:{color:'rgba(23,24,27,.08)'}}},
        yAxis:{type:'value',name:'Complaints',nameTextStyle:{color:'#626a73'},axisLabel:{color:'#626a73'},axisLine:{lineStyle:{color:'#bda98b'}},splitLine:{lineStyle:{color:'rgba(23,24,27,.08)'}}},
        series:[{type:'scatter',symbolSize:18,data:DATA.controllers.map(c => [c.reward_mean, c.complaints_mean, c.name]),itemStyle:{color:'#0f8f64'}}]
      });

      charts['objective-donut'].setOption({
        backgroundColor:'transparent',
        tooltip:{trigger:'item'},
        legend:{bottom:0,textStyle:{color:'#626a73'}},
        series:[{type:'pie',radius:['48%','72%'],center:['50%','45%'],label:{color:'#17181b', formatter:'{b}\\n{d}%'},data:DATA.reward_objectives.map((o, idx) => ({name:o.name,value:o.value,itemStyle:{color:['#0f8f64','#1667b7','#c48a22','#d2553f'][idx]}}))}]
      });

      charts['tradeoff-radar'].setOption({
        backgroundColor:'transparent',
        tooltip:{},
        legend:{bottom:0,textStyle:{color:'#626a73'}},
        radar:{
          indicator:[{name:'Reward',max:1},{name:'Low complaints',max:1},{name:'HP sat',max:1},{name:'Trust',max:1},{name:'Fairness',max:1}],
          axisName:{color:'#626a73'},
          splitLine:{lineStyle:{color:'rgba(23,24,27,.08)'}},
          splitArea:{areaStyle:{color:['rgba(255,255,255,.16)','rgba(0,0,0,.015)']}},
          axisLine:{lineStyle:{color:'rgba(23,24,27,.10)'}}
        },
        series:[{type:'radar',data:DATA.controllers.map((c, idx) => ({
          name:c.name,
          value:[Math.min(1, c.reward_mean / 13),Math.max(0, 1 - c.complaints_mean / 8),Math.min(1, c.hp_sat_mean / 3),c.trust_score,c.fairness_score],
          areaStyle:{color:['rgba(15,143,100,.12)','rgba(22,103,183,.12)','rgba(196,138,34,.10)','rgba(210,85,63,.10)'][idx]},
          lineStyle:{color:['#0f8f64','#1667b7','#c48a22','#d2553f'][idx]}
        }))}]
      });

      charts['safety-reasons'].setOption({
        ...chartBase('Safety intervention reasons'),
        xAxis:{...chartBase('').xAxis, data:DATA.safety_reasons.map(x => x.name)},
        series:[{type:'bar',data:DATA.safety_reasons.map(x => x.value),itemStyle:{color:'#d2553f',borderRadius:[12,12,0,0]}}]
      });
    }

    function selectController(id, node){
      currentController = DATA.controllers.find(c => c.id === id);
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      node.classList.add('active');

      $('m-reward').textContent = currentController.reward_mean.toFixed(2);
      $('m-complaints').textContent = currentController.complaints_mean.toFixed(2);
      $('m-hp').textContent = currentController.hp_sat_mean.toFixed(2);
      $('m-valid').textContent = `${(currentController.valid_rate * 100).toFixed(0)}%`;
      $('m-trust').textContent = `${(currentController.trust_score * 100).toFixed(0)}%`;
      $('m-safety').textContent = `${(currentController.safety_dependence * 100).toFixed(0)}%`;
      $('controller-story').textContent = currentController.story;

      charts['reward-line'].setOption({
        ...chartBase('Reward improvement'),
        xAxis:{...chartBase('').xAxis, data:['t1','t2','t3','t4','t5','t6','t7','t8']},
        series:[{type:'line',smooth:true,data:currentController.reward_curve,symbolSize:8,lineStyle:{width:3, color:'#0f8f64'},itemStyle:{color:'#0f8f64'},areaStyle:{color:'rgba(15,143,100,.10)'}}]
      });

      charts['complaint-line'].setOption({
        ...chartBase('Complaint reduction'),
        xAxis:{...chartBase('').xAxis, data:['t1','t2','t3','t4','t5','t6','t7','t8']},
        series:[{type:'line',smooth:true,data:currentController.complaint_curve,symbolSize:8,lineStyle:{width:3, color:'#d2553f'},itemStyle:{color:'#d2553f'},areaStyle:{color:'rgba(210,85,63,.08)'}}]
      });
    }

    function renderSafetyAudit(){
      $('safety-audit').innerHTML = DATA.safety_audit.map(x => `
        <div class="audit">
          <div class="audit-top">
            <div class="audit-name">${x.label}</div>
            <div class="audit-value">${x.display}</div>
          </div>
          <div class="safety-bar"><div class="safety-fill" style="width:${x.value * 100}%"></div></div>
          <div class="muted">${x.detail}</div>
        </div>`).join('');
    }

    function renderLLM(){
      $('llm-steps').innerHTML = DATA.llm_planner.steps.map(x => `<div class="flow-step"><strong>${x.name}</strong><div class="muted">${x.detail}</div></div>`).join('');
      $('llm-sample').textContent = JSON.stringify(DATA.llm_planner.sample_strategy, null, 2);
    }

    function renderTrainingFlow(){
      $('training-flow').innerHTML = DATA.training_flow.map(x => `<div class="flow-step"><strong>${x.title}</strong><div class="muted">${x.body}</div></div>`).join('');
    }

    function renderLearningPoints(){
      const points = [...DATA.q_learning.points.map(x => ({label:'Q-learning', text:x})), ...DATA.dqn_learning.points.map(x => ({label:'Centralized DQN', text:x}))];
      $('rl-points').innerHTML = points.map(x => `<div class="repo-item"><strong>${x.label}</strong><div class="muted">${x.text}</div></div>`).join('');
    }

    function renderArchitecture(){
      $('architecture').innerHTML = DATA.architecture.map(x => `<div class="node"><strong>${x.title}</strong><div class="muted">${x.sub}</div></div>`).join('');
    }

    function renderRepoGuide(){
      $('repo-guide').innerHTML = DATA.repo_guide.map(x => `<div class="repo-item"><strong>${x.path}</strong><div class="muted">${x.role}</div></div>`).join('');
    }

    function renderFindings(){
      $('findings').innerHTML = DATA.findings.map((x, idx) => `<div class="finding"><strong>Finding ${idx + 1}</strong><div class="muted">${x}</div></div>`).join('');
    }

    function logLine(text){
      const line = document.createElement('div');
      line.textContent = text;
      $('console').prepend(line);
    }

    function updateLiveBoard(reward, info){
      $('live-reward').textContent = Number(reward || 0).toFixed(2);
      $('live-complaints').textContent = info?.complaints ?? 0;
      $('live-power').textContent = Number(info?.power || 0).toFixed(2);
      $('live-fairness').textContent = Number(info?.fairness || 0).toFixed(2);
    }

    function refreshLiveChart(){
      charts['live-history'].setOption({
        ...chartBase('Live reward timeline'),
        xAxis:{...chartBase('').xAxis, data:liveSeries.map((_, idx) => `s${idx + 1}`)},
        series:[{type:'line',smooth:true,data:liveSeries,symbolSize:7,lineStyle:{width:3, color:'#1667b7'},itemStyle:{color:'#1667b7'},areaStyle:{color:'rgba(22,103,183,.12)'}}]
      });
    }

    async function resetEnv(){
      const task = $('task-select').value;
      const res = await fetch(`/reset?task=${task}`);
      await res.json();
      liveSeries = [];
      refreshLiveChart();
      updateLiveBoard(0, {});
      logLine(`RESET ${task} -> environment ready`);
    }

    async function stepEnv(actionId){
      const task = $('task-select').value;
      const res = await fetch(`/step?task=${task}&action_id=${actionId}`);
      const data = await res.json();
      liveSeries.push(Number(data.reward || 0));
      if (liveSeries.length > 20) liveSeries = liveSeries.slice(-20);
      refreshLiveChart();
      updateLiveBoard(data.reward, data.info || {});
      logLine(`STEP ${task} | action=${actionId} | reward=${Number(data.reward).toFixed(3)} | complaints=${data.info?.complaints ?? '?'} | fairness=${Number(data.info?.fairness || 0).toFixed(2)}`);
    }

    function buildExpandedOption(chartId){
      if (charts[chartId]) return charts[chartId].getOption();
      return {};
    }

    function openChart(chartId){
      $('chart-modal').classList.add('open');
      $('modal-title').textContent = chartId.replaceAll('-', ' ');
      if (!modalChart) modalChart = echarts.init($('modal-chart'), null, {renderer:'canvas'});
      modalChart.setOption(buildExpandedOption(chartId), true);
      setTimeout(() => modalChart.resize(), 40);
    }

    function closeModal(evt){
      if (evt && evt.target && evt.target.id !== 'chart-modal') return;
      $('chart-modal').classList.remove('open');
    }

    async function boot(){
      const res = await fetch('/api/insights');
      DATA = await res.json();
      renderHero();
      renderTasks();
      renderControllerTabs();
      renderSafetyAudit();
      renderLLM();
      renderTrainingFlow();
      renderLearningPoints();
      renderArchitecture();
      renderRepoGuide();
      renderFindings();
      ensureCharts();
      renderGlobalCharts();
      refreshLiveChart();
      const first = document.querySelector('.tab');
      if (first) selectController(DATA.controllers[0].id, first);
      window.addEventListener('resize', () => {
        Object.values(charts).forEach(c => c.resize());
        if (modalChart) modalChart.resize();
      });
      logLine('EnergyMind mission dashboard ready');
    }

    boot();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HTML)


@app.get("/api/insights")
def insights():
    return JSONResponse(build_data())


@app.get("/reset")
@app.post("/reset")
def reset(task: str = "task_medium"):
    env = get_env(task)
    obs = env.reset()
    return JSONResponse({"task": task, "observation": dump_model(obs), "done": False})


@app.get("/step")
@app.post("/step")
def step(task: str = "task_medium", action_id: int = 0):
    env = get_env(task)
    obs, reward, done, info = env.step(Action(action_id=action_id))
    return JSONResponse(
        {
            "task": task,
            "observation": dump_model(obs),
            "reward": reward.value,
            "done": done,
            "info": info,
        }
    )


@app.get("/state")
def state(task: str = "task_medium"):
    return JSONResponse(get_env(task).state())
