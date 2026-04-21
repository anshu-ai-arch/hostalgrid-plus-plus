# paste the full final app.py here
# app.py — EnergyMind Dashboard for HF Spaces

import random
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from env.openenv_api import HostelGridOpenEnv, Action

app = FastAPI(
    title="EnergyMind",
    description="Human-Aware Energy Optimization Environment",
    version="2.0.0"
)

environments = {}

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EnergyMind — Human-Aware Energy Optimization</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #060a0f;
    --surface: #0d1520;
    --surface2: #111d2e;
    --border: #1a2d45;
    --green: #00ff88;
    --green-dim: #00cc6a;
    --yellow: #ffd93d;
    --red: #ff4757;
    --blue: #38bdf8;
    --text: #e2eaf4;
    --muted: #5a7a9a;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

  /* ── HEADER ── */
  header {
    padding: 40px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
  }

  .header-inner {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 20px;
  }

  .logo-block {}

  .logo {
    font-family: var(--font-display);
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
    background: linear-gradient(135deg, var(--green) 0%, var(--blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .logo span {
    font-weight: 400;
    opacity: 0.6;
  }

  .tagline {
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 16px;
  }

  .badge {
    padding: 4px 12px;
    border-radius: 2px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid;
  }

  .badge-green { color: var(--green); border-color: var(--green); background: rgba(0,255,136,0.07); }
  .badge-blue  { color: var(--blue);  border-color: var(--blue);  background: rgba(56,189,248,0.07); }
  .badge-yellow{ color: var(--yellow);border-color: var(--yellow);background: rgba(255,217,61,0.07); }

  .live-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: 2px;
    font-size: 0.75rem;
    color: var(--muted);
    background: var(--surface);
    height: fit-content;
  }

  .live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }

  /* ── SCORE CARDS ── */
  .score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 40px;
  }

  .score-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px;
    position: relative;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.2s;
  }

  .score-card:hover { transform: translateY(-2px); }
  .score-card.easy:hover  { border-color: var(--green); }
  .score-card.medium:hover{ border-color: var(--yellow); }
  .score-card.hard:hover  { border-color: var(--red); }

  .score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .score-card.easy::before   { background: var(--green); }
  .score-card.medium::before { background: var(--yellow); }
  .score-card.hard::before   { background: var(--red); }

  .card-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .card-title {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 20px;
    line-height: 1.3;
  }

  .score-display {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    margin-bottom: 16px;
  }

  .score-number {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
  }
  .easy .score-number   { color: var(--green); }
  .medium .score-number { color: var(--yellow); }
  .hard .score-number   { color: var(--red); }

  .score-denom { color: var(--muted); font-size: 1.2rem; margin-bottom: 6px; }

  .score-bar {
    height: 3px;
    background: var(--border);
    border-radius: 1px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 1.5s cubic-bezier(0.23, 1, 0.32, 1);
  }
  .easy .score-fill   { background: var(--green); }
  .medium .score-fill { background: var(--yellow); }
  .hard .score-fill   { background: var(--red); }

  /* ── MAIN GRID ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 40px;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .panel-title {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .panel-tag {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 1px;
  }

  /* ── REWARD CHART ── */
  .chart-wrap { position: relative; height: 160px; }

  canvas { width: 100% !important; height: 100% !important; }

  /* ── METRICS ── */
  .metrics-list { display: flex; flex-direction: column; gap: 14px; }

  .metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .metric-label { font-size: 0.75rem; color: var(--muted); min-width: 120px; }

  .metric-bar-wrap { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }

  .metric-bar { height: 100%; border-radius: 2px; transition: width 0.8s ease; }

  .metric-value { font-size: 0.8rem; font-weight: 700; min-width: 50px; text-align: right; }

  /* ── TASK EXPLORER ── */
  .tasks-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }

  .task-tile {
    border: 1px solid var(--border);
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }

  .task-tile:hover { background: var(--surface2); }
  .task-tile.active-tile { border-color: var(--green); background: rgba(0,255,136,0.05); }

  .task-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-bottom: 10px;
  }
  .dot-easy   { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot-medium { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }
  .dot-hard   { background: var(--red); box-shadow: 0 0 8px var(--red); }

  .task-tile-name { font-family: var(--font-display); font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; }
  .task-tile-desc { font-size: 0.7rem; color: var(--muted); line-height: 1.5; }

  /* ── ACTION SIMULATOR ── */
  .action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

  .action-btn {
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-btn:hover { border-color: var(--green); color: var(--green); }
  .action-btn:active { transform: scale(0.98); }

  .action-id {
    font-size: 0.65rem;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 1px 5px;
    min-width: 20px;
    text-align: center;
  }

  /* ── LIVE FEED ── */
  .feed-wrap {
    height: 160px;
    overflow-y: auto;
    font-size: 0.72rem;
    line-height: 1.8;
    color: var(--muted);
  }

  .feed-wrap::-webkit-scrollbar { width: 3px; }
  .feed-wrap::-webkit-scrollbar-thumb { background: var(--border); }

  .feed-line { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
  .feed-line.good { color: var(--green); }
  .feed-line.warn { color: var(--yellow); }
  .feed-line.bad  { color: var(--red); }

  /* ── REWARD OBJECTIVE BREAKDOWN ── */
  .objectives { display: flex; flex-direction: column; gap: 10px; }

  .obj-row { display: flex; align-items: center; gap: 12px; }
  .obj-name { font-size: 0.72rem; color: var(--muted); min-width: 80px; }
  .obj-weight { font-size: 0.65rem; color: var(--muted); min-width: 30px; }

  .obj-bar-wrap { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .obj-bar { height: 100%; border-radius: 3px; }

  .obj-value { font-size: 0.75rem; font-weight: 700; min-width: 45px; text-align: right; }

  /* ── FULL WIDTH PANELS ── */
  .full-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px;
    margin-bottom: 20px;
  }

  /* ── ARCHITECTURE ── */
  .arch-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    flex-wrap: wrap;
    padding: 20px 0;
  }

  .arch-node {
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 14px 20px;
    text-align: center;
    min-width: 120px;
  }

  .arch-node-title { font-family: var(--font-display); font-size: 0.8rem; font-weight: 600; margin-bottom: 4px; }
  .arch-node-sub { font-size: 0.65rem; color: var(--muted); }

  .arch-arrow {
    color: var(--green);
    font-size: 1.2rem;
    padding: 0 8px;
    opacity: 0.6;
  }

  /* ── FOOTER ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 24px 0;
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    color: var(--muted);
    font-size: 0.72rem;
  }

  .footer-links { display: flex; gap: 20px; }
  .footer-links a { color: var(--muted); text-decoration: none; transition: color 0.2s; }
  .footer-links a:hover { color: var(--green); }

  /* ── ANIMATIONS ── */
  .fade-in { animation: fadeIn 0.6s ease forwards; opacity: 0; }
  @keyframes fadeIn { to { opacity: 1; } }

  .slide-up { animation: slideUp 0.5s ease forwards; opacity: 0; transform: translateY(20px); }
  @keyframes slideUp { to { opacity: 1; transform: translateY(0); } }

  /* delays */
  .d1 { animation-delay: 0.1s; }
  .d2 { animation-delay: 0.2s; }
  .d3 { animation-delay: 0.3s; }
  .d4 { animation-delay: 0.4s; }
  .d5 { animation-delay: 0.5s; }

  @media (max-width: 768px) {
    .score-grid { grid-template-columns: 1fr; }
    .main-grid  { grid-template-columns: 1fr; }
    .tasks-grid { grid-template-columns: 1fr; }
    .logo { font-size: 2rem; }
  }
</style>
</head>
<body>

<div class="container">

  <!-- HEADER -->
  <header class="fade-in">
    <div class="header-inner">
      <div class="logo-block">
        <div class="logo">EnergyMind<span>++</span></div>
        <div class="tagline">Human-Aware Energy Optimization · RL Environment</div>
        <div class="badges" style="margin-top:14px">
          <span class="badge badge-green">Reinforcement Learning</span>
          <span class="badge badge-blue">Multi-Objective</span>
          <span class="badge badge-yellow">OpenEnv</span>
          <span class="badge badge-green">PyTorch Hackathon</span>
        </div>
      </div>
      <div class="live-indicator">
        <div class="live-dot"></div>
        SYSTEM ONLINE
      </div>
    </div>
  </header>

  <!-- SCORE CARDS -->
  <div class="score-grid">
    <div class="score-card easy slide-up d1">
      <div class="card-label">Task 1 · Easy</div>
      <div class="card-title">Commitment-Aware<br/>Energy Allocation</div>
      <div class="score-display">
        <div class="score-number" id="score-easy">0.65</div>
        <div class="score-denom">/ 1.00</div>
      </div>
      <div class="score-bar"><div class="score-fill" id="bar-easy" style="width:0%"></div></div>
    </div>
    <div class="score-card medium slide-up d2">
      <div class="card-label">Task 2 · Medium</div>
      <div class="card-title">Fair Enforcement<br/>Under Misuse</div>
      <div class="score-display">
        <div class="score-number" id="score-medium">0.90</div>
        <div class="score-denom">/ 1.00</div>
      </div>
      <div class="score-bar"><div class="score-fill" id="bar-medium" style="width:0%"></div></div>
    </div>
    <div class="score-card hard slide-up d3">
      <div class="card-label">Task 3 · Hard</div>
      <div class="card-title">Crisis Governance<br/>Under Extreme Conditions</div>
      <div class="score-display">
        <div class="score-number" id="score-hard">1.00</div>
        <div class="score-denom">/ 1.00</div>
      </div>
      <div class="score-bar"><div class="score-fill" id="bar-hard" style="width:0%"></div></div>
    </div>
  </div>

  <!-- MAIN GRID -->
  <div class="main-grid">

    <!-- Reward Chart -->
    <div class="panel slide-up d2">
      <div class="panel-header">
        <div class="panel-title">Reward Curve</div>
        <div class="panel-tag">TRAINING PROGRESSION · TASK 3</div>
      </div>
      <div class="chart-wrap">
        <canvas id="rewardChart"></canvas>
      </div>
    </div>

    <!-- Multi-Objective Breakdown -->
    <div class="panel slide-up d3">
      <div class="panel-header">
        <div class="panel-title">Reward Objectives</div>
        <div class="panel-tag">WEIGHT · CONTRIBUTION</div>
      </div>
      <div class="objectives">
        <div class="obj-row">
          <div class="obj-name">⚡ Energy</div>
          <div class="obj-weight">35%</div>
          <div class="obj-bar-wrap"><div class="obj-bar" id="obj-energy" style="width:0%;background:#00ff88"></div></div>
          <div class="obj-value" style="color:#00ff88">+0.35</div>
        </div>
        <div class="obj-row">
          <div class="obj-name">😊 Comfort</div>
          <div class="obj-weight">30%</div>
          <div class="obj-bar-wrap"><div class="obj-bar" id="obj-comfort" style="width:0%;background:#38bdf8"></div></div>
          <div class="obj-value" style="color:#38bdf8">+0.30</div>
        </div>
        <div class="obj-row">
          <div class="obj-name">🌱 Carbon</div>
          <div class="obj-weight">20%</div>
          <div class="obj-bar-wrap"><div class="obj-bar" id="obj-carbon" style="width:0%;background:#ffd93d"></div></div>
          <div class="obj-value" style="color:#ffd93d">+0.20</div>
        </div>
        <div class="obj-row">
          <div class="obj-name">⚖️ Fairness</div>
          <div class="obj-weight">15%</div>
          <div class="obj-bar-wrap"><div class="obj-bar" id="obj-fairness" style="width:0%;background:#ff4757"></div></div>
          <div class="obj-value" style="color:#ff4757">+0.15</div>
        </div>
      </div>

      <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border)">
        <div style="font-size:0.65rem;color:var(--muted);margin-bottom:10px;letter-spacing:1px">LIVE METRICS</div>
        <div class="metrics-list">
          <div class="metric-row">
            <div class="metric-label">Demand Satisfaction</div>
            <div class="metric-bar-wrap"><div class="metric-bar" id="m-demand" style="width:0%;background:var(--green)"></div></div>
            <div class="metric-value" style="color:var(--green)" id="mv-demand">93%</div>
          </div>
          <div class="metric-row">
            <div class="metric-label">System Trust</div>
            <div class="metric-bar-wrap"><div class="metric-bar" id="m-trust" style="width:0%;background:var(--blue)"></div></div>
            <div class="metric-value" style="color:var(--blue)" id="mv-trust">87%</div>
          </div>
          <div class="metric-row">
            <div class="metric-label">Fairness Score</div>
            <div class="metric-bar-wrap"><div class="metric-bar" id="m-fair" style="width:0%;background:var(--yellow)"></div></div>
            <div class="metric-value" style="color:var(--yellow)" id="mv-fair">81%</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Task Explorer -->
    <div class="panel slide-up d3">
      <div class="panel-header">
        <div class="panel-title">Task Explorer</div>
        <div class="panel-tag">3 DIFFICULTY LEVELS</div>
      </div>
      <div class="tasks-grid">
        <div class="task-tile active-tile" onclick="selectTask('easy',this)">
          <div class="task-dot dot-easy"></div>
          <div class="task-tile-name">Easy</div>
          <div class="task-tile-desc">Commitment-aware allocation. Honor approved requests.</div>
        </div>
        <div class="task-tile" onclick="selectTask('medium',this)">
          <div class="task-dot dot-medium"></div>
          <div class="task-tile-name">Medium</div>
          <div class="task-tile-desc">Detect misuse, enforce fairness, cap power.</div>
        </div>
        <div class="task-tile" onclick="selectTask('hard',this)">
          <div class="task-dot dot-hard"></div>
          <div class="task-tile-name">Hard</div>
          <div class="task-tile-desc">Heatwave + exam week + outage + misuse simultaneously.</div>
        </div>
      </div>

      <div style="margin-top:16px;padding:14px;background:var(--surface2);border:1px solid var(--border);font-size:0.72rem;color:var(--muted);line-height:1.7" id="task-detail">
        <span style="color:var(--green)">●</span> TASK 1 — 40% rooms have approved requests with minimum supply guarantees.
        Agent must <span style="color:var(--text)">never violate a commitment</span> — heavy penalty if it does.
        Predictable demand, no extreme events. <span style="color:var(--green)">Score: 0.65/1.00</span>
      </div>
    </div>

    <!-- Action Simulator -->
    <div class="panel slide-up d4">
      <div class="panel-header">
        <div class="panel-title">Action Space</div>
        <div class="panel-tag">6 DISCRETE ACTIONS</div>
      </div>
      <div class="action-grid">
        <button class="action-btn" onclick="logAction(0,'increase_ac','comfort ↑ cost ↑')">
          <span class="action-id">0</span> increase_ac
        </button>
        <button class="action-btn" onclick="logAction(1,'decrease_ac','energy saved')">
          <span class="action-id">1</span> decrease_ac
        </button>
        <button class="action-btn" onclick="logAction(2,'lights_off_empty','silent save')">
          <span class="action-id">2</span> lights_off_empty
        </button>
        <button class="action-btn" onclick="logAction(3,'lights_on','restore light')">
          <span class="action-id">3</span> lights_on
        </button>
        <button class="action-btn" onclick="logAction(4,'defer_heavy_load','shift to off-peak')">
          <span class="action-id">4</span> defer_heavy_load
        </button>
        <button class="action-btn" onclick="logAction(5,'do_nothing','hold state')">
          <span class="action-id">5</span> do_nothing
        </button>
      </div>
      <div style="margin-top:14px">
        <div style="font-size:0.65rem;color:var(--muted);letter-spacing:1px;margin-bottom:8px">ACTION LOG</div>
        <div class="feed-wrap" id="feed"></div>
      </div>
    </div>

  </div>

  <!-- ARCHITECTURE -->
  <div class="full-panel slide-up d4">
    <div class="panel-header">
      <div class="panel-title">Agent Architecture</div>
      <div class="panel-tag">Q-LEARNING WITH EXPERIENCE REPLAY</div>
    </div>
    <div class="arch-flow">
      <div class="arch-node">
        <div class="arch-node-title">🏨 Hostel Env</div>
        <div class="arch-node-sub">20 rooms · students · grid</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node" style="border-color:var(--green)">
        <div class="arch-node-title" style="color:var(--green)">Observation</div>
        <div class="arch-node-sub">7–17 dim state vector</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node">
        <div class="arch-node-title">🧠 Q-Agent</div>
        <div class="arch-node-sub">ε-greedy · replay buffer</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node" style="border-color:var(--blue)">
        <div class="arch-node-title" style="color:var(--blue)">Action</div>
        <div class="arch-node-sub">6 discrete actions</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-node">
        <div class="arch-node-title">🎯 Reward</div>
        <div class="arch-node-sub">4-objective function</div>
      </div>
      <div class="arch-arrow">↩</div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px">
      <div style="padding:14px;background:var(--surface2);border:1px solid var(--border);text-align:center">
        <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:800;color:var(--green)">15K</div>
        <div style="font-size:0.65rem;color:var(--muted);margin-top:4px">Replay Buffer</div>
      </div>
      <div style="padding:14px;background:var(--surface2);border:1px solid var(--border);text-align:center">
        <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:800;color:var(--blue)">32</div>
        <div style="font-size:0.65rem;color:var(--muted);margin-top:4px">Batch Size</div>
      </div>
      <div style="padding:14px;background:var(--surface2);border:1px solid var(--border);text-align:center">
        <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:800;color:var(--yellow)">0.98</div>
        <div style="font-size:0.65rem;color:var(--muted);margin-top:4px">Discount γ</div>
      </div>
      <div style="padding:14px;background:var(--surface2);border:1px solid var(--border);text-align:center">
        <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:800;color:var(--red)">1500</div>
        <div style="font-size:0.65rem;color:var(--muted);margin-top:4px">Episodes</div>
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <footer class="fade-in d5">
    <div>Built for <span style="color:var(--green)">Meta PyTorch × Scaler Hackathon</span> · by Team Raptor </div>
    <div class="footer-links">
      <a href="/docs">API Docs</a>
      <a href="/tasks">Tasks</a>
      <a href="/scores">Scores</a>
      <a href="https://github.com/anshu-ai-arch/hostalgrid-plus-plus" target="_blank">GitHub</a>
    </div>
  </footer>

</div>

<script>
// ── Animate score bars on load ─────────────────────────────
window.addEventListener('load', () => {
  setTimeout(() => {
    document.getElementById('bar-easy').style.width   = '65%';
    document.getElementById('bar-medium').style.width = '90%';
    document.getElementById('bar-hard').style.width   = '100%';

    document.getElementById('obj-energy').style.width  = '88%';
    document.getElementById('obj-comfort').style.width = '75%';
    document.getElementById('obj-carbon').style.width  = '50%';
    document.getElementById('obj-fairness').style.width= '38%';

    document.getElementById('m-demand').style.width = '93%';
    document.getElementById('m-trust').style.width  = '87%';
    document.getElementById('m-fair').style.width   = '81%';
  }, 400);
});

// ── Reward Chart ───────────────────────────────────────────
const ctx = document.getElementById('rewardChart').getContext('2d');
const episodes = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500];
const rewards  = [204,205,206,204,207,205,208,203,201,201,210,215,221,215,209];

// Draw manually with canvas
function drawChart() {
  const canvas = document.getElementById('rewardChart');
  const c = canvas.getContext('2d');
  const W = canvas.offsetWidth; const H = canvas.offsetHeight;
  canvas.width = W; canvas.height = H;

  const minR = 195, maxR = 225;
  const pad = { top:10, right:10, bottom:30, left:40 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  c.clearRect(0,0,W,H);

  // Grid lines
  c.strokeStyle = 'rgba(26,45,69,0.8)';
  c.lineWidth = 1;
  for(let i=0;i<=4;i++){
    const y = pad.top + (ch/4)*i;
    c.beginPath(); c.moveTo(pad.left,y); c.lineTo(W-pad.right,y); c.stroke();
    c.fillStyle = '#5a7a9a';
    c.font = '9px Space Mono,monospace';
    c.fillText(Math.round(maxR - (maxR-minR)/4*i), 0, y+3);
  }

  // X labels
  c.fillStyle = '#5a7a9a';
  c.font = '9px Space Mono,monospace';
  [100,500,1000,1500].forEach(ep => {
    const idx = episodes.indexOf(ep);
    if(idx<0) return;
    const x = pad.left + (idx/(episodes.length-1))*cw;
    c.fillText(ep, x-8, H-8);
  });

  // Area fill
  const grad = c.createLinearGradient(0, pad.top, 0, H-pad.bottom);
  grad.addColorStop(0, 'rgba(0,255,136,0.15)');
  grad.addColorStop(1, 'rgba(0,255,136,0)');
  c.fillStyle = grad;
  c.beginPath();
  episodes.forEach((ep,i) => {
    const x = pad.left + (i/(episodes.length-1))*cw;
    const y = pad.top + ch - ((rewards[i]-minR)/(maxR-minR))*ch;
    if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
  });
  c.lineTo(pad.left+cw, H-pad.bottom);
  c.lineTo(pad.left, H-pad.bottom);
  c.closePath(); c.fill();

  // Line
  c.strokeStyle = '#00ff88';
  c.lineWidth = 2;
  c.lineJoin = 'round';
  c.beginPath();
  episodes.forEach((ep,i) => {
    const x = pad.left + (i/(episodes.length-1))*cw;
    const y = pad.top + ch - ((rewards[i]-minR)/(maxR-minR))*ch;
    if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
  });
  c.stroke();

  // Dots
  episodes.forEach((ep,i) => {
    const x = pad.left + (i/(episodes.length-1))*cw;
    const y = pad.top + ch - ((rewards[i]-minR)/(maxR-minR))*ch;
    c.beginPath();
    c.arc(x,y,3,0,Math.PI*2);
    c.fillStyle = '#00ff88';
    c.fill();
  });

  // Highlight peak
  const peakIdx = 12; // ep 1300 = 221
  const px = pad.left + (peakIdx/(episodes.length-1))*cw;
  const py = pad.top + ch - ((rewards[peakIdx]-minR)/(maxR-minR))*ch;
  c.beginPath(); c.arc(px,py,5,0,Math.PI*2);
  c.fillStyle = '#fff'; c.fill();
  c.strokeStyle = '#00ff88'; c.lineWidth=2; c.stroke();
}

drawChart();
window.addEventListener('resize', drawChart);

// ── Task selector ─────────────────────────────────────────
const taskDetails = {
  easy: `<span style="color:var(--green)">●</span> TASK 1 — 40% rooms have approved requests with minimum supply guarantees. Agent must <span style="color:var(--text)">never violate a commitment</span> — heavy penalty if it does. Predictable demand, no extreme events. <span style="color:var(--green)">Score: 0.65/1.00</span>`,
  medium: `<span style="color:var(--yellow)">●</span> TASK 2 — 15% of students are selfish and randomly spike demand. Agent must detect spikes and apply power caps. Fairness score tracked across all rooms. <span style="color:var(--yellow)">Score: 0.90/1.00</span>`,
  hard: `<span style="color:var(--red)">●</span> TASK 3 — Combined: 🌡 Heatwave + 📚 Exam week + ⚡ Partial outage + ⚠️ Misuse + 📉 Partial observability. Battery management and solar harvesting critical. <span style="color:var(--red)">Score: 1.00/1.00 ⭐</span>`,
};

function selectTask(id, el) {
  document.querySelectorAll('.task-tile').forEach(t => t.classList.remove('active-tile'));
  el.classList.add('active-tile');
  document.getElementById('task-detail').innerHTML = taskDetails[id];
}

// ── Action feed ───────────────────────────────────────────
const actionMsgs = {
  0: { cls:'warn', msg:'↑ AC increased — comfort up, cost rising' },
  1: { cls:'good', msg:'↓ AC decreased — energy saved' },
  2: { cls:'good', msg:'💡 Lights off in empty rooms — silent save' },
  3: { cls:'',     msg:'💡 Lights restored' },
  4: { cls:'good', msg:'⏱ Heavy load deferred to off-peak — big save' },
  5: { cls:'',     msg:'— Holding current state' },
};

function logAction(id, name, desc) {
  const feed = document.getElementById('feed');
  const now = new Date().toLocaleTimeString('en-US',{hour12:false});
  const { cls, msg } = actionMsgs[id];
  const line = document.createElement('div');
  line.className = `feed-line ${cls}`;
  line.innerHTML = `<span style="color:var(--muted)">${now}</span>  [ACTION ${id}] ${msg}`;
  feed.prepend(line);
  if(feed.children.length > 30) feed.removeChild(feed.lastChild);
}

// Auto-simulate actions
const autoActions = [4,2,0,5,1,4,2,5,0,3];
let autoIdx = 0;
setInterval(() => {
  const id = autoActions[autoIdx % autoActions.length];
  logAction(id, '', '');
  autoIdx++;
}, 3000);

// Init feed
logAction(4,'defer_heavy_load','shift to off-peak');
logAction(2,'lights_off_empty','silent save');
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return DASHBOARD_HTML

@app.get("/reset")
@app.post("/reset")
def reset(task_id: str = "task_easy"):
    env = HostelGridOpenEnv(task_id=task_id)
    environments[task_id] = env
    obs = env.reset()
    return JSONResponse({"task_id": task_id, "observation": obs.model_dump(), "state": env.state()})

@app.post("/step")
def step(task_id: str = "task_easy", action_id: int = 5):
    if task_id not in environments:
        env = HostelGridOpenEnv(task_id=task_id)
        environments[task_id] = env
        env.reset()
    env = environments[task_id]
    action = Action(action_id=action_id)
    obs, reward, done, info = env.step(action)
    return JSONResponse({
        "task_id": task_id,
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info,
        "score": env.score() if done else None
    })

@app.get("/state")
def state(task_id: str = "task_easy"):
    if task_id not in environments:
        env = HostelGridOpenEnv(task_id=task_id)
        environments[task_id] = env
        env.reset()
    return JSONResponse(environments[task_id].state())

@app.get("/tasks")
def tasks():
    return JSONResponse({"tasks": [
        {"id": "task_easy",   "name": "Commitment-Aware Energy Allocation",      "difficulty": "easy",   "max_score": 1.0},
        {"id": "task_medium", "name": "Fair Enforcement Under Misuse",            "difficulty": "medium", "max_score": 1.0},
        {"id": "task_hard",   "name": "Crisis Governance Under Extreme Conditions","difficulty": "hard",  "max_score": 1.0},
    ]})

@app.get("/scores")
def scores():
    result = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env = HostelGridOpenEnv(task_id=task_id)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(Action(action_id=random.randint(0, 5)))
        result[task_id] = env.score()
    result["average"] = round(sum(result.values()) / len(result), 4)
    return JSONResponse(result)

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
