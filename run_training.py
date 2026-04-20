"""
run_training.py
===============
Complete training script for HostelGrid++.

Usage:
    python run_training.py                   # both agents, all modes, 500 ep
    python run_training.py --agent q         # Q-Learning only
    python run_training.py --agent dqn       # DQN only
    python run_training.py --mode medium     # one mode only
    python run_training.py --episodes 300    # quick test
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs         import get_config
from training        import train_q, train_dqn, evaluate_all
from utils           import TrainingMetrics, plot_dashboard, plot_curriculum
from graders         import grade
from tasks           import print_all_tasks


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def banner(text: str):
    print()
    print("=" * 58)
    print(f"  {text}")
    print("=" * 58)


def get_raw_rewards(agent, mode: str, n: int = 100) -> list:
    """Greedy evaluation — returns list of episode rewards."""
    import random
    from env.hostelgrid_env import HostelGridEnv
    env = HostelGridEnv(mode=mode, seed=0)
    rewards = []
    for _ in range(n):
        states = env.reset().to_vectors()
        total  = 0
        for _ in range(50):
            actions        = agent.select_actions(states, greedy=True)
            obs, r, done, _ = env.step(actions)
            states             = obs.to_vectors()
            total        += r.total if hasattr(r, 'total') else r
            if done:
                break
        rewards.append(total)
    return rewards


def get_random_rewards(mode: str, n: int = 100) -> list:
    """Random policy baseline."""
    import random
    from env.hostelgrid_env import HostelGridEnv
    env = HostelGridEnv(mode=mode, seed=0)
    rewards = []
    for _ in range(n):
        env.reset()
        total = 0
        for _ in range(50):
            actions       = [random.randrange(8) for _ in range(10)]
            _, r, done, _ = env.step(actions)
            total        += r.total if hasattr(r, 'total') else r
            if done:
                break
        rewards.append(total)
    return rewards


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def run(modes: list, agents: list, n_episodes: int, seed: int):
    cfg = get_config()
    os.makedirs(cfg.logging.log_dir,  exist_ok=True)
    os.makedirs(cfg.logging.plot_dir, exist_ok=True)

    banner("HostelGrid++ — Training")
    print(f"  Modes    : {modes}")
    print(f"  Agents   : {agents}")
    print(f"  Episodes : {n_episodes}")
    print()
    print_all_tasks()

    all_q_agents   = {}
    all_dqn_agents = {}
    all_q_hist     = {}
    all_dqn_hist   = {}

    # ── TRAIN ─────────────────────────────────────────────────
    for mode in modes:
        banner(f"TRAINING — {mode.upper()}")

        if "q" in agents:
            print("\n  [Q-Learning]")
            q_agent, q_hist = train_q(
                mode=mode, n_episodes=n_episodes, seed=seed)
            all_q_agents[mode] = q_agent
            all_q_hist[mode]   = q_hist
            print(TrainingMetrics(q_hist).summary())

        if "dqn" in agents:
            print("\n  [DQN — Shared Policy]")
            dqn_agent, dqn_hist = train_dqn(
                mode=mode, n_episodes=n_episodes, seed=seed)
            all_dqn_agents[mode] = dqn_agent
            all_dqn_hist[mode]   = dqn_hist
            print(TrainingMetrics(dqn_hist).summary())

    # ── EVALUATE ──────────────────────────────────────────────
    banner("EVALUATION")

    for mode in modes:
        q_agent   = all_q_agents.get(mode)
        dqn_agent = all_dqn_agents.get(mode)

        if q_agent and dqn_agent:
            eval_res = evaluate_all(q_agent, dqn_agent, mode=mode)
            eval_res["random"]["rewards"]    = get_random_rewards(mode)
            eval_res["q_agent"]["rewards"]   = get_raw_rewards(q_agent,   mode)
            eval_res["dqn_agent"]["rewards"] = get_raw_rewards(dqn_agent, mode)

            plot_dashboard(
                all_q_hist[mode], all_dqn_hist[mode],
                eval_res, mode=mode,
                save_dir=cfg.logging.plot_dir
            )

    # curriculum plot — only when all 3 modes trained with both agents
    if (len(modes) == 3
            and all_q_hist
            and all_dqn_hist):
        plot_curriculum(
            {m: {"q": all_q_hist[m], "dqn": all_dqn_hist[m]}
             for m in modes if m in all_q_hist and m in all_dqn_hist},
            save_dir=cfg.logging.plot_dir
        )

    # ── GRADE ─────────────────────────────────────────────────
    banner("GRADING")

    for agent_name, agents_dict in [("Q-Learning", all_q_agents),
                                     ("DQN",        all_dqn_agents)]:
        if not agents_dict:
            continue

        print(f"\n{'='*50}")
        print(f"  {agent_name} — graded on its own trained mode")
        print(f"{'='*50}")

        scores = []
        for mode in modes:
            if mode not in agents_dict:
                continue
            print(f"\n  [{agent_name}] mode = {mode.upper()}")
            report = grade(agents_dict[mode], task_name=mode)
            scores.append(report["score"])

        if scores:
            overall = np.mean(scores)
            if   overall >= 0.75: og = "A"
            elif overall >= 0.60: og = "B"
            elif overall >= 0.45: og = "C"
            elif overall >= 0.30: og = "D"
            else:               og = "F"
            print(f"\n  {agent_name} Overall : "
                  f"{overall:.4f} / 1.0 →  Grade {og}")

    # ── DONE ──────────────────────────────────────────────────
    banner("DONE")
    print(f"  Plots saved → {cfg.logging.plot_dir}")
    print(f"  Logs  saved → {cfg.logging.log_dir}")
    print()
    print("  Commands:")
    print("    python run_training.py                  # full run")
    print("    python run_training.py --agent dqn      # DQN only")
    print("    python run_training.py --mode hard      # hard only")
    print("    python run_training.py --episodes 300   # quick test")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HostelGrid++ — Full Training Pipeline")
    parser.add_argument(
        "--agent", default="both",
        choices=["q", "dqn", "both"],
        help="Agent to train (default: both)")
    parser.add_argument(
        "--mode", default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Task mode (default: all)")
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Episodes per mode (default: 500)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    args = parser.parse_args()

    modes  = (["easy", "medium", "hard"]
               if args.mode == "all" else [args.mode])
    agents = (["q", "dqn"]
               if args.agent == "both" else [args.agent])

    run(modes=modes, agents=agents,
        n_episodes=args.episodes, seed=args.seed)