"""
training/train_q.py

Improved Q-learning with configurable teacher gating:
- teacher mode can be none, rule, or llm
- tracks whether llm mode really used llm or rule fallback
- saves structured experiment summaries to results/experiments
- Q-agent suggests actions
- per-room gate decides which action to execute
- safe action wrapper applied after gating
- logs teacher %, agent %, agreement %
"""

import argparse
import json
import os
import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv
from agent.q_agent import QAgent
from training.policy_boost import (
    teacher_actions,
    safe_actions,
    teacher_prob,
    get_last_teacher_source,
)

N_EPISODES = 500
PRINT_EVERY = 100
EXPERIMENT_DIR = "results/experiments"

Q_CFG = {
    "easy": dict(lr=0.10, gamma=0.96, eps_end=0.03, eps_decay=0.995),
    "medium": dict(lr=0.08, gamma=0.98, eps_end=0.03, eps_decay=0.994),
    "hard": dict(lr=0.06, gamma=0.985, eps_end=0.02, eps_decay=0.9935),
}


def _reward_value(reward):
    return float(reward.total) if hasattr(reward, "total") else float(reward)


def gate_actions(agent, states, teacher_acts, agent_acts, p_teacher, teacher_mode):
    final_actions = []
    teacher_used = 0
    agent_used = 0
    agree = 0

    for room_idx, state in enumerate(states):
        t = int(teacher_acts[room_idx])
        a = int(agent_acts[room_idx])

        if teacher_mode == "none":
            final_actions.append(a)
            agent_used += 1
            if t == a:
                agree += 1
            continue

        if t == a:
            final_actions.append(a)
            agree += 1
            agent_used += 1
            continue

        if random.random() < p_teacher:
            final_actions.append(t)
            teacher_used += 1
            continue

        key = agent._key(state)
        qvals = agent.Q[key]

        q_teacher = float(qvals[t])
        q_agent = float(qvals[a])

        if q_teacher > q_agent + 0.10:
            final_actions.append(t)
            teacher_used += 1
        else:
            final_actions.append(a)
            agent_used += 1

    stats = {
        "teacher_used": teacher_used,
        "agent_used": agent_used,
        "agree": agree,
        "n_rooms": len(states),
    }
    return final_actions, stats


def save_experiment_summary(history, mode, teacher_mode, seed, n_episodes):
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    early = np.mean(history["rewards"][:100]) if len(history["rewards"]) >= 100 else np.mean(history["rewards"])
    late = np.mean(history["rewards"][-100:]) if len(history["rewards"]) >= 100 else np.mean(history["rewards"])

    summary = {
        "mode": mode,
        "teacher_mode": teacher_mode,
        "seed": seed,
        "episodes": n_episodes,
        "reward_early": float(early),
        "reward_late": float(late),
        "reward_improving": bool(late > early),
        "reward_mean": float(np.mean(history["rewards"])) if history["rewards"] else 0.0,
        "satisfied_mean": float(np.mean(history["satisfied"])) if history["satisfied"] else 0.0,
        "complaints_mean": float(np.mean(history["complaints"])) if history["complaints"] else 0.0,
        "hp_sat_mean": float(np.mean(history["hp_sat"])) if history["hp_sat"] else 0.0,
        "teacher_pct_mean": float(np.mean(history["teacher_pct"])) if history["teacher_pct"] else 0.0,
        "agent_pct_mean": float(np.mean(history["agent_pct"])) if history["agent_pct"] else 0.0,
        "agree_pct_mean": float(np.mean(history["agree_pct"])) if history["agree_pct"] else 0.0,
        "teacher_source_counts": {
            "llm": int(history["teacher_source_llm"]),
            "rule": int(history["teacher_source_rule"]),
            "rule_fallback": int(history["teacher_source_rule_fallback"]),
            "none": int(history["teacher_source_none"]),
            "unknown": int(history["teacher_source_unknown"]),
        },
    }

    filename = f"{mode}_{teacher_mode}_seed{seed}_ep{n_episodes}.json"
    path = os.path.join(EXPERIMENT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return path, summary


def train(mode: str = "easy", n_episodes: int = N_EPISODES, seed: int = 42, teacher_mode: str = "llm"):
    random.seed(seed)
    np.random.seed(seed)

    env = HostelGridEnv(mode=mode, seed=seed)
    agent = QAgent(**Q_CFG[mode])

    history = {
        "rewards": [],
        "satisfied": [],
        "complaints": [],
        "hp_sat": [],
        "teacher_pct": [],
        "agent_pct": [],
        "agree_pct": [],
        "teacher_source_llm": 0,
        "teacher_source_rule": 0,
        "teacher_source_rule_fallback": 0,
        "teacher_source_none": 0,
        "teacher_source_unknown": 0,
        "mode": mode,
        "teacher_mode": teacher_mode,
    }

    print(f"  Q-Learning | mode={mode} | teacher={teacher_mode} | episodes={n_episodes}")
    print(f"  {'Ep':>4} | {'avgR':>8} | {'sat':>7} | {'comp':>8} | {'hp':>5} | {'teach%':>7} | {'agent%':>7} | {'agree%':>7}")

    for ep in range(n_episodes):
        states = env.reset().to_vectors()
        total_rew = 0.0

        teacher_steps = 0
        agent_steps = 0
        agree_steps = 0
        gate_total = 0

        while True:
            p_t = teacher_prob(ep, n_episodes, start=0.85, end=0.05, frac=0.60)

            teacher_acts = teacher_actions(env, teacher_mode=teacher_mode)
            source = get_last_teacher_source()
            source_key = f"teacher_source_{source}"
            if source_key in history:
                history[source_key] += 1
            else:
                history["teacher_source_unknown"] += 1

            agent_acts = agent.select_actions(states)

            actions, gate_stats = gate_actions(agent, states, teacher_acts, agent_acts, p_t, teacher_mode)
            actions = safe_actions(env, actions)

            teacher_steps += gate_stats["teacher_used"]
            agent_steps += gate_stats["agent_used"]
            agree_steps += gate_stats["agree"]
            gate_total += gate_stats["n_rooms"]

            obs, reward, done, info = env.step(actions)
            next_states = obs.to_vectors()

            total_reward = _reward_value(reward)
            room_rewards = np.asarray(info["per_room"], dtype=np.float32)

            global_delta = total_reward - float(room_rewards.mean())
            train_room_rewards = (room_rewards + global_delta).tolist()

            agent.update(states, actions, train_room_rewards, next_states, done)

            states = next_states
            total_rew += total_reward

            if done:
                break

        agent.decay_epsilon()

        history["rewards"].append(total_rew)
        history["satisfied"].append(info["satisfied"])
        history["complaints"].append(info["complaints"])
        history["hp_sat"].append(info.get("hp_satisfied", 0))
        history["teacher_pct"].append(100.0 * teacher_steps / max(1, gate_total))
        history["agent_pct"].append(100.0 * agent_steps / max(1, gate_total))
        history["agree_pct"].append(100.0 * agree_steps / max(1, gate_total))

        if (ep + 1) % PRINT_EVERY == 0:
            avg_r = np.mean(history["rewards"][-PRINT_EVERY:])
            avg_sat = np.mean(history["satisfied"][-PRINT_EVERY:])
            avg_comp = np.mean(history["complaints"][-PRINT_EVERY:])
            avg_hp = np.mean(history["hp_sat"][-PRINT_EVERY:])
            avg_tp = np.mean(history["teacher_pct"][-PRINT_EVERY:])
            avg_ap = np.mean(history["agent_pct"][-PRINT_EVERY:])
            avg_gp = np.mean(history["agree_pct"][-PRINT_EVERY:])

            print(
                f"  {ep+1:4d} | {avg_r:8.3f} | {avg_sat:7.2f} | {avg_comp:8.2f} | "
                f"{avg_hp:5.2f} | {avg_tp:7.2f} | {avg_ap:7.2f} | {avg_gp:7.2f}"
            )

    return agent, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher", default="llm", choices=["none", "rule", "llm"])
    args = parser.parse_args()

    agent, history = train(
        mode=args.mode,
        n_episodes=args.episodes,
        seed=args.seed,
        teacher_mode=args.teacher,
    )
    print(agent.summary())

    path, summary = save_experiment_summary(
        history=history,
        mode=args.mode,
        teacher_mode=args.teacher,
        seed=args.seed,
        n_episodes=args.episodes,
    )

    print(f"  Reward: early={summary['reward_early']:.3f} -> late={summary['reward_late']:.3f} | improving={summary['reward_improving']}")
    print(f"  Mean teacher %: {summary['teacher_pct_mean']:.2f}")
    print(f"  Mean agent %:   {summary['agent_pct_mean']:.2f}")
    print(f"  Mean agree %:   {summary['agree_pct_mean']:.2f}")
    print(f"  Teacher source counts: llm={summary['teacher_source_counts']['llm']}, rule={summary['teacher_source_counts']['rule']}, rule_fallback={summary['teacher_source_counts']['rule_fallback']}, none={summary['teacher_source_counts']['none']}, unknown={summary['teacher_source_counts']['unknown']}")
    print(f"  Saved summary: {path}")
