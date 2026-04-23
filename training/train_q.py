"""
training/train_q.py

Improved Q-learning with configurable teacher gating:
- teacher mode can be none, rule, or llm
- tracks whether llm mode really used llm or rule fallback
- saves structured experiment summaries to results/experiments
- tracks safety metrics:
  - over-budget rate
  - safe-actions modification rate
  - empty-room waste rate
  - safety-fix reason counts
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
    get_last_safe_actions_changed,
    get_last_safe_actions_reason_counts,
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


def count_empty_room_waste_before_step(env, actions) -> int:
    waste = 0
    for room, action in zip(env.hostel.rooms, actions):
        if room.occupancy == 0 and int(action) != 0:
            waste += 1
    return waste


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
        "over_budget_count": int(sum(history["over_budget"])),
        "over_budget_rate": float(np.mean(history["over_budget"])) if history["over_budget"] else 0.0,
        "safe_action_changed_count": int(sum(history["safe_action_changed"])),
        "safe_action_changed_rate": float(np.mean(history["safe_action_changed"])) if history["safe_action_changed"] else 0.0,
        "empty_room_waste_episode_count": int(sum(history["empty_room_waste_episode"])),
        "empty_room_waste_episode_rate": float(np.mean(history["empty_room_waste_episode"])) if history["empty_room_waste_episode"] else 0.0,
        "empty_room_waste_step_count": int(sum(history["empty_room_waste_steps"])),
        "empty_room_waste_step_rate": float(sum(history["empty_room_waste_steps"]) / max(1, sum(history["episode_steps"]))) if history["episode_steps"] else 0.0,
        "safe_action_reason_totals": {
            "empty_room_off": int(history["safe_reason_empty_room_off"]),
            "urgent_room_protection": int(history["safe_reason_urgent_room_protection"]),
            "budget_downgrade": int(history["safe_reason_budget_downgrade"]),
        },
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
        "over_budget": [],
        "safe_action_changed": [],
        "empty_room_waste_episode": [],
        "empty_room_waste_steps": [],
        "episode_steps": [],
        "safe_reason_empty_room_off": 0,
        "safe_reason_urgent_room_protection": 0,
        "safe_reason_budget_downgrade": 0,
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
        episode_over_budget = 0
        episode_safe_changed = 0
        episode_empty_waste = 0
        episode_steps = 0

        teacher_steps = 0
        agent_steps = 0
        agree_steps = 0
        gate_total = 0

        while True:
            p_t = teacher_prob(ep, n_episodes, start=0.85, end=0.05, frac=0.60)
            episode_steps += 1

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

            if get_last_safe_actions_changed():
                episode_safe_changed += 1

            reasons = get_last_safe_actions_reason_counts()
            history["safe_reason_empty_room_off"] += int(reasons.get("empty_room_off", 0))
            history["safe_reason_urgent_room_protection"] += int(reasons.get("urgent_room_protection", 0))
            history["safe_reason_budget_downgrade"] += int(reasons.get("budget_downgrade", 0))

            teacher_steps += gate_stats["teacher_used"]
            agent_steps += gate_stats["agent_used"]
            agree_steps += gate_stats["agree"]
            gate_total += gate_stats["n_rooms"]

            obs, reward, done, info = env.step(actions)
            next_states = obs.to_vectors()

            if info.get("over_budget", False):
                episode_over_budget += 1

            episode_empty_waste += count_empty_room_waste_before_step(env, actions)

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
        history["over_budget"].append(1 if episode_over_budget > 0 else 0)
        history["safe_action_changed"].append(1 if episode_safe_changed > 0 else 0)
        history["empty_room_waste_episode"].append(1 if episode_empty_waste > 0 else 0)
        history["empty_room_waste_steps"].append(episode_empty_waste)
        history["episode_steps"].append(episode_steps)

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
    print(f"  Over-budget count: {summary['over_budget_count']}")
    print(f"  Over-budget rate:  {summary['over_budget_rate']:.2f}")
    print(f"  Safe-action changed count: {summary['safe_action_changed_count']}")
    print(f"  Safe-action changed rate:  {summary['safe_action_changed_rate']:.2f}")
    print(f"  Empty-room waste episode count: {summary['empty_room_waste_episode_count']}")
    print(f"  Empty-room waste episode rate:  {summary['empty_room_waste_episode_rate']:.2f}")
    print(f"  Empty-room waste step count: {summary['empty_room_waste_step_count']}")
    print(f"  Empty-room waste step rate:  {summary['empty_room_waste_step_rate']:.2f}")
    print(f"  Safe-action reason totals: empty_room_off={summary['safe_action_reason_totals']['empty_room_off']}, urgent_room_protection={summary['safe_action_reason_totals']['urgent_room_protection']}, budget_downgrade={summary['safe_action_reason_totals']['budget_downgrade']}")
    print(f"  Teacher source counts: llm={summary['teacher_source_counts']['llm']}, rule={summary['teacher_source_counts']['rule']}, rule_fallback={summary['teacher_source_counts']['rule_fallback']}, none={summary['teacher_source_counts']['none']}, unknown={summary['teacher_source_counts']['unknown']}")
    print(f"  Saved summary: {path}")
