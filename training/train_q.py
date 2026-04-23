"""
training/train_q.py

Improved Q-learning with LLM-guided teacher gating:
- teacher suggests actions
- Q-agent suggests actions
- per-room gate decides which action to execute
- safe action wrapper applied after gating
- logs teacher %, agent %, agreement %
"""

import argparse
import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv
from agent.q_agent import QAgent
from training.policy_boost import teacher_actions, safe_actions, teacher_prob

N_EPISODES = 500
PRINT_EVERY = 100

Q_CFG = {
    "easy": dict(lr=0.10, gamma=0.96, eps_end=0.03, eps_decay=0.995),
    "medium": dict(lr=0.08, gamma=0.98, eps_end=0.03, eps_decay=0.994),
    "hard": dict(lr=0.06, gamma=0.985, eps_end=0.02, eps_decay=0.9935),
}


def _reward_value(reward):
    return float(reward.total) if hasattr(reward, "total") else float(reward)


def gate_actions(agent, states, teacher_acts, agent_acts, p_teacher):
    final_actions = []
    teacher_used = 0
    agent_used = 0
    agree = 0

    for room_idx, state in enumerate(states):
        t = int(teacher_acts[room_idx])
        a = int(agent_acts[room_idx])

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


def train(mode: str = "easy", n_episodes: int = N_EPISODES, seed: int = 42):
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
        "mode": mode,
    }

    print(f"  Q-Learning | mode={mode} | episodes={n_episodes}")
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

            teacher_acts = teacher_actions(env)
            agent_acts = agent.select_actions(states)

            actions, gate_stats = gate_actions(agent, states, teacher_acts, agent_acts, p_t)
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
    args = parser.parse_args()

    agent, history = train(mode=args.mode, n_episodes=args.episodes, seed=args.seed)
    print(agent.summary())

    early = np.mean(history["rewards"][:100]) if len(history["rewards"]) >= 100 else np.mean(history["rewards"])
    late = np.mean(history["rewards"][-100:]) if len(history["rewards"]) >= 100 else np.mean(history["rewards"])
    print(f"  Reward: early={early:.3f} -> late={late:.3f} | improving={late > early}")
    print(f"  Mean teacher %: {np.mean(history['teacher_pct']):.2f}")
    print(f"  Mean agent %:   {np.mean(history['agent_pct']):.2f}")
    print(f"  Mean agree %:   {np.mean(history['agree_pct']):.2f}")
