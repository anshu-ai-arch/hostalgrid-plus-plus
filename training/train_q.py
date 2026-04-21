"""
training/train_q.py

Improved Q-learning:
- rolling averages in logs
- teacher forcing early in training
- safe action wrapper
- trains on per-room reward + shared global delta
"""

import argparse
import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv
from agent.q_agent import QAgent
from training.policy_boost import teacher_actions, safe_actions, teacher_prob

N_EPISODES  = 500
PRINT_EVERY = 100

Q_CFG = {
    "easy":   dict(lr=0.10, gamma=0.96,  eps_end=0.03, eps_decay=0.995),
    "medium": dict(lr=0.08, gamma=0.98,  eps_end=0.03, eps_decay=0.994),
    "hard":   dict(lr=0.06, gamma=0.985, eps_end=0.02, eps_decay=0.9935),
}


def _reward_value(reward):
    return float(reward.total) if hasattr(reward, "total") else float(reward)


def train(mode: str = "easy",
          n_episodes: int = N_EPISODES,
          seed: int = 42) -> tuple:
    random.seed(seed)
    np.random.seed(seed)

    env   = HostelGridEnv(mode=mode, seed=seed)
    agent = QAgent(**Q_CFG[mode])

    history = {
        "rewards":    [],
        "satisfied":  [],
        "complaints": [],
        "hp_sat":     [],
        "mode":       mode,
    }

    print(f"  Q-Learning | mode={mode} | episodes={n_episodes}")

    for ep in range(n_episodes):
        states    = env.reset().to_vectors()
        total_rew = 0.0

        while True:
            p_teacher = teacher_prob(ep, n_episodes, start=0.85, end=0.05, frac=0.60)

            if random.random() < p_teacher:
                actions = teacher_actions(env)
            else:
                actions = agent.select_actions(states)

            actions = safe_actions(env, actions)

            obs, reward, done, info = env.step(actions)
            next_states = obs.to_vectors()

            total_reward = _reward_value(reward)
            room_rewards = np.asarray(info["per_room"], dtype=np.float32)

            # Teach global budget effect to every room update.
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

        if (ep + 1) % PRINT_EVERY == 0:
            avg_r    = np.mean(history["rewards"][-PRINT_EVERY:])
            avg_sat  = np.mean(history["satisfied"][-PRINT_EVERY:])
            avg_comp = np.mean(history["complaints"][-PRINT_EVERY:])
            avg_hp   = np.mean(history["hp_sat"][-PRINT_EVERY:])

            print(
                f"    Ep {ep+1:4d} | avg={avg_r:7.3f}"
                f" | sat={avg_sat:5.2f}/10"
                f" | complaints={avg_comp:6.2f}"
                f" | hp={avg_hp:4.2f}"
                f" | ε={agent.epsilon:.3f}"
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
    late  = np.mean(history["rewards"][-100:]) if len(history["rewards"]) >= 100 else np.mean(history["rewards"])
    print(f"  Reward: early={early:.3f} -> late={late:.3f} | improving={late > early}")
