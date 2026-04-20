"""
training/train_q.py

Q-Learning training loop.

Usage:
    python -m training.train_q --mode easy
    python -m training.train_q --mode medium
    python -m training.train_q --mode hard
"""

import argparse
import numpy as np
from env.hostelgrid_env import HostelGridEnv
from agent.q_agent      import QAgent

N_EPISODES = 500
PRINT_EVERY = 100


def train(mode: str = "easy",
          n_episodes: int = N_EPISODES,
          seed: int = 42) -> tuple:

    env   = HostelGridEnv(mode=mode, seed=seed)
    agent = QAgent()
    history = {
        "rewards":    [],
        "satisfied":  [],
        "complaints": [],
        "mode":       mode,
    }

    print(f"  Q-Learning | mode={mode} | episodes={n_episodes}")

    for ep in range(n_episodes):
        states     = env.reset().to_vectors()
        total_rew = 0

        while True:
            actions = agent.select_actions(states)
            obs, reward, done, info = env.step(actions)
            next_states = obs.to_vectors()
            agent.update(states, actions, info["per_room"], next_states)
            states      = next_states
            total_rew += reward.total
            if done: break

        agent.decay_epsilon()
        history["rewards"].append(total_rew)
        history["satisfied"].append(info["satisfied"])
        history["complaints"].append(info["complaints"])

        if (ep + 1) % PRINT_EVERY == 0:
            avg = np.mean(history["rewards"][-PRINT_EVERY:])
            print(f"    Ep {ep+1:4d} | avg={avg:7.3f}"
                  f" | satisfied={info['satisfied']:2d}/10"
                  f" | complaints={info['complaints']:2d}"
                  f" | ε={agent.epsilon:.3f}")

    return agent, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, history = train(mode=args.mode,
                           n_episodes=args.episodes,
                           seed=args.seed)
    print(agent.summary())
    early = np.mean(history["rewards"][:100])
    late  = np.mean(history["rewards"][-100:])
    print(f"  Reward: early={early:.3f} → late={late:.3f}"
          f" | improving={late > early}")