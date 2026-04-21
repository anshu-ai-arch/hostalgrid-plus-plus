"""
training/train_dqn.py

Centralized DQN training:
- full-hostel input
- budget-aware joint decoding
- teacher heuristic early in training
- Double DQN targets
"""

import argparse
import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv
from agent.dqn_agent import DQNAgent
from env.action import ACTION_MAP

AC_WATTS    = 900.0
FAN_WATTS   = 75.0
LIGHT_WATTS = 15.0

N_EPISODES       = 1200
PRINT_EVERY      = 100
UPDATES_PER_STEP = 1

DQN_CFG = {
    "easy": dict(
        lr=5e-4, gamma=0.97, eps_end=0.03, eps_decay=0.997,
        batch_size=64, buffer_size=15000, warmup=1000, target_update=250
    ),
    "medium": dict(
        lr=4e-4, gamma=0.985, eps_end=0.02, eps_decay=0.996,
        batch_size=64, buffer_size=20000, warmup=1500, target_update=250
    ),
    "hard": dict(
        lr=3e-4, gamma=0.99, eps_end=0.02, eps_decay=0.996,
        batch_size=64, buffer_size=25000, warmup=2000, target_update=250
    ),
}


def action_power(action_id: int) -> float:
    ac, fan, light = ACTION_MAP[int(action_id)]
    return ac * AC_WATTS + fan * FAN_WATTS + light * LIGHT_WATTS


def teacher_prob(ep: int, total_eps: int, start: float = 0.70, end: float = 0.05, frac: float = 0.50) -> float:
    cutoff = max(1, int(total_eps * frac))
    if ep >= cutoff:
        return end
    alpha = ep / cutoff
    return start * (1.0 - alpha) + end * alpha


def teacher_actions(env) -> list:
    rooms = env.hostel.rooms
    actions = [0] * len(rooms)
    budget = float(env.hostel.power_budget)
    used = 0.0

    order = sorted(
        range(len(rooms)),
        key=lambda i: (
            -rooms[i].priority,
            -rooms[i].occupancy,
            -getattr(rooms[i], "complaint", 0),
            -getattr(rooms[i], "consecutive_ignored", 0),
        )
    )

    for i in order:
        room = rooms[i]
        if room.occupancy == 0:
            continue

        if room.priority >= 3 or getattr(room, "complaint", 0) >= 2:
            cands = [7, 6, 4, 5, 2, 3, 0]
        elif room.priority == 2 or getattr(room, "complaint", 0) >= 1:
            cands = [6, 7, 2, 3, 4, 5, 0]
        else:
            cands = [2, 3, 6, 7, 0]

        for cand in cands:
            if used + action_power(cand) <= budget + 1e-6:
                actions[i] = cand
                used += action_power(cand)
                break

    return actions


def _reward_value(reward):
    return float(reward.total) if hasattr(reward, "total") else float(reward)


def train(mode: str = "medium",
          n_episodes: int = N_EPISODES,
          seed: int = 42) -> tuple:
    random.seed(seed)
    np.random.seed(seed)

    env   = HostelGridEnv(mode=mode, seed=seed)
    agent = DQNAgent(**DQN_CFG[mode])

    history = {
        "rewards":    [],
        "satisfied":  [],
        "complaints": [],
        "hp_sat":     [],
        "losses":     [],
        "mode":       mode,
    }

    print(f"  Centralized DQN | mode={mode} | episodes={n_episodes}")

    for ep in range(n_episodes):
        obs = env.reset()
        total_rew = 0.0
        ep_losses = []

        while True:
            p_teacher = teacher_prob(
                ep, n_episodes,
                start=0.65 if mode == "easy" else 0.75,
                end=0.05,
                frac=0.50 if mode != "hard" else 0.60
            )

            if random.random() < p_teacher:
                actions = teacher_actions(env)
            else:
                actions = agent.select_actions(obs)

            next_obs, reward, done, info = env.step(actions)

            total_reward = _reward_value(reward)
            room_rewards = np.asarray(info["per_room"], dtype=np.float32)

            global_delta = total_reward - float(room_rewards.mean())
            train_room_rewards = (room_rewards + global_delta).tolist()

            agent.store(obs, actions, train_room_rewards, next_obs, done)

            for _ in range(UPDATES_PER_STEP):
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            obs = next_obs
            total_rew += total_reward

            if done:
                break

        agent.decay_epsilon()

        history["rewards"].append(total_rew)
        history["satisfied"].append(info["satisfied"])
        history["complaints"].append(info["complaints"])
        history["hp_sat"].append(info.get("hp_satisfied", 0))
        history["losses"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (ep + 1) % PRINT_EVERY == 0:
            avg_r    = np.mean(history["rewards"][-PRINT_EVERY:])
            avg_sat  = np.mean(history["satisfied"][-PRINT_EVERY:])
            avg_comp = np.mean(history["complaints"][-PRINT_EVERY:])
            avg_hp   = np.mean(history["hp_sat"][-PRINT_EVERY:])
            avg_loss = np.mean(history["losses"][-PRINT_EVERY:])

            print(
                f"    Ep {ep+1:4d} | avg={avg_r:7.3f}"
                f" | sat={avg_sat:5.2f}/10"
                f" | complaints={avg_comp:6.2f}"
                f" | hp={avg_hp:4.2f}"
                f" | loss={avg_loss:.4f}"
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
