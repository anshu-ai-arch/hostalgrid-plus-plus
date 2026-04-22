from __future__ import annotations

import os
import numpy as np
import pandas as pd

from env.hostelgrid_env import HostelGridEnv
from env.action import Action
from training.llm_strategy import LLMStrategyPlanner


def heuristic_policy(env):
    actions = [0] * 10
    used = 0.0
    budget = env.hostel.power_budget

    order = sorted(
        range(10),
        key=lambda i: (
            -env.hostel.rooms[i].priority,
            -env.hostel.rooms[i].occupancy,
            -env.hostel.rooms[i].complaint,
        ),
    )

    for i in order:
        room = env.hostel.rooms[i]
        if room.occupancy == 0:
            continue

        if used + 990 <= budget:
            actions[i] = 7
            used += 990
        elif used + 90 <= budget:
            actions[i] = 6
            used += 90

    return actions


def evaluate_policy(name, controller_type="heuristic", mode="medium", seeds=None, episodes_per_seed=1, steps=10, dqn_agent=None, llm_planner=None):
    if seeds is None:
        seeds = [100, 200, 300]

    rewards = []
    complaints = []
    hp_sats = []
    valid_rates = []

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)

        for _ in range(episodes_per_seed):
            obs = env.reset()
            total_reward = 0.0
            valid_count = 0
            step_count = 0
            last_info = None

            for _ in range(steps):
                if controller_type == "heuristic":
                    actions = heuristic_policy(env)
                    valid = True

                elif controller_type == "dqn":
                    states = obs.to_vectors()
                    actions = dqn_agent.select_actions(states, greedy=True)
                    valid = True

                elif controller_type == "llm":
                    result = llm_planner.policy(env)
                    actions = result["actions"]
                    valid = result["valid"]

                else:
                    raise ValueError(f"Unknown controller_type: {controller_type}")

                obs, reward, done, info = env.step(Action.from_list(actions))
                total_reward += reward.total
                last_info = info
                valid_count += int(valid)
                step_count += 1

                if done:
                    break

            rewards.append(total_reward)
            complaints.append(last_info["complaints"])
            hp_sats.append(last_info["hp_satisfied"])
            valid_rates.append(valid_count / max(1, step_count))

    return {
        "policy": name,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "complaints_mean": float(np.mean(complaints)),
        "hp_sat_mean": float(np.mean(hp_sats)),
        "valid_rate": float(np.mean(valid_rates)),
    }


def run_comparison(mode="medium", dqn_agent=None, output_dir="eval_results"):
    llm_planner = LLMStrategyPlanner()

    results = [
        evaluate_policy("Heuristic", "heuristic", mode=mode),
        evaluate_policy("Centralized DQN", "dqn", mode=mode, dqn_agent=dqn_agent),
        evaluate_policy("LLM Planner + Executor", "llm", mode=mode, llm_planner=llm_planner),
    ]

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"controller_comparison_{mode}.csv")
    df.to_csv(out_path, index=False)

    print(df)
    print(f"\nSaved: {out_path}")
    return df
