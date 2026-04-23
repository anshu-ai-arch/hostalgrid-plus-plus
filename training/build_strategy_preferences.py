from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.hostelgrid_env import HostelGridEnv
from env.action import Action
from training.llm_strategy import execute_strategy, make_strategy_prompt

STRATEGIES = [
    "priority_safe",
    "complaint_rescue",
    "energy_saver",
    "comfort_all",
    "shutdown_empty",
    "do_nothing",
]


def rollout_strategy_once(env, strategy: str):
    actions = execute_strategy(env, strategy)
    _, reward, done, info = env.step(Action.from_list(actions))
    return {
        "strategy": strategy,
        "reward": float(reward.total),
        "complaints": info["complaints"],
        "hp_satisfied": info["hp_satisfied"],
        "power_used": info["power_used"],
        "done": done,
    }


def rank_strategies_for_state(mode="medium", seed=100):
    ranked = []
    for strategy in STRATEGIES:
        env = HostelGridEnv(mode=mode, seed=seed)
        env.reset()
        ranked.append(rollout_strategy_once(env, strategy))
    return sorted(ranked, key=lambda x: x["reward"], reverse=True)


def build_preference_pairs(modes=("easy", "medium", "hard"), seeds=range(30)):
    pairs = []

    for mode in modes:
        for seed in seeds:
            env = HostelGridEnv(mode=mode, seed=seed)
            env.reset()
            prompt = make_strategy_prompt(env)

            ranked = rank_strategies_for_state(mode=mode, seed=seed)
            chosen = ranked[0]["strategy"]
            rejected = ranked[-1]["strategy"]

            pairs.append({
                "prompt": prompt,
                "chosen": json.dumps({"strategy": chosen}),
                "rejected": json.dumps({"strategy": rejected}),
                "mode": mode,
                "seed": seed,
                "best_reward": ranked[0]["reward"],
                "worst_reward": ranked[-1]["reward"],
            })

    return pairs


def main():
    out_path = os.path.join(REPO_ROOT, "data", "llm_strategy_preferences.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pairs = build_preference_pairs()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    print("Saved:", out_path)
    print("Pairs:", len(pairs))
    if pairs:
        print(pairs[0])


if __name__ == "__main__":
    main()
