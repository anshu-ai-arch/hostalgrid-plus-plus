from __future__ import annotations

import argparse
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


def rank_strategies_for_state(mode="hard", seed=100):
    ranked = []
    for strategy in STRATEGIES:
        env = HostelGridEnv(mode=mode, seed=seed)
        env.reset()
        ranked.append(rollout_strategy_once(env, strategy))
    return sorted(ranked, key=lambda x: x["reward"], reverse=True)


def build_preference_pairs(mode="hard", seed_start=0, num_seeds=1000, min_gap=0.05):
    pairs = []

    for seed in range(seed_start, seed_start + num_seeds):
        env = HostelGridEnv(mode=mode, seed=seed)
        env.reset()
        prompt = make_strategy_prompt(env)

        ranked = rank_strategies_for_state(mode=mode, seed=seed)
        chosen = ranked[0]["strategy"]
        rejected = ranked[-1]["strategy"]
        gap = ranked[0]["reward"] - ranked[-1]["reward"]

        if gap < min_gap:
            continue

        pairs.append(
            {
                "prompt": prompt,
                "chosen": json.dumps({"strategy": chosen}),
                "rejected": json.dumps({"strategy": rejected}),
                "mode": mode,
                "seed": seed,
                "best_reward": ranked[0]["reward"],
                "worst_reward": ranked[-1]["reward"],
                "reward_gap": gap,
            }
        )

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1000)
    parser.add_argument("--min_gap", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "llm_strategy_preferences_hard_large.jsonl"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pairs = build_preference_pairs(
        mode="hard",
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        min_gap=args.min_gap,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    print("Saved:", args.out)
    print("Pairs:", len(pairs))
    if pairs:
        print("First:", pairs[0])
        print("Last:", pairs[-1])


if __name__ == "__main__":
    main()
