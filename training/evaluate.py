"""
training/evaluate.py

Compact evaluation:
- no giant reward-list JSON
- supports centralized DQN and older Q-agent
"""

import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv

EVAL_SEEDS   = [100, 200, 300, 400, 500]
STRESS_SEEDS = [999, 1001, 1337, 2024]

N_EVAL_EPISODES = 100
STEPS           = 50


def _reward_value(r):
    return float(r.total) if hasattr(r, "total") else float(r)


def _summarize(all_rewards, all_satisfied, all_complaints, all_hp_sat):
    return {
        "mean":       float(np.mean(all_rewards)),
        "std":        float(np.std(all_rewards)),
        "satisfied":  float(np.mean(all_satisfied)),
        "complaints": float(np.mean(all_complaints)),
        "hp_sat":     float(np.mean(all_hp_sat)),
    }


def _random_policy(env) -> list:
    return [random.randrange(8) for _ in range(10)]


def _heuristic_policy(env) -> list:
    actions = []
    for room in env.hostel.rooms:
        if room.occupancy == 1:
            actions.append(7)
        else:
            actions.append(0)
    return actions


def _priority_heuristic(env) -> list:
    actions = [0] * 10
    budget  = env.hostel.power_budget
    used    = 0.0

    order = sorted(
        range(10),
        key=lambda i: (
            -env.hostel.rooms[i].priority,
            -env.hostel.rooms[i].occupancy
        )
    )

    for i in order:
        room = env.hostel.rooms[i]
        if room.occupancy == 1:
            cost_all = 900 + 75 + 15
            if used + cost_all <= budget:
                actions[i] = 7
                used += cost_all
            else:
                cost_partial = 75 + 15
                if used + cost_partial <= budget:
                    actions[i] = 6
                    used += cost_partial

    return actions


def _run_policy(policy_fn, mode: str, seeds: list, n: int = N_EVAL_EPISODES) -> dict:
    all_rewards    = []
    all_satisfied  = []
    all_complaints = []
    all_hp_sat     = []

    eps_per_seed = max(1, n // len(seeds))

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)
        for _ in range(eps_per_seed):
            obs = env.reset()
            total = 0.0
            last_info = {"satisfied": 0, "complaints": 0, "hp_satisfied": 0}

            for _ in range(STEPS):
                actions = policy_fn(env)
                obs, r, done, info = env.step(actions)
                total += _reward_value(r)
                last_info = info
                if done:
                    break

            all_rewards.append(total)
            all_satisfied.append(last_info["satisfied"])
            all_complaints.append(last_info["complaints"])
            all_hp_sat.append(last_info.get("hp_satisfied", 0))

    return _summarize(all_rewards, all_satisfied, all_complaints, all_hp_sat)


def _run_agent(agent, mode: str, seeds: list, n: int = N_EVAL_EPISODES) -> dict:
    all_rewards    = []
    all_satisfied  = []
    all_complaints = []
    all_hp_sat     = []

    eps_per_seed = max(1, n // len(seeds))

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)
        for _ in range(eps_per_seed):
            obs = env.reset()
            total = 0.0
            last_info = {"satisfied": 0, "complaints": 0, "hp_satisfied": 0}

            for _ in range(STEPS):
                if getattr(agent, "centralized", False):
                    actions = agent.select_actions(obs, greedy=True)
                else:
                    actions = agent.select_actions(obs.to_vectors(), greedy=True)

                obs, r, done, info = env.step(actions)
                total += _reward_value(r)
                last_info = info
                if done:
                    break

            all_rewards.append(total)
            all_satisfied.append(last_info["satisfied"])
            all_complaints.append(last_info["complaints"])
            all_hp_sat.append(last_info.get("hp_satisfied", 0))

    return _summarize(all_rewards, all_satisfied, all_complaints, all_hp_sat)


def evaluate_all(q_agent=None, dqn_agent=None, mode: str = "medium") -> dict:
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {mode.upper()}  (seeds={EVAL_SEEDS}, UNSEEN during training)")
    print(f"{'='*60}")

    results = {
        "random":      _run_policy(_random_policy, mode, EVAL_SEEDS),
        "heuristic":   _run_policy(_heuristic_policy, mode, EVAL_SEEDS),
        "p_heuristic": _run_policy(_priority_heuristic, mode, EVAL_SEEDS),
        "mode":        mode,
    }

    if q_agent is not None:
        results["q_agent"] = _run_agent(q_agent, mode, EVAL_SEEDS)
        results["q_stress"] = _run_agent(q_agent, mode, STRESS_SEEDS, n=40)

    if dqn_agent is not None:
        results["dqn_agent"] = _run_agent(dqn_agent, mode, EVAL_SEEDS)
        results["dqn_stress"] = _run_agent(dqn_agent, mode, STRESS_SEEDS, n=40)

    print(f"  {'Policy':20s} | {'Mean':>7} | {'Std':>6} | {'Sat':>5} | {'Compl':>6} | {'HP':>5}")
    print("  " + "-" * 62)

    rows = [
        ("Random", results["random"]),
        ("Heuristic", results["heuristic"]),
        ("Priority-Heuristic", results["p_heuristic"]),
    ]

    if "q_agent" in results:
        rows.append(("Q-Learning", results["q_agent"]))
    if "dqn_agent" in results:
        rows.append(("Central DQN", results["dqn_agent"]))

    for label, res in rows:
        print(
            f"  {label:20s} | {res['mean']:7.3f} | {res['std']:6.3f}"
            f" | {res['satisfied']:5.1f} | {res['complaints']:6.1f}"
            f" | {res['hp_sat']:5.1f}"
        )

    print(f"\n  STRESS TEST (adversarial seeds={STRESS_SEEDS})")
    print("  " + "-" * 40)

    if "q_stress" in results:
        print(f"  {'Q-Learning stress':20s} | {results['q_stress']['mean']:7.3f} | complaints={results['q_stress']['complaints']:.1f}")
    if "dqn_stress" in results:
        print(f"  {'Central DQN stress':20s} | {results['dqn_stress']['mean']:7.3f} | complaints={results['dqn_stress']['complaints']:.1f}")

    return results
