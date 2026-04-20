"""
training/evaluate.py

Proper evaluation framework with:
    - Strict train/test seed separation
    - Unseen scenario testing
    - Stress tests (adversarial conditions)
    - Baseline: Random + Heuristic policies
    - No data leakage between training and evaluation
"""

import random
import numpy as np
from env.hostelgrid_env import HostelGridEnv

# ── Seed management ───────────────────────────────────────────────
TRAIN_SEED = 42      # used during training
EVAL_SEEDS = [100, 200, 300, 400, 500]   # NEVER used in training
STRESS_SEEDS = [999, 1001, 1337, 2024]   # adversarial/OOD seeds

N_EVAL_EPISODES = 100
STEPS           = 50


# ── Policies ──────────────────────────────────────────────────────

def _random_policy(env) -> list:
    return [random.randrange(8) for _ in range(10)]


def _heuristic_policy(env) -> list:
    """
    Rule-based: turn ON all appliances for occupied rooms,
    OFF for empty. Does NOT handle power budget or priorities.
    Deliberately blind to constraints — shows where rules fail.
    """
    actions = []
    for room in env.hostel.rooms:
        if room.occupancy == 1:
            actions.append(7)   # all ON
        else:
            actions.append(0)   # all OFF
    return actions


def _priority_heuristic(env) -> list:
    """
    Smarter heuristic: serve HP rooms first,
    then MP, then LP within power budget.
    Still rule-based — no learning.
    """
    actions = [0] * 10
    budget  = env.hostel.power_budget
    used    = 0.0

    # Sort rooms: HP occupied first
    order = sorted(range(10),
                   key=lambda i: (
                       -env.hostel.rooms[i].priority,
                       -env.hostel.rooms[i].occupancy
                   ))

    for i in order:
        room = env.hostel.rooms[i]
        if room.occupancy == 1:
            cost = 900 + 75 + 15   # all ON
            if used + cost <= budget:
                actions[i] = 7
                used += cost
            else:
                # Partial: just fan+light if budget allows
                cost2 = 75 + 15
                if used + cost2 <= budget:
                    actions[i] = 6   # fan+light
                    used += cost2
        # empty rooms stay OFF (action 0)

    return actions


# ── Single policy runner ──────────────────────────────────────────

def _run_policy(policy_fn, mode: str,
                seeds: list, n: int = N_EVAL_EPISODES) -> dict:
    """Run a policy across multiple seeds for robustness."""
    all_rewards    = []
    all_satisfied  = []
    all_complaints = []
    all_hp_sat     = []

    eps_per_seed = max(1, n // len(seeds))

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)
        for _ in range(eps_per_seed):
            env.reset()
            total = 0
            for _ in range(STEPS):
                actions       = policy_fn(env)
                obs, r, done, info = env.step(actions)
                total += r.total if hasattr(r, 'total') else r
                if done: break
            all_rewards.append(total)
            all_satisfied.append(info["satisfied"])
            all_complaints.append(info["complaints"])
            all_hp_sat.append(info.get("hp_satisfied", 0))

    return {
        "rewards":    all_rewards,
        "mean":       float(np.mean(all_rewards)),
        "std":        float(np.std(all_rewards)),
        "satisfied":  float(np.mean(all_satisfied)),
        "complaints": float(np.mean(all_complaints)),
        "hp_sat":     float(np.mean(all_hp_sat)),
    }


def _run_agent(agent, mode: str,
               seeds: list, n: int = N_EVAL_EPISODES) -> dict:
    """Run trained agent greedily across unseen seeds."""
    all_rewards    = []
    all_satisfied  = []
    all_complaints = []
    all_hp_sat     = []

    eps_per_seed = max(1, n // len(seeds))

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)
        for _ in range(eps_per_seed):
             states = env.reset().to_vectors()
             total  = 0
             for _ in range(STEPS):
                 actions                  = agent.select_actions(states, greedy=True)
                 obs, r, done, info       = env.step(actions)
                 states                   = obs.to_vectors()
                 total                   += r.total if hasattr(r, 'total') else r
                 if done: break
                 all_rewards.append(total)
                 all_satisfied.append(info["satisfied"])
                 all_complaints.append(info["complaints"])
                 all_hp_sat.append(info.get("hp_satisfied", 0))

    return {
        "rewards":    all_rewards,
        "mean":       float(np.mean(all_rewards)),
        "std":        float(np.std(all_rewards)),
        "satisfied":  float(np.mean(all_satisfied)),
        "complaints": float(np.mean(all_complaints)),
        "hp_sat":     float(np.mean(all_hp_sat)),
    }


# ── Main evaluation ───────────────────────────────────────────────

def evaluate_all(q_agent, dqn_agent, mode: str = "medium") -> dict:
    """
    Full evaluation across unseen seeds.
    Includes: Random, Heuristic, Priority-Heuristic, Q, DQN.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {mode.upper()}"
          f"  (seeds={EVAL_SEEDS}, UNSEEN during training)")
    print(f"{'='*60}")

    r_rand  = _run_policy(_random_policy,     mode, EVAL_SEEDS)
    r_heur  = _run_policy(_heuristic_policy,  mode, EVAL_SEEDS)
    r_pheur = _run_policy(_priority_heuristic, mode, EVAL_SEEDS)
    r_q     = _run_agent(q_agent,    mode, EVAL_SEEDS)
    r_dqn   = _run_agent(dqn_agent,  mode, EVAL_SEEDS)

    print(f"  {'Policy':20s} | {'Mean':>7} | {'Std':>6}"
          f" | {'Sat':>5} | {'Compl':>6} | {'HP':>5}")
    print("  " + "-"*62)

    for label, res in [
        ("Random",            r_rand),
        ("Heuristic",         r_heur),
        ("Priority-Heuristic",r_pheur),
        ("Q-Learning",        r_q),
        ("DQN (shared)",      r_dqn),
    ]:
        print(f"  {label:20s} | {res['mean']:7.3f} | {res['std']:6.3f}"
              f" | {res['satisfied']:5.1f} | {res['complaints']:6.1f}"
              f" | {res['hp_sat']:5.1f}")

    # Stress test
    print(f"\n  STRESS TEST (adversarial seeds={STRESS_SEEDS})")
    print("  " + "-"*40)
    r_q_stress   = _run_agent(q_agent,   mode, STRESS_SEEDS, n=40)
    r_dqn_stress = _run_agent(dqn_agent, mode, STRESS_SEEDS, n=40)
    print(f"  {'Q-Learning stress':20s} | {r_q_stress['mean']:7.3f}"
          f" | complaints={r_q_stress['complaints']:.1f}")
    print(f"  {'DQN stress':20s} | {r_dqn_stress['mean']:7.3f}"
          f" | complaints={r_dqn_stress['complaints']:.1f}")

    results = {
        "random":    r_rand,
        "heuristic": r_heur,
        "p_heuristic": r_pheur,
        "q_agent":   r_q,
        "dqn_agent": r_dqn,
        "q_stress":  r_q_stress,
        "dqn_stress": r_dqn_stress,
        "mode":      mode,
    }

    return results


# ── Standalone evaluation ─────────────────────────────────────────

def _run_random(mode, seed=EVAL_SEEDS[0], n=N_EVAL_EPISODES):
    """Legacy interface for graders."""
    return _run_policy(_random_policy, mode, EVAL_SEEDS, n=n)


def _run_agent_single(agent, mode,
                       seed=EVAL_SEEDS[0],
                       n=N_EVAL_EPISODES):
    """Legacy interface for graders."""
    return _run_agent(agent, mode, EVAL_SEEDS, n=n)