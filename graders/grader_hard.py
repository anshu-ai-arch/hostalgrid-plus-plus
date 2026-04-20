"""
graders/grader_hard.py

Hard task grader — Meta OpenEnv compliant.
Returns score in [0.0, 1.0].

Objective:
    Survive adversarial conditions under tight budget (3500W).
    Prioritise HP rooms under scarcity.
    Manage heatwave and peak hour conflicts.

Perfect score (1.0) is IMPOSSIBLE by design —
budget is too tight to serve everyone.
A score of 0.55+ is considered excellent.

Success criteria (deterministic):
    score = weighted combination of:
        - HP rooms served under scarcity (50%)
        - complaint rate under adversarial load (30%)
        - robustness across adversarial seeds (20%)
"""

import numpy as np
from env.hostelgrid_env import HostelGridEnv
from env.action         import Action

TASK_NAME    = "hard"
N_EPISODES   = 100
EVAL_SEEDS   = [100, 200, 300, 400, 500]
STRESS_SEEDS = [999, 1001, 1337, 2024]


def grade(agent) -> dict:
    hp_scores      = []
    compl_scores   = []
    stress_scores  = []

    # Standard eval
    eps_per_seed = N_EPISODES // len(EVAL_SEEDS)
    for seed in EVAL_SEEDS:
        env = HostelGridEnv(mode=TASK_NAME, seed=seed)
        for _ in range(eps_per_seed):
            states = env.reset().to_vectors()
            ep_hp    = []
            ep_compl = []

            for _ in range(50):
                actions           = agent.select_actions(states, greedy=True)
                obs, reward, done, info = env.step(Action.from_list(actions))
                states            = obs.to_vectors()

                # HP satisfaction under scarcity
                hp_rooms = [r for r in obs.rooms if r.priority > 0.8]
                hp_sat   = sum(
                    r.ac > 0.5 and r.fan > 0.5 and r.light > 0.5
                    for r in hp_rooms
                ) / max(len(hp_rooms), 1)
                ep_hp.append(hp_sat)

                # Complaint control
                avg_c = np.mean([r.complaint_level for r in obs.rooms])
                ep_compl.append(1.0 - avg_c)

                if done: break

            hp_scores.append(np.mean(ep_hp))
            compl_scores.append(np.mean(ep_compl))

    # Stress test on adversarial seeds
    for seed in STRESS_SEEDS:
        env = HostelGridEnv(mode=TASK_NAME, seed=seed)
        for _ in range(10):
            states = env.reset().to_vectors()
            ep_hp  = []
            for _ in range(50):
                actions           = agent.select_actions(states, greedy=True)
                obs, reward, done, info = env.step(Action.from_list(actions))
                states            = obs.to_vectors()
                hp_rooms = [r for r in obs.rooms if r.priority > 0.8]
                hp_sat   = sum(
                    r.ac > 0.5 and r.fan > 0.5 and r.light > 0.5
                    for r in hp_rooms
                ) / max(len(hp_rooms), 1)
                ep_hp.append(hp_sat)
                if done: break
            stress_scores.append(np.mean(ep_hp))

    hp_score     = float(np.mean(hp_scores))
    compl_score  = float(np.mean(compl_scores))
    stress_score = float(np.mean(stress_scores))
    score        = 0.50*hp_score + 0.30*compl_score + 0.20*stress_score

    # Hard mode thresholds — ceiling ~0.55
    if   score >= 0.55: letter = "A"
    elif score >= 0.42: letter = "B"
    elif score >= 0.30: letter = "C"
    elif score >= 0.18: letter = "D"
    else:               letter = "F"

    report = {
        "task":          TASK_NAME,
        "score":         round(score, 4),
        "grade":         letter,
        "hp_score":      round(hp_score, 4),
        "complaint":     round(compl_score, 4),
        "stress":        round(stress_score, 4),
        "beats_random":  score > 0.15,
        "note":          "Hard mode ceiling ~0.55 by design (budget too tight for perfect service)"
    }

    _print(report)
    return report


def _print(r: dict):
    print(f"\n{'='*48}")
    print(f"  GRADE REPORT — {r['task'].upper()}")
    print(f"{'='*48}")
    print(f"  Score        : {r['score']:.4f} / 1.0")
    print(f"  Grade        : {r['grade']}")
    print(f"  HP Sat       : {r['hp_score']:.4f}  (50%)")
    print(f"  Complaints   : {r['complaint']:.4f}  (30%)")
    print(f"  Stress       : {r['stress']:.4f}  (20%)")
    print(f"  Beats random?: {'YES ✓' if r['beats_random'] else 'NO ✗'}")
    print(f"  Note: {r['note']}")
    print(f"{'='*48}")