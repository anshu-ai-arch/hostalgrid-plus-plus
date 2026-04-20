"""
graders/grader_medium.py

Medium task grader — Meta OpenEnv compliant.
Returns score in [0.0, 1.0].

Objective:
    Serve rooms fairly under power budget (6000W).
    Reduce complaints. Prioritise HP rooms.

Success criteria (deterministic):
    score = weighted combination of:
        - HP satisfaction rate (40%)
        - complaint control (35%)
        - power efficiency (25%)
"""

import numpy as np
from env.hostelgrid_env import HostelGridEnv
from env.action         import Action

TASK_NAME   = "medium"
N_EPISODES  = 100
EVAL_SEEDS  = [100, 200, 300, 400, 500]
MAX_COMPLAINT = 10


def grade(agent) -> dict:
    hp_scores       = []
    complaint_scores = []
    power_scores    = []

    eps_per_seed = N_EPISODES // len(EVAL_SEEDS)

    for seed in EVAL_SEEDS:
        env = HostelGridEnv(mode=TASK_NAME, seed=seed)
        for _ in range(eps_per_seed):
            states = env.reset().to_vectors()
            ep_hp    = []
            ep_compl = []
            ep_power = []

            for _ in range(50):
                actions           = agent.select_actions(states, greedy=True)
                obs, reward, done, info = env.step(Action.from_list(actions))
                states            = obs.to_vectors()

                # HP satisfaction
                hp_rooms = [r for r in obs.rooms if r.priority > 0.8]
                hp_sat   = sum(
                    r.ac > 0.5 and r.fan > 0.5 and r.light > 0.5
                    for r in hp_rooms
                ) / max(len(hp_rooms), 1)
                ep_hp.append(hp_sat)

                # Complaint control
                avg_complaint = np.mean(
                    [r.complaint_level for r in obs.rooms])
                ep_compl.append(1.0 - avg_complaint)

                # Power efficiency — within budget
                util = obs.power_used / obs.power_budget
                power_score = 1.0 if util <= 1.0 else max(0.0, 2.0 - util)
                ep_power.append(power_score)

                if done: break

            hp_scores.append(np.mean(ep_hp))
            complaint_scores.append(np.mean(ep_compl))
            power_scores.append(np.mean(ep_power))

    hp_score      = float(np.mean(hp_scores))
    compl_score   = float(np.mean(complaint_scores))
    power_score   = float(np.mean(power_scores))
    score         = 0.40*hp_score + 0.35*compl_score + 0.25*power_score

    if   score >= 0.75: letter = "A"
    elif score >= 0.60: letter = "B"
    elif score >= 0.45: letter = "C"
    elif score >= 0.30: letter = "D"
    else:               letter = "F"

    report = {
        "task":          TASK_NAME,
        "score":         round(score, 4),
        "grade":         letter,
        "hp_score":      round(hp_score, 4),
        "complaint":     round(compl_score, 4),
        "power":         round(power_score, 4),
        "beats_random":  score > 0.25,
    }

    _print(report)
    return report


def _print(r: dict):
    print(f"\n{'='*48}")
    print(f"  GRADE REPORT — {r['task'].upper()}")
    print(f"{'='*48}")
    print(f"  Score        : {r['score']:.4f} / 1.0")
    print(f"  Grade        : {r['grade']}")
    print(f"  HP Sat       : {r['hp_score']:.4f}  (40%)")
    print(f"  Complaints   : {r['complaint']:.4f}  (35%)")
    print(f"  Power eff    : {r['power']:.4f}  (25%)")
    print(f"  Beats random?: {'YES ✓' if r['beats_random'] else 'NO ✗'}")
    print(f"{'='*48}")