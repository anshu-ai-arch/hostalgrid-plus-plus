"""
graders/grader_easy.py

Easy task grader — Meta OpenEnv compliant.
Returns score in [0.0, 1.0].

Objective:
    Turn ON appliances in occupied rooms, OFF in empty rooms.
    No power constraint. No complaints.

Success criteria (deterministic):
    score = weighted combination of:
        - satisfaction rate (70%)
        - no waste in empty rooms (30%)
"""

import numpy as np
from env.hostelgrid_env import HostelGridEnv
from env.action         import Action

TASK_NAME   = "easy"
N_EPISODES  = 100
EVAL_SEEDS  = [100, 200, 300, 400, 500]


def grade(agent) -> dict:
    """
    Grade agent on easy task.

    Returns:
        dict with score (0.0-1.0), grade letter, and details
    """
    satisfaction_scores = []
    efficiency_scores   = []

    eps_per_seed = N_EPISODES // len(EVAL_SEEDS)

    for seed in EVAL_SEEDS:
        env = HostelGridEnv(mode=TASK_NAME, seed=seed)
        for _ in range(eps_per_seed):
            states = env.reset().to_vectors()
            ep_sat  = []
            ep_eff  = []

            for _ in range(50):
                actions           = agent.select_actions(states, greedy=True)
                obs, reward, done, info = env.step(Action.from_list(actions))
                states            = obs.to_vectors()

                # Satisfaction: occupied rooms fully served
                occ   = sum(r.occupancy > 0.5 for r in obs.rooms)
                sat   = info["satisfied"] / max(occ, 1)
                ep_sat.append(sat)

                # Efficiency: empty rooms should be off
                empty_waste = sum(
                    (r.ac + r.fan + r.light)
                    for r in obs.rooms if r.occupancy < 0.5
                )
                max_waste  = 3 * sum(1 for r in obs.rooms if r.occupancy < 0.5)
                eff        = 1.0 - (empty_waste / max(max_waste, 1))
                ep_eff.append(eff)

                if done: break

            satisfaction_scores.append(np.mean(ep_sat))
            efficiency_scores.append(np.mean(ep_eff))

    sat_score  = float(np.clip(np.mean(satisfaction_scores), 0.0, 1.0))
    eff_score  = float(np.clip(np.mean(efficiency_scores),   0.0, 1.0))
    score      = float(np.clip(0.70 * sat_score + 0.30 * eff_score, 0.0, 1.0))

    # Grade letter
    if   score >= 0.85: letter = "A"
    elif score >= 0.70: letter = "B"
    elif score >= 0.55: letter = "C"
    elif score >= 0.40: letter = "D"
    else:               letter = "F"

    report = {
        "task":          TASK_NAME,
        "score":         round(score, 4),       # 0.0 – 1.0
        "grade":         letter,
        "satisfaction":  round(sat_score, 4),
        "efficiency":    round(eff_score, 4),
        "beats_random":  score > 0.35,          # random baseline ~0.35
    }

    _print(report)
    return report


def _print(r: dict):
    print(f"\n{'='*48}")
    print(f"  GRADE REPORT — {r['task'].upper()}")
    print(f"{'='*48}")
    print(f"  Score        : {r['score']:.4f} / 1.0")
    print(f"  Grade        : {r['grade']}")
    print(f"  Satisfaction : {r['satisfaction']:.4f}")
    print(f"  Efficiency   : {r['efficiency']:.4f}")
    print(f"  Beats random?: {'YES ✓' if r['beats_random'] else 'NO ✗'}")
    print(f"{'='*48}")