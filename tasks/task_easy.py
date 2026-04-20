"""
tasks/task_easy.py

EASY task.
Objective: turn ON appliances when occupied, OFF when empty.
No complaints, no fairness, no peak hours.
"""

from env.hostelgrid_env import HostelGridEnv

TASK_NAME   = "easy"
DESCRIPTION = "Basic occupancy control. No complaints, no fairness."
N_EPISODES  = 500


def make_env(seed: int = 42) -> HostelGridEnv:
    return HostelGridEnv(mode=TASK_NAME, seed=seed)


def task_info() -> dict:
    return {
        "name":        TASK_NAME,
        "mode":        TASK_NAME,
        "episodes":    N_EPISODES,
        "description": DESCRIPTION,
        "difficulty":  "LOW",
        "goal":        "Serve occupied rooms. Avoid waste in empty rooms.",
    }