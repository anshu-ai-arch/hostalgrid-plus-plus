"""
tasks/task_hard.py

HARD task.
Adds complaint system, peak load hours, and heatwave events.
Agent must prioritize HP rooms and manage energy cost.
"""

from env.hostelgrid_env import HostelGridEnv

TASK_NAME   = "hard"
DESCRIPTION = "Complaints + peak hours + heatwave. Priority override required."
N_EPISODES  = 500


def make_env(seed: int = 42) -> HostelGridEnv:
    return HostelGridEnv(mode=TASK_NAME, seed=seed)


def task_info() -> dict:
    return {
        "name":        TASK_NAME,
        "mode":        TASK_NAME,
        "episodes":    N_EPISODES,
        "description": DESCRIPTION,
        "difficulty":  "HIGH",
        "goal":        "Zero complaints. Survive peak hours and heatwaves.",
    }