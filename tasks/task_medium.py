"""
tasks/task_medium.py

MEDIUM task.
Adds fairness penalty and comfort priority on top of easy.
Complaints accumulate if rooms are repeatedly ignored.
"""

from env.hostelgrid_env import HostelGridEnv

TASK_NAME   = "medium"
DESCRIPTION = "Fairness + comfort priority. Avoid repeated room neglect."
N_EPISODES  = 500


def make_env(seed: int = 42) -> HostelGridEnv:
    return HostelGridEnv(mode=TASK_NAME, seed=seed)


def task_info() -> dict:
    return {
        "name":        TASK_NAME,
        "mode":        TASK_NAME,
        "episodes":    N_EPISODES,
        "description": DESCRIPTION,
        "difficulty":  "MEDIUM",
        "goal":        "Serve all rooms fairly. Reduce complaints to 0.",
    }