from tasks.task_easy   import make_env as make_easy_env,   task_info as easy_info
from tasks.task_medium import make_env as make_medium_env, task_info as medium_info
from tasks.task_hard   import make_env as make_hard_env,   task_info as hard_info

TASKS = {
    "easy":   {"make_env": make_easy_env,   "info": easy_info},
    "medium": {"make_env": make_medium_env, "info": medium_info},
    "hard":   {"make_env": make_hard_env,   "info": hard_info},
}


def make_env(task_name: str, seed: int = 42):
    assert task_name in TASKS, \
        f"Unknown task '{task_name}'. Choose: {list(TASKS.keys())}"
    return TASKS[task_name]["make_env"](seed=seed)


def task_info(task_name: str) -> dict:
    assert task_name in TASKS
    return TASKS[task_name]["info"]()


def print_all_tasks():
    for name, t in TASKS.items():
        info = t["info"]()
        print(f"  {name:8s} | {info['difficulty']:6s} | {info['description']}")


__all__ = ["make_env", "task_info", "print_all_tasks", "TASKS"]