"""graders/__init__.py"""

import numpy as np
from graders.grader_easy   import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard   import grade as grade_hard

_GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(agent, task_name: str) -> dict:
    assert task_name in _GRADERS, \
        f"Unknown task. Choose: {list(_GRADERS.keys())}"
    return _GRADERS[task_name](agent)


def grade_all(agent) -> dict:
    reports = {}
    for task in ("easy", "medium", "hard"):
        reports[task] = grade(agent, task)

    print(f"\n{'='*52}")
    print("  FINAL SUMMARY (Meta OpenEnv — scores 0.0–1.0)")
    print(f"{'='*52}")
    print(f"  {'Task':8s} | {'Score':>8} | {'Grade':>5}")
    print("  " + "-"*30)
    for task, r in reports.items():
        print(f"  {task:8s} | {r['score']:8.4f} | {r['grade']:>5}")

    overall = np.mean([r["score"] for r in reports.values()])
    if   overall >= 0.75: og = "A"
    elif overall >= 0.60: og = "B"
    elif overall >= 0.45: og = "C"
    elif overall >= 0.30: og = "D"
    else:                 og = "F"
    print(f"  Overall: {overall:.4f} / 1.0  →  Grade {og}")
    print(f"{'='*52}")
    print(f"  Overall: {overall:.4f} / 1.0")
    print(f"{'='*52}")

    reports["overall"] = {"score": round(overall, 4)}
    return reports


__all__ = ["grade", "grade_all"]