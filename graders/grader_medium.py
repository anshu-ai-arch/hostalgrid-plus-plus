# graders/grader_medium.py
# Grader for Task 2 — Fair Enforcement Under Misuse

def grade_medium(stats):
    """
    Evaluates Task 2 agent performance.
    stats keys:
        demand_satisfaction : float (0-1)
        violations          : int
        total_cost          : float
        fairness            : float (0-1)
        misuse_handled      : float (0-1)
        complaints          : int
        total_reward        : float
    """
    score    = 0.0
    feedback = []

    # ── 1. Demand Satisfaction (25%) ──────────────────────────
    ds = stats.get("demand_satisfaction", 0)
    if ds > 0.90:
        score += 0.25
        feedback.append(("✅", "Demand satisfaction > 90%",
                         f"{ds:.3f}", "+0.25"))
    elif ds > 0.75:
        score += 0.10
        feedback.append(("⚠️ ", "Demand satisfaction > 75%",
                         f"{ds:.3f}", "+0.10"))
    else:
        feedback.append(("❌", "Demand satisfaction too low",
                         f"{ds:.3f}", "+0.00"))

    # ── 2. Priority Violations (25%) ──────────────────────────
    v = stats.get("violations", 999)
    if v == 0:
        score += 0.25
        feedback.append(("✅", "Zero priority violations",
                         f"{v}", "+0.25"))
    elif v <= 8:
        score += 0.10
        feedback.append(("⚠️ ", "Few violations (≤ 8)",
                         f"{v}", "+0.10"))
    else:
        feedback.append(("❌", "Too many violations",
                         f"{v}", "+0.00"))

    # ── 3. Cost Efficiency (20%) ──────────────────────────────
    cost = stats.get("total_cost", 9999)
    if cost < 500:
        score += 0.20
        feedback.append(("✅", "Cost efficient (< 500)",
                         f"{cost:.1f}", "+0.20"))
    elif cost < 700:
        score += 0.10
        feedback.append(("⚠️ ", "Acceptable cost (< 700)",
                         f"{cost:.1f}", "+0.10"))
    else:
        feedback.append(("❌", "Cost too high",
                         f"{cost:.1f}", "+0.00"))

    # ── 4. Fairness Score (15%) ───────────────────────────────
    fairness = stats.get("fairness", 0)
    if fairness > 0.70:
        score += 0.15
        feedback.append(("✅", "Fairness maintained (> 0.70)",
                         f"{fairness:.3f}", "+0.15"))
    elif fairness > 0.50:
        score += 0.08
        feedback.append(("⚠️ ", "Acceptable fairness (> 0.50)",
                         f"{fairness:.3f}", "+0.08"))
    else:
        feedback.append(("❌", "Fairness too low",
                         f"{fairness:.3f}", "+0.00"))

    # ── 5. Misuse Handled (15%) ───────────────────────────────
    misuse = stats.get("misuse_handled", 0)
    if misuse > 0.60:
        score += 0.15
        feedback.append(("✅", "Misuse handled well (> 60%)",
                         f"{misuse:.3f}", "+0.15"))
    elif misuse > 0.35:
        score += 0.08
        feedback.append(("⚠️ ", "Partial misuse handling (> 35%)",
                         f"{misuse:.3f}", "+0.08"))
    else:
        feedback.append(("❌", "Misuse not handled",
                         f"{misuse:.3f}", "+0.00"))

    # ── Print Report ──────────────────────────────────────────
    print("\n" + "="*60)
    print("📋  Task 2 Grader — Fair Enforcement Under Misuse")
    print("="*60)
    print(f"  {'':2} {'Metric':<35} {'Value':>10}  {'Points':>6}")
    print("-"*60)
    for icon, label, value, points in feedback:
        print(f"  {icon} {label:<35} {value:>10}  {points:>6}")
    print("="*60)
    print(f"  🏆  Final Score : {score:.2f} / 1.00")
    print("="*60)
    return score


# ── Standalone runner ─────────────────────────────────────────
if __name__ == "__main__":
    from tasks.task_medium import train
    import numpy as np
    import random

    print("🚀 Running Task 2 training then grading...")
    agent = train(episodes=500)

    from tasks.task_medium import Task2Env
    env = Task2Env()
    obs = env.reset()
    done = False

    total_reward     = 0
    total_violations = 0
    total_complaints = 0
    total_cost       = 0
    total_ds         = 0
    total_fairness   = 0
    total_misuse     = 0
    steps            = 0

    while not done:
        state = tuple(np.round(obs, 1))
        if state in agent.q_table:
            action = int(np.argmax(agent.q_table[state]))
        else:
            action = random.randint(0, 5)
        obs, reward, done, info = env.step(action)
        total_reward     += reward
        total_violations += info["violations"]
        total_complaints += info["complaints"]
        total_cost       += info["cost"]
        total_ds         += info["demand_satisfaction"]
        total_fairness   += info["fairness_score"]
        total_misuse     += info["misuse_count"]
        steps            += 1

    misuse_handled = min(1.0, total_misuse / max(1, steps) * 2)

    grade_medium({
        "demand_satisfaction" : total_ds / max(1, steps),
        "violations"          : total_violations,
        "total_cost"          : total_cost,
        "fairness"            : total_fairness / max(1, steps),
        "misuse_handled"      : misuse_handled,
        "complaints"          : total_complaints,
        "total_reward"        : total_reward,
    })