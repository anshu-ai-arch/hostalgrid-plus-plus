# graders/grader_easy.py
# Grader for Task 1 — Commitment-Aware Energy Allocation

def grade_easy(stats):
    """
    Evaluates Task 1 agent performance.
    stats keys:
        demand_satisfaction : float (0-1)
        violations          : int
        total_cost          : float
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

    # ── 2. Commitment Violations (25%) ────────────────────────
    v = stats.get("violations", 999)
    if v == 0:
        score += 0.25
        feedback.append(("✅", "Zero commitment violations",
                         f"{v}", "+0.25"))
    elif v <= 5:
        score += 0.10
        feedback.append(("⚠️ ", "Few violations (≤ 5)",
                         f"{v}", "+0.10"))
    else:
        feedback.append(("❌", "Too many violations",
                         f"{v}", "+0.00"))

    # ── 3. Cost Efficiency (20%) ──────────────────────────────
    cost = stats.get("total_cost", 9999)
    if cost < 400:
        score += 0.20
        feedback.append(("✅", "Cost efficient (< 400)",
                         f"{cost:.1f}", "+0.20"))
    elif cost < 600:
        score += 0.10
        feedback.append(("⚠️ ", "Acceptable cost (< 600)",
                         f"{cost:.1f}", "+0.10"))
    else:
        feedback.append(("❌", "Cost too high",
                         f"{cost:.1f}", "+0.00"))

    # ── 4. Complaint Management (15%) ─────────────────────────
    complaints = stats.get("complaints", 999)
    if complaints < 10:
        score += 0.15
        feedback.append(("✅", "Complaints under control (< 10)",
                         f"{complaints}", "+0.15"))
    elif complaints < 20:
        score += 0.08
        feedback.append(("⚠️ ", "Moderate complaints (< 20)",
                         f"{complaints}", "+0.08"))
    else:
        feedback.append(("❌", "Too many complaints",
                         f"{complaints}", "+0.00"))

    # ── 5. Total Reward (15%) ─────────────────────────────────
    reward = stats.get("total_reward", -999)
    if reward > 5.0:
        score += 0.15
        feedback.append(("✅", "Strong total reward (> 5.0)",
                         f"{reward:.4f}", "+0.15"))
    elif reward > 0:
        score += 0.08
        feedback.append(("⚠️ ", "Positive reward",
                         f"{reward:.4f}", "+0.08"))
    else:
        feedback.append(("❌", "Negative reward",
                         f"{reward:.4f}", "+0.00"))

    # ── Print Report ──────────────────────────────────────────
    print("\n" + "="*60)
    print("📋  Task 1 Grader — Commitment-Aware Energy Allocation")
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
    from tasks.task_easy import train
    print("🚀 Running Task 1 training then grading...")
    agent = train(episodes=300)

    # Collect final episode stats manually
    import numpy as np
    from tasks.task_easy import Task1Env
    env = Task1Env()
    obs = env.reset()
    done = False
    total_reward     = 0
    total_violations = 0
    total_complaints = 0
    total_cost       = 0
    total_ds         = 0

    while not done:
        state = tuple(np.round(obs, 1))
        if state in agent.q_table:
            action = int(np.argmax(agent.q_table[state]))
        else:
            import random
            action = random.randint(0, 5)
        obs, reward, done, info = env.step(action)
        total_reward     += reward
        total_violations += info["violations"]
        total_complaints += info["complaints"]
        total_cost       += info["cost"]
        total_ds         += info["demand_satisfaction"]

    grade_easy({
        "demand_satisfaction" : total_ds / 24,
        "violations"          : total_violations,
        "total_cost"          : total_cost,
        "complaints"          : total_complaints,
        "total_reward"        : total_reward,
    })