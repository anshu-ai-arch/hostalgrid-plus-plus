# graders/grader_hard.py
# Grader for Task 3 — Crisis Governance Under Extreme Conditions

def grade_hard(stats):
    """
    Evaluates Task 3 agent performance.
    stats keys (matches Task3Env.episode_stats()):
        demand_satisfaction : float (0-1)
        violations          : int
        total_cost          : float
        total_carbon        : float
        fairness_score      : float
        peak_violations     : int
        system_trust        : float (0-1)
        events_encountered  : int
    """
    score    = 0.0
    feedback = []

    # ── 1. Demand Satisfaction (20%) ──────────────────────────
    ds = stats.get("demand_satisfaction", 0)
    if ds > 0.85:
        score += 0.20
        feedback.append(("✅", "Demand satisfaction > 85%",
                         f"{ds:.3f}", "+0.20"))
    elif ds > 0.70:
        score += 0.10
        feedback.append(("⚠️ ", "Demand satisfaction > 70%",
                         f"{ds:.3f}", "+0.10"))
    else:
        feedback.append(("❌", "Demand satisfaction too low",
                         f"{ds:.3f}", "+0.00"))

    # ── 2. Priority Violations (20%) ──────────────────────────
    v = stats.get("violations", 999)
    if v < 15:
        score += 0.20
        feedback.append(("✅", "Violations under control (< 15)",
                         f"{v}", "+0.20"))
    elif v < 30:
        score += 0.10
        feedback.append(("⚠️ ", "Moderate violations (< 30)",
                         f"{v}", "+0.10"))
    else:
        feedback.append(("❌", "Too many violations",
                         f"{v}", "+0.00"))

    # ── 3. Cost Efficiency (15%) ──────────────────────────────
    cost = stats.get("total_cost", 9999)
    if cost < 500:
        score += 0.15
        feedback.append(("✅", "Cost efficient (< 500)",
                         f"{cost:.1f}", "+0.15"))
    elif cost < 700:
        score += 0.08
        feedback.append(("⚠️ ", "Acceptable cost (< 700)",
                         f"{cost:.1f}", "+0.08"))
    else:
        feedback.append(("❌", "Cost too high",
                         f"{cost:.1f}", "+0.00"))

    # ── 4. Carbon Footprint (10%) ─────────────────────────────
    carbon = stats.get("total_carbon", 9999)
    if carbon < 200:
        score += 0.10
        feedback.append(("✅", "Low carbon footprint (< 200)",
                         f"{carbon:.1f}", "+0.10"))
    elif carbon < 300:
        score += 0.05
        feedback.append(("⚠️ ", "Moderate carbon (< 300)",
                         f"{carbon:.1f}", "+0.05"))
    else:
        feedback.append(("❌", "High carbon footprint",
                         f"{carbon:.1f}", "+0.00"))

    # ── 5. Fairness Under Crisis (15%) ────────────────────────
    fv = stats.get("fairness_score", 999)
    if fv < 0.70:
        score += 0.15
        feedback.append(("✅", "Fairness maintained (< 0.70)",
                         f"{fv:.3f}", "+0.15"))
    elif fv < 0.90:
        score += 0.08
        feedback.append(("⚠️ ", "Acceptable fairness (< 0.90)",
                         f"{fv:.3f}", "+0.08"))
    else:
        feedback.append(("❌", "Fairness collapsed",
                         f"{fv:.3f}", "+0.00"))

    # ── 6. Peak Management (10%) ──────────────────────────────
    pv = stats.get("peak_violations", 999)
    if pv < 3:
        score += 0.10
        feedback.append(("✅", "Peak violations minimal (< 3)",
                         f"{pv}", "+0.10"))
    elif pv < 6:
        score += 0.05
        feedback.append(("⚠️ ", "Some peak violations (< 6)",
                         f"{pv}", "+0.05"))
    else:
        feedback.append(("❌", "Too many peak violations",
                         f"{pv}", "+0.00"))

    # ── 7. System Trust (10%) ─────────────────────────────────
    trust = stats.get("system_trust", 0)
    if trust > 0.80:
        score += 0.10
        feedback.append(("✅", "System trust high (> 0.80)",
                         f"{trust:.3f}", "+0.10"))
    elif trust > 0.60:
        score += 0.05
        feedback.append(("⚠️ ", "Moderate trust (> 0.60)",
                         f"{trust:.3f}", "+0.05"))
    else:
        feedback.append(("❌", "Trust collapsed",
                         f"{trust:.3f}", "+0.00"))

    # ── Bonus: Survived events (informational) ────────────────
    events = stats.get("events_encountered", 0)
    bonus  = 0.0
    if events >= 3 and trust > 0.80 and v < 15:
        bonus   = 0.05
        score  += bonus
        feedback.append(("⭐", f"Bonus: survived {events} events with trust intact",
                         f"{events}", f"+{bonus:.2f}"))

    # ── Print Report ──────────────────────────────────────────
    print("\n" + "="*65)
    print("📋  Task 3 Grader — Crisis Governance Under Extreme Conditions")
    print("="*65)
    print(f"  {'':2} {'Metric':<38} {'Value':>10}  {'Points':>6}")
    print("-"*65)
    for icon, label, value, points in feedback:
        print(f"  {icon} {label:<38} {value:>10}  {points:>6}")
    print("="*65)
    print(f"  🏆  Final Score : {score:.2f} / 1.00"
          + (f"  (includes ⭐ bonus)" if bonus > 0 else ""))
    print("="*65)
    return score


# ── Standalone runner ─────────────────────────────────────────
if __name__ == "__main__":
    from tasks.task_hard import train, Task3Env
    import numpy as np
    import random

    print("🚀 Running Task 3 training then grading...")
    agent = train(episodes=1500)

    # Run one clean evaluation episode
    env = Task3Env()
    obs = env.reset()
    done = False

    while not done:
        s = tuple(np.clip(
            (np.asarray(obs, dtype=np.float32) * 10).astype(int),
            -100, 100
        ))
        if s in agent.q_table:
            action = int(np.argmax(agent.q_table[s]))
        else:
            action = random.randint(0, 5)
        obs, reward, done, info = env.step(action)

    # Use built-in episode_stats from Task3Env
    final_stats = env.episode_stats()
    grade_hard(final_stats)