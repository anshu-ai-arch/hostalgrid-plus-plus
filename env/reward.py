# env/reward.py
# Next-level reward — more factors, fixed violation strictness per task

import numpy as np


def calculate_reward(
    power_saved:      float,
    complaint_delta:  float,
    carbon_saved:     float,
    fairness_score:   float,
    power_usage:      float = 0.0,
    min_power_floor:  float = 0.3,
) -> float:
    """
    Base multi-objective reward.
    Fixed: agent penalized for going below minimum power floor.
    Prevents the 'turn everything off' reward hack.
    """
    w_energy   = 0.35
    w_comfort  = 0.30
    w_carbon   = 0.20
    w_fairness = 0.15

    reward = (
          w_energy   *  power_saved
        - w_comfort  *  complaint_delta
        + w_carbon   *  carbon_saved
        + w_fairness *  fairness_score
    )

    # Minimum power floor penalty
    if power_usage < min_power_floor:
        shortfall = min_power_floor - power_usage
        reward -= shortfall * 3.0

    return float(np.clip(reward, -50.0, 50.0))


def calculate_task1_reward(
    demand_sat:      float,
    violations:      int,
    feasible:        bool,
    total_required:  float,
    total_power:     float,
    cost:            float,
    complaints:      int,
    power_usage:     float,
    penalty_weights: dict,
) -> float:
    """
    Task 1 — FIXED violation strictness.
    Feasible violation = agent fault = harsh.
    Infeasible violation = system constraint = soft.
    """
    reward = 0.0

    # Primary: demand satisfaction
    reward += demand_sat * 5.0

    # Violation — scaled by feasibility
    if violations > 0:
        if feasible:
            reward -= violations * penalty_weights["priority_violation"] * 1.5
        else:
            shortfall_ratio = max(0, total_required - total_power) / max(1, total_required)
            reward -= shortfall_ratio * 1.5

    # Zero violation bonus
    if feasible and violations == 0:
        reward += 2.5

    # Cost light penalty
    reward -= cost * penalty_weights.get("cost", 0.05) * 0.3

    # Complaint moderate penalty
    reward -= complaints * penalty_weights.get("complaints", 0.1) * 0.3

    # Minimum power floor
    if power_usage < 0.5:
        reward -= (0.5 - power_usage) * 5.0

    return float(np.clip(reward, -50.0, 50.0))


def calculate_task2_reward(
    demand_sat:        float,
    violations:        int,
    feasible:          bool,
    fairness_score:    float,
    misuse_count:      int,
    misuse_handled:    int,
    enforcement_bonus: float,
    cost:              float,
    complaints:        int,
    power_usage:       float,
    penalty_weights:   dict,
) -> float:
    """Task 2 — fairness + misuse enforcement."""
    reward = 0.0

    reward += demand_sat * 4.0

    if fairness_score > 0.8:
        reward += 1.5
    elif fairness_score > 0.6:
        reward += 0.5
    else:
        reward -= (1.0 - fairness_score) * penalty_weights.get("fairness", 0.5)

    if misuse_count > 0:
        misuse_ratio = misuse_handled / misuse_count
        reward += misuse_ratio * 2.0
        if misuse_ratio < 0.3:
            reward -= 1.5
    reward += enforcement_bonus

    if violations > 0:
        if feasible:
            reward -= violations * penalty_weights["priority_violation"]
        else:
            reward -= violations * 0.5

    reward -= cost       * penalty_weights.get("cost", 0.05) * 0.3
    reward -= complaints * penalty_weights.get("complaints", 0.1) * 0.4

    if power_usage < 0.5:
        reward -= (0.5 - power_usage) * 5.0

    return float(np.clip(reward, -50.0, 50.0))


def calculate_task3_reward(
    demand_sat:         float,
    violations:         int,
    feasible:           bool,
    exam_satisfaction:  float,
    fairness_violation: float,
    misuse_ratio:       float,
    flagged_count:      int,
    battery_ratio:      float,
    cost:               float,
    carbon:             float,
    peak_violation:     bool,
    system_trust:       float,
    power_usage:        float,
    penalty_weights:    dict,
) -> float:
    """Task 3 — full crisis governance, 11 reward components."""
    reward = 0.0

    reward += demand_sat * 4.0

    if violations > 0:
        if feasible:
            reward -= violations * penalty_weights["priority_violation"]
        else:
            reward -= violations * 1.0
    else:
        reward += 1.5

    reward += exam_satisfaction * 1.5

    if fairness_violation > 1.0:
        reward -= fairness_violation * penalty_weights["fairness_violation"]
    elif fairness_violation < 0.6:
        reward += 1.0

    if flagged_count > 0:
        reward += misuse_ratio * 1.5
        if misuse_ratio < 0.3:
            reward -= 1.0

    if 0.3 < battery_ratio < 0.7:
        reward += 0.5
    elif battery_ratio < 0.1 or battery_ratio > 0.95:
        reward -= 0.5

    reward -= (cost / 100.0)  * penalty_weights["cost"]
    reward -= (carbon / 50.0) * penalty_weights["carbon_penalty"]

    if peak_violation:
        reward -= penalty_weights["peak_violation"]
    else:
        reward += 0.3

    if system_trust > 0.9:
        reward += 0.5
    elif system_trust < 0.5:
        reward -= 2.0

    if power_usage < 0.3:
        reward -= (0.3 - power_usage) * 8.0

    return float(np.clip(reward, -50.0, 50.0))


def time_of_day_bonus(hour: int, power_usage: float) -> float:
    if 9 <= hour <= 12 or 18 <= hour <= 22:
        return 0.3 if power_usage < 5.0 else -0.2
    elif 0 <= hour <= 5:
        return 0.1
    return 0.0


def complaint_momentum_penalty(complaint_history: list) -> float:
    if len(complaint_history) < 2:
        return 0.0
    recent = list(complaint_history)[-3:]
    if all(c > 0 for c in recent):
        return -0.5 * len(recent)
    return 0.0


def solar_harvest_bonus(solar_output: float, hour: int, power_usage: float) -> float:
    if solar_output > 0.5 and 9 <= hour <= 15:
        return 0.2
    return 0.0