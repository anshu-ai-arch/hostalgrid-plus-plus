# env/reward.py
#
# MULTI-OBJECTIVE REWARD FUNCTION — MATHEMATICAL SPECIFICATION
#
# FULL FORMULA:
#
#   R(s,a) = α₁·DS - α₂·CR - α₃·V + α₄·FI - α₅·(C/Cₙ) - α₆·(CO₂/COₙ) + α₇·T + B
#
# Where:
#   DS  = demand_satisfaction ∈ [0,1]
#   CR  = complaint_ratio ∈ [0,1]  = total_complaints / num_rooms
#   V   = violations (integer)
#   FI  = fairness_index ∈ [0,1]  = 1 - std(supply)/mean(supply)
#   C   = hour_cost (Rs)
#   Cₙ  = 200.0 (normalizer)
#   CO₂ = carbon (gCO2)
#   COₙ = 100.0 (normalizer)
#   T   = system_trust ∈ [0,1]
#   B   = bonus for simultaneous excellence
#
# WEIGHTS (with justification):
#   α₁ = 4.0   — demand is PRIMARY objective
#   α₂ = 3.0   — student welfare is core KPI
#   α₃ = 2.0   — violations are hard constraint (trust decay adds more pressure)
#   α₄ = 1.5   — fairness matters but acute discomfort is worse
#   α₅ = 1/200 — cost normalized, tertiary
#   α₆ = 1/100 — carbon normalized, sustainability
#   α₇ = 0.5   — long-horizon reputation signal
#   B  = 1.0   — bonus for simultaneous excellence

import numpy as np
from dataclasses import dataclass
from typing import Optional

W_DEMAND_SAT = 4.0
W_COMPLAINTS = 3.0
W_VIOLATIONS = 2.0
W_FAIRNESS   = 1.5
W_COST       = 1/200
W_CARBON     = 1/100
W_TRUST      = 0.5
BONUS_CLEAN  = 1.0
CLIP_LOW     = -50.0
CLIP_HIGH    = 50.0


@dataclass(frozen=True)
class RewardWeights:
    """
    Shared multi-objective weights.

    Canonical task reward:
        r_t =
            w_d * D_t
            - w_c * C_t
            - alpha * P_t
            - beta * U_t
            - w_f * F_t
            - w_co2 * E_t
            + gamma * R_t
            + w_tr * T_t

    where
        D_t   = demand satisfaction
        C_t   = normalized electricity cost
        P_t   = peak-load penalty
        U_t   = discomfort penalty
        F_t   = fairness penalty
        E_t   = normalized carbon penalty
        R_t   = renewable usage ratio
        T_t   = system trust
    """

    demand: float = 5.0
    cost: float = 1.2
    peak: float = 1.4
    discomfort: float = 2.0
    fairness: float = 1.4
    carbon: float = 0.9
    renewable: float = 1.0
    trust: float = 0.6
    clean_bonus: float = 0.5


@dataclass(frozen=True)
class RewardNormalizers:
    cost_rs: float = 120.0
    carbon_g: float = 40.0
    fairness: float = 1.0


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def multi_objective_reward(
    *,
    demand_satisfaction: float,
    electricity_cost: float,
    peak_load_penalty: float,
    discomfort_penalty: float,
    renewable_usage: float,
    fairness_penalty: float = 0.0,
    carbon_penalty: float = 0.0,
    system_trust: float = 1.0,
    weights: Optional[RewardWeights] = None,
    normalizers: Optional[RewardNormalizers] = None,
):
    weights = weights or RewardWeights()
    normalizers = normalizers or RewardNormalizers()

    cost_term = electricity_cost / max(normalizers.cost_rs, 1e-6)
    carbon_term = carbon_penalty / max(normalizers.carbon_g, 1e-6)
    fairness_term = fairness_penalty / max(normalizers.fairness, 1e-6)

    components = {
        "demand": weights.demand * clip01(demand_satisfaction),
        "cost": -weights.cost * cost_term,
        "peak": -weights.peak * clip01(peak_load_penalty),
        "discomfort": -weights.discomfort * clip01(discomfort_penalty),
        "fairness": -weights.fairness * clip01(fairness_term),
        "carbon": -weights.carbon * clip01(carbon_term),
        "renewable": weights.renewable * clip01(renewable_usage),
        "trust": weights.trust * clip01(system_trust),
        "bonus": 0.0,
    }

    if demand_satisfaction > 0.95 and peak_load_penalty < 0.05 and discomfort_penalty < 0.10:
        components["bonus"] = weights.clean_bonus

    reward = sum(components.values())
    return float(np.clip(reward, CLIP_LOW, CLIP_HIGH)), components


def base_reward(demand_satisfaction, complaint_ratio, violations,
                fairness_index, hour_cost, carbon, system_trust):
    """Core multi-objective reward. All tasks build on this."""
    reward = (
          W_DEMAND_SAT * demand_satisfaction
        - W_COMPLAINTS * complaint_ratio
        - W_VIOLATIONS * violations
        + W_FAIRNESS   * fairness_index
        - W_COST       * hour_cost
        - W_CARBON     * carbon
        + W_TRUST      * system_trust
    )
    if violations == 0 and demand_satisfaction > 0.8:
        reward += BONUS_CLEAN
    return float(np.clip(reward, CLIP_LOW, CLIP_HIGH))


def task1_shaping(violations, demand_satisfaction):
    """
    Task 1: Commitment-Aware Allocation shaping.
    Extra signal to make violation penalty stronger when 40% rooms have priority.
    R_extra = +2.0 if clean step, -1.5*V if violations
    """
    if violations == 0 and demand_satisfaction > 0.9:
        return 2.0
    return -1.5 * violations


def task2_shaping(flagged_rooms, handled_rooms, fairness_violation, violations, demand_satisfaction):
    """
    Task 2: Fair Enforcement shaping.
    R_extra = misuse_ratio*2.0 + fairness_bonus - fairness_penalty + clean_bonus
    """
    extra = 0.0
    if flagged_rooms > 0:
        ratio = handled_rooms / flagged_rooms
        extra += ratio * 2.0
        if ratio < 0.3:
            extra -= 1.5
    if fairness_violation < 0.4:
        extra += 0.8
    elif fairness_violation > 0.8:
        extra -= fairness_violation * 1.2
    if violations == 0 and demand_satisfaction > 0.85:
        extra += 1.5
    return extra


def task3_shaping(active_events, system_trust, carbon, fairness_violation, is_done):
    """
    Task 3: Crisis Governance shaping.
    R_extra = crisis_survival + carbon_bonus + fairness_bonus + end_trust_bonus
    """
    extra = 0.0
    if active_events > 0:
        extra += 0.5
    if carbon < 20:
        extra += 0.3
    if fairness_violation < 0.4:
        extra += 0.5
    if is_done and system_trust > 0.8:
        extra += 2.0
    return extra


class RunningNormalizer:
    """
    Welford online algorithm for reward normalization.
    Stabilizes Q-value updates when reward scale varies across tasks.
    
    Use: wrap reward before replay buffer storage.
    Do NOT use for final score computation.
    """
    def __init__(self, clip=5.0, epsilon=1e-8):
        self.mean    = 0.0
        self.var     = 1.0
        self.count   = 0
        self.clip    = clip
        self.epsilon = epsilon
        self._M2     = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._M2 += delta * delta2
        if self.count > 1:
            self.var = self._M2 / (self.count - 1)
        std = max(np.sqrt(self.var), self.epsilon)
        return float(np.clip((x - self.mean) / std, -self.clip, self.clip))

    def stats(self):
        return {"mean": round(self.mean, 4), "std": round(float(np.sqrt(self.var)), 4)}


def compute_snr(env_factory, n_trials=500):
    """
    Signal-to-Noise Ratio check.
    SNR = (max_action_mean - min_action_mean) / mean(per_action_std)
    Must be > 1.0 before training. HostelGrid++ achieves ~8.49.
    """
    import random
    by_action = {i: [] for i in range(6)}
    for _ in range(n_trials):
        env = env_factory()
        env.reset()
        a = random.randint(0, 5)
        _, r, _, _ = env.step(a)
        by_action[a].append(r)
    means = [float(np.mean(v)) for v in by_action.values() if v]
    stds  = [float(np.std(v))  for v in by_action.values() if v]
    if not means:
        return {"snr": 0.0}
    snr = (max(means) - min(means)) / (float(np.mean(stds)) + 1e-9)
    return {
        "snr":    round(snr, 2),
        "status": "GOOD" if snr > 1.0 else "POOR — redesign reward",
    }


# Legacy compatibility — old code imports these names
def calculate_reward(power_saved, complaint_delta, carbon_saved, fairness_score,
                     power_usage=0.0, min_power_floor=0.3, violations=0):
    """Backward-compatible wrapper for base_reward."""
    return base_reward(
        demand_satisfaction = max(0.0, min(1.0, power_saved)),
        complaint_ratio     = complaint_delta,
        violations          = violations,
        fairness_index      = fairness_score,
        hour_cost           = 0.0,
        carbon              = max(0.0, -carbon_saved * 100),
        system_trust        = 0.8,
    )
def time_of_day_bonus(*args, **kwargs):
    return 0.0