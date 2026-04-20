"""
env/reward.py

Reward designed with COMPETING OBJECTIVES — not additive dominance.

Objectives (in tension):
    1. Comfort  — serve occupied rooms
    2. Fairness — don't ignore any room repeatedly
    3. Efficiency — don't waste power in empty rooms
    4. Constraint — stay within power budget

These CANNOT all be maximised simultaneously in HARD mode.
The agent must learn genuine trade-offs.

Per-room reward range: roughly [-2, +1]
Total (mean across rooms): roughly [-2, +1]

IMPORTANT DESIGN CHOICES:
    - Comfort reward is NOT linear — diminishing returns
    - Complaint penalty is NON-LINEAR — escalates fast
    - Power violation overrides everything else
    - HP rooms create CONFLICTS in hard mode (budget too small)
"""

import numpy as np

PEAK_HOURS    = {9, 10, 11, 12, 13, 14, 18, 19, 20, 21}
MAX_COMPLAINT = 10

# Priority weights — HP rooms matter more
PRIORITY_WEIGHT = {1: 0.4, 2: 0.7, 3: 1.0}


def room_reward(room,
                mode:      str,
                heatwave:  bool,
                hour:      int,
                power_ratio: float) -> float:
    """
    Per-room reward with competing objectives.

    Returns float in roughly [-2.0, +1.0].
    Deliberately NOT possible to max in hard mode.
    """
    pw  = PRIORITY_WEIGHT[room.priority]
    r   = 0.0
    app = room.appliance_sum()

    # ── Comfort objective ────────────────────────────────────────
    if room.occupancy == 1:
        # Diminishing returns — partial service gets less credit
        comfort = (app / 3.0) ** 1.5 * pw   # non-linear
        r += comfort

        # Full satisfaction bonus (hard to achieve in hard mode)
        if room.is_satisfied():
            r += 0.2 * pw

        # Inaction penalty — doing nothing in occupied room
        if app == 0:
            r -= 0.4 * pw

    # ── Efficiency objective ─────────────────────────────────────
    else:
        # Penalise running appliances in empty room
        if app > 0:
            r -= app * 0.15   # per device wasted

    # ── Fairness objective (medium/hard) ─────────────────────────
    if mode in ("medium", "hard"):
        # NON-LINEAR fairness penalty — gets much worse over time
        ignored = room.consecutive_ignored
        if ignored > 0:
            fairness_penalty = -0.05 * (ignored ** 1.4)
            r += max(fairness_penalty, -0.5)   # cap at -0.5

    # ── Complaint penalty (hard escalation) ──────────────────────
    if mode in ("medium", "hard") and room.complaint > 0:
        # Quadratic escalation — first few complaints small,
        # later complaints very expensive
        complaint_ratio = room.complaint / MAX_COMPLAINT
        r -= (complaint_ratio ** 2) * 0.8 * pw

    # ── Energy cost (hard mode / peak hours) ─────────────────────
    if mode == "hard":
        if hour in PEAK_HOURS and room.ac == 1:
            r -= 0.2   # AC during peak is expensive

        # Heatwave conflict — AC needed but increases power usage
        # Creates impossible choice in hard mode (budget too small)
        if heatwave and room.occupancy == 1 and room.ac == 0:
            r -= 0.3   # penalty for no AC during heatwave

    # ── Power constraint signal ───────────────────────────────────
    # Low power_ratio means budget almost exhausted
    # This signal tells agent to be conservative
    if power_ratio < 0.2 and app > 0:
        r -= (0.2 - power_ratio) * 0.5   # soft penalty near budget

    return float(np.clip(r, -2.0, 1.0))


def compute_reward(hostel) -> tuple:
    """
    Compute total reward and per-room breakdown.

    Total = mean per-room reward.
    Additional global penalty if over power budget.

    Returns:
        total_reward : float in [-2, +1]
        per_room     : list of 10 floats
    """
    power_ratio = hostel.power_ratio()

    per_room = [
        room_reward(r,
                    mode=hostel.mode,
                    heatwave=hostel.heatwave,
                    hour=hostel.hour,
                    power_ratio=power_ratio)
        for r in hostel.rooms
    ]

    total = float(np.mean(per_room))

    # Hard budget violation penalty (global)
    if hostel.total_power() > hostel.power_budget:
        overage = (hostel.total_power() - hostel.power_budget)
        violation = -(overage / hostel.power_budget) * 0.5
        total += violation

    return float(np.clip(total, -2.0, 1.0)), per_room