"""
env/reward.py

Reward aligned more closely with the behavior needed to beat the heuristic:
- reward occupied HP service strongly
- penalize complaints and repeated neglect
- penalize wasted energy in empty rooms
- penalize over-budget operation strongly
"""

import numpy as np

PEAK_HOURS    = {9, 10, 11, 12, 13, 14, 18, 19, 20, 21}
MAX_COMPLAINT = 10

PRIORITY_WEIGHT = {1: 0.60, 2: 0.80, 3: 1.00}


def room_reward(room, mode: str, heatwave: bool, hour: int, power_ratio: float) -> float:
    pw  = PRIORITY_WEIGHT.get(room.priority, 0.75)
    app = room.appliance_sum()
    occ = int(room.occupancy == 1)
    r   = 0.0

    if occ:
        # Saturate service after 2 appliances so "all_on" is not always optimal.
        service = min(1.0, app / 2.0)
        r += (service ** 1.2) * (0.55 + 0.45 * pw)

        # Small energy penalty even when occupied, so efficient service can beat blind "all_on".
        energy_pen = 0.04 * app
        if room.priority >= 3 and (heatwave or getattr(room, "complaint", 0) >= 2):
            energy_pen *= 0.5
        r -= energy_pen

        if room.is_satisfied():
            sat_bonus = {1: 0.12, 2: 0.22, 3: 0.34}[room.priority]
            r += sat_bonus
        else:
            lack_pen = {1: 0.05, 2: 0.16, 3: 0.30}[room.priority]
            r -= lack_pen

        if app == 0:
            r -= 0.45 * pw

    else:
        if app > 0:
            r -= 0.18 * app
            if getattr(room, "ac", 0) == 1:
                r -= 0.10

    if mode in ("medium", "hard"):
        ignored = float(getattr(room, "consecutive_ignored", 0))
        if ignored > 0:
            fairness_penalty = 0.06 * (ignored ** 1.35) * pw
            r -= min(fairness_penalty, 0.55)

        complaint = float(getattr(room, "complaint", 0))
        if complaint > 0:
            complaint_ratio = min(1.0, complaint / MAX_COMPLAINT)
            r -= (complaint_ratio ** 2) * (0.90 + 0.40 * pw)

    if mode in ("medium", "hard") and power_ratio < 0.25 and app > 0:
        r -= (0.25 - power_ratio) * (0.50 + 0.30 * pw)

    if mode == "hard":
        if hour in PEAK_HOURS and getattr(room, "ac", 0) == 1:
            r -= 0.22

        if heatwave and occ and getattr(room, "ac", 0) == 0:
            r -= 0.25 * pw

    return float(np.clip(r, -2.5, 1.5))


def compute_reward(hostel) -> tuple:
    power_ratio = float(np.clip(hostel.power_ratio(), 0.0, 1.0))

    per_room = [
        room_reward(
            r,
            mode=hostel.mode,
            heatwave=hostel.heatwave,
            hour=hostel.hour,
            power_ratio=power_ratio,
        )
        for r in hostel.rooms
    ]

    total = float(np.mean(per_room))

    occupied_hp = [r for r in hostel.rooms if r.occupancy == 1 and r.priority == 3]
    if occupied_hp:
        hp_sat = float(np.mean([1.0 if r.is_satisfied() else 0.0 for r in occupied_hp]))
        total += 0.80 * hp_sat - 0.20

    occupied = [r for r in hostel.rooms if r.occupancy == 1]
    if occupied and hostel.mode in ("medium", "hard"):
        avg_complaint = float(np.mean([r.complaint / MAX_COMPLAINT for r in occupied]))
        total += 0.25 * (1.0 - avg_complaint)

    if hostel.total_power() > hostel.power_budget:
        overage_ratio = (
            (hostel.total_power() - hostel.power_budget) /
            max(hostel.power_budget, 1e-6)
        )
        penalty_scale = {
            "easy": 0.8,
            "medium": 1.4,
            "hard": 2.0,
        }[hostel.mode]
        total -= penalty_scale * overage_ratio

    return float(np.clip(total, -3.0, 2.0)), per_room
