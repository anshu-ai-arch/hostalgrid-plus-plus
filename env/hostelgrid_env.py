# env/hostelgrid_env.py
# Fixed:
# 1. system_trust now updated every step from EpisodeState
# 2. battery_level charges from solar, discharges at peak
# 3. demand_sat = 1.0 guard when no priority rooms (now returns avg supply ratio)
# 4. complaint_delta uses level-based penalty instead of delta to prevent gaming

import random
import numpy as np
from collections import deque

from env.observation import Observation
from env.action import (
    ACTIONS, get_action_count,
    get_power_delta, get_comfort_delta, get_temp_delta, get_min_power
)
from env.reward import calculate_reward, time_of_day_bonus


class BaseRoom:
    """
    Realistic room model — individual complaint logic,
    temperature physics, demand constraints.
    """
    def __init__(self, room_id: int, num_rooms: int = 20):
        self.room_id     = room_id
        self.num_rooms   = num_rooms

        # Occupancy
        self.is_occupied = random.choice([True, False])

        # Temperature — individual comfort threshold
        self.temperature          = random.uniform(22.0, 30.0)
        self.comfort_threshold    = random.uniform(24.0, 28.0)
        self.temperature_tolerance = random.uniform(1.0, 3.0)

        # Power
        self.current_supply  = 0.0
        self.current_demand  = random.uniform(0.5, 2.5)
        self.base_demand     = self.current_demand
        self.power_cap       = 5.0

        # Commitment
        self.has_approved_request = random.random() < 0.30
        self.min_required_supply  = 1.5 if self.has_approved_request else 0.0

        # Complaint tracking — REALISTIC (not just counter)
        self.complaint_level    = 0      # 0=none 1=mild 2=moderate 3=severe
        self.complaint_history  = deque(maxlen=5)
        self.consecutive_hot    = 0      # steps above threshold
        self.consecutive_dark   = 0      # steps with lights off while occupied

        # Appliances
        self.ac_on     = self.is_occupied
        self.lights_on = self.is_occupied

        # Crisis features
        self.in_exam_center = random.random() < 0.2
        self.exam_mode      = False
        self.flagged_for_misuse     = False
        self.misuse_penalty_timer   = 0
        self.demand_history         = deque(maxlen=5)

        # Priority level
        if self.has_approved_request:
            self.priority_level = "critical"
        elif self.in_exam_center:
            self.priority_level = "high"
        else:
            self.priority_level = "normal"

    def update_temperature(self, action_temp_delta: float):
        ambient = 30.0
        if self.ac_on:
            self.temperature += (22.0 - self.temperature) * 0.1
        else:
            self.temperature += (ambient - self.temperature) * 0.15
        self.temperature += action_temp_delta
        self.temperature  = float(np.clip(self.temperature, 10.0, 45.0))

    def update_complaints(self):
        if not self.is_occupied:
            self.complaint_level = 0
            self.consecutive_hot  = 0
            self.consecutive_dark = 0
            return

        hot  = self.temperature > self.comfort_threshold + self.temperature_tolerance
        dark = not self.lights_on

        if hot and not self.ac_on:
            self.consecutive_hot += 1
        else:
            self.consecutive_hot = max(0, self.consecutive_hot - 1)

        if dark:
            self.consecutive_dark += 1
        else:
            self.consecutive_dark = 0

        discomfort = self.consecutive_hot + self.consecutive_dark

        if discomfort == 0:
            self.complaint_level = 0
        elif discomfort <= 1:
            self.complaint_level = 1
        elif discomfort <= 3:
            self.complaint_level = 2
        else:
            self.complaint_level = 3

        if self.exam_mode and self.complaint_level > 0:
            self.complaint_level = min(3, self.complaint_level + 1)

        self.complaint_history.append(self.complaint_level)

    def update_demand(self):
        variation = random.uniform(0.85, 1.15)
        self.current_demand = float(np.clip(
            self.base_demand * variation, 0.1, 5.0
        ))
        self.demand_history.append(self.current_demand)
        return self.current_demand

    def check_violation(self) -> bool:
        if self.has_approved_request:
            return self.current_supply < self.min_required_supply
        return False

    def get_power_consumption(self) -> float:
        power = 0.0
        if self.ac_on:
            power += 1.5
        if self.lights_on:
            power += 0.1
        return power


class HostelGridEnv:
    """
    Next-level base RL environment.
    Fixed:
    - system_trust synced from EpisodeState each step
    - battery_level updated from solar/peak logic
    - demand_sat no longer returns 1.0 when no priority rooms
    - complaint penalty uses level, not delta, to prevent gaming
    """

    def __init__(self, num_rooms: int = 20, episode_hours: int = 24):
        self.num_rooms     = num_rooms
        self.episode_hours = episode_hours
        self.current_hour  = 0
        self.done          = False

        self.obs   = None
        self.rooms = []

        # Episode tracking
        self._ep_total_complaints = 0
        self._ep_total_cost       = 0.0
        self._ep_total_violations = 0
        self._ep_demand_sat_sum   = 0.0
        self._ep_steps            = 0
        self._complaint_history   = deque(maxlen=10)

        # System trust — tracked internally
        self._system_trust = 1.0

        # Power floor — hard constraint
        self.MIN_POWER_KW = 0.5

    def reset(self):
        self.current_hour  = 0
        self.done          = False
        self._system_trust = 1.0  # FIX: reset trust

        self.rooms = [BaseRoom(i, self.num_rooms) for i in range(self.num_rooms)]

        self._ep_total_complaints = 0
        self._ep_total_cost       = 0.0
        self._ep_total_violations = 0
        self._ep_demand_sat_sum   = 0.0
        self._ep_steps            = 0
        self._complaint_history   = deque(maxlen=10)

        self.obs = Observation(self.num_rooms)
        self.obs.power_usage       = sum(r.get_power_consumption() for r in self.rooms)
        self.obs.avg_temperature   = float(np.mean([r.temperature for r in self.rooms]))
        self.obs.avg_occupancy     = sum(r.is_occupied for r in self.rooms) / self.num_rooms
        self.obs.complaint_level   = 0
        self.obs.time_of_day       = 0
        self.obs.carbon_rate       = self._get_carbon_rate(0)
        self.obs.current_cost      = 0.0
        self.obs.peak_hour         = self._is_peak(0)
        self.obs.solar_output      = self._get_solar(0)
        self.obs.battery_level     = 0.5
        self.obs.system_trust      = 1.0  # FIX: initialize properly

        return self.obs.to_vector()

    def step(self, action: int):
        assert 0 <= action < get_action_count(), f"Invalid action {action}"

        prev_power      = self.obs.power_usage
        prev_complaints = self.obs.complaint_level

        power_delta   = get_power_delta(action)
        comfort_delta = get_comfort_delta(action)
        temp_delta    = get_temp_delta(action)
        min_power     = get_min_power(action)

        for room in self.rooms:
            room.update_temperature(temp_delta / self.num_rooms)

        if action in [0, 1]:
            for room in self.rooms:
                if room.is_occupied:
                    room.ac_on = True
        elif action in [7]:
            for room in self.rooms:
                room.ac_on = False
        elif action in [4]:
            for room in self.rooms:
                if not room.has_approved_request:
                    room.ac_on = random.random() > 0.3
        elif action == 2:
            for room in self.rooms:
                if room.is_occupied:
                    room.lights_on = True
        elif action == 5:
            for room in self.rooms:
                if not room.is_occupied:
                    room.lights_on = False
        elif action == 9:
            for room in self.rooms:
                if not room.has_approved_request:
                    room.ac_on     = False
                    room.lights_on = room.is_occupied

        for room in self.rooms:
            room.update_demand()

        raw_power = self.obs.power_usage + power_delta
        new_power = max(self.MIN_POWER_KW, raw_power)
        new_power = float(np.clip(new_power, self.MIN_POWER_KW, 50.0))

        total_demand = sum(r.current_demand for r in self.rooms)
        for room in self.rooms:
            if total_demand > 0:
                room.current_supply = (room.current_demand / total_demand) * new_power
            else:
                room.current_supply = new_power / self.num_rooms

        for room in self.rooms:
            room.update_complaints()

        total_complaints = sum(r.complaint_level for r in self.rooms)
        self._complaint_history.append(total_complaints)

        violations = sum(1 for r in self.rooms if r.check_violation())

        priority_rooms = [r for r in self.rooms if r.has_approved_request]

        # FIX: demand_sat — when no priority rooms, use overall supply ratio
        # instead of returning a misleading 1.0
        if len(priority_rooms) > 0:
            satisfied  = sum(1 for r in priority_rooms if not r.check_violation())
            demand_sat = satisfied / len(priority_rooms)
        else:
            # No priority rooms — use supply vs demand ratio as proxy
            demand_sat = min(1.0, new_power / max(total_demand, 0.1))

        supplies       = [r.current_supply for r in self.rooms]
        avg_supply     = np.mean(supplies)
        fairness_score = max(0.0, 1.0 - float(
            np.std(supplies) / (avg_supply + 1e-5)
        )) if avg_supply > 0 else 0.0

        # FIX: battery_level — charge from solar, discharge during peak
        solar_now = self._get_solar(self.current_hour)
        if solar_now > 0.3:
            self.obs.battery_level = min(1.0, self.obs.battery_level + solar_now * 0.05)
        elif self._is_peak(self.current_hour):
            self.obs.battery_level = max(0.0, self.obs.battery_level - 0.03)

        # FIX: system_trust — decay on violations/high complaints, recover otherwise
        if violations > 0:
            self._system_trust *= (0.95 ** violations)
        elif total_complaints > 5:
            self._system_trust *= 0.98
        else:
            self._system_trust = min(1.0, self._system_trust * 1.02)
        self._system_trust = float(np.clip(self._system_trust, 0.0, 1.0))

        self.current_hour        += 1
        self.obs.time_of_day      = self.current_hour
        self.obs.carbon_rate      = self._get_carbon_rate(self.current_hour)
        self.obs.peak_hour        = self._is_peak(self.current_hour)
        self.obs.solar_output     = self._get_solar(self.current_hour)
        self.obs.system_trust     = self._system_trust  # FIX: sync to observation

        tariff                    = self._get_tariff(self.current_hour)
        hour_cost                 = float(np.clip(new_power * tariff, 0, 1000))
        self.obs.current_cost    += hour_cost

        self.obs.power_usage     = new_power
        self.obs.avg_temperature = float(np.mean([r.temperature for r in self.rooms]))
        self.obs.avg_occupancy   = sum(r.is_occupied for r in self.rooms) / self.num_rooms
        self.obs.complaint_level = total_complaints
        self.obs.fairness_score  = fairness_score
        self.obs.total_demand    = total_demand
        self.obs.demand_supply_ratio = total_demand / max(new_power, 0.1)
        self.obs.violations_this_step = violations
        self.obs.update_from_rooms(self.rooms)

        self._ep_steps            += 1
        self._ep_total_complaints += total_complaints
        self._ep_total_cost       += hour_cost
        self._ep_total_violations += violations
        self._ep_demand_sat_sum   += demand_sat

        power_saved  = max(0, prev_power - new_power)
        carbon_saved = power_saved * self.obs.carbon_rate
        tod_bonus    = time_of_day_bonus(self.current_hour, new_power)

        # FIX: use current complaint LEVEL as penalty signal, not delta
        # Delta allows gaming (stay high = 0 penalty); level doesn't
        complaint_level_penalty = total_complaints / max(self.num_rooms, 1)

        reward = calculate_reward(
            power_saved     = power_saved,
            complaint_delta = complaint_level_penalty,  # FIX: was raw delta
            carbon_saved    = carbon_saved,
            fairness_score  = fairness_score,
            power_usage     = new_power,
            min_power_floor = self.MIN_POWER_KW,
        ) + tod_bonus

        self.done = self.current_hour >= self.episode_hours

        info = {
            "hour"              : self.current_hour,
            "power"             : round(new_power, 3),
            "complaints"        : total_complaints,
            "cost"              : round(hour_cost, 4),
            "carbon_rate"       : self.obs.carbon_rate,
            "violations"        : violations,
            "demand_sat"        : round(demand_sat, 3),
            "fairness"          : round(fairness_score, 3),
            "peak_hour"         : self.obs.peak_hour,
            "solar"             : self.obs.solar_output,
            "system_trust"      : round(self._system_trust, 3),  # FIX: expose in info
            "battery_level"     : round(self.obs.battery_level, 3),  # FIX: expose in info
        }

        return self.obs.to_vector(), round(reward, 4), self.done, info

    def _get_carbon_rate(self, hour: int) -> float:
        if 9 <= hour <= 12 or 18 <= hour <= 22:
            return 0.82
        elif 0 <= hour <= 5:
            return 0.45
        return 0.63

    def _get_tariff(self, hour: int) -> float:
        if 9 <= hour <= 12 or 18 <= hour <= 22:
            return 8.5
        elif 0 <= hour <= 5:
            return 4.0
        return 6.0

    def _get_solar(self, hour: int) -> float:
        if 6 <= hour <= 18:
            return max(0.0, 1.0 - abs(hour - 12) / 7.0)
        return 0.0

    def _is_peak(self, hour: int) -> bool:
        return (9 <= hour <= 12) or (18 <= hour <= 22)

    def episode_stats(self) -> dict:
        steps = max(1, self._ep_steps)
        return {
            "demand_satisfaction" : self._ep_demand_sat_sum / steps,
            "violations"          : self._ep_total_violations,
            "total_cost"          : self._ep_total_cost,
            "total_complaints"    : self._ep_total_complaints,
            "system_trust"        : self._system_trust,
        }