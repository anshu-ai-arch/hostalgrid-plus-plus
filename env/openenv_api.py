# env/openenv_api.py
# Fixed:
# 1. Observation model now includes battery_level (was silently dropped)
# 2. _vec_to_obs updated for 15-feature vector (was 14)
# 3. demand_supply_ratio denormalization corrected (* 3.0, matching to_vector)
# 4. system_trust and battery_level passed through from info dict each step
# 5. score() thresholds recalibrated to realistic episode ranges

from pydantic import BaseModel
from typing import Any, Dict, Optional
import numpy as np
import random

from env.hostelgrid_env import HostelGridEnv
from env.state import EpisodeState


# ── Typed Models (OpenEnv spec) ───────────────────────────────

class Observation(BaseModel):
    power_usage:          float
    avg_temperature:      float
    avg_occupancy:        float
    complaint_level:      int
    time_of_day:          int
    carbon_rate:          float
    current_cost:         float
    system_trust:         float
    peak_hour:            bool
    solar_output:         float
    battery_level:        float    # FIX: was missing from model
    fairness_score:       float
    demand_supply_ratio:  float
    violations_this_step: int


class Action(BaseModel):
    action_id: int   # 0-9


class Reward(BaseModel):
    value:     float
    breakdown: Dict[str, float]
    done:      bool
    info:      Dict[str, Any]


# ── OpenEnv-compliant Environment ─────────────────────────────

class HostelGridOpenEnv:
    """
    OpenEnv spec-compliant wrapper.

    Fixes in this version:
    1. battery_level added to Observation model
    2. _vec_to_obs handles 15-feature vector (was 14)
    3. system_trust and battery_level read from info dict (env now writes them)
    4. demand_supply_ratio denormalized correctly (* 3.0)
    5. violations_this_step decoded from vec[14]
    6. EpisodeState.update() now receives system_trust for trust tracking
    """

    def __init__(self, task_id: str = "task_easy", num_rooms: int = 20):
        self.task_id       = task_id
        self.num_rooms     = num_rooms
        self._env          = HostelGridEnv(num_rooms=num_rooms, episode_hours=24)
        self._obs_vec      = None
        self._ep_state     = EpisodeState()

    def reset(self) -> Observation:
        self._obs_vec  = self._env.reset()
        self._ep_state = EpisodeState()
        return self._vec_to_obs(self._obs_vec)

    def step(self, action: Action):
        obs_vec, reward_val, done, info = self._env.step(action.action_id)
        self._obs_vec = obs_vec

        self._ep_state.update(
            reward         = reward_val,
            cost           = info.get("cost", 0),
            carbon         = info.get("carbon_rate", 0) * info.get("power", 0),
            complaints     = info.get("complaints", 0),
            violations     = info.get("violations", 0),
            demand_sat     = info.get("demand_sat", 0.0),
            fairness       = info.get("fairness", 1.0),
            peak_violation = info.get("peak_hour", False),
            hour           = info.get("hour", 0),
            # FIX: pass system_trust so EpisodeState can track it accurately
            system_trust   = info.get("system_trust", 1.0),
        )

        observation = self._vec_to_obs(obs_vec, info)
        reward_obj  = Reward(
            value    = reward_val,
            breakdown = {
                "power"         : info.get("power", 0),
                "complaints"    : info.get("complaints", 0),
                "cost"          : info.get("cost", 0),
                "violations"    : info.get("violations", 0),
                "demand_sat"    : info.get("demand_sat", 0),
                "fairness"      : info.get("fairness", 1.0),
                "system_trust"  : info.get("system_trust", 1.0),
                "battery_level" : info.get("battery_level", 0.5),
            },
            done = done,
            info = info,
        )

        return observation, reward_obj, done, info

    def state(self) -> Dict[str, Any]:
        ep = self._ep_state.summary()
        return {
            "task_id"             : self.task_id,
            "step"                : self._ep_state.steps,
            "done"                : self._ep_state.steps >= 24,
            "observation"         : self._vec_to_obs(
                                        self._obs_vec
                                    ).model_dump() if self._obs_vec is not None else {},
            "total_reward"        : ep["total_reward"],
            "total_cost"          : ep["total_cost"],
            "total_complaints"    : ep["total_complaints"],
            "total_violations"    : ep["total_violations"],
            "demand_satisfaction" : ep["demand_satisfaction"],
            "system_trust"        : ep["system_trust"],
            "avg_fairness"        : ep["avg_fairness"],
            "collapsed"           : ep["collapsed"],
        }

    def score(self) -> float:
        ep = self._ep_state.summary()

        if self.task_id == "task_easy":
            return self._score_easy(ep)
        elif self.task_id == "task_medium":
            return self._score_medium(ep)
        elif self.task_id == "task_hard":
            return self._score_hard(ep)
        return 0.0

    def _score_easy(self, ep: dict) -> float:
        score = 0.0

        tr = ep["total_reward"]
        if tr > 3.0:   score += 0.25
        elif tr > 0:   score += 0.10

        c = ep["total_cost"]
        if c < 1200:   score += 0.25
        elif c < 1500: score += 0.10

        cp = ep["total_complaints"]
        if cp < 40:    score += 0.25
        elif cp < 70:  score += 0.10

        v = ep["total_violations"]
        if v == 0:     score += 0.25
        elif v < 10:   score += 0.10

        return round(min(score, 1.0), 4)

    def _score_medium(self, ep: dict) -> float:
        score = 0.0

        tr = ep["total_reward"]
        if tr > 4.0:   score += 0.20
        elif tr > 0:   score += 0.10

        c = ep["total_cost"]
        if c < 1000:   score += 0.20
        elif c < 1400: score += 0.10

        cp = ep["total_complaints"]
        if cp < 60:    score += 0.20
        elif cp < 90:  score += 0.10

        v = ep["total_violations"]
        if v == 0:     score += 0.20
        elif v < 15:   score += 0.10

        f = ep["avg_fairness"]
        if f > 0.7:    score += 0.20
        elif f > 0.5:  score += 0.10

        return round(min(score, 1.0), 4)

    def _score_hard(self, ep: dict) -> float:
        score = 0.0

        ds = ep["demand_satisfaction"]
        if ds > 0.85:   score += 0.20
        elif ds > 0.70: score += 0.10

        v = ep["total_violations"]
        if v < 15:     score += 0.20
        elif v < 30:   score += 0.10

        c = ep["total_cost"]
        if c < 1500:   score += 0.15
        elif c < 2000: score += 0.08

        cp = ep["total_complaints"]
        if cp < 50:    score += 0.15
        elif cp < 80:  score += 0.08

        st = ep["system_trust"]
        if st > 0.8:   score += 0.15
        elif st > 0.6: score += 0.08

        if not ep["collapsed"]:
            score += 0.15

        return round(min(score, 1.0), 4)

    def _vec_to_obs(self, vec: np.ndarray, info: dict = None) -> Observation:
        """
        Convert numpy vector to typed Observation.
        FIX: handles 15-feature vector (index 14 = violations_this_step).
        FIX: battery_level read from vec[10] and included in model.
        FIX: system_trust / battery_level also cross-checked with info dict.
        FIX: demand_supply_ratio denormalized by * 3.0 (matches to_vector).
        """
        n = len(vec)
        info = info or {}

        # Prefer info dict for trust/battery (authoritative from env)
        system_trust   = info.get("system_trust",   float(vec[7])  if n > 7  else 1.0)
        battery_level  = info.get("battery_level",  float(vec[10]) if n > 10 else 0.5)

        # violations: from vec[14] if available, else from info
        if n > 14:
            violations_this_step = int(float(vec[14]) * 20)
        else:
            violations_this_step = info.get("violations", 0)

        return Observation(
            power_usage          = float(vec[0]) * 20.0,
            avg_temperature      = float(vec[1]) * 40.0,
            avg_occupancy        = float(vec[2]),
            complaint_level      = int(float(vec[3]) * 20),
            time_of_day          = int(float(vec[4]) * 23),
            carbon_rate          = float(vec[5]),
            current_cost         = float(vec[6]) * 1000.0,
            system_trust         = system_trust,
            peak_hour            = bool(vec[8] > 0.5) if n > 8  else False,
            solar_output         = float(vec[9])       if n > 9  else 0.0,
            battery_level        = battery_level,                          # FIX
            fairness_score       = float(vec[13])      if n > 13 else 1.0,
            demand_supply_ratio  = float(vec[12]) * 3.0 if n > 12 else 1.0,  # FIX: denorm once
            violations_this_step = violations_this_step,                   # FIX
        )