# env/openenv_api.py
# Next-level OpenEnv API — all bugs fixed:
# - total_violations now properly tracked
# - score() no longer always gives +0.25 for violations
# - demand_satisfaction properly computed
# - trust tracked and exposed

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
    system_trust:         float    # NEW
    peak_hour:            bool     # NEW
    solar_output:         float    # NEW
    fairness_score:       float    # NEW
    demand_supply_ratio:  float    # NEW
    violations_this_step: int      # NEW


class Action(BaseModel):
    action_id: int   # 0-9 (expanded action space)


class Reward(BaseModel):
    value:     float
    breakdown: Dict[str, float]
    done:      bool
    info:      Dict[str, Any]


# ── OpenEnv-compliant Environment ─────────────────────────────

class HostelGridOpenEnv:
    """
    OpenEnv spec-compliant wrapper.
    
    Bugs fixed vs previous version:
    1. self._total_violations now incremented every step
    2. score() uses actual violation count (not always 0)
    3. demand_satisfaction computed from EpisodeState (not hardcoded)
    4. system_trust exposed in observation and state()
    5. Scoring thresholds tuned to real episode ranges
    """

    def __init__(self, task_id: str = "task_easy", num_rooms: int = 20):
        self.task_id       = task_id
        self.num_rooms     = num_rooms
        self._env          = HostelGridEnv(num_rooms=num_rooms, episode_hours=24)
        self._obs_vec      = None
        self._ep_state     = EpisodeState()

    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Reset environment — returns typed Observation."""
        self._obs_vec  = self._env.reset()
        self._ep_state = EpisodeState()
        return self._vec_to_obs(self._obs_vec)

    # ------------------------------------------------------------------
    def step(self, action: Action):
        """
        Apply action — returns (Observation, Reward, done, info).
        FIXED: violations tracked every step via EpisodeState.
        """
        obs_vec, reward_val, done, info = self._env.step(action.action_id)
        self._obs_vec = obs_vec

        # FIXED: update EpisodeState with all real values
        self._ep_state.update(
            reward       = reward_val,
            cost         = info.get("cost", 0),
            carbon       = info.get("carbon_rate", 0) * info.get("power", 0),
            complaints   = info.get("complaints", 0),
            violations   = info.get("violations", 0),   # FIXED: was never passed
            demand_sat   = info.get("demand_sat", 0.0), # FIXED: was never passed
            fairness     = info.get("fairness", 1.0),
            peak_violation = info.get("peak_hour", False),
            hour         = info.get("hour", 0),
        )

        observation = self._vec_to_obs(obs_vec)
        reward_obj  = Reward(
            value    = reward_val,
            breakdown = {
                "power"      : info.get("power", 0),
                "complaints" : info.get("complaints", 0),
                "cost"       : info.get("cost", 0),
                "violations" : info.get("violations", 0),
                "demand_sat" : info.get("demand_sat", 0),
                "fairness"   : info.get("fairness", 1.0),
            },
            done = done,
            info = info,
        )

        return observation, reward_obj, done, info

    # ------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Full current state — OpenEnv spec."""
        ep = self._ep_state.summary()
        return {
            "task_id"             : self.task_id,
            "step"                : self._ep_state.steps,
            "done"                : self._ep_state.steps >= 24,
            "observation"         : self._vec_to_obs(
                                        self._obs_vec
                                    ).model_dump() if self._obs_vec is not None else {},
            # FIXED: all these now actually have real values
            "total_reward"        : ep["total_reward"],
            "total_cost"          : ep["total_cost"],
            "total_complaints"    : ep["total_complaints"],
            "total_violations"    : ep["total_violations"],     # FIXED
            "demand_satisfaction" : ep["demand_satisfaction"],  # FIXED
            "system_trust"        : ep["system_trust"],         # FIXED
            "avg_fairness"        : ep["avg_fairness"],
            "collapsed"           : ep["collapsed"],
        }

    # ------------------------------------------------------------------
    def score(self) -> float:
        """
        Returns normalized score 0.0-1.0 for current episode.
        FIXED: scoring uses actual tracked values, not hardcoded assumptions.
        The old version always gave +0.25 for violations because
        self._total_violations was never updated. Now fixed.
        """
        ep = self._ep_state.summary()

        if self.task_id == "task_easy":
            return self._score_easy(ep)
        elif self.task_id == "task_medium":
            return self._score_medium(ep)
        elif self.task_id == "task_hard":
            return self._score_hard(ep)
        return 0.0

    # ------------------------------------------------------------------
    def _score_easy(self, ep: dict) -> float:
        """
        Task 1 scoring — FIXED violation check.
        Old bug: total_violations was always 0 → always got +0.25.
        Now uses real violation count.
        """
        score = 0.0

        # Reward quality (25%)
        tr = ep["total_reward"]
        if tr > 3.0:   score += 0.25
        elif tr > 0:   score += 0.10

        # Cost (25%)
        c = ep["total_cost"]
        if c < 1200:   score += 0.25
        elif c < 1500: score += 0.10

        # Complaints (25%)
        cp = ep["total_complaints"]
        if cp < 40:    score += 0.25
        elif cp < 70:  score += 0.10

        # Violations — FIXED: now uses real count
        v = ep["total_violations"]
        if v == 0:     score += 0.25
        elif v < 10:   score += 0.10
        # else: 0 points (old bug gave 0.25 always)

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    def _score_medium(self, ep: dict) -> float:
        """Task 2 scoring — fairness + misuse + violations."""
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

        # Violations — FIXED
        v = ep["total_violations"]
        if v == 0:     score += 0.20
        elif v < 15:   score += 0.10

        # Fairness — new factor
        f = ep["avg_fairness"]
        if f > 0.7:    score += 0.20
        elif f > 0.5:  score += 0.10

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    def _score_hard(self, ep: dict) -> float:
        """Task 3 scoring — full crisis metrics."""
        score = 0.0

        # Demand satisfaction — FIXED: now actually tracked
        ds = ep["demand_satisfaction"]
        if ds > 0.85:  score += 0.20
        elif ds > 0.70: score += 0.10

        # Violations — FIXED
        v = ep["total_violations"]
        if v < 15:     score += 0.20
        elif v < 30:   score += 0.10

        # Cost
        c = ep["total_cost"]
        if c < 1500:   score += 0.15
        elif c < 2000: score += 0.08

        # Complaints
        cp = ep["total_complaints"]
        if cp < 50:    score += 0.15
        elif cp < 80:  score += 0.08

        # System trust — FIXED: now actually tracked
        st = ep["system_trust"]
        if st > 0.8:   score += 0.15
        elif st > 0.6: score += 0.08

        # No collapse bonus
        if not ep["collapsed"]:
            score += 0.15

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    def _vec_to_obs(self, vec: np.ndarray) -> Observation:
        """Convert numpy vector to typed Observation."""
        # Base 14 features from new observation space
        # Handle both old (7) and new (14) observation sizes
        n = len(vec)
        return Observation(
            power_usage          = float(vec[0]) * 20.0,   # denormalize
            avg_temperature      = float(vec[1]) * 40.0,
            avg_occupancy        = float(vec[2]),
            complaint_level      = int(float(vec[3]) * 20),
            time_of_day          = int(float(vec[4]) * 23),
            carbon_rate          = float(vec[5]),
            current_cost         = float(vec[6]) * 1000.0,
            system_trust         = float(vec[7])  if n > 7  else 1.0,
            peak_hour            = bool(vec[8] > 0.5) if n > 8  else False,
            solar_output         = float(vec[9])  if n > 9  else 0.0,
            fairness_score       = float(vec[13]) if n > 13 else 1.0,
            demand_supply_ratio  = float(vec[12]) * 3.0 if n > 12 else 1.0,
            violations_this_step = int(self._ep_state.violation_history[-1])
                                   if self._ep_state.violation_history else 0,
        )