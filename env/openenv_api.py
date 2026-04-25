from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from env.hostelgrid_env import HostelGridEnv
from env.action import Action as EnvAction, ACTION_MAP
from env.observation import Observation as EnvObservation
from env.reward_model import Reward as EnvReward

TASK_TO_MODE = {
    "task_easy": "easy",
    "task_medium": "medium",
    "task_hard": "hard",
}

BITS_TO_ACTION = {tuple(v): k for k, v in ACTION_MAP.items()}


class ObservationView:
    def __init__(self, obs: EnvObservation):
        self._obs = obs

    def model_dump(self) -> Dict[str, Any]:
        return self._obs.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return self._obs.to_dict()

    def __getattr__(self, name: str):
        return getattr(self._obs, name)


class RewardView:
    def __init__(self, reward: EnvReward, done: bool, info: Dict[str, Any]):
        self._reward = reward
        self.done = done
        self.info = info

    @property
    def value(self) -> float:
        return float(self._reward.total)

    @property
    def total(self) -> float:
        return float(self._reward.total)

    @property
    def normalized(self) -> float:
        return float(np.clip((self._reward.total + 2.0) / 3.0, 0.0, 1.0))

    @property
    def breakdown(self) -> Dict[str, Any]:
        return {
            "per_room": self._reward.per_room,
            "satisfied_rooms": self._reward.satisfied_rooms,
            "total_complaints": self._reward.total_complaints,
            "hp_satisfied": self._reward.hp_satisfied,
            "power_used": self._reward.power_used,
            "power_budget": self._reward.power_budget,
            "over_budget": self._reward.over_budget,
        }

    def model_dump(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "normalized": self.normalized,
            "done": self.done,
            "breakdown": self.breakdown,
            "info": self.info,
        }


@dataclass
class Action:
    room_actions: Optional[List[int]] = None
    action_id: Optional[int] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "room_actions": self.room_actions,
            "action_id": self.action_id,
        }

    def to_env_action(self, env: "HostelGridOpenEnv") -> EnvAction:
        if self.room_actions is not None:
            return EnvAction.from_list(self.room_actions)

        if self.action_id is not None:
            return EnvAction.from_list(_legacy_action_to_room_actions(env._env, self.action_id))

        raise ValueError("Action must provide either room_actions or action_id.")


def _bits_to_action_id(ac: int, fan: int, light: int) -> int:
    return BITS_TO_ACTION[(int(ac), int(fan), int(light))]


def _current_room_actions(env: HostelGridEnv) -> List[int]:
    actions = []
    for room in env.hostel.rooms:
        actions.append(_bits_to_action_id(room.ac, room.fan, room.light))
    return actions


def _legacy_action_to_room_actions(env: HostelGridEnv, action_id: int) -> List[int]:
    """
    Backward-compatibility shim for old 0..5 coarse actions.

    Canonical interface is room_actions[10] with values 0..7.
    This exists only so older app/demo paths do not crash immediately.
    """
    action_id = int(action_id)
    if action_id not in range(6):
        action_id = 5

    next_actions = _current_room_actions(env)

    for i, room in enumerate(env.hostel.rooms):
        ac = int(room.ac)
        fan = int(room.fan)
        light = int(room.light)

        if action_id == 0 and room.occupancy == 1:
            ac = 1
        elif action_id == 1 and room.occupancy == 1:
            ac = 0
        elif action_id == 2 and room.occupancy == 0:
            light = 0
        elif action_id == 3 and room.occupancy == 1:
            light = 1
        elif action_id == 4:
            if room.occupancy == 0:
                ac, fan, light = 0, 0, 0
            else:
                ac = 0
        elif action_id == 5:
            pass

        next_actions[i] = _bits_to_action_id(ac, fan, light)

    return next_actions


class HostelGridOpenEnv:
    """
    Thin OpenEnv wrapper around the current 10-room HostelGridEnv.

    Canonical action interface:
        Action(room_actions=[0..7] * 10)

    Backward-compatible legacy interface:
        Action(action_id=0..5)
    """

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        if task_id not in TASK_TO_MODE:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task_id = task_id
        self.mode = TASK_TO_MODE[task_id]
        self.seed = seed
        self._env = HostelGridEnv(mode=self.mode, seed=seed)
        self._obs: Optional[ObservationView] = None
        self._done = False
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self._total_reward = 0.0
        self._steps = 0
        self._over_budget_steps = 0
        self._metric_history = {
            "satisfaction": [],
            "efficiency": [],
            "hp_satisfaction": [],
            "complaint_control": [],
            "power_efficiency": [],
            "budget_ok": [],
        }

    def reset(self) -> ObservationView:
        obs = self._env.reset()
        self._obs = ObservationView(obs)
        self._done = False
        self._reset_episode_stats()
        return self._obs

    def step(self, action: Action | EnvAction | List[int]):
        if isinstance(action, list):
            env_action = EnvAction.from_list(action)
        elif isinstance(action, EnvAction):
            env_action = action
        elif isinstance(action, Action):
            env_action = action.to_env_action(self)
        else:
            raise TypeError("action must be Action, env.action.Action, or list[int].")

        obs, reward, done, info = self._env.step(env_action)
        self._obs = ObservationView(obs)
        self._done = done

        self._steps += 1
        self._total_reward += float(reward.total)
        if info.get("over_budget", False):
            self._over_budget_steps += 1

        self._update_metrics(obs, info)

        reward_view = RewardView(reward, done, info)
        return self._obs, reward_view, done, info

    def _update_metrics(self, obs: EnvObservation, info: Dict[str, Any]) -> None:
        occupied = max(obs.occupied_rooms, 1)
        satisfaction = info["satisfied"] / occupied

        empty_rooms = [r for r in obs.rooms if r.occupancy < 0.5]
        empty_waste = sum((r.ac + r.fan + r.light) for r in empty_rooms)
        max_empty_waste = max(3 * len(empty_rooms), 1)
        efficiency = 1.0 - (empty_waste / max_empty_waste)

        hp_rooms = [r for r in obs.rooms if r.priority > 0.8]
        hp_sat = sum(
            r.ac > 0.5 and r.fan > 0.5 and r.light > 0.5
            for r in hp_rooms
        ) / max(len(hp_rooms), 1)

        avg_complaint = float(np.mean([r.complaint_level for r in obs.rooms])) if obs.rooms else 0.0
        complaint_control = 1.0 - avg_complaint

        util = obs.power_used / max(obs.power_budget, 1.0)
        power_efficiency = 1.0 if util <= 1.0 else max(0.0, 2.0 - util)

        budget_ok = 1.0 if not info.get("over_budget", False) else 0.0

        self._metric_history["satisfaction"].append(float(np.clip(satisfaction, 0.0, 1.0)))
        self._metric_history["efficiency"].append(float(np.clip(efficiency, 0.0, 1.0)))
        self._metric_history["hp_satisfaction"].append(float(np.clip(hp_sat, 0.0, 1.0)))
        self._metric_history["complaint_control"].append(float(np.clip(complaint_control, 0.0, 1.0)))
        self._metric_history["power_efficiency"].append(float(np.clip(power_efficiency, 0.0, 1.0)))
        self._metric_history["budget_ok"].append(float(np.clip(budget_ok, 0.0, 1.0)))

    def _mean_metric(self, key: str) -> float:
        values = self._metric_history[key]
        return float(np.mean(values)) if values else 0.0

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "seed": self.seed,
            "step": self._env.current_step,
            "done": self._done,
            "observation": self._obs.model_dump() if self._obs is not None else None,
            "episode": {
                "total_reward": round(self._total_reward, 4),
                "score": self.score(),
                "avg_satisfaction": round(self._mean_metric("satisfaction"), 4),
                "avg_efficiency": round(self._mean_metric("efficiency"), 4),
                "avg_hp_satisfaction": round(self._mean_metric("hp_satisfaction"), 4),
                "avg_complaint_control": round(self._mean_metric("complaint_control"), 4),
                "avg_power_efficiency": round(self._mean_metric("power_efficiency"), 4),
                "over_budget_steps": self._over_budget_steps,
            },
        }

    def score(self) -> float:
        if self._steps == 0:
            return 0.0

        if self.mode == "easy":
            score = (
                0.70 * self._mean_metric("satisfaction") +
                0.30 * self._mean_metric("efficiency")
            )
        elif self.mode == "medium":
            score = (
                0.40 * self._mean_metric("hp_satisfaction") +
                0.35 * self._mean_metric("complaint_control") +
                0.25 * self._mean_metric("power_efficiency")
            )
        else:
            score = (
                0.50 * self._mean_metric("hp_satisfaction") +
                0.30 * self._mean_metric("complaint_control") +
                0.20 * self._mean_metric("budget_ok")
            )

        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def close(self) -> None:
        self._obs = None
        self._done = True
