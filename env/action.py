"""
env/action.py
Typed Action model — Meta OpenEnv compliant.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing      import List, Dict
import random as _random

ACTION_MAP: Dict[int, List[int]] = {
    0: [0,0,0], 1: [1,0,0], 2: [0,1,0], 3: [0,0,1],
    4: [1,1,0], 5: [1,0,1], 6: [0,1,1], 7: [1,1,1],
}

ACTION_NAMES: Dict[int, str] = {
    0: "all_off",  1: "ac_only",   2: "fan_only",  3: "light_only",
    4: "ac_fan",   5: "ac_light",  6: "fan_light",  7: "all_on",
}

NUM_ROOMS        = 10
NUM_ROOM_ACTIONS = 8


@dataclass
class Action:
    room_actions: List[int]

    def __post_init__(self):
        assert len(self.room_actions) == NUM_ROOMS, \
            f"Need {NUM_ROOMS} actions, got {len(self.room_actions)}"
        for i, a in enumerate(self.room_actions):
            assert 0 <= a <= 7, \
                f"Room {i} action {a} invalid — must be 0-7"

    @classmethod
    def from_list(cls, actions: List[int]) -> "Action":
        return cls(room_actions=list(actions))

    @classmethod
    def all_off(cls) -> "Action":
        return cls(room_actions=[0]*NUM_ROOMS)

    @classmethod
    def all_on(cls) -> "Action":
        return cls(room_actions=[7]*NUM_ROOMS)

    @classmethod
    def random(cls) -> "Action":
        return cls(room_actions=[_random.randrange(8)
                                  for _ in range(NUM_ROOMS)])

    def to_list(self) -> List[int]:
        return self.room_actions

    def to_dict(self) -> dict:
        return {
            "room_actions": self.room_actions,
            "action_names": [ACTION_NAMES[a] for a in self.room_actions],
        }