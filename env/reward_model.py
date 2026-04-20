from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Reward:
    total:            float
    per_room:         List[float]
    satisfied_rooms:  int
    total_complaints: int
    hp_satisfied:     int
    power_used:       float
    power_budget:     float
    over_budget:      bool
    step:             int

    @property
    def normalized(self) -> float:
        return (self.total + 2.0) / 3.0

    @property
    def power_utilization(self) -> float:
        return min(self.power_used / self.power_budget, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_hostel(cls, total, per_room, hostel, step):
        info = hostel.get_info()
        return cls(
            total            = total,
            per_room         = per_room,
            satisfied_rooms  = info["satisfied"],
            total_complaints = info["complaints"],
            hp_satisfied     = info["hp_satisfied"],
            power_used       = info["power_used"],
            power_budget     = info["power_budget"],
            over_budget      = info["over_budget"],
            step             = step,
        )