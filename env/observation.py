"""
env/observation.py
Typed Observation model — Meta OpenEnv compliant.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing      import List, Dict, Any


@dataclass
class RoomState:
    room_id:             int
    occupancy:           float
    priority:            float
    complaint_level:     float
    consecutive_ignored: float
    ac:                  float
    fan:                 float
    light:               float
    peak_hour:           float
    power_ratio:         float

    def to_vector(self) -> List[float]:
        return [
            self.occupancy, self.priority, self.complaint_level,
            self.consecutive_ignored, self.ac, self.fan, self.light,
            self.peak_hour, self.power_ratio,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    step:             int
    rooms:            List[RoomState]
    hour:             int
    heatwave:         bool
    power_used:       float
    power_budget:     float
    occupied_rooms:   int
    total_complaints: int
    mode:             str

    def to_vectors(self) -> List[List[float]]:
        return [r.to_vector() for r in self.rooms]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":             self.step,
            "rooms":            [r.to_dict() for r in self.rooms],
            "hour":             self.hour,
            "heatwave":         self.heatwave,
            "power_used":       self.power_used,
            "power_budget":     self.power_budget,
            "occupied_rooms":   self.occupied_rooms,
            "total_complaints": self.total_complaints,
            "mode":             self.mode,
        }

    @classmethod
    def from_hostel(cls, hostel, step: int) -> "Observation":
        from simulation.hostel import MAX_COMPLAINT, PEAK_HOURS
        power_ratio = hostel.power_ratio()
        hour        = hostel.hour
        rooms = []
        for i, r in enumerate(hostel.rooms):
            rooms.append(RoomState(
                room_id             = i,
                occupancy           = float(r.occupancy),
                priority            = float(r.priority) / 3.0,
                complaint_level     = float(r.complaint) / MAX_COMPLAINT,
                consecutive_ignored = min(float(r.consecutive_ignored)/10.0, 1.0),
                ac                  = float(r.ac),
                fan                 = float(r.fan),
                light               = float(r.light),
                peak_hour           = float(hour in PEAK_HOURS),
                power_ratio         = power_ratio,
            ))
        return cls(
            step             = step,
            rooms            = rooms,
            hour             = hour,
            heatwave         = hostel.heatwave,
            power_used       = hostel.total_power(),
            power_budget     = float(hostel.power_budget),
            occupied_rooms   = sum(r.occupancy for r in hostel.rooms),
            total_complaints = sum(r.complaint for r in hostel.rooms),
            mode             = hostel.mode,
        )