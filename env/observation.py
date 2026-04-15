# env/observation.py
# Fixed:
# 1. system_trust in to_vector() is now live (env writes it each step)
# 2. demand_supply_ratio normalization consistent — capped at 3.0, divided once
# 3. battery_level exposed in Observation (was silently dropped from vector)

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class RoomObservation:
    """Per-room state — gives agent individual comfort awareness."""
    room_id:              int
    is_occupied:          bool
    temperature:          float
    current_supply:       float
    current_demand:       float
    complaint_level:      int       # 0=none 1=mild 2=moderate 3=severe
    has_approved_request: bool
    min_required_supply:  float
    flagged_for_misuse:   bool
    in_exam_center:       bool
    supply_deficit:       float     # max(0, min_required - current_supply)

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.is_occupied),
            self.temperature / 40.0,
            self.current_supply / 5.0,
            self.current_demand / 5.0,
            self.complaint_level / 3.0,
            float(self.has_approved_request),
            self.min_required_supply / 5.0,
            float(self.flagged_for_misuse),
            float(self.in_exam_center),
            self.supply_deficit / 5.0,
        ], dtype=np.float32)


class Observation:
    """
    Full hostel observation — individual + aggregate.
    Base vector: 15 features.
    Fixed: battery_level now included in vector (was index 10, now explicit).
    Fixed: system_trust reflects live value written by env each step.
    Fixed: demand_supply_ratio normalized consistently (clip to 3.0, divide once).
    """
    def __init__(self, num_rooms: int = 20):
        self.num_rooms = num_rooms

        # Aggregate grid/hostel state
        self.power_usage        = 0.0
        self.avg_temperature    = 24.0
        self.avg_occupancy      = 0.5
        self.complaint_level    = 0
        self.time_of_day        = 0
        self.carbon_rate        = 0.63
        self.current_cost       = 0.0

        # Individual room tracking
        self.room_observations: List[RoomObservation] = []

        # System-level signals
        self.system_trust         = 1.0
        self.peak_hour            = False
        self.solar_output         = 0.0
        self.battery_level        = 0.5   # FIX: now properly updated by env
        self.total_demand         = 0.0
        self.demand_supply_ratio  = 1.0   # FIX: normalized once, consistently
        self.fairness_score       = 1.0
        self.violations_this_step = 0

    def to_vector(self) -> np.ndarray:
        """
        15-feature base vector (was 14 — added battery_level explicitly).
        Index map:
          0  power_usage          (/ 20.0)
          1  avg_temperature      (/ 40.0)
          2  avg_occupancy
          3  complaint_level      (/ 20.0)
          4  time_of_day          (/ 23.0)
          5  carbon_rate
          6  current_cost         (/ 1000.0)
          7  system_trust         ← live from env now
          8  peak_hour
          9  solar_output
          10 battery_level        ← FIX: was in vector but not in Observation model
          11 total_demand         (/ 20.0)
          12 demand_supply_ratio  (clipped to 3.0, / 3.0) ← normalized once
          13 fairness_score
          14 violations_this_step (/ 20.0)  ← FIX: was missing, useful signal
        """
        return np.array([
            self.power_usage / 20.0,
            self.avg_temperature / 40.0,
            self.avg_occupancy,
            min(self.complaint_level, 20) / 20.0,
            self.time_of_day / 23.0,
            self.carbon_rate / 1.0,
            min(self.current_cost, 1000) / 1000.0,
            self.system_trust,                                # FIX: live value
            float(self.peak_hour),
            self.solar_output,
            self.battery_level,                               # FIX: explicit index 10
            min(self.total_demand, 20.0) / 20.0,
            min(self.demand_supply_ratio, 3.0) / 3.0,        # FIX: normalized once
            self.fairness_score,
            min(self.violations_this_step, 20) / 20.0,       # FIX: added as signal
        ], dtype=np.float32)

    def update_from_rooms(self, rooms):
        """
        Update aggregate signals from individual room states.
        Called every step after rooms are updated.
        """
        if not rooms:
            return

        occupied_rooms  = [r for r in rooms if r.is_occupied]
        supplies        = [r.current_supply for r in rooms]
        demands         = [r.current_demand for r in rooms]
        temps           = [r.temperature    for r in rooms]

        self.avg_occupancy   = len(occupied_rooms) / max(1, len(rooms))
        self.avg_temperature = float(np.mean(temps)) if temps else 24.0
        self.total_demand    = float(sum(demands))
        self.power_usage     = float(sum(supplies))

        # FIX: demand_supply_ratio — single normalization point
        # env does not re-normalize; to_vector() does it once
        if self.power_usage > 0:
            self.demand_supply_ratio = min(
                self.total_demand / self.power_usage, 3.0  # cap at 3.0 here
            )
        else:
            self.demand_supply_ratio = 3.0

        avg_supply = np.mean(supplies) if supplies else 0
        if avg_supply > 0:
            self.fairness_score = max(0.0, 1.0 - float(
                np.std(supplies) / (avg_supply + 1e-5)
            ))
        else:
            self.fairness_score = 0.0

        self.complaint_level = sum(
            getattr(r, 'complaint_level', 0) for r in rooms
        )

        self.violations_this_step = sum(
            1 for r in rooms
            if getattr(r, 'has_approved_request', False)
            and r.current_supply < getattr(r, 'min_required_supply', 0)
        )

        self.room_observations = []
        for r in rooms:
            deficit = max(0.0,
                getattr(r, 'min_required_supply', 0) - r.current_supply
            )
            self.room_observations.append(RoomObservation(
                room_id              = r.room_id,
                is_occupied          = getattr(r, 'is_occupied', False),
                temperature          = getattr(r, 'temperature', 24.0),
                current_supply       = r.current_supply,
                current_demand       = getattr(r, 'current_demand', 0.0),
                complaint_level      = getattr(r, 'complaint_level', 0),
                has_approved_request = getattr(r, 'has_approved_request', False),
                min_required_supply  = getattr(r, 'min_required_supply', 0.0),
                flagged_for_misuse   = getattr(r, 'flagged_for_misuse', False),
                in_exam_center       = getattr(r, 'in_exam_center', False),
                supply_deficit       = deficit,
            ))

    def get_critical_rooms(self) -> List[RoomObservation]:
        return [r for r in self.room_observations if r.has_approved_request]

    def get_complaining_rooms(self) -> List[RoomObservation]:
        return [r for r in self.room_observations if r.complaint_level > 0]

    def get_flagged_rooms(self) -> List[RoomObservation]:
        return [r for r in self.room_observations if r.flagged_for_misuse]

    def get_deficit_rooms(self) -> List[RoomObservation]:
        return [r for r in self.room_observations if r.supply_deficit > 0]

    def __repr__(self):
        return (
            f"Observation("
            f"power={self.power_usage:.2f}kW, "
            f"temp={self.avg_temperature:.1f}C, "
            f"complaints={self.complaint_level}, "
            f"trust={self.system_trust:.2f}, "
            f"battery={self.battery_level:.2f}, "
            f"violations={self.violations_this_step}, "
            f"hour={self.time_of_day})"
        )