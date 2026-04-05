# simulation/hostel.py

import random
import numpy as np
from simulation.student import Student


class Room:
    def __init__(self, room_id):
        self.room_id          = room_id
        self.is_occupied      = random.choice([True, False])
        self.temperature      = random.uniform(22.0, 30.0)
        self.ac_on            = self.is_occupied
        self.lights_on        = self.is_occupied
        self.power_consumption = 0.0

        # Task 1 — commitment
        self.has_approved_request = random.random() < 0.30
        self.min_required_supply  = 1.5 if self.has_approved_request else 0.0
        self.current_supply       = 0.0

        # Task 2 — misuse
        self.flagged              = False
        self.penalty_timer        = 0
        self.base_demand          = random.uniform(0.4, 2.5)
        self.current_demand       = 0.0

        # Task 3 — crisis
        self.in_exam_center       = random.random() < 0.2
        self.exam_mode            = False
        self.priority_level       = (
            "critical" if self.has_approved_request
            else "high" if self.in_exam_center
            else "normal"
        )
        self.power_cap            = 5.0

    def update(self):
        self.power_consumption = 0.0
        if self.ac_on:
            self.power_consumption += 1.5
        if self.lights_on:
            self.power_consumption += 0.1
        return self.power_consumption

    def check_violation(self):
        if self.has_approved_request:
            return self.current_supply < self.min_required_supply
        return False

    def __repr__(self):
        return (
            f"Room {self.room_id:02d} | "
            f"Occupied: {self.is_occupied} | "
            f"Temp: {self.temperature:.1f}C | "
            f"Priority: {self.priority_level} | "
            f"Power: {self.power_consumption:.2f}kW"
        )


class Hostel:
    def __init__(self, num_rooms=20):
        self.num_rooms       = num_rooms
        self.rooms           = [Room(i) for i in range(num_rooms)]
        self.total_power     = 0.0
        self.total_complaints = 0
        self.students        = [
            Student(i, i) for i in range(num_rooms)
            if self.rooms[i].is_occupied
        ]

    def update_all_rooms(self):
        self.total_power = sum(r.update() for r in self.rooms)
        return self.total_power

    def get_occupancy(self):
        return [1 if r.is_occupied else 0 for r in self.rooms]

    def get_temperatures(self):
        return [r.temperature for r in self.rooms]

    def get_total_power(self):
        return round(self.total_power, 3)

    def get_priority_rooms(self):
        return [r for r in self.rooms if r.has_approved_request]

    def get_exam_rooms(self):
        return [r for r in self.rooms if r.exam_mode]

    def get_flagged_rooms(self):
        return [r for r in self.rooms if r.flagged]

    def simulate_complaints(self):
        complaints = 0
        for room in self.rooms:
            if room.is_occupied:
                if not room.ac_on and room.temperature > 27:
                    complaints += 1
                if not room.lights_on:
                    complaints += 1
        self.total_complaints = complaints
        return complaints

    def get_fairness_score(self):
        """Used by graders — std/mean of supply across rooms"""
        supplies = [r.current_supply for r in self.rooms]
        avg      = np.mean(supplies)
        if avg == 0:
            return 0.0
        return float(np.std(supplies) / (avg + 1e-5))

    def summary(self):
        print(f"\n🏨 Hostel Summary")
        print(f"   Total Rooms    : {self.num_rooms}")
        print(f"   Occupied       : {sum(self.get_occupancy())}")
        print(f"   Priority Rooms : {len(self.get_priority_rooms())}")
        print(f"   Exam Rooms     : {len(self.get_exam_rooms())}")
        print(f"   Flagged Rooms  : {len(self.get_flagged_rooms())}")
        print(f"   Total Power    : {self.get_total_power()} kW")
        print(f"   Complaints     : {self.simulate_complaints()}")
        print(f"   Avg Temp       : "
              f"{sum(self.get_temperatures())/self.num_rooms:.1f}C")