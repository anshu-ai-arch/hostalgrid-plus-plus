# simulation/student.py

import random
import numpy as np
from collections import deque


class Student:
    """
    Models individual student behavior including:
    - Comfort threshold (personal preference)
    - Complaint generation
    - Misuse behavior (Task 2 & 3)
    - Exam mode sensitivity (Task 3)
    """
    def __init__(self, student_id, room_id):
        self.student_id        = student_id
        self.room_id           = room_id
        self.comfort_threshold = random.uniform(24.0, 28.0)
        self.complaint_count   = 0
        self.is_selfish        = random.random() < 0.15   # 15% selfish
        self.in_exam_mode      = False
        self.demand_history    = deque(maxlen=5)

    def check_comfort(self, room_temperature, ac_on, lights_on):
        """
        Returns True if student is complaining this step.
        Used by hostel.simulate_complaints()
        """
        complaining = False

        if room_temperature > self.comfort_threshold and not ac_on:
            self.complaint_count += 1
            complaining = True

        if not lights_on:
            self.complaint_count += 1
            complaining = True

        # Exam mode — more sensitive
        if self.in_exam_mode and room_temperature > self.comfort_threshold - 1:
            self.complaint_count += 1
            complaining = True

        return complaining

    def generate_demand(self, base_demand):
        """
        Returns actual demand this step.
        Selfish students randomly spike usage (Task 2 & 3).
        """
        if self.is_selfish and random.random() < 0.20:
            # Sudden spike — induction stove, high AC etc.
            demand = base_demand * random.uniform(2.0, 3.5)
        elif self.in_exam_mode:
            # Exam mode — steady moderate demand
            demand = base_demand * random.uniform(1.2, 1.5)
        else:
            demand = base_demand * random.uniform(0.7, 1.3)

        self.demand_history.append(demand)
        return round(demand, 3)

    def is_spiking(self):
        """
        Detect if this student is causing a demand spike.
        Used by Task 2 misuse detection.
        """
        if len(self.demand_history) < 2:
            return False
        recent_avg = np.mean(list(self.demand_history)[:-1])
        current    = list(self.demand_history)[-1]
        return current > recent_avg * 1.5 and current > 2.5

    def __repr__(self):
        return (
            f"Student {self.student_id:02d} | "
            f"Room {self.room_id:02d} | "
            f"Threshold: {self.comfort_threshold:.1f}C | "
            f"Selfish: {self.is_selfish} | "
            f"Complaints: {self.complaint_count}"
        )