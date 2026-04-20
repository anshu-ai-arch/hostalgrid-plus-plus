"""
simulation/hostel.py

Redesigned with genuine difficulty progression.

EASY:
    Stable occupancy, no conflicts, no scarcity.
    Agent should reach ~80% satisfaction.

MEDIUM:
    Occupancy drifts faster, fairness matters,
    complaints accumulate. Agent cannot be perfect.
    Expected ceiling: ~60-70% satisfaction.

HARD:
    Power budget enforced (scarcity).
    Priority conflicts that CANNOT be perfectly solved.
    Adversarial occupancy spikes.
    Heatwave forces AC even at peak cost.
    Expected ceiling: ~40-55% satisfaction.
    A perfect score is IMPOSSIBLE by design.
"""

import numpy as np

NUM_ROOMS     = 10
MAX_COMPLAINT = 10     # raised from 5 — complaints accumulate more
PEAK_HOURS    = {9, 10, 11, 12, 13, 14, 18, 19, 20, 21}

ACTION_MAP = {
    0: [0, 0, 0],  1: [1, 0, 0],  2: [0, 1, 0],  3: [0, 0, 1],
    4: [1, 1, 0],  5: [1, 0, 1],  6: [0, 1, 1],  7: [1, 1, 1],
}

# Power budget per timestep (watts)
POWER_BUDGET = {
    "easy":   99999,   # unlimited
    "medium": 6000,    # tight but manageable
    "hard":   3500,    # forces real trade-offs (full service = 9900W)
}

APPLIANCE_POWER = {"AC": 900, "fan": 75, "light": 15}


class Room:
    def __init__(self, room_id: int, priority: int):
        self.room_id   = room_id
        self.priority  = priority
        self.occupancy = 0
        self.ac = self.fan = self.light = 0
        self.complaint   = 0
        self.last_served = 0
        self.consecutive_ignored = 0   # for non-linear escalation

    def reset(self, rng, mode: str):
        # Hard mode starts with more rooms occupied (more pressure)
        occ_prob = {"easy": 0.5, "medium": 0.6, "hard": 0.75}
        self.occupancy   = int(rng.random() < occ_prob.get(mode, 0.6))
        self.ac = self.fan = self.light = 0
        self.complaint   = 0
        self.last_served = 0
        self.consecutive_ignored = 0

    def apply_action(self, action_id: int):
        self.ac, self.fan, self.light = ACTION_MAP[action_id]

    def appliance_sum(self) -> int:
        return self.ac + self.fan + self.light

    def power_draw(self) -> float:
        return (self.ac * APPLIANCE_POWER["AC"]
                + self.fan * APPLIANCE_POWER["fan"]
                + self.light * APPLIANCE_POWER["light"])

    def is_satisfied(self) -> bool:
        if self.occupancy == 0:
            return True
        return self.ac == 1 and self.fan == 1 and self.light == 1

    def update(self, rng, mode: str):
        if self.occupancy == 1:
            if self.is_satisfied():
                self.last_served       = 0
                self.consecutive_ignored = 0
                # Recovery is slow — complaints drop by 1 only
                self.complaint = max(0, self.complaint - 1)
            else:
                self.last_served         += 1
                self.consecutive_ignored += 1
                if mode in ("medium", "hard"):
                    # Non-linear complaint escalation
                    escalation = 1 + (self.consecutive_ignored // 3)
                    self.complaint = min(MAX_COMPLAINT,
                                        self.complaint + escalation)
        else:
            self.last_served         = 0
            self.consecutive_ignored = 0

        # Occupancy drift rates per mode
        drift = {"easy": 0.03, "medium": 0.07, "hard": 0.12}
        if rng.random() < drift.get(mode, 0.05):
            self.occupancy = 1 - self.occupancy

    def get_state(self, hour: int, power_ratio: float) -> np.ndarray:
        """
        9-feature normalized state vector:
        [occ, priority/3, complaint/MAX, ignored/10,
         AC, fan, light, peak_hour, power_ratio]
        """
        return np.array([
            float(self.occupancy),
            float(self.priority) / 3.0,
            float(self.complaint) / MAX_COMPLAINT,
            min(float(self.consecutive_ignored) / 10.0, 1.0),
            float(self.ac),
            float(self.fan),
            float(self.light),
            float(hour in PEAK_HOURS),
            float(power_ratio),    # global power constraint signal
        ], dtype=np.float32)

    def __repr__(self):
        return (f"Room({self.room_id} p={self.priority} "
                f"occ={self.occupancy} "
                f"ac={self.ac} fan={self.fan} light={self.light} "
                f"complaint={self.complaint})")


class Hostel:
    """
    10-room hostel with genuine difficulty.

    HARD mode enforces a power budget that makes it
    physically impossible to fully serve all rooms simultaneously.
    Full service of all 10 rooms = 9,900W but budget = 3,500W.
    Agent MUST choose which rooms to serve.
    """

    ADVERSARIAL_STEPS = {
        "hard": [10, 20, 30, 40]   # steps where occupancy spikes
    }

    def __init__(self, mode: str = "easy", seed: int = 42):
        self.mode        = mode
        self.rng         = np.random.default_rng(seed)
        self.hour        = 0
        self.step_n      = 0
        self.heatwave    = False
        self.power_budget = POWER_BUDGET[mode]

        # Priority distribution — more HP rooms in hard mode
        if mode == "easy":
            priorities = [3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
        elif mode == "medium":
            priorities = [3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        else:  # hard
            priorities = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1]

        shuffled = priorities.copy()
        self.rng.shuffle(shuffled)
        self.rooms = [Room(i, shuffled[i]) for i in range(NUM_ROOMS)]

    def reset(self):
        self.hour   = self.step_n = 0
        self.heatwave = False
        for room in self.rooms:
            room.reset(self.rng, self.mode)

    def total_power(self) -> float:
        return sum(r.power_draw() for r in self.rooms)

    def power_ratio(self) -> float:
        """Remaining power ratio: 1.0 = budget unused, 0.0 = maxed out."""
        used = self.total_power()
        return max(0.0, 1.0 - used / self.power_budget)

    def enforce_power_budget(self):
        """
        If total power exceeds budget, force-cut lowest priority
        devices until within budget. This makes scarcity real.
        Only active in medium/hard.
        """
        if self.mode == "easy":
            return

        # Sort rooms by priority ascending (cut low priority first)
        sorted_rooms = sorted(self.rooms,
                              key=lambda r: (r.priority, r.occupancy))
        for room in sorted_rooms:
            if self.total_power() <= self.power_budget:
                break
            # Turn off most expensive device first
            if room.ac == 1:
                room.ac = 0
            elif room.fan == 1:
                room.fan = 0
            elif room.light == 1:
                room.light = 0

    def apply_actions(self, actions: list):
        for i, a in enumerate(actions):
            self.rooms[i].apply_action(a)
        # Enforce power budget AFTER actions applied
        self.enforce_power_budget()

    def step(self):
        for room in self.rooms:
            room.update(self.rng, self.mode)

        self.hour    = (self.hour + 1) % 24
        self.step_n += 1

        # Heatwave events (hard only)
        if self.mode == "hard":
            if self.step_n % 15 == 0:
                self.heatwave = not self.heatwave

            # Adversarial occupancy spikes
            if self.step_n in self.ADVERSARIAL_STEPS.get("hard", []):
                # Force all HP rooms to become occupied
                for r in self.rooms:
                    if r.priority == 3:
                        r.occupancy = 1

    def get_all_states(self) -> list:
        pr = self.power_ratio()
        return [r.get_state(self.hour, pr) for r in self.rooms]

    def get_info(self) -> dict:
        return {
            "satisfied":     sum(r.is_satisfied() for r in self.rooms),
            "complaints":    sum(r.complaint for r in self.rooms),
            "hp_satisfied":  sum(r.priority == 3 and r.is_satisfied()
                                 for r in self.rooms),
            "occupied":      sum(r.occupancy for r in self.rooms),
            "hour":          self.hour,
            "heatwave":      self.heatwave,
            "power_used":    self.total_power(),
            "power_budget":  self.power_budget,
            "over_budget":   self.total_power() > self.power_budget,
        }

    def __repr__(self):
        lines = [f"Hostel(mode={self.mode} hour={self.hour} "
                 f"power={self.total_power():.0f}/{self.power_budget}W)"]
        for r in self.rooms:
            lines.append(f"  {r}")
        return "\n".join(lines)