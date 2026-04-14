# env/state.py
# Next-level episode state tracker
# Fixed: violations tracked, demand_sat tracked, trust updated properly

import numpy as np
from collections import deque
from typing import List, Dict, Any


class EpisodeState:
    """
    Complete episode state tracker.
    Fixed bugs:
    - violations now properly accumulated each step
    - demand_satisfaction tracked every step and averaged
    - system_trust updated with decay + recovery logic
    - complaint momentum tracked (not just total)
    """

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        # Core metrics
        self.steps                  = 0
        self.total_reward           = 0.0
        self.total_cost             = 0.0
        self.total_carbon           = 0.0
        self.total_complaints       = 0
        self.total_violations       = 0       # FIXED: was never updated before
        self.total_misuse_events    = 0
        self.total_misuse_handled   = 0

        # Per-step averages
        self.demand_sat_sum         = 0.0     # FIXED: was never tracked before
        self.fairness_sum           = 0.0
        self.trust_sum              = 0.0

        # Peak/grid metrics
        self.peak_violations        = 0
        self.solar_harvested        = 0.0

        # System trust — FIXED: properly updated each step
        self.system_trust           = 1.0
        self.trust_collapse_steps   = 0       # steps where trust < 0.3
        self.collapsed              = False

        # History buffers for momentum tracking
        self.reward_history         = deque(maxlen=100)
        self.complaint_history      = deque(maxlen=10)
        self.violation_history      = deque(maxlen=10)
        self.trust_history          = deque(maxlen=10)
        self.demand_sat_history     = deque(maxlen=10)
        self.cost_history           = deque(maxlen=24)

        # Per-hour logs (full episode)
        self.hourly_log: List[Dict] = []

        # Events encountered (Task 3)
        self.events_encountered: List[str] = []

    # ------------------------------------------------------------------
    def update(
        self,
        reward:       float,
        cost:         float,
        carbon:       float,
        complaints:   int,
        violations:   int,      # FIXED: now actually used
        demand_sat:   float,    # FIXED: now actually tracked
        fairness:     float,
        peak_violation: bool   = False,
        misuse_count:   int    = 0,
        misuse_handled: int    = 0,
        solar_harvest:  float  = 0.0,
        event_name:     str    = None,
        hour:           int    = 0,
    ):
        self.steps             += 1
        self.total_reward      += reward
        self.total_cost        += cost
        self.total_carbon      += carbon
        self.total_complaints  += complaints
        self.total_violations  += violations     # FIXED
        self.total_misuse_events  += misuse_count
        self.total_misuse_handled += misuse_handled
        self.demand_sat_sum    += demand_sat     # FIXED
        self.fairness_sum      += fairness
        self.solar_harvested   += solar_harvest
        self.peak_violations   += int(peak_violation)

        # History
        self.reward_history.append(reward)
        self.complaint_history.append(complaints)
        self.violation_history.append(violations)
        self.demand_sat_history.append(demand_sat)
        self.trust_history.append(self.system_trust)
        self.cost_history.append(cost)

        # Trust update — FIXED: proper decay + recovery
        self._update_trust(violations, complaints)
        self.trust_sum += self.system_trust

        # Collapse detection
        if self.system_trust < 0.3:
            self.trust_collapse_steps += 1
        if self.trust_collapse_steps >= 3:
            self.collapsed = True

        # Events
        if event_name:
            self.events_encountered.append(event_name)

        # Hourly log
        self.hourly_log.append({
            "hour"       : hour,
            "reward"     : round(reward, 4),
            "cost"       : round(cost, 4),
            "complaints" : complaints,
            "violations" : violations,
            "demand_sat" : round(demand_sat, 3),
            "fairness"   : round(fairness, 3),
            "trust"      : round(self.system_trust, 3),
        })

    # ------------------------------------------------------------------
    def _update_trust(self, violations: int, complaints: int):
        """
        FIXED trust update — was never called before.
        Trust decays with violations and high complaints.
        Trust recovers slowly when everything is good.
        """
        if violations > 0:
            self.system_trust *= (0.95 ** violations)   # decay per violation
        elif complaints > 5:
            self.system_trust *= 0.98                    # slow decay for complaints
        else:
            # Recovery when performing well
            self.system_trust = min(1.0, self.system_trust * 1.02)

        self.system_trust = float(np.clip(self.system_trust, 0.0, 1.0))

    # ------------------------------------------------------------------
    def avg_demand_satisfaction(self) -> float:
        """FIXED: properly averaged over all steps."""
        return self.demand_sat_sum / max(1, self.steps)

    def avg_fairness(self) -> float:
        return self.fairness_sum / max(1, self.steps)

    def avg_trust(self) -> float:
        return self.trust_sum / max(1, self.steps)

    def avg_cost_per_hour(self) -> float:
        return self.total_cost / max(1, self.steps)

    def misuse_handle_rate(self) -> float:
        if self.total_misuse_events == 0:
            return 1.0
        return self.total_misuse_handled / self.total_misuse_events

    # ------------------------------------------------------------------
    def is_collapsed(self, complaint_threshold: int = 100) -> bool:
        return self.collapsed or self.total_complaints > complaint_threshold

    def recent_complaint_trend(self) -> str:
        """Is complaint situation improving or worsening?"""
        if len(self.complaint_history) < 4:
            return "stable"
        recent = list(self.complaint_history)[-4:]
        if recent[-1] > recent[0] * 1.5:
            return "worsening"
        elif recent[-1] < recent[0] * 0.7:
            return "improving"
        return "stable"

    def recent_violation_trend(self) -> str:
        if len(self.violation_history) < 4:
            return "stable"
        recent = list(self.violation_history)[-4:]
        avg_early = np.mean(recent[:2])
        avg_late  = np.mean(recent[2:])
        if avg_late > avg_early * 1.5:
            return "worsening"
        elif avg_late < avg_early * 0.5:
            return "improving"
        return "stable"

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Full episode summary — used by graders."""
        return {
            "steps"               : self.steps,
            "total_reward"        : round(self.total_reward, 4),
            "total_cost"          : round(self.total_cost, 4),
            "total_carbon"        : round(self.total_carbon, 4),
            "total_complaints"    : self.total_complaints,
            "total_violations"    : self.total_violations,       # FIXED
            "demand_satisfaction" : round(self.avg_demand_satisfaction(), 4),  # FIXED
            "avg_fairness"        : round(self.avg_fairness(), 4),
            "system_trust"        : round(self.system_trust, 4),  # FIXED
            "avg_trust"           : round(self.avg_trust(), 4),
            "peak_violations"     : self.peak_violations,
            "collapsed"           : self.collapsed,
            "events_encountered"  : len(self.events_encountered),
            "misuse_handle_rate"  : round(self.misuse_handle_rate(), 4),
            "complaint_trend"     : self.recent_complaint_trend(),
            "violation_trend"     : self.recent_violation_trend(),
            "solar_harvested"     : round(self.solar_harvested, 4),
        }

    # ------------------------------------------------------------------
    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*55}")
        print(f"📊 Episode Summary")
        print(f"{'='*55}")
        print(f"  Steps              : {s['steps']}")
        print(f"  Total Reward       : {s['total_reward']:.4f}")
        print(f"  Total Cost         : Rs {s['total_cost']:.2f}")
        print(f"  Demand Satisfaction: {s['demand_satisfaction']:.3f}")
        print(f"  Total Violations   : {s['total_violations']}")
        print(f"  Total Complaints   : {s['total_complaints']}")
        print(f"  Avg Fairness       : {s['avg_fairness']:.3f}")
        print(f"  System Trust       : {s['system_trust']:.3f}")
        print(f"  Collapsed          : {s['collapsed']}")
        print(f"  Complaint Trend    : {s['complaint_trend']}")
        print(f"  Violation Trend    : {s['violation_trend']}")
        print(f"{'='*55}")