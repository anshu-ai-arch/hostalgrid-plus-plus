"""
utils/metrics.py
Computes summary stats from training history dicts.
"""

import numpy as np


class TrainingMetrics:
    """
    Wraps a history dict from train_q() or train_dqn().
    history keys: rewards, satisfied, complaints, mode
    """

    def __init__(self, history: dict):
        self.rewards    = np.array(history["rewards"])
        self.satisfied  = np.array(history["satisfied"])
        self.complaints = np.array(history["complaints"])
        self.mode       = history.get("mode", "unknown")
        self.n          = len(self.rewards)

    def mean(self)      -> float: return float(np.mean(self.rewards))
    def std(self)       -> float: return float(np.std(self.rewards))
    def early_avg(self, n=100) -> float:
        return float(np.mean(self.rewards[:n]))
    def late_avg(self, n=100)  -> float:
        return float(np.mean(self.rewards[-n:]))
    def is_improving(self) -> bool:
        return self.late_avg() > self.early_avg()

    def rolling_mean(self, window=30) -> np.ndarray:
        return np.convolve(self.rewards,
                           np.ones(window)/window,
                           mode="valid")

    def avg_satisfied(self) -> float:
        return float(np.mean(self.satisfied))

    def avg_complaints(self) -> float:
        return float(np.mean(self.complaints))

    def summary(self) -> str:
        return (
            f"─── Metrics ({self.mode}) ──────────────────────────\n"
            f"  Episodes      : {self.n}\n"
            f"  Early avg     : {self.early_avg():.3f}\n"
            f"  Late  avg     : {self.late_avg():.3f}\n"
            f"  Improving?    : {'YES ✓' if self.is_improving() else 'NO ✗'}\n"
            f"  Avg satisfied : {self.avg_satisfied():.1f}/10\n"
            f"  Avg complaints: {self.avg_complaints():.1f}\n"
            f"──────────────────────────────────────────────────"
        )