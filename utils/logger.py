"""
utils/logger.py
Logs training progress to console and timestamped file.
"""

import os
from datetime import datetime


class Logger:
    def __init__(self, mode: str = "medium",
                 agent: str = "Q",
                 log_dir: str = "logs"):
        self.mode  = mode
        self.agent = agent
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir,
                                 f"{agent}_{mode}_{ts}.log")
        self._f   = open(self.path, "w")
        self._write(f"HostelGrid++ | agent={agent} mode={mode}")
        self._write(f"Started: {datetime.now()}")
        self._write("=" * 52)

    def _write(self, line: str):
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def episode(self, ep: int, reward: float,
                satisfied: int, complaints: int,
                epsilon: float, interval: int = 100):
        if ep % interval == 0:
            self._write(
                f"  Ep {ep:4d} | reward={reward:7.3f}"
                f" | satisfied={satisfied:2d}/10"
                f" | complaints={complaints:2d}"
                f" | ε={epsilon:.3f}"
            )

    def eval_result(self, label: str, mean: float,
                    std: float, satisfied: float,
                    complaints: float):
        self._write(
            f"  {label:14s} | mean={mean:7.3f}"
            f" | std={std:6.3f}"
            f" | satisfied={satisfied:.1f}"
            f" | complaints={complaints:.1f}"
        )

    def info(self, msg: str):
        self._write(f"  [INFO] {msg}")

    def close(self):
        self._write(f"\nLog saved → {self.path}")
        self._f.close()