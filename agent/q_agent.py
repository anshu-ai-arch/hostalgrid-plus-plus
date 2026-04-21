"""
agent/q_agent.py

Improved tabular Q-learning:
- shared Q-table across rooms
- stronger room-state discretization
- uses complaint / ignored / appliance / peak / budget scarcity
"""

import random
import numpy as np
from collections import defaultdict

NUM_ACTIONS = 8
NUM_ROOMS   = 10


class QAgent:
    """
    Shared Q-table across rooms.

    State key:
        (occupancy, priority, complaint_bucket, ignored_bucket,
         appliances_on, peak, budget_scarcity)
    """

    def __init__(self,
                 lr         = 0.08,
                 gamma      = 0.98,
                 eps_start  = 1.0,
                 eps_end    = 0.03,
                 eps_decay  = 0.994,
                 init_q     = 0.05):
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.init_q    = init_q

        self.Q = defaultdict(
            lambda: np.full(NUM_ACTIONS, self.init_q, dtype=np.float32)
        )

        self.action_counts = np.zeros((NUM_ROOMS, NUM_ACTIONS), dtype=int)
        self.episodes      = 0

    def _bucket01(self, value: float, bins: int) -> int:
        value = float(np.clip(value, 0.0, 0.999999))
        return min(bins - 1, int(value * bins))

    def _priority_bin(self, value: float) -> int:
        # observation stores priority normalized by /3.0
        raw = int(round(float(np.clip(value, 0.0, 1.0)) * 3.0))
        raw = min(3, max(1, raw))
        return raw - 1

    def _key(self, state: np.ndarray) -> tuple:
        s = np.asarray(state, dtype=np.float32)

        occupancy  = int(s[0] > 0.5)
        priority   = self._priority_bin(s[1])
        complaint  = self._bucket01(s[2], 4)
        ignored    = self._bucket01(s[3], 4)
        appliances = int(s[4] > 0.5) + int(s[5] > 0.5) + int(s[6] > 0.5)
        peak       = int(s[7] > 0.5)

        # Low power_ratio = high scarcity, so invert it
        scarcity = self._bucket01(1.0 - float(np.clip(s[8], 0.0, 1.0)), 5)

        return (
            occupancy,
            priority,
            complaint,
            ignored,
            appliances,
            peak,
            scarcity,
        )

    def select_actions(self, states: list, greedy: bool = False) -> list:
        actions = []

        for i, state in enumerate(states):
            key = self._key(state)

            if not greedy and random.random() < self.epsilon:
                a = random.randrange(NUM_ACTIONS)
            else:
                q = self.Q[key]
                best = np.flatnonzero(q == q.max())
                a = int(np.random.choice(best))

            actions.append(a)
            self.action_counts[i, a] += 1

        return actions

    def update(self, states, actions, per_room_rewards, next_states, done: bool):
        for i in range(NUM_ROOMS):
            s  = self._key(states[i])
            a  = int(actions[i])
            r  = float(per_room_rewards[i])
            ns = self._key(next_states[i])

            best_next = 0.0 if done else float(np.max(self.Q[ns]))
            td_target = r + self.gamma * best_next
            td_error  = td_target - self.Q[s][a]
            self.Q[s][a] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.episodes += 1

    def summary(self) -> str:
        return (
            f"─── QAgent (Shared Table) ─────────────────────────\n"
            f"  Episodes : {self.episodes}\n"
            f"  Epsilon  : {self.epsilon:.4f}\n"
            f"  Q-states : {len(self.Q):,}\n"
            f"──────────────────────────────────────────────────"
        )
