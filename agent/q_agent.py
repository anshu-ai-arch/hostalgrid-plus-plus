"""
agent/q_agent.py

Q-Learning with separate Q-table per room.
State discretized to 5-element tuple → 8 Q-values.
"""

import random
import numpy as np
from collections import defaultdict

NUM_ACTIONS = 8
NUM_ROOMS   = 10


class QAgent:
    """
    Separate Q-table per room.
    State key: (occupancy, priority_bucket, complaint_flag, appliances, peak)
    """

    def __init__(self,
                 lr         = 0.1,
                 gamma      = 0.95,
                 eps_start  = 1.0,
                 eps_end    = 0.02,
                 eps_decay  = 0.997):
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # One Q-table per room
        self.Q = [
            defaultdict(lambda: np.zeros(NUM_ACTIONS))
            for _ in range(NUM_ROOMS)
        ]
        self.action_counts = np.zeros((NUM_ROOMS, NUM_ACTIONS), dtype=int)
        self.episodes      = 0

    def _key(self, state: np.ndarray) -> tuple:
        """Discretize 8-float state → 5-element tuple."""
        return (
            int(state[0]),                    # occupancy
            int(round(state[1] * 2)),         # priority bucket (0,1,2)
            int(state[2] > 0.4),             # complaint flag
            (int(state[4]), int(state[5]), int(state[6])),  # appliances
            int(state[7]),                    # peak hour
        )

    def select_actions(self, states: list,
                       greedy: bool = False) -> list:
        actions = []
        for i, state in enumerate(states):
            if not greedy and random.random() < self.epsilon:
                a = random.randrange(NUM_ACTIONS)
            else:
                a = int(np.argmax(self.Q[i][self._key(state)]))
            actions.append(a)
            self.action_counts[i, a] += 1
        return actions

    def update(self, states, actions, per_room_rewards, next_states):
        for i in range(NUM_ROOMS):
            s  = self._key(states[i])
            a  = actions[i]
            r  = per_room_rewards[i]
            ns = self._key(next_states[i])
            best_next      = np.max(self.Q[i][ns])
            td             = r + self.gamma * best_next - self.Q[i][s][a]
            self.Q[i][s][a] += self.lr * td

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end,
                           self.epsilon * self.eps_decay)
        self.episodes += 1

    def summary(self) -> str:
        total_entries = sum(len(q) for q in self.Q)
        return (
            f"─── QAgent ────────────────────────────────────────\n"
            f"  Episodes : {self.episodes}\n"
            f"  Epsilon  : {self.epsilon:.4f}\n"
            f"  Q-entries: {total_entries} across {NUM_ROOMS} tables\n"
            f"──────────────────────────────────────────────────"
        )