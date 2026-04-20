"""
agent/dqn_agent.py

DQN with SHARED POLICY — one network applied to each room.
Room 1's experience trains the same weights as Room 7.
No PyTorch required — pure numpy MLP.

Input : 8-feature room state vector
Output: 8 Q-values (one per action)
"""

import random
import numpy as np
from collections import deque

NUM_ACTIONS = 8
NUM_ROOMS   = 10
STATE_SIZE  = 9


# ── Numpy MLP ──────────────────────────────────────────────────

class NumpyMLP:
    """
    2-hidden-layer MLP: 8 → 64 → 64 → 8
    Huber loss + Adam optimiser.
    """
    def __init__(self, lr: float = 0.001):
        self.lr = lr
        rng = np.random.default_rng(0)

        def he(i, o):
            return rng.standard_normal((i, o)) * np.sqrt(2 / i)

        self.W1 = he(9, 64); self.b1 = np.zeros((1, 64))
        self.W2 = he(64, 64);         self.b2 = np.zeros((1, 64))
        self.W3 = he(64, NUM_ACTIONS); self.b3 = np.zeros((1, NUM_ACTIONS))

        self._t  = 0
        params   = self._params()
        self._ms = [np.zeros_like(p) for p in params]
        self._vs = [np.zeros_like(p) for p in params]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3]

    def _relu(self, x):  return np.maximum(0, x)
    def _drelu(self, x): return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x  = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3

    def backward(self, q_pred: np.ndarray,
                 q_target: np.ndarray) -> float:
        """Huber loss gradient."""
        err  = q_pred - q_target
        grad = np.where(np.abs(err) < 1.0, err, np.sign(err))
        n    = len(grad)

        dW3 = self._a2.T @ grad / n
        db3 = grad.mean(axis=0, keepdims=True)
        d2  = grad @ self.W3.T * self._drelu(self._z2)
        dW2 = self._a1.T @ d2 / n
        db2 = d2.mean(axis=0, keepdims=True)
        d1  = d2 @ self.W2.T * self._drelu(self._z1)
        dW1 = self._x.T @ d1 / n
        db1 = d1.mean(axis=0, keepdims=True)

        self._adam([dW1, db1, dW2, db2, dW3, db3])
        return float(np.mean(np.abs(err)))

    def _adam(self, grads,
              beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for i, (g, p) in enumerate(zip(grads, self._params())):
            self._ms[i] = beta1 * self._ms[i] + (1 - beta1) * g
            self._vs[i] = beta2 * self._vs[i] + (1 - beta2) * g**2
            m_hat = self._ms[i] / (1 - beta1**self._t)
            v_hat = self._vs[i] / (1 - beta2**self._t)
            p    -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other: "NumpyMLP"):
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
        self.W3 = other.W3.copy(); self.b3 = other.b3.copy()


# ── Replay Buffer ───────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQN Agent ───────────────────────────────────────────────────

class DQNAgent:
    """
    Shared-policy DQN.

    Key insight: same network processes every room's state.
    Learns general comfort policy, not room-specific habits.
    Advantage over Q-table: no per-room state space explosion.
    """

    def __init__(self,
                 lr         = 0.001,
                 gamma      = 0.9,
                 eps_start  = 1.0,
                 eps_end    = 0.05,
                 eps_decay  = 0.995,
                 batch_size = 64):
        self.gamma      = gamma
        self.epsilon    = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch_size = batch_size

        self.policy_net = NumpyMLP(lr=lr)
        self.target_net = NumpyMLP(lr=lr)
        self.target_net.copy_from(self.policy_net)

        self.buffer  = ReplayBuffer()
        self.steps   = 0
        self.episodes = 0

    def select_actions(self, states: list,
                       greedy: bool = False) -> list:
        """
        Pass all 10 room states through shared network at once.
        Each room gets its own action from its own Q-values.
        """
        states_arr = np.array(states, dtype=np.float32)  # (10, 8)
        q_vals     = self.policy_net.forward(states_arr)  # (10, 8)

        actions = []
        for i in range(NUM_ROOMS):
            if not greedy and random.random() < self.epsilon:
                actions.append(random.randrange(NUM_ACTIONS))
            else:
                actions.append(int(np.argmax(q_vals[i])))
        return actions

    def store(self, states, actions, per_room_rewards,
              next_states, done):
        """Store one transition per room in shared buffer."""
        for i in range(NUM_ROOMS):
            self.buffer.push(
                states[i], actions[i],
                per_room_rewards[i],
                next_states[i], float(done)
            )

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s_arr  = np.array(s,  dtype=np.float32)
        ns_arr = np.array(ns, dtype=np.float32)
        a_arr  = np.array(a,  dtype=int)
        r_arr  = np.array(r,  dtype=np.float32)
        d_arr  = np.array(d,  dtype=np.float32)

        # Current Q-values
        q_all = self.policy_net.forward(s_arr)   # (B, 8)

        # Bellman targets
        q_next = self.target_net.forward(ns_arr).max(axis=1)  # (B,)
        q_tgt  = r_arr + self.gamma * q_next * (1 - d_arr)

        # Only update Q-value of taken action
        q_target_full = q_all.copy()
        q_target_full[np.arange(len(a_arr)), a_arr] = q_tgt

        loss = self.policy_net.backward(q_all, q_target_full)

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.copy_from(self.policy_net)

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end,
                           self.epsilon * self.eps_decay)
        self.episodes += 1

    def summary(self) -> str:
        return (
            f"─── DQNAgent (Shared Policy) ───────────────────────\n"
            f"  Episodes : {self.episodes}\n"
            f"  Epsilon  : {self.epsilon:.4f}\n"
            f"  Buffer   : {len(self.buffer):,}\n"
            f"  Steps    : {self.steps:,}\n"
            f"──────────────────────────────────────────────────"
        )