"""
agent/dqn_agent.py

Centralized Double DQN:
- input is the full hostel state: 10 rooms x 9 features = 90 dims
- output is 10 x 8 Q-values conditioned on the full hostel
- decoder converts those scores into one budget-aware 10-room action plan
"""

import random
import numpy as np
from collections import deque
from typing import Optional
from env.action import ACTION_MAP

NUM_ROOMS        = 10
ROOM_STATE_SIZE  = 9
NUM_ACTIONS      = 8
STATE_SIZE       = NUM_ROOMS * ROOM_STATE_SIZE
OUTPUT_SIZE      = NUM_ROOMS * NUM_ACTIONS

AC_WATTS    = 900.0
FAN_WATTS   = 75.0
LIGHT_WATTS = 15.0

ACTION_COSTS = np.array([
    ACTION_MAP[a][0] * AC_WATTS +
    ACTION_MAP[a][1] * FAN_WATTS +
    ACTION_MAP[a][2] * LIGHT_WATTS
    for a in range(NUM_ACTIONS)
], dtype=np.float32)


class NumpyMLP:
    """
    90 -> 256 -> 256 -> 80
    Huber loss + Adam.
    """

    def __init__(self, lr: float = 3e-4, hidden: int = 128):
        self.lr = lr
        self.hidden = hidden
        rng = np.random.default_rng(0)

        def he(i, o):
            return rng.standard_normal((i, o)) * np.sqrt(2.0 / i)

        self.W1 = he(STATE_SIZE, hidden)
        self.b1 = np.zeros((1, hidden))
        self.W2 = he(hidden, hidden)
        self.b2 = np.zeros((1, hidden))
        self.W3 = he(hidden, OUTPUT_SIZE)
        self.b3 = np.zeros((1, OUTPUT_SIZE))

        self._t = 0
        params = self._params()
        self._ms = [np.zeros_like(p) for p in params]
        self._vs = [np.zeros_like(p) for p in params]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _drelu(self, x):
        return (x > 0.0).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x  = np.asarray(x, dtype=np.float32)
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x  = np.asarray(x, dtype=np.float32)
        self._z1 = self._x @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3

    def backward(self, q_pred: np.ndarray, q_target: np.ndarray) -> float:
        err = q_pred - q_target
        abs_err = np.abs(err)

        quad = np.minimum(abs_err, 1.0)
        lin  = abs_err - quad
        loss = np.mean(0.5 * quad**2 + lin)

        grad = np.where(abs_err < 1.0, err, np.sign(err)).astype(np.float32)
        n = max(1, len(grad))

        dW3 = self._a2.T @ grad / n
        db3 = grad.mean(axis=0, keepdims=True)

        d2  = (grad @ self.W3.T) * self._drelu(self._z2)
        dW2 = self._a1.T @ d2 / n
        db2 = d2.mean(axis=0, keepdims=True)

        d1  = (d2 @ self.W2.T) * self._drelu(self._z1)
        dW1 = self._x.T @ d1 / n
        db1 = d1.mean(axis=0, keepdims=True)

        self._adam([dW1, db1, dW2, db2, dW3, db3])
        return float(loss)

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for i, (g, p) in enumerate(zip(grads, self._params())):
            g = np.clip(g, -5.0, 5.0)
            self._ms[i] = beta1 * self._ms[i] + (1 - beta1) * g
            self._vs[i] = beta2 * self._vs[i] + (1 - beta2) * (g ** 2)
            m_hat = self._ms[i] / (1 - beta1 ** self._t)
            v_hat = self._vs[i] / (1 - beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other: "NumpyMLP"):
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
        self.W3 = other.W3.copy(); self.b3 = other.b3.copy()


class ReplayBuffer:
    def __init__(self, capacity: int = 30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, rewards, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Centralized Double DQN.
    """

    def __init__(self,
                 lr            = 3e-4,
                 gamma         = 0.99,
                 eps_start     = 1.0,
                 eps_end       = 0.02,
                 eps_decay     = 0.996,
                 batch_size    = 64,
                 buffer_size   = 20000,
                 warmup        = 1500,
                 target_update = 250):
        self.gamma         = gamma
        self.epsilon       = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.warmup        = warmup
        self.target_update = target_update
        self.centralized   = True

        self.policy_net = NumpyMLP(lr=lr, hidden=128)
        self.target_net = NumpyMLP(lr=lr, hidden=128)
        self.target_net.copy_from(self.policy_net)

        self.buffer   = ReplayBuffer(capacity=buffer_size)
        self.steps    = 0
        self.episodes = 0

    def _obs_to_room_states(self, obs_or_states):
        if hasattr(obs_or_states, "to_vectors"):
            states = np.asarray(obs_or_states.to_vectors(), dtype=np.float32)
            budget = float(obs_or_states.power_budget)
        else:
            states = np.asarray(obs_or_states, dtype=np.float32).reshape(NUM_ROOMS, ROOM_STATE_SIZE)
            budget = None
        return states, budget

    def _flatten_states(self, obs_or_states):
        states, _ = self._obs_to_room_states(obs_or_states)
        return states.reshape(STATE_SIZE)

    def _to_q_tensor(self, flat_q: np.ndarray) -> np.ndarray:
        return flat_q.reshape(-1, NUM_ROOMS, NUM_ACTIONS)

    def _urgency(self, room_state: np.ndarray, room_q: np.ndarray) -> float:
        occ       = float(room_state[0])
        priority  = float(room_state[1])
        complaint = float(room_state[2])
        ignored   = float(room_state[3])
        scarcity  = 1.0 - float(room_state[8])
        service_adv = float(np.max(room_q[1:]) - room_q[0])

        return (
            3.0 * occ +
            2.8 * priority +
            2.0 * complaint +
            1.2 * ignored +
            0.8 * service_adv +
            0.3 * scarcity
        )

    def _candidate_actions(self, room_state: np.ndarray):
        occ       = float(room_state[0])
        priority  = int(round(float(room_state[1]) * 3.0))
        complaint = float(room_state[2])
        ignored   = float(room_state[3])
        scarcity  = 1.0 - float(room_state[8])

        if occ < 0.5:
            return [0]

        urgent = priority >= 3 or complaint >= 0.35 or ignored >= 0.35
        medium = priority == 2 or complaint >= 0.15 or ignored >= 0.15

        if urgent:
            if scarcity > 0.55:
                return [6, 2, 3, 7, 4, 5, 1, 0]
            return [7, 6, 4, 5, 2, 3, 1, 0]

        if medium:
            if scarcity > 0.60:
                return [6, 2, 3, 7, 4, 5, 1, 0]
            return [6, 7, 2, 3, 4, 5, 1, 0]

        if scarcity > 0.60:
            return [2, 3, 6, 0, 7, 4, 5, 1]
        return [6, 2, 3, 7, 0, 4, 5, 1]

    def _decode_joint_actions(self, states: np.ndarray, q_vals: np.ndarray, budget: float) -> list:
        scarcity = 1.0 - float(np.mean(states[:, 8]))
        actions = [0] * NUM_ROOMS
        remaining = float(budget)

        order = sorted(
            range(NUM_ROOMS),
            key=lambda i: self._urgency(states[i], q_vals[i]),
            reverse=True
        )

        bias = {
            0: 0.00,
            2: 0.05,
            3: 0.03,
            6: 0.10,
            7: 0.06,
            4: 0.02,
            5: 0.01,
            1: -0.05,
        }

        for i in order:
            cands = self._candidate_actions(states[i])
            ranked = sorted(
                cands,
                key=lambda a: float(q_vals[i, a] + bias[a] - 0.00025 * scarcity * ACTION_COSTS[a]),
                reverse=True,
            )

            chosen = 0
            for a in ranked:
                if ACTION_COSTS[a] <= remaining + 1e-6:
                    chosen = int(a)
                    break

            actions[i] = chosen
            remaining -= float(ACTION_COSTS[chosen])

        # Greedy upgrade pass: use any leftover budget where Q gain per extra watt is best.
        while True:
            best = None

            for i in range(NUM_ROOMS):
                current = actions[i]
                cands = self._candidate_actions(states[i])

                for cand in cands:
                    if ACTION_COSTS[cand] <= ACTION_COSTS[current]:
                        continue

                    extra = float(ACTION_COSTS[cand] - ACTION_COSTS[current])
                    if extra > remaining + 1e-6:
                        continue

                    gain = float(q_vals[i, cand] - q_vals[i, current])
                    score = gain / (extra + 1e-6)

                    if best is None or score > best[0]:
                        best = (score, i, cand, extra)

            if best is None or best[0] <= 0.0:
                break

            _, i, cand, extra = best
            actions[i] = int(cand)
            remaining -= extra

        return actions

    def _random_budgeted_actions(self, states: np.ndarray, budget: float) -> list:
        actions = [0] * NUM_ROOMS
        remaining = float(budget)

        order = list(range(NUM_ROOMS))
        random.shuffle(order)

        for i in order:
            cands = list(self._candidate_actions(states[i]))
            random.shuffle(cands)

            chosen = 0
            for a in cands:
                if ACTION_COSTS[a] <= remaining + 1e-6:
                    chosen = int(a)
                    break

            actions[i] = chosen
            remaining -= float(ACTION_COSTS[chosen])

        return actions

    def select_actions(self, obs_or_states, greedy: bool = False) -> list:
        states, budget = self._obs_to_room_states(obs_or_states)
        if budget is None:
            # fallback if only raw states are passed
            budget = 6000.0

        flat_state = states.reshape(1, STATE_SIZE)
        q_vals = self._to_q_tensor(self.policy_net.predict(flat_state))[0]

        if not greedy and random.random() < self.epsilon:
            return self._random_budgeted_actions(states, budget)

        return self._decode_joint_actions(states, q_vals, budget)

    def store(self, obs_or_states, actions, per_room_rewards, next_obs_or_states, done):
        s  = self._flatten_states(obs_or_states)
        ns = self._flatten_states(next_obs_or_states)
        self.buffer.push(s, actions, per_room_rewards, ns, done)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None

        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s_arr  = np.asarray(s, dtype=np.float32).reshape(-1, STATE_SIZE)
        ns_arr = np.asarray(ns, dtype=np.float32).reshape(-1, STATE_SIZE)
        a_arr  = np.asarray(a, dtype=np.int64).reshape(-1, NUM_ROOMS)
        r_arr  = np.asarray(r, dtype=np.float32).reshape(-1, NUM_ROOMS)
        d_arr  = np.asarray(d, dtype=np.float32).reshape(-1)

        q_all_flat = self.policy_net.forward(s_arr)
        q_all = self._to_q_tensor(q_all_flat)

        next_online = self._to_q_tensor(self.policy_net.predict(ns_arr))
        next_actions = np.argmax(next_online, axis=2)

        next_target = self._to_q_tensor(self.target_net.predict(ns_arr))

        batch_idx = np.arange(len(a_arr))[:, None]
        room_idx  = np.arange(NUM_ROOMS)[None, :]
        q_next = next_target[batch_idx, room_idx, next_actions]

        q_tgt = r_arr + self.gamma * q_next * (1.0 - d_arr[:, None])

        q_target_full = q_all.copy()
        q_target_full[batch_idx, room_idx, a_arr] = q_tgt

        loss = self.policy_net.backward(
            q_all_flat,
            q_target_full.reshape(-1, OUTPUT_SIZE)
        )

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.copy_from(self.policy_net)

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.episodes += 1

    def summary(self) -> str:
        return (
            f"─── Centralized DQNAgent ──────────────────────────\n"
            f"  Episodes : {self.episodes}\n"
            f"  Epsilon  : {self.epsilon:.4f}\n"
            f"  Buffer   : {len(self.buffer):,}\n"
            f"  Steps    : {self.steps:,}\n"
            f"  Input    : {STATE_SIZE} dims (full hostel)\n"
            f"  Output   : {OUTPUT_SIZE} Q-values (10 rooms x 8 actions)\n"
            f"──────────────────────────────────────────────────"
        )
