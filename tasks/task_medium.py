# tasks/task_medium.py
# TASK 2: Fair Enforcement Under Misuse

import numpy as np
import random
from collections import deque
from env.hostelgrid_env import HostelGridEnv
from simulation.grid import Grid


# ── Task Config ───────────────────────────────────────────────
class TaskConfig:
    enable_events   = False
    enable_misuse   = True      # ← CRITICAL: detect selfish behavior
    enable_priority = True      # ← still have commitments
    demand_variance = "medium"  # ← higher variability (spikes)
    solar_variance  = "medium"
    penalty_weights = {
        "priority_violation": 2.5,   # commitment failures still matter
        "cost":               0.08,
        "complaints":         0.15,  # ← higher weight (fairness signal)
        "misuse_penalty":     1.5,   # ← NEW: penalize selfish behavior
        "fairness_violation": 1.2,   # ← NEW: track unfairness
    }


# ── Room with Commitment + Misuse Tracking ───────────────────
class EnforcementRoom:
    def __init__(self, room_id):
        self.room_id               = room_id
        self.is_occupied           = random.choice([True, False])
        
        # Commitment system (from Task 1)
        self.has_approved_request  = random.random() < 0.35
        self.min_required_supply   = 1.5 if self.has_approved_request else 0.0
        
        # Misuse detection
        self.base_demand           = np.random.uniform(0.5, 2.0)
        self.current_demand        = 0.0
        self.current_supply        = 0.0
        self.flagged_for_misuse    = False
        self.misuse_penalty_timer  = 0
        self.power_cap             = 4.0  # max allowed power
        
        # History
        self.demand_history        = deque(maxlen=5)
        self.is_spike              = False

    def check_violation(self):
        """Commitment violation check."""
        if self.has_approved_request:
            return self.current_supply < self.min_required_supply
        return False

    def detect_misuse(self):
        """
        Detect sudden usage spike without approval.
        Misuse = current_demand >> base_demand AND no_approved_request
        """
        self.demand_history.append(self.current_demand)
        
        if len(self.demand_history) < 2:
            return False
        
        # Spike detection: sudden jump (>150% of recent average)
        recent_avg = np.mean(list(self.demand_history)[:-1])
        current = self.current_demand
        
        self.is_spike = current > recent_avg * 1.5 and current > 2.5
        
        # Misuse: spike without approval
        if self.is_spike and not self.has_approved_request:
            self.flagged_for_misuse = True
            self.misuse_penalty_timer = 12  # 12-step punishment window
            return True
        
        return False

    def apply_power_cap(self):
        """Limit power supply if flagged for misuse."""
        if self.flagged_for_misuse and self.misuse_penalty_timer > 0:
            self.power_cap = 1.5  # Harsh cap during punishment
            self.misuse_penalty_timer -= 1
        else:
            self.power_cap = 4.0
            if self.misuse_penalty_timer == 0:
                self.flagged_for_misuse = False

    def update_fairness_score(self):
        """
        Fairness metric: how much is this room using vs. its fair share?
        fair_share = avg_supply per room
        usage_ratio = current_supply / fair_share
        """
        # Return ratio for aggregate fairness calculation
        return self.current_supply


# ── Experience Replay Buffer ──────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=15000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, action, reward, next_obs, done = zip(*batch)
        return obs, action, reward, next_obs, done
    
    def __len__(self):
        return len(self.buffer)


# ── Improved Discretizer ──────────────────────────────────────
def _discretize(obs, n_bins=10):
    obs_array = np.asarray(obs, dtype=np.float32)
    lower = np.percentile(obs_array, 1)
    upper = np.percentile(obs_array, 99)
    if upper <= lower:
        upper = lower + 1.0
    clipped = np.clip(obs_array, lower, upper)
    scaled = ((clipped - lower) / (upper - lower) * (n_bins - 1)).astype(int)
    return tuple(np.clip(scaled, 0, n_bins - 1))


# ── Task 2 Environment ────────────────────────────────────────
class Task2Env(HostelGridEnv):
    def __init__(self, num_rooms=20, episode_hours=24):
        super().__init__(num_rooms, episode_hours)
        self.config = TaskConfig()
        self.rooms = []
        self.grid = Grid()

        # Episode tracking
        self._ep_steps = 0
        self._ep_ds_sum = 0.0
        self._ep_violations = 0
        self._ep_priority_count = 0
        self._ep_misuse_detected = 0
        self._ep_misuse_handled = 0
        self._ep_fairness_score = 0.0
        self._ep_step_count = 0

    # ------------------------------------------------------------------
    def reset(self):
        obs = super().reset()
        self.rooms = [EnforcementRoom(i) for i in range(self.num_rooms)]
        
        self._ep_steps = 0
        self._ep_ds_sum = 0.0
        self._ep_violations = 0
        self._ep_priority_count = 0
        self._ep_misuse_detected = 0
        self._ep_misuse_handled = 0
        self._ep_fairness_score = 0.0
        self._ep_step_count = 0
        
        return self._augment_obs(obs)

    # ------------------------------------------------------------------
    def _augment_obs(self, base_obs):
        """
        Augmented observation with enforcement-critical features.
        
        Extra features (6):
          - fraction of flagged rooms
          - average demand (detect spike pressure)
          - fraction with approved requests
          - average fairness violation (demand ratio)
          - step progress
          - misuse severity (sum of excess demand in flagged rooms)
        """
        priority_rooms = [r for r in self.rooms if r.has_approved_request]
        flagged_rooms = [r for r in self.rooms if r.flagged_for_misuse]
        
        frac_flagged = len(flagged_rooms) / max(1, self.num_rooms)
        avg_demand = np.mean([r.current_demand for r in self.rooms])
        frac_priority = len(priority_rooms) / max(1, self.num_rooms)
        
        # Fairness: how unequal is distribution?
        fair_share = avg_demand
        fairness_violations = sum(max(0, r.current_demand - fair_share * 2.0) 
                                  for r in self.rooms)
        fairness_score = fairness_violations / (self.num_rooms * 2.0)
        
        # Misuse severity
        misuse_severity = sum(max(0, r.current_demand - r.base_demand * 1.5)
                             for r in flagged_rooms)
        misuse_severity = misuse_severity / (self.num_rooms * 2.0)
        
        step_frac = self._ep_steps / max(1, self.episode_hours)
        
        extra = np.array([
            frac_flagged, 
            avg_demand / 3.0,  # normalize
            frac_priority,
            fairness_score,
            step_frac,
            misuse_severity
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, extra])

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Action: enforcement intensity (0-5)
          0: no enforcement (full power for everyone)
          1-3: medium enforcement (selective caps)
          4-5: strict enforcement (harsh caps + deep fairness)
        """
        obs_base, base_reward, done, info = super().step(action)
        self._ep_steps += 1
        self._ep_step_count += 1

        # ── Demand generation with misuse potential ───────────────────
        total_power = info["power"] * random.uniform(0.80, 1.0)

        for room in self.rooms:
            # Base demand + optional spike (misuse)
            spike_prob = 0.15 if not room.has_approved_request else 0.05
            if random.random() < spike_prob:
                room.current_demand = room.base_demand * random.uniform(1.8, 3.0)
            else:
                room.current_demand = room.base_demand * random.uniform(0.8, 1.2)
            
            # Detect misuse
            room.detect_misuse()
            room.apply_power_cap()

        # ── Enforcement Action (agent decides) ────────────────────────
        # action: 0-5 enforcement intensity
        enforcement_level = action / 5.0  # normalize to [0, 1]
        
        # Adjust power caps based on enforcement
        for room in self.rooms:
            if room.flagged_for_misuse:
                # Enforcement reduces access to misusing rooms
                cap_reduction = enforcement_level * 0.6
                room.power_cap = max(0.5, room.power_cap * (1.0 - cap_reduction))

        # ── Power allocation ──────────────────────────────────────────
        priority_rooms = [r for r in self.rooms if r.has_approved_request]
        flagged_rooms = [r for r in self.rooms if r.flagged_for_misuse]
        normal_rooms = [r for r in self.rooms 
                       if not r.has_approved_request and not r.flagged_for_misuse]

        power_remaining = total_power

        # Step 1: Satisfy priority commitments
        for room in priority_rooms:
            allocation = min(room.min_required_supply, room.power_cap)
            room.current_supply = allocation
            power_remaining -= allocation

        # Step 2: Distribute to flagged (misusing) rooms (limited)
        if flagged_rooms and power_remaining > 0:
            per_flagged = power_remaining / len(flagged_rooms)
            for room in flagged_rooms:
                allocation = min(per_flagged, room.power_cap)
                room.current_supply = allocation
                power_remaining -= allocation

        # Step 3: Distribute to normal rooms (fairness)
        if normal_rooms and power_remaining > 0:
            per_normal = power_remaining / len(normal_rooms)
            for room in normal_rooms:
                room.current_supply = max(0.0, per_normal)

        # ── Metrics ───────────────────────────────────────────────────
        violations = sum(1 for r in self.rooms if r.check_violation())
        misuse_detected = sum(1 for r in self.rooms if r.flagged_for_misuse)
        
        # Fairness metric: Gini coefficient approximation
        supplies = [r.current_supply for r in self.rooms]
        avg_supply = np.mean(supplies)
        fairness_violation = np.std(supplies) / (avg_supply + 0.1)
        
        # Misuse handling: did enforcement cap flagged rooms?
        misuse_handled = sum(1 for r in flagged_rooms 
                            if r.current_supply < r.current_demand * 0.8)

        satisfied = sum(1 for r in priority_rooms if not r.check_violation())
        n_priority = max(1, len(priority_rooms))
        demand_sat = satisfied / n_priority

        self._ep_ds_sum += demand_sat
        self._ep_violations += violations
        self._ep_priority_count += n_priority
        self._ep_misuse_detected += misuse_detected
        self._ep_misuse_handled += misuse_handled
        self._ep_fairness_score += fairness_violation

        # ── Reward Engineering (Task 2 Specific) ──────────────────────
        reward = 0.0

        # 1. Primary: demand satisfaction (same as Task 1)
        reward += demand_sat * 4.5

        # 2. Priority violations (still critical)
        if len(priority_rooms) > 0:
            feasible_priority = sum(r.min_required_supply for r in priority_rooms) <= total_power
            if feasible_priority:
                reward -= violations * self.config.penalty_weights["priority_violation"] * 1.3
            else:
                shortfall = sum(r.min_required_supply for r in priority_rooms) - total_power
                reward -= (shortfall / max(1, sum(r.min_required_supply for r in priority_rooms))) * 1.5

        # 3. Fairness enforcement (NEW FOR TASK 2)
        # Penalize if power distribution is too unequal
        if fairness_violation > 0.8:
            reward -= fairness_violation * self.config.penalty_weights["fairness_violation"]
        
        # Bonus for maintaining fairness
        if fairness_violation < 0.5:
            reward += 0.8

        # 4. Misuse handling (NEW FOR TASK 2)
        if misuse_detected > 0:
            # Agent effectively handled misuse if it capped flagged rooms
            misuse_handling_ratio = misuse_handled / max(1, misuse_detected)
            reward += misuse_handling_ratio * 2.0
            
            # But penalize if misuse goes unchecked
            if misuse_handled == 0:
                reward -= 1.5

        # 5. Enforcement cost (don't over-enforce)
        # Higher enforcement = higher operational cost
        enforcement_cost = enforcement_level * 0.3
        reward -= enforcement_cost

        # 6. Secondary costs
        reward -= info["cost"] * self.config.penalty_weights["cost"] * 0.4
        reward -= info["complaints"] * self.config.penalty_weights["complaints"] * 0.6

        # 7. Bonus for zero violations
        if violations == 0:
            reward += 1.5

        info["violations"] = violations
        info["demand_satisfaction"] = round(demand_sat, 3)
        info["misuse_detected"] = misuse_detected
        info["misuse_handled"] = misuse_handled
        info["fairness_violation"] = round(fairness_violation, 3)
        info["enforcement_level"] = round(enforcement_level, 3)

        aug_obs = self._augment_obs(obs_base)
        return aug_obs, round(reward, 4), done, info

    # ------------------------------------------------------------------
    def episode_stats(self):
        steps = max(1, self._ep_steps)
        return {
            "demand_satisfaction": self._ep_ds_sum / steps,
            "violations": self._ep_violations,
            "misuse_detected": self._ep_misuse_detected,
            "misuse_handled": self._ep_misuse_handled,
            "fairness_score": self._ep_fairness_score / steps,
        }


# ── Q-Learning Agent with Experience Replay ───────────────────
class EnforcementAgent:
    def __init__(self, obs_dim, action_count=6, n_bins=10):
        self.action_count = action_count
        self.n_bins = n_bins
        self.q_table = {}

        self.replay_buffer = ReplayBuffer(capacity=15000)
        self.batch_size = 32

        self.alpha = 0.15
        self.gamma = 0.97

        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.05

        self.init_value = 1.0
        self.train_frequency = 4

    # ------------------------------------------------------------------
    def _state(self, obs):
        return _discretize(np.asarray(obs, dtype=np.float32), self.n_bins)

    # ------------------------------------------------------------------
    def choose_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)

        s = self._state(obs)
        if s not in self.q_table:
            return random.randint(0, self.action_count - 1)

        q_values = self.q_table[s]
        max_q = np.max(q_values)
        best_actions = [a for a in range(self.action_count) if q_values[a] == max_q]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    def _ensure_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = np.full(self.action_count, self.init_value, dtype=np.float32)

    # ------------------------------------------------------------------
    def learn(self, obs, action, reward, next_obs, done, step_count):
        s = self._state(obs)
        s2 = self._state(next_obs)

        self._ensure_state(s)
        self._ensure_state(s2)

        best_next = 0.0 if done else np.max(self.q_table[s2])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s][action]

        self.q_table[s][action] += self.alpha * td_error

        self.replay_buffer.push(obs, action, reward, next_obs, done)

        if step_count % self.train_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self._replay_batch()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    def _replay_batch(self):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)

        for obs, action, reward, next_obs, done in zip(
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
        ):
            s = self._state(obs)
            s2 = self._state(next_obs)

            self._ensure_state(s)
            self._ensure_state(s2)

            best_next = 0.0 if done else np.max(self.q_table[s2])
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_table[s][action]

            self.q_table[s][action] += (self.alpha * 0.7) * td_error


# ── Grader ────────────────────────────────────────────────────
def grade(stats):
    score = 0.0

    ds = stats["demand_satisfaction"]
    v = stats["violations"]
    c = stats["total_cost"]
    cp = stats["complaints"]
    tr = stats["total_reward"]
    mh = stats["misuse_handled"]
    fv = stats["fairness_violation"]

    # Demand satisfaction
    score += 0.20 if ds > 0.85 else (0.08 if ds > 0.70 else 0.0)

    # Violations
    score += 0.20 if v < 20 else (0.08 if v < 40 else 0.0)

    # Cost efficiency
    score += 0.15 if c < 450 else (0.08 if c < 650 else 0.0)

    # Complaints (fairness signal)
    score += 0.15 if cp < 15 else (0.08 if cp < 25 else 0.0)

    # Total reward
    score += 0.15 if tr > 10.0 else (0.08 if tr > 5.0 else 0.0)

    # Misuse handling (NEW)
    score += 0.10 if mh > 10 else (0.05 if mh > 5 else 0.0)

    # Fairness maintenance (NEW)
    score += 0.05 if fv < 0.6 else (0.02 if fv < 0.8 else 0.0)

    print(f"\n📊 Grader Breakdown:")
    print(f"   Demand Satisfaction : {ds:.3f}  → {'✅' if ds > 0.85 else '⚠️ '}")
    print(f"   Violations          : {v}       → {'✅' if v  < 20  else '⚠️ '}")
    print(f"   Total Cost          : {c:.1f}   → {'✅' if c  < 450 else '⚠️ '}")
    print(f"   Complaints          : {cp}      → {'✅' if cp < 15  else '⚠️ '}")
    print(f"   Best Reward         : {tr:.4f}  → {'✅' if tr > 10  else '⚠️ '}")
    print(f"   Misuse Handled      : {mh}      → {'✅' if mh > 10  else '⚠️ '}")
    print(f"   Fairness Score      : {fv:.3f}  → {'✅' if fv < 0.6 else '⚠️ '}")
    print(f"\n🏆 Task 2 Score: {score:.2f} / 1.00")
    return score


# ── Training Loop ─────────────────────────────────────────────
def train(episodes=1000):
    env = Task2Env(num_rooms=20, episode_hours=24)

    sample_obs = env.reset()
    obs_dim = len(sample_obs)

    agent = EnforcementAgent(obs_dim=obs_dim, action_count=6, n_bins=10)

    print(f"\n🟡 Task 2 — Fair Enforcement Under Misuse (Quality RL)")
    print(f"   Obs dim (augmented): {obs_dim}")
    print(f"   Features: base + [flagged_frac, avg_demand, priority_frac, fairness, step_frac, misuse_severity]")
    print(f"   Experience Replay: Enabled (buffer={agent.replay_buffer.buffer.maxlen}, batch={agent.batch_size})")
    print("=" * 70)

    best_reward = float('-inf')
    reward_log = []
    violation_log = []

    best_stats = None
    global_step = 0

    for ep in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        total_complaints = 0
        total_cost = 0.0
        total_misuse_handled = 0
        total_fairness = 0.0
        done = False
        step_in_ep = 0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done, global_step)
            obs = next_obs

            total_reward += reward
            total_complaints += info["complaints"]
            total_cost += info["cost"]
            total_misuse_handled += info["misuse_handled"]
            total_fairness += info["fairness_violation"]

            step_in_ep += 1
            global_step += 1

        ep_stats = env.episode_stats()
        reward_log.append(total_reward)
        violation_log.append(ep_stats["violations"])

        if total_reward > best_reward:
            best_reward = total_reward
            best_stats = {
                "demand_satisfaction": ep_stats["demand_satisfaction"],
                "violations": ep_stats["violations"],
                "total_cost": total_cost,
                "complaints": total_complaints,
                "total_reward": total_reward,
                "misuse_handled": total_misuse_handled,
                "fairness_violation": total_fairness / max(1, step_in_ep),
            }

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(reward_log[-100:])
            avg_violations = np.mean(violation_log[-100:])
            print(f"  Episode {ep+1:04d} | "
                  f"Avg Reward: {avg_reward:+7.4f} | "
                  f"Violations: {avg_violations:5.1f} | "
                  f"Best: {best_reward:+7.4f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Q-states: {len(agent.q_table):6d}")

    print("=" * 70)
    print(f"\n📈 Final Training Summary:")
    print(f"   Episodes trained: {episodes}")
    print(f"   Best reward: {best_reward:.4f}")
    print(f"   Q-table size: {len(agent.q_table)}")

    grade(best_stats)
    return agent


if __name__ == "__main__":
    train(episodes=1000)