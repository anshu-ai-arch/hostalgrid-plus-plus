"""
env/hostelgrid_env.py

HostelGridEnv — Meta OpenEnv compliant environment.

Interface:
    env = HostelGridEnv(mode="easy")
    obs  = env.reset()           → Observation (Pydantic)
    obs, reward, done, info = env.step(action)
                                 → Observation, Reward, bool, dict
    cur  = env.state()           → Observation (current)
"""

from simulation.hostel  import Hostel
from env.observation    import Observation
from env.action         import Action
from env.reward_model   import Reward
from env.reward         import compute_reward

MAX_STEPS  = 50
STATE_SIZE = 9    # features per room


class HostelGridEnv:
    """
    Meta OpenEnv compliant 10-room hostel environment.

    Modes:
        easy   — stable occupancy, no constraints
        medium — fairness + power budget (6000W)
        hard   — adversarial + tight budget (3500W) + complaints
    """

    def __init__(self, mode: str = "easy", seed: int = 42):
        assert mode in ("easy", "medium", "hard"), \
            f"mode must be easy/medium/hard, got {mode}"
        self.mode     = mode
        self.hostel   = Hostel(mode=mode, seed=seed)
        self.step_n   = 0
        self._current_obs: Observation = None

    # ── OpenEnv required interface ────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        Returns typed Observation (Pydantic model).
        """
        self.hostel.reset()
        self.step_n = 0
        obs = Observation.from_hostel(self.hostel, step=0)
        self._current_obs = obs
        return obs

    def step(self, action: Action) -> tuple:
        """
        Apply action, advance environment, return results.

        Args:
            action : Action (Pydantic model)
                     Use Action.from_list([0..7]*10) to construct.

        Returns:
            obs    : Observation
            reward : Reward
            done   : bool
            info   : dict
        """
        # Accept both Action model and raw list
        if isinstance(action, list):
            action = Action.from_list(action)

        # 1. Apply actions to hostel
        self.hostel.apply_actions(action.to_list())

        # 2. Compute reward
        total_reward, per_room = compute_reward(self.hostel)

        # 3. Build typed Reward
        reward = Reward.from_hostel(
            total    = total_reward,
            per_room = per_room,
            hostel   = self.hostel,
            step     = self.step_n,
        )

        # 4. Advance simulation
        self.hostel.step()
        self.step_n += 1
        done = self.step_n >= MAX_STEPS

        # 5. Build next Observation
        obs = Observation.from_hostel(self.hostel, step=self.step_n)
        self._current_obs = obs

        # 6. Info dict
        info = {
            "step":            self.step_n,
            "per_room": per_room,
            "satisfied":       reward.satisfied_rooms,
            "complaints":      reward.total_complaints,
            "hp_satisfied":    reward.hp_satisfied,
            "power_used":      reward.power_used,
            "power_budget":    reward.power_budget,
            "over_budget":     reward.over_budget,
            "heatwave":        self.hostel.heatwave,
        }

        return obs, reward, done, info

    def state(self) -> Observation:
        """
        Return current environment state without advancing.
        Required by Meta OpenEnv spec.
        """
        if self._current_obs is None:
            return self.reset()
        return self._current_obs

    # ── Convenience ───────────────────────────────────────────────

    def render(self):
        print(self.hostel)

    @property
    def num_rooms(self) -> int:
        return 10

    @property
    def state_size(self) -> int:
        return STATE_SIZE

    @property
    def num_actions(self) -> int:
        return 8

    @property
    def current_step(self) -> int:
        return self.step_n

    def __repr__(self):
        return (f"HostelGridEnv(mode={self.mode} "
                f"step={self.step_n}/{MAX_STEPS})")