from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.digital_twin.pharmacy_simulator import PharmacyDepotSimulator, WEIGHT_NAMES
from src.rl_env.state_builder import STATE_FIELDS, state_dict_to_array


class PriorityWeightingEnv(gym.Env):
    """Gym wrapper that safely nudges allocation-weight logits over time."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        catalogue_scored_path,
        simulator_config=None,
        max_step_size=0.3,
        episode_length=26,
        reward_scale=1e-3,
    ):
        super().__init__()
        self.sim = PharmacyDepotSimulator(catalogue_scored_path, simulator_config)
        self.simulator = self.sim
        self.max_step_size = float(max_step_size)
        self.episode_length = int(episode_length)
        self.reward_scale = float(reward_scale)

        if self.max_step_size <= 0:
            raise ValueError("max_step_size must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(WEIGHT_NAMES),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(STATE_FIELDS),),
            dtype=np.float32,
        )
        self._weight_logits = np.zeros(len(WEIGHT_NAMES), dtype=np.float64)
        self.t = 0

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.simulator.rng = np.random.default_rng(seed)

        self._weight_logits = np.zeros(len(WEIGHT_NAMES), dtype=np.float64)
        self.t = 0
        state = self.simulator.reset()
        observation = state_dict_to_array(state)
        info = {"weights": self._softmax(self._weight_logits).astype(np.float32)}
        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        bounded_delta = np.clip(action, -1.0, 1.0) * self.max_step_size
        self._weight_logits += bounded_delta
        weights = self._softmax(self._weight_logits)

        state, raw_reward, _, kpis = self.simulator.step(weights)
        self.t += 1
        observation = state_dict_to_array(state)
        reward = float(raw_reward * self.reward_scale)
        terminated = False
        truncated = self.t >= self.episode_length
        info = {"weights": weights.astype(np.float32), **kpis}
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


def main():
    from gymnasium.utils.env_checker import check_env

    project_root = Path(__file__).resolve().parents[2]
    catalogue_path = project_root / "data" / "processed" / "catalogue_scored.parquet"
    environment = PriorityWeightingEnv(catalogue_path)
    check_env(environment)
    print("Gymnasium check_env passed")


if __name__ == "__main__":
    main()
