from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src.rl_env.priority_weighting_env import PriorityWeightingEnv


BASELINE_WEIGHTS = {
    "equal": [1 / 6] * 6,
    "criticality_heavy": [0.55, 0.15, 0.1, 0.1, 0.05, 0.05],
}
KPI_NAMES = [
    "chronic_unfulfilled",
    "cold_chain_unfulfilled",
    "critical_delay",
    "overstock",
]


def train(catalogue_path, output_dir, total_timesteps=30_000, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PriorityWeightingEnv(catalogue_path, simulator_config={"seed": seed})
    eval_env = PriorityWeightingEnv(
        catalogue_path, simulator_config={"seed": seed + 1}
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=2_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.95,
        seed=seed,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(output_dir / "ppo_priority_weighting.zip")
    eval_env.close()
    return model, env


def evaluate_fixed_weights(env, weights, n_episodes, seed, return_kpis=False):
    weights = np.asarray(weights, dtype=float)
    episode_totals = []
    for episode in range(n_episodes):
        env.sim.rng = np.random.default_rng(seed + episode)
        env.sim.reset()
        totals = {name: 0.0 for name in KPI_NAMES}
        totals["total_reward"] = 0.0
        for _ in range(env.episode_length):
            _, raw_reward, _, kpis = env.sim.step(weights)
            for name in KPI_NAMES:
                totals[name] += kpis[name]
            totals["total_reward"] += raw_reward * env.reward_scale
        episode_totals.append(totals)
    means = {
        name: float(np.mean([episode[name] for episode in episode_totals]))
        for name in [*KPI_NAMES, "total_reward"]
    }
    return means if return_kpis else means["total_reward"]


def evaluate_policy(env, model, n_episodes, seed, return_kpis=False):
    episode_totals = []
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        totals = {name: 0.0 for name in KPI_NAMES}
        totals["total_reward"] = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            for name in KPI_NAMES:
                totals[name] += info[name]
            totals["total_reward"] += reward
        episode_totals.append(totals)
    means = {
        name: float(np.mean([episode[name] for episode in episode_totals]))
        for name in [*KPI_NAMES, "total_reward"]
    }
    return means if return_kpis else means["total_reward"]


def _print_comparison(results):
    print(f"{'Approach':<24} {'Mean episode reward':>20}")
    print(f"{'-' * 24} {'-' * 20}")
    for name, reward in results:
        print(f"{name:<24} {reward:>20.6f}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    catalogue_path = project_root / "data" / "processed" / "catalogue_scored.parquet"
    output_dir = project_root / "models" / "rl_agent"

    model, env = train(catalogue_path, output_dir)
    evaluation_seed = 999
    n_episodes = 20
    results = [
        (
            "trained",
            evaluate_policy(env, model, n_episodes, evaluation_seed),
        ),
        (
            "equal",
            evaluate_fixed_weights(
                env,
                BASELINE_WEIGHTS["equal"],
                n_episodes,
                evaluation_seed,
            ),
        ),
        (
            "criticality_heavy",
            evaluate_fixed_weights(
                env,
                BASELINE_WEIGHTS["criticality_heavy"],
                n_episodes,
                evaluation_seed,
            ),
        ),
    ]
    _print_comparison(results)

    trained_reward = results[0][1]
    assert trained_reward > results[1][1], "Trained PPO did not beat equal weights"
    assert trained_reward > results[2][1], (
        "Trained PPO did not beat criticality-heavy weights"
    )
    env.close()


if __name__ == "__main__":
    main()
