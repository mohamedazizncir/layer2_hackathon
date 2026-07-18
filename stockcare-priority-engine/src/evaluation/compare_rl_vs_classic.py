from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.rl_agent.train_ppo import (
    BASELINE_WEIGHTS,
    KPI_NAMES,
    evaluate_fixed_weights,
    evaluate_policy,
)
from src.rl_env.priority_weighting_env import PriorityWeightingEnv


LABELS = {
    "chronic_unfulfilled": "Chronic unfulfilled",
    "cold_chain_unfulfilled": "Cold-chain unfulfilled",
    "critical_delay": "Critical delay",
    "overstock": "Overstock",
    "total_reward": "Total reward",
}


def main():
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    env = PriorityWeightingEnv(processed / "catalogue_scored.parquet")
    model = PPO.load(root / "models" / "rl_agent" / "ppo_priority_weighting.zip")
    classic = evaluate_fixed_weights(
        env, BASELINE_WEIGHTS["criticality_heavy"], 20, 999, return_kpis=True
    )
    rl = evaluate_policy(env, model, 20, 999, return_kpis=True)
    metrics = [*KPI_NAMES, "total_reward"]

    lines = [
        "| KPI | Classic (fixed weights) | RL (adaptive) | % improvement |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        improvement_text = "N/A"
        if classic[metric] != 0:
            improvement = (classic[metric] - rl[metric]) / classic[metric] * 100
            improvement_text = f"{improvement:.2f}%"
        lines.append(
            f"| {LABELS[metric]} | {classic[metric]:.4f} | "
            f"{rl[metric]:.4f} | {improvement_text} |"
        )
    table = "\n".join(lines)
    print(table)
    (processed / "rl_vs_classic_comparison.md").write_text(table + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    comparable = [m for m in KPI_NAMES if classic[m] != 0]
    x, width = np.arange(len(comparable)), 0.36
    fig, (ax, reward_ax) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax.bar(x - width / 2, [100] * len(x), width, label="Classic", color="#64748b")
    ax.bar(x + width / 2, [rl[m] / classic[m] * 100 for m in comparable],
           width, label="RL", color="#0f766e")
    ax.set_xticks(x, [LABELS[m] for m in comparable], rotation=15, ha="right")
    ax.set_ylabel("KPI index (Classic = 100; lower is better)")
    ax.set_title("Operational KPIs — 20 held-out episodes")
    ax.legend(frameon=False)
    reward_ax.bar(["Classic", "RL"], [classic["total_reward"], rl["total_reward"]],
                  color=["#64748b", "#0f766e"])
    reward_ax.set_title("Mean total reward\n(higher is better)")
    reward_ax.set_ylabel("Scaled episode reward")
    fig.suptitle("StockCare: Adaptive RL vs Best Static Heuristic", fontsize=16)
    fig.tight_layout()
    fig.savefig(processed / "rl_vs_classic_comparison.png", dpi=160)
    plt.close(fig)

    horizons, classic_projection, rl_projection = [26, 52, 104], [], []
    for horizon in horizons:
        env.episode_length = horizon
        classic_projection.append(evaluate_fixed_weights(
            env, BASELINE_WEIGHTS["criticality_heavy"], 20, 999
        ) / horizon)
        rl_projection.append(evaluate_policy(env, model, 20, 999) / horizon)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(horizons, classic_projection, "o-", lw=2.5, label="Classic", color="#64748b")
    ax.plot(horizons, rl_projection, "o-", lw=2.5, label="RL", color="#0f766e")
    ax.set(xlabel="Evaluation horizon (weeks)", ylabel="Mean reward per week",
           title="SIMULATED PROJECTION — Not a Real-World Benchmark")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(processed / "rl_vs_classic_long_term_projection.png", dpi=160)
    plt.close(fig)
    env.close()


if __name__ == "__main__":
    main()
