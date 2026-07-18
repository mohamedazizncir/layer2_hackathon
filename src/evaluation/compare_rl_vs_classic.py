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
    for index, metric in enumerate(comparable):
        ax.text(index - width / 2, 102, f"{classic[metric]:,.2f}",
                ha="center", va="bottom", fontsize=8, rotation=90)
        rl_index = rl[metric] / classic[metric] * 100
        ax.text(index + width / 2, rl_index + 2, f"{rl[metric]:,.2f}",
                ha="center", va="bottom", fontsize=8, rotation=90)
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

    # Literature-informed target scenario, not an evaluation of the saved model.
    # The 5%-15% effect-size range is an explicit conservative assumption.
    horizons = np.array([26, 52, 104])
    assumed_reductions = np.array([0.05, 0.10, 0.15])
    classic_projection = np.repeat(classic["chronic_unfulfilled"], len(horizons))
    rl_projection = classic_projection * (1 - assumed_reductions)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(horizons, classic_projection, "o-", lw=2.5, label="Classic", color="#64748b")
    ax.plot(horizons, rl_projection, "o-", lw=2.5, label="RL", color="#0f766e")
    for weeks, classic_value, rl_value in zip(horizons, classic_projection, rl_projection):
        ax.annotate(f"{classic_value:.2f}", (weeks, classic_value),
                    xytext=(0, 8), textcoords="offset points", ha="center")
        ax.annotate(f"{rl_value:.2f}", (weeks, rl_value),
                    xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set(xlabel="Illustrative deployment horizon (weeks)",
           ylabel="Chronic unfulfilled demand per episode (lower is better)",
           title="LITERATURE-INFORMED TARGET SCENARIO — NOT A FORECAST")
    ax.text(0.01, 0.02, "Assumed RL reduction vs classic: 5% / 10% / 15%",
            transform=ax.transAxes, fontsize=9, color="#475569")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(processed / "rl_vs_classic_long_term_projection.png", dpi=160)
    plt.close(fig)
    env.close()


if __name__ == "__main__":
    main()
