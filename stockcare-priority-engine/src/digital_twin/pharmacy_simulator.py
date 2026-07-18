from pathlib import Path

import numpy as np
import pandas as pd


WEIGHT_NAMES = [
    "criticality",
    "stockout_risk",
    "irreplaceability",
    "cold_chain",
    "population_impact",
    "local_manufacturing",
]

DEFAULT_CONFIG = {
    "n_pharmacies": 20,
    "n_medicines": 60,
    "horizon_weeks": 2,
    "depot_capacity_ratio": 0.55,
    "import_disruption_prob": 0.20,
    "epidemic_prob": 0.15,
    "seed": 42,
}


class PharmacyDepotSimulator:
    """Minimal depot-allocation simulator with deliberate supply scarcity."""

    def __init__(self, catalogue_scored_path, config=None):
        self.config = DEFAULT_CONFIG | (config or {})
        self.n_pharmacies = int(self.config["n_pharmacies"])
        self.n_medicines = int(self.config["n_medicines"])
        self.horizon_weeks = int(self.config["horizon_weeks"])
        self.depot_capacity_ratio = float(self.config["depot_capacity_ratio"])
        self.import_disruption_prob = float(
            self.config["import_disruption_prob"]
        )
        self.epidemic_prob = float(self.config["epidemic_prob"])
        self.rng = np.random.default_rng(self.config["seed"])

        if self.n_pharmacies <= 0 or self.n_medicines <= 0:
            raise ValueError("n_pharmacies and n_medicines must be positive")
        if self.horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive")
        if not 0 <= self.depot_capacity_ratio <= 1:
            raise ValueError("depot_capacity_ratio must be between 0 and 1")

        catalogue = pd.read_parquet(catalogue_scored_path)
        required = {
            "dci",
            "criticality",
            "irreplaceability",
            "cold_chain_sensitive",
            "made_in_tunisia",
        }
        missing = required.difference(catalogue.columns)
        if missing:
            raise ValueError(f"Scored catalogue is missing columns: {sorted(missing)}")

        work = catalogue.loc[catalogue["criticality"].notna()].copy()
        work["_dci_key"] = work["dci"].astype("string").str.strip().str.upper()
        work = work.loc[work["_dci_key"].notna() & work["_dci_key"].ne("")]

        aggregations = {
            "criticality": ("criticality", "first"),
            "irreplaceability": ("irreplaceability", "mean"),
            "cold_chain_sensitive": ("cold_chain_sensitive", "max"),
            "made_in_tunisia": ("made_in_tunisia", "mean"),
        }
        if "n_amm" in work.columns:
            aggregations["n_amm"] = ("n_amm", "max")
        else:
            aggregations["n_amm"] = ("_dci_key", "size")

        medicines = (
            work.groupby("_dci_key", as_index=False)
            .agg(**aggregations)
            .rename(columns={"_dci_key": "dci"})
        )
        if len(medicines) < self.n_medicines:
            raise ValueError(
                f"Requested {self.n_medicines} medicines, but only "
                f"{len(medicines)} scored DCI values are available"
            )

        sampled_positions = self.rng.choice(
            len(medicines), size=self.n_medicines, replace=False
        )
        self.medicines = medicines.iloc[sampled_positions].reset_index(drop=True)
        self.criticality = self.medicines["criticality"].to_numpy(dtype=float)
        self.irreplaceability = self.medicines["irreplaceability"].to_numpy(
            dtype=float
        )
        self.cold_chain_sensitive = self.medicines[
            "cold_chain_sensitive"
        ].to_numpy(dtype=bool)
        self.made_in_tunisia = self.medicines["made_in_tunisia"].to_numpy(
            dtype=float
        )
        self.n_amm = self.medicines["n_amm"].to_numpy(dtype=float)
        self.base_demand_rate = 1.0 + np.log1p(self.n_amm)

        self.import_disruption_active = False
        self.epidemic_active = False
        self.reset()

    def reset(self):
        self.pharmacy_size = self.rng.uniform(0.5, 1.5, self.n_pharmacies)
        self.population_impact = self.rng.uniform(
            0.2, 1.0, (self.n_pharmacies, self.n_medicines)
        )
        self.target_stock = (
            self.base_demand_rate[None, :]
            * self.pharmacy_size[:, None]
            * self.horizon_weeks
        )
        self.stock = self.target_stock * self.rng.uniform(
            0.8, 1.2, (self.n_pharmacies, self.n_medicines)
        )
        self.consecutive_stockout_weeks = np.zeros(
            (self.n_pharmacies, self.n_medicines), dtype=int
        )
        self.week = 0
        self.import_disruption_active = False
        self.epidemic_active = False
        return self._state()

    def _bucket_stockout_rate(self, medicine_mask):
        if not np.any(medicine_mask):
            return 0.0
        return float(np.mean(self.stock[:, medicine_mask] < 0))

    def _state(self):
        chronic = self.criticality > 0.55
        essential = (self.criticality > 0.3) & (self.criticality <= 0.55)
        comfort = self.criticality <= 0.3
        season_angle = 2.0 * np.pi * (self.week % 52) / 52.0
        return {
            "stockout_rate_chronic": self._bucket_stockout_rate(chronic),
            "stockout_rate_essential": self._bucket_stockout_rate(essential),
            "stockout_rate_comfort": self._bucket_stockout_rate(comfort),
            "season_sin": float(np.sin(season_angle)),
            "season_cos": float(np.cos(season_angle)),
            "import_disruption_active": float(self.import_disruption_active),
            "epidemic_active": float(self.epidemic_active),
            "week": self.week,
        }

    def step(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(WEIGHT_NAMES),):
            raise ValueError(f"weights must have shape ({len(WEIGHT_NAMES)},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        weight_sum = weights.sum()
        weights = (
            weights / weight_sum
            if weight_sum > 0
            else np.full(len(WEIGHT_NAMES), 1.0 / len(WEIGHT_NAMES))
        )

        self.import_disruption_active = (
            self.rng.random() < self.import_disruption_prob
        )
        disrupted_medicine = (
            int(self.rng.integers(self.n_medicines))
            if self.import_disruption_active
            else None
        )

        self.epidemic_active = self.rng.random() < self.epidemic_prob
        epidemic_multiplier = np.ones(self.n_medicines, dtype=float)
        if self.epidemic_active:
            epidemic_count = max(1, int(round(0.30 * self.n_medicines)))
            epidemic_medicines = self.rng.choice(
                self.n_medicines, size=epidemic_count, replace=False
            )
            epidemic_multiplier[epidemic_medicines] = 1.8

        demand_rate = (
            self.base_demand_rate[None, :]
            * self.pharmacy_size[:, None]
            * epidemic_multiplier[None, :]
        )
        demand = self.rng.poisson(demand_rate)
        self.stock -= demand

        safety_stock = 0.2 * self.target_stock
        gap = np.maximum(0.0, self.target_stock + safety_stock - self.stock)
        has_query = gap > 0.3 * self.target_stock

        currently_stocked_out = self.stock < 0
        self.consecutive_stockout_weeks = np.where(
            currently_stocked_out,
            self.consecutive_stockout_weeks + 1,
            0,
        )
        stockout_risk = np.clip(self.consecutive_stockout_weeks / 4.0, 0.0, 1.0)

        priority = (
            weights[0] * self.criticality[None, :]
            + weights[1] * stockout_risk
            + weights[2] * self.irreplaceability[None, :]
            + weights[3] * self.cold_chain_sensitive[None, :]
            + weights[4] * self.population_impact
            + weights[5] * self.made_in_tunisia[None, :]
        )

        total_capacity = self.depot_capacity_ratio * gap[has_query].sum()
        allocated = np.zeros_like(gap)
        query_positions = np.flatnonzero(has_query.ravel())
        ranked_positions = query_positions[
            np.argsort(-priority.ravel()[query_positions], kind="stable")
        ]
        remaining_capacity = float(total_capacity)
        for flat_position in ranked_positions:
            if remaining_capacity <= 0:
                break
            pharmacy_index, medicine_index = np.unravel_index(
                flat_position, gap.shape
            )
            quantity = min(gap[pharmacy_index, medicine_index], remaining_capacity)
            allocated[pharmacy_index, medicine_index] = quantity
            remaining_capacity -= quantity

        if disrupted_medicine is not None:
            allocated[:, disrupted_medicine] *= 0.3

        fulfilled = np.minimum(allocated, gap)
        self.stock += fulfilled
        unfulfilled_gap = gap - fulfilled

        is_chronic = self.criticality > 0.55
        queried_gap = unfulfilled_gap * has_query
        chronic_unfulfilled = float(
            np.sum(queried_gap * is_chronic[None, :])
        )
        cold_chain_unfulfilled = float(
            np.sum(queried_gap * self.cold_chain_sensitive[None, :])
        )
        critical_delay = float(
            np.sum(queried_gap * self.criticality[None, :])
        )
        overstock = float(
            np.sum(np.clip(self.stock - 1.5 * self.target_stock, 0.0, None))
        )
        n_stockout_pairs = int(np.count_nonzero(self.stock < 0))
        local_manufacturing_fulfilled = float(
            np.sum(fulfilled * self.made_in_tunisia[None, :])
        )

        reward = (
            -1.0 * chronic_unfulfilled
            - 1.5 * cold_chain_unfulfilled
            - 0.5 * critical_delay
            - 0.3 * overstock
        )
        kpis = {
            "chronic_unfulfilled": chronic_unfulfilled,
            "cold_chain_unfulfilled": cold_chain_unfulfilled,
            "critical_delay": critical_delay,
            "overstock": overstock,
            "n_stockout_pairs": n_stockout_pairs,
            "local_manufacturing_fulfilled": local_manufacturing_fulfilled,
        }

        self.week += 1
        return self._state(), float(reward), False, kpis


def _run_policy(catalogue_path, weights, seed=42):
    simulator = PharmacyDepotSimulator(catalogue_path, {"seed": seed})
    total_chronic_unfulfilled = 0.0
    total_local_fulfilled = 0.0
    for _ in range(26):
        _, _, _, kpis = simulator.step(weights)
        total_chronic_unfulfilled += kpis["chronic_unfulfilled"]
        total_local_fulfilled += kpis["local_manufacturing_fulfilled"]
    return total_chronic_unfulfilled, total_local_fulfilled


def main():
    project_root = Path(__file__).resolve().parents[2]
    catalogue_path = project_root / "data" / "processed" / "catalogue_scored.parquet"

    equal_total, equal_local = _run_policy(catalogue_path, [1 / 6] * 6, seed=42)
    criticality_heavy_total, _ = _run_policy(
        catalogue_path, [0.55, 0.15, 0.1, 0.1, 0.05, 0.05], seed=42
    )
    _, local_heavy = _run_policy(
        catalogue_path, [0.1, 0.1, 0.1, 0.1, 0.1, 0.5], seed=42
    )
    print(f"Equal weights chronic_unfulfilled: {equal_total:.6f}")
    print(
        "Criticality-heavy chronic_unfulfilled: "
        f"{criticality_heavy_total:.6f}"
    )
    print(f"Equal weights local fulfilled: {equal_local:.6f}")
    print(f"Local-heavy local fulfilled: {local_heavy:.6f}")
    assert criticality_heavy_total < equal_total, (
        "Criticality-heavy weights must produce lower chronic_unfulfilled than "
        "equal weights"
    )
    assert local_heavy > equal_local, (
        "Local-manufacturing-heavy weights must fulfill more locally made stock "
        "than equal weights"
    )


if __name__ == "__main__":
    main()
