import asyncio
import contextlib
from datetime import datetime, timezone
import logging
import os
from pathlib import Path

from fastapi import FastAPI
import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel
from stable_baselines3 import PPO

from src.digital_twin.pharmacy_simulator import WEIGHT_NAMES
from src.rl_env.state_builder import STATE_FIELDS, state_dict_to_array
from src.static_scoring.train_criticality_regressor import aggregate_dci_properties


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "rl_agent" / "ppo_priority_weighting.zip"
CATALOGUE_PATH = ROOT / "data" / "processed" / "catalogue_scored.parquet"
FALLBACK_WEIGHTS = np.array([0.55, 0.15, 0.1, 0.1, 0.05, 0.05], dtype=float)
current_context = {field: 0.0 for field in STATE_FIELDS}
app = FastAPI(title="StockCare Priority Service")


class DepotContext(BaseModel):
    stockout_rate_chronic: float
    stockout_rate_essential: float
    stockout_rate_comfort: float
    season_sin: float
    season_cos: float
    import_disruption_active: float
    epidemic_active: float


class PriorityRequest(BaseModel):
    pharmacy_id: str
    dci: str
    stockout_risk: float
    population_impact: float


def _weights_dict(weights):
    return {name: float(value) for name, value in zip(WEIGHT_NAMES, weights)}


def _compute_weights():
    fallback_used = False
    try:
        observation = state_dict_to_array(current_context)
        action, _ = app.state.model.predict(observation, deterministic=True)
        logits = np.asarray(action, dtype=float).reshape(-1)
        if logits.shape != (len(WEIGHT_NAMES),) or not np.isfinite(logits).all():
            raise ValueError("Model returned invalid logits")
        shifted = logits - logits.max()
        weights = np.exp(shifted) / np.exp(shifted).sum()
        if not np.isfinite(weights).all() or np.any(weights > 0.85):
            raise ValueError("Model returned unsafe weights")
    except Exception as error:
        LOGGER.warning("Using priority-weight fallback: %s", error)
        weights = FALLBACK_WEIGHTS.copy()
        fallback_used = True
    return weights, fallback_used


@app.on_event("startup")
async def startup():
    app.state.model = PPO.load(MODEL_PATH)
    catalogue = pd.read_parquet(CATALOGUE_PATH)
    dci_rows = aggregate_dci_properties(catalogue, include_criticality=True)
    app.state.dci_lookup = dci_rows.set_index("_dci_key")[
        [
            "criticality",
            "irreplaceability",
            "cold_chain_sensitive",
            "made_in_tunisia",
        ]
    ].to_dict("index")
    app.state.push_task = asyncio.create_task(_push_loop())


@app.on_event("shutdown")
async def shutdown():
    app.state.push_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await app.state.push_task


@app.post("/context")
def update_context(context: DepotContext):
    global current_context
    current_context = context.model_dump()
    return {"context": current_context}


@app.get("/weights")
def get_weights():
    weights, fallback_used = _compute_weights()
    return {"weights": _weights_dict(weights), "fallback_used": fallback_used}


@app.post("/priority-scores")
def priority_scores(items: list[PriorityRequest]):
    weights, fallback_used = _compute_weights()
    weights_used = _weights_dict(weights)
    results = []
    for item in items:
        properties = app.state.dci_lookup.get(item.dci.strip().upper())
        result = {
            "pharmacy_id": item.pharmacy_id,
            "dci": item.dci,
            "weights_used": weights_used,
            "fallback_used": fallback_used,
        }
        if properties is None:
            results.append(result | {"pri": None, "note": "dci_not_found"})
            continue
        pri = (
            weights[0] * properties["criticality"]
            + weights[1] * item.stockout_risk
            + weights[2] * properties["irreplaceability"]
            + weights[3] * float(properties["cold_chain_sensitive"])
            + weights[4] * item.population_impact
            + weights[5] * properties["made_in_tunisia"]
        )
        results.append(result | {"pri": float(np.clip(pri, 0.0, 1.0))})
    return results


async def _push_loop():
    interval = float(os.getenv("REFRESH_INTERVAL_SECONDS", "60"))
    webhook_url = os.getenv("LAYER3_WEBHOOK_URL")
    while True:
        await asyncio.sleep(interval)
        weights, _ = _compute_weights()
        if webhook_url:
            payload = {
                "weights": _weights_dict(weights),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(webhook_url, json=payload, timeout=10)
            except Exception as error:
                LOGGER.warning("Layer 3 weight push failed: %s", error)
