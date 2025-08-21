from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AssetColumns:
    name: str
    latitude: str
    longitude: str
    pof: str
    cof: str
    cost: str
    condition: str


def generate_synthetic_assets(num_assets: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    names = [f"Asset_{i:04d}" for i in range(num_assets)]

    # Simulate a service territory roughly around a location (e.g., US lat/lon ranges)
    lats = rng.uniform(29.0, 49.0, size=num_assets)
    lons = rng.uniform(-123.0, -67.0, size=num_assets)

    # Condition score (1=poor, 5=excellent)
    condition = rng.integers(1, 6, size=num_assets)

    # PoF: Higher for worse condition
    base_pof = np.clip(rng.beta(2, 5, size=num_assets), 0.01, 0.99)
    pof = np.clip(base_pof * (6 - condition) / 5.0, 0.005, 0.99)

    # CoF: Monetized impact in dollars
    cof = rng.lognormal(mean=11.0, sigma=0.6, size=num_assets)

    # Cost: Repair/replacement cost in dollars
    cost = rng.lognormal(mean=10.5, sigma=0.5, size=num_assets)

    df = pd.DataFrame(
        {
            "name": names,
            "lat": lats,
            "lon": lons,
            "condition_score": condition,
            "pof": pof,
            "cof": cof,
            "cost": cost,
        }
    )
    return compute_risk_fields(df, AssetColumns(
        name="name",
        latitude="lat",
        longitude="lon",
        pof="pof",
        cof="cof",
        cost="cost",
        condition="condition_score",
    ))


def compute_risk_fields(df: pd.DataFrame, cols: AssetColumns) -> pd.DataFrame:
    # Ensure columns exist
    required = [cols.name, cols.latitude, cols.longitude, cols.pof, cols.cof, cols.cost]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    result["risk"] = result[cols.pof].astype(float) * result[cols.cof].astype(float)
    result["risk_per_cost"] = result["risk"] / result[cols.cost].astype(float)
    return result


def map_columns(config_columns: dict) -> AssetColumns:
    return AssetColumns(
        name=config_columns.get("name", "name"),
        latitude=config_columns.get("latitude", "lat"),
        longitude=config_columns.get("longitude", "lon"),
        pof=config_columns.get("pof", "pof"),
        cof=config_columns.get("cof", "cof"),
        cost=config_columns.get("cost", "cost"),
        condition=config_columns.get("condition", "condition_score"),
    )
