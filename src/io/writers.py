from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(df: pd.DataFrame, output_dir: str, filename: str = "asset_prioritization_results.csv") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def write_geojson(df: pd.DataFrame, output_dir: str, filename: str = "asset_prioritization_results.geojson") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)

    # Build features
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])],
            },
            "properties": {k: (None if pd.isna(v) else v) for k, v in row.drop(["lat", "lon"]).to_dict().items()},
        }
        features.append(feature)

    obj = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def write_summary(summary: Dict, output_dir: str, filename: str = "summary.json") -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path
