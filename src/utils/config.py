import argparse
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List

import yaml


@dataclass
class AppConfig:
    budget: float
    optimizer: str
    random_seed: int
    num_assets: int
    output_dir: str
    columns: Dict[str, str]


def load_yaml_config(config_path: str) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(
        budget=float(data.get("budget", 1_000_000)),
        optimizer=str(data.get("optimizer", "greedy")).lower(),
        random_seed=int(data.get("random_seed", 42)),
        num_assets=int(data.get("num_assets", 150)),
        output_dir=str(data.get("output_dir", "outputs")),
        columns=data.get(
            "columns",
            {
                "name": "name",
                "latitude": "lat",
                "longitude": "lon",
                "pof": "pof",
                "cof": "cof",
                "cost": "cost",
                "condition": "condition_score",
            },
        ),
    )


def _parse_number(value: Optional[Union[str, float, int]], default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "").replace("_", "")
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError("Invalid numeric value: {}".format(value))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk-based asset prioritization")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "config.yaml"))
    parser.add_argument("--budget", type=str, default=None)
    parser.add_argument("--optimizer", type=str, choices=["greedy", "ilp"], default=None)
    parser.add_argument("--random-seed", type=str, default=None)
    parser.add_argument("--num-assets", type=str, default=None, help="Num synthetic assets to generate if no CSV provided")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to asset CSV. If omitted, synthetic data is generated")
    parser.add_argument("--lat-col", type=str, default=None)
    parser.add_argument("--lon-col", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def merge_config_with_args(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    budget = _parse_number(args.budget, config.budget)
    optimizer = str(args.optimizer).lower() if args.optimizer is not None else config.optimizer
    random_seed = int(_parse_number(args.random_seed, config.random_seed))
    num_assets = int(_parse_number(args.num_assets, config.num_assets))
    output_dir = str(args.output_dir) if args.output_dir is not None else config.output_dir

    columns = dict(config.columns)
    if args.lat_col:
        columns["latitude"] = args.lat_col
    if args.lon_col:
        columns["longitude"] = args.lon_col

    return AppConfig(
        budget=budget,
        optimizer=optimizer,
        random_seed=random_seed,
        num_assets=num_assets,
        output_dir=output_dir,
        columns=columns,
    )
