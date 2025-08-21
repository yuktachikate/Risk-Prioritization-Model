import os
import sys
from typing import Optional

import pandas as pd

from src.utils.config import load_yaml_config, parse_args, merge_config_with_args
from src.model.risk import generate_synthetic_assets, compute_risk_fields, map_columns
from src.model.optimization import greedy_select, ilp_select
from src.io.writers import write_csv, write_geojson, write_summary, ensure_dir


def load_input_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    config = merge_config_with_args(config, args)

    ensure_dir(config.output_dir)

    if args.input_csv:
        df = load_input_csv(args.input_csv)
        cols = map_columns(config.columns)
        df = compute_risk_fields(df, cols)
        # Ensure standardized columns for writers
        if cols.latitude != "lat":
            df = df.rename(columns={cols.latitude: "lat"})
        if cols.longitude != "lon":
            df = df.rename(columns={cols.longitude: "lon"})
        if cols.cost != "cost":
            df = df.rename(columns={cols.cost: "cost"})
    else:
        df = generate_synthetic_assets(config.num_assets, config.random_seed)

    if config.optimizer == "greedy":
        result = greedy_select(df, config.budget)
    else:
        result = ilp_select(df, config.budget)

    selected_mask = df.index.isin(result.selected_indices)
    df_out = df.copy()
    df_out["selected"] = selected_mask

    csv_path = write_csv(df_out, config.output_dir)
    geojson_path = write_geojson(df_out, config.output_dir)

    total_risk = float(df_out["risk"].sum())
    selected_risk = float(df_out.loc[selected_mask, "risk"].sum())
    risk_reduction = selected_risk  # baseline assumes unrepaired risk; selected mitigates that amount

    summary = {
        "optimizer": config.optimizer,
        "budget": config.budget,
        "total_assets": int(len(df_out)),
        "selected_assets": int(selected_mask.sum()),
        "total_cost_selected": float(df_out.loc[selected_mask, "cost"].sum()),
        "total_risk": total_risk,
        "selected_risk": selected_risk,
        "risk_reduction": risk_reduction,
        "outputs": {
            "csv": os.path.abspath(csv_path),
            "geojson": os.path.abspath(geojson_path),
        },
    }
    summary_path = write_summary(summary, config.output_dir)

    print("Generated outputs:")
    print(" -", os.path.abspath(csv_path))
    print(" -", os.path.abspath(geojson_path))
    print(" -", os.path.abspath(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
