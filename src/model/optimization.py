from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class OptimizationResult:
    selected_indices: List[int]
    total_cost: float
    total_risk_selected: float


def greedy_select(df: pd.DataFrame, budget: float) -> OptimizationResult:
    # Sort by risk-per-cost descending, then pick until budget is exhausted
    sorted_df = df.sort_values("risk_per_cost", ascending=False).reset_index(drop=False)
    selected = []
    total_cost = 0.0
    total_risk = 0.0
    for _, row in sorted_df.iterrows():
        cost = float(row["cost"]) if "cost" in row else float(row.get("Cost", 0))
        if total_cost + cost <= budget:
            selected.append(int(row["index"]))
            total_cost += cost
            total_risk += float(row["risk"]) if "risk" in row else float(row.get("Risk", 0))
    return OptimizationResult(selected_indices=selected, total_cost=total_cost, total_risk_selected=total_risk)


def ilp_select(df: pd.DataFrame, budget: float) -> OptimizationResult:
    # 0/1 knapsack: maximize risk subject to cost <= budget
    try:
        import pulp as pl
    except Exception as exc:
        raise RuntimeError("PuLP is required for ILP optimizer. Install with `pip install pulp`.") from exc

    n = len(df)
    model = pl.LpProblem("AssetSelection", pl.LpMaximize)
    x = [pl.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pl.LpBinary) for i in range(n)]

    risk = df["risk"].astype(float).tolist()
    cost = df["cost"].astype(float).tolist()

    model += pl.lpSum(risk[i] * x[i] for i in range(n))
    model += pl.lpSum(cost[i] * x[i] for i in range(n)) <= float(budget)

    model.solve(pl.PULP_CBC_CMD(msg=False))

    selected = [i for i in range(n) if x[i].value() is not None and x[i].value() >= 0.99]
    total_cost = float(sum(cost[i] for i in selected))
    total_risk = float(sum(risk[i] for i in selected))
    return OptimizationResult(selected_indices=selected, total_cost=total_cost, total_risk_selected=total_risk)
