"""Turn forecasts into stock targets: safety stock and order-up-to levels.

Safety stock = z(service level) x sigma(weekly forecast error) x sqrt(lead time + review period)
Order-up-to  = forecast demand over (lead time + review period) + safety stock

Sigma comes from the model's own backtest errors for that series (non-anomalous weeks),
so series that are harder to forecast carry more buffer.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def plan(forecast: pd.DataFrame, backtest_preds: pd.DataFrame, model: str = "gbm+clean",
         service_level: float = 0.95, lead_time_weeks: int = 2, review_weeks: int = 1) -> pd.DataFrame:
    z = NormalDist().inv_cdf(service_level)
    cover = lead_time_weeks + review_weeks
    errors = backtest_preds[(backtest_preds["model"] == model) & ~backtest_preds["in_anomaly"]]
    sigma = (errors.assign(err=errors["actual"] - errors["forecast"])
             .groupby(["product", "region"])["err"].std().rename("error_sd"))
    demand = (forecast[forecast["h"] <= cover].groupby(["product", "region", "therapy_class"])["forecast"]
              .sum().rename("demand_over_cover").reset_index())
    out = demand.merge(sigma.reset_index(), on=["product", "region"], how="left")
    out["safety_stock"] = z * out["error_sd"] * np.sqrt(cover)
    out["order_up_to"] = out["demand_over_cover"] + out["safety_stock"]
    out["buffer_pct"] = 100 * out["safety_stock"] / out["demand_over_cover"]
    cols = ["demand_over_cover", "error_sd", "safety_stock", "order_up_to", "buffer_pct"]
    out[cols] = out[cols].round(0)
    return out.sort_values("order_up_to", ascending=False).reset_index(drop=True)
