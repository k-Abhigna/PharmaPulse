"""Demand forecasting models and a rolling-origin backtest.

Models (8-week horizon, weekly data):
- naive          last observed week, repeated
- moving_avg_4   mean of the last 4 weeks
- seasonal_naive same week last year
- ets            Holt-Winters on log units (damped trend + 52-week seasonality), one model per series
- gbm            one global gradient-boosting model across all 48 series: lags, last-year values,
                 week of year, and the promotion calendar (promotions are planned, so known ahead)

The "+clean" variants first replace weeks the anomaly detector flags (stockouts, spikes,
data errors) with expected demand, using only history up to the forecast origin.

    python -m pharmapulse.forecast      # runs the backtest, writes reports/
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .anomaly import detect

HORIZON = 8
ORIGINS_FROM_END = [33, 25, 17, 9]  # 4 folds x 8 weeks = the last 32 weeks are tested
REPORTS = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class Panel:
    """Sales as aligned arrays: one row per (product, region) series, one column per week."""
    y: np.ndarray            # units
    promo: np.ndarray        # 0/1
    weeks: pd.DatetimeIndex
    meta: pd.DataFrame       # product, region, therapy_class per row

    @classmethod
    def from_sales(cls, sales: pd.DataFrame) -> "Panel":
        sales = sales.sort_values(["product", "region", "week"])
        y = sales.pivot_table(index=["product", "region"], columns="week", values="units").sort_index()
        p = sales.pivot_table(index=["product", "region"], columns="week", values="promo").sort_index()
        meta = (sales.drop_duplicates(["product", "region"]).set_index(["product", "region"])
                .loc[y.index, ["therapy_class"]].reset_index())
        return cls(y.to_numpy(float), p.to_numpy(int), pd.DatetimeIndex(y.columns), meta)

    def cleaned(self, origin: int) -> np.ndarray:
        """History up to `origin` with detected anomalies replaced by expected demand."""
        hist = pd.DataFrame({
            "week": np.tile(self.weeks[: origin + 1], len(self.meta)),
            "product": np.repeat(self.meta["product"].values, origin + 1),
            "region": np.repeat(self.meta["region"].values, origin + 1),
            "units": self.y[:, : origin + 1].ravel(),
            "promo": self.promo[:, : origin + 1].ravel(),
        })
        flagged = detect(hist)
        units = np.where(flagged["is_anomaly"], flagged["expected"], flagged["units"])
        out = self.y.copy()
        out[:, : origin + 1] = units.reshape(len(self.meta), origin + 1)
        return out


# ------------------------------------------------------------------ local models
def naive(y: np.ndarray, origin: int, h: int = HORIZON) -> np.ndarray:
    return np.repeat(y[:, origin : origin + 1], h, axis=1)


def moving_avg_4(y: np.ndarray, origin: int, h: int = HORIZON) -> np.ndarray:
    return np.repeat(y[:, origin - 3 : origin + 1].mean(axis=1, keepdims=True), h, axis=1)


def seasonal_naive(y: np.ndarray, origin: int, h: int = HORIZON) -> np.ndarray:
    return y[:, origin + 1 - 52 : origin + 1 - 52 + h]


def ets(y: np.ndarray, origin: int, h: int = HORIZON) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    preds = []
    for row in y[:, : origin + 1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(np.log1p(row), trend="add", damped_trend=True,
                                       seasonal="add", seasonal_periods=52).fit()
        preds.append(np.expm1(fit.forecast(h)))
    return np.clip(np.array(preds), 0, None)


# ------------------------------------------------------------------ global ML model
def _features(y: np.ndarray, promo: np.ndarray, weeks: pd.DatetimeIndex, therapy_codes: np.ndarray,
              t: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for every series at origin t, horizon h. Returns (X, scale)."""
    scale = y[:, t - 51 : t + 1].mean(axis=1) + 1.0
    target = t + h
    seas2 = y[:, target - 104] / scale if target - 104 >= 0 else np.full(len(y), np.nan)
    X = np.column_stack([
        np.full(len(y), h),
        y[:, t] / scale, y[:, t - 1] / scale, y[:, t - 2] / scale, y[:, t - 3] / scale,
        y[:, t - 3 : t + 1].mean(axis=1) / scale,
        y[:, t - 12 : t + 1].mean(axis=1) / scale,
        y[:, target - 52] / scale,
        y[:, target - 53 : target - 50].mean(axis=1) / scale,
        seas2,
        np.full(len(y), weeks[min(target, len(weeks) - 1)].isocalendar().week if target < len(weeks)
                else (weeks[-1] + pd.Timedelta(weeks=target - len(weeks) + 1)).isocalendar().week),
        promo[:, target] if target < promo.shape[1] else np.zeros(len(y)),
        promo[:, t],
        therapy_codes,
    ])
    return X, scale


class GlobalGBM:
    def __init__(self, panel: Panel, seed: int = 0):
        self.panel = panel
        self.codes = panel.meta["therapy_class"].astype("category").cat.codes.to_numpy()
        self.seed = seed

    def fit_predict(self, y: np.ndarray, origin: int, h: int = HORIZON, promo_future: np.ndarray | None = None):
        promo = self.panel.promo
        if promo_future is not None:  # forecasting beyond the data: append the planned promo calendar
            promo = np.concatenate([promo[:, : origin + 1], promo_future], axis=1)
        Xs, ys = [], []
        for t in range(52, origin - h + 1, 2):  # every other origin keeps training fast
            for hh in range(1, h + 1):
                X, scale = _features(y, promo, self.panel.weeks, self.codes, t, hh)
                Xs.append(X)
                ys.append(y[:, t + hh] / scale)
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                              max_leaf_nodes=31, categorical_features=[13],
                                              random_state=self.seed)
        model.fit(np.vstack(Xs), np.concatenate(ys))
        preds = []
        for hh in range(1, h + 1):
            X, scale = _features(y, promo, self.panel.weeks, self.codes, origin, hh)
            preds.append(model.predict(X) * scale)
        return np.clip(np.column_stack(preds), 0, None)


# ------------------------------------------------------------------ backtest
def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / actual.sum())


def backtest(sales: pd.DataFrame, labels: pd.DataFrame | None = None, models: list[str] | None = None,
             origins_from_end: list[int] = ORIGINS_FROM_END) -> pd.DataFrame:
    panel = Panel.from_sales(sales)
    n = panel.y.shape[1]
    gbm = GlobalGBM(panel)
    runners = {
        "naive": lambda y, o: naive(y, o),
        "moving_avg_4": lambda y, o: moving_avg_4(y, o),
        "seasonal_naive": lambda y, o: seasonal_naive(y, o),
        "ets": lambda y, o: ets(y, o),
        "gbm": lambda y, o: gbm.fit_predict(y, o),
    }
    plan = models or ["naive", "moving_avg_4", "seasonal_naive", "ets", "ets+clean", "gbm", "gbm+clean"]

    # Weeks inside a labelled anomaly (plus the rebound week) cannot be forecast by anyone;
    # they are flagged so metrics can be reported with and without them.
    anomalous = np.zeros_like(panel.y, dtype=bool)
    if labels is not None and len(labels):
        idx = {k: i for i, k in enumerate(zip(panel.meta["product"], panel.meta["region"]))}
        for r in labels.itertuples():
            i = idx[(r.product, r.region)]
            span = (panel.weeks >= r.start) & (panel.weeks <= r.end + pd.Timedelta(days=7))
            anomalous[i, span] = True

    rows = []
    for back in origins_from_end:
        origin = n - back
        cleaned = panel.cleaned(origin) if any("+clean" in m for m in plan) else None
        for name in plan:
            base, _, variant = name.partition("+")
            pred = runners[base](cleaned if variant == "clean" else panel.y, origin)
            for i in range(len(panel.meta)):
                for hh in range(HORIZON):
                    w = origin + 1 + hh
                    rows.append((name, origin, hh + 1, panel.weeks[w], panel.meta.at[i, "product"],
                                 panel.meta.at[i, "region"], panel.meta.at[i, "therapy_class"],
                                 panel.y[i, w], pred[i, hh], anomalous[i, w]))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "week", "product", "region", "therapy_class",
                                       "actual", "forecast", "in_anomaly"])


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        clean = g[~g["in_anomaly"]]
        return pd.Series({
            "wape_pct": 100 * wape(clean["actual"].values, clean["forecast"].values),
            "bias_pct": 100 * (clean["forecast"].sum() - clean["actual"].sum()) / clean["actual"].sum(),
            "wape_all_weeks_pct": 100 * wape(g["actual"].values, g["forecast"].values),
        })
    out = preds.groupby("model").apply(stats, include_groups=False).round(2)
    return out.sort_values("wape_pct")


def forecast_future(sales: pd.DataFrame, h: int = HORIZON, promo_future: np.ndarray | None = None) -> pd.DataFrame:
    """Forecast the h weeks after the data ends with the production model (gbm on cleaned history)."""
    panel = Panel.from_sales(sales)
    origin = panel.y.shape[1] - 1
    if promo_future is None:
        promo_future = np.zeros((len(panel.meta), h), dtype=int)
    pred = GlobalGBM(panel).fit_predict(panel.cleaned(origin), origin, h, promo_future)
    future_weeks = pd.date_range(panel.weeks[-1] + pd.Timedelta(weeks=1), periods=h, freq="W-MON")
    out = []
    for i in range(len(panel.meta)):
        for hh in range(h):
            out.append((future_weeks[hh], panel.meta.at[i, "product"], panel.meta.at[i, "region"],
                        panel.meta.at[i, "therapy_class"], hh + 1, round(float(pred[i, hh]), 1)))
    return pd.DataFrame(out, columns=["week", "product", "region", "therapy_class", "h", "forecast"])


def main() -> None:
    from .data import load

    sales, labels = load()
    REPORTS.mkdir(exist_ok=True)
    preds = backtest(sales, labels)
    preds.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    summary = summarize(preds)
    summary.to_csv(REPORTS / "backtest_summary.csv")
    by_class = (preds[~preds["in_anomaly"]].groupby(["therapy_class", "model"])
                .apply(lambda g: 100 * wape(g["actual"].values, g["forecast"].values), include_groups=False)
                .unstack().round(1))
    by_class.to_csv(REPORTS / "wape_by_therapy_class.csv")
    forecast_future(sales).to_csv(REPORTS / "forecast_next_8_weeks.csv", index=False)
    print("Backtest: 4 rolling origins x 8-week horizon x 48 series (WAPE excludes labelled anomaly weeks)\n")
    print(summary.to_string())
    print("\nWAPE % by therapy class\n", by_class[["seasonal_naive", "ets", "gbm+clean"]].to_string())


if __name__ == "__main__":
    main()
