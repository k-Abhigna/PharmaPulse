"""Robust anomaly detection for weekly demand series.

For each series:
1. Expected demand = centred rolling median of non-promo weeks, multiplied by the
   series' own promo uplift on promo weeks (so planned promotions are not flagged).
2. Score = log(actual / expected), standardised with the median absolute deviation (MAD),
   which a few extreme weeks cannot inflate the way a standard deviation can.
3. |robust z| above the threshold is flagged, then typed:
   data_error (zero or >5x / <2% of expected), stockout (below), spike (above).
   A high week straight after a stockout is the catch-up rebound, not a new spike.

`naive_zscore` is the comparison baseline: a plain 3-sigma rule on raw units.

    python -m pharmapulse.anomaly      # scores both methods against the injected labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["product", "region"]


def _expected(series: pd.DataFrame, window: int = 9) -> np.ndarray:
    y = series["units"].astype(float)
    non_promo = y.where(series["promo"] == 0)
    baseline = non_promo.rolling(window, center=True, min_periods=3).median()
    baseline = baseline.bfill().ffill()
    promo_mask = series["promo"] == 1
    uplift = (y[promo_mask] / baseline[promo_mask]).median() if promo_mask.any() else 1.0
    uplift = float(np.clip(uplift if np.isfinite(uplift) else 1.0, 1.0, 2.0))
    return np.where(promo_mask, baseline * uplift, baseline)


def detect(sales: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """Return `sales` with expected, robust_z, is_anomaly and anomaly_type columns."""
    out = []
    for _, s in sales.sort_values("week").groupby(KEYS, sort=False):
        s = s.copy()
        s["expected"] = _expected(s)
        r = np.log((s["units"] + 1) / (s["expected"] + 1))
        mad = np.median(np.abs(r - np.median(r))) or 1e-9
        s["robust_z"] = (r - np.median(r)) / (1.4826 * mad)
        flag = s["robust_z"].abs() > threshold
        ratio = (s["units"] + 1) / (s["expected"] + 1)
        kind = np.where((s["units"] == 0) | (ratio > 5) | (ratio < 0.02), "data_error",
                        np.where(s["robust_z"] < 0, "stockout", "spike"))
        kind = pd.Series(np.where(flag, kind, ""), index=s.index)
        # Catch-up week after a stockout belongs to the stockout event.
        prev = kind.shift(1).fillna("")
        kind = kind.mask((kind == "spike") & (prev == "stockout"), "stockout_rebound")
        s["anomaly_type"] = kind
        s["is_anomaly"] = flag
        out.append(s)
    return pd.concat(out).sort_index()


def naive_zscore(sales: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Baseline: |units - mean| / std > 3 per series. Blind to seasonality and promotions."""
    s = sales.copy()
    g = s.groupby(KEYS)["units"]
    z = (s["units"] - g.transform("mean")) / g.transform("std")
    s["is_anomaly"] = z.abs() > threshold
    s["anomaly_type"] = np.where(s["is_anomaly"], np.where(z < 0, "stockout", "spike"), "")
    return s


def _events(flags: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive flagged weeks of one series into events."""
    rows = []
    for key, s in flags[flags["is_anomaly"]].sort_values("week").groupby(KEYS):
        start = prev = None
        for w in s["week"]:
            if prev is not None and (w - prev).days <= 7:
                prev = w
                continue
            if start is not None:
                rows.append((*key, start, prev))
            start = prev = w
        rows.append((*key, start, prev))
    return pd.DataFrame(rows, columns=[*KEYS, "start", "end"])


def score(flags: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Event-level precision/recall. A labelled event counts as found if any flagged week
    falls inside it; the week after a stockout (the rebound) also counts as inside it."""
    events = _events(flags)
    lab = labels.copy()
    lab["end"] = lab["end"] + pd.Timedelta(days=7)  # include the rebound week

    def any_overlap(start, end, product, region, table) -> bool:
        t = table[(table["product"] == product) & (table["region"] == region)]
        return bool(((t["start"] <= end) & (t["end"] >= start)).any())

    found = pd.Series([any_overlap(r.start, r.end, r.product, r.region, events) for r in lab.itertuples()],
                      index=lab.index, dtype=bool)
    tp = sum(any_overlap(e.start, e.end, e.product, e.region, lab) for e in events.itertuples())
    precision = tp / len(events) if len(events) else 0.0
    recall = float(found.mean()) if len(lab) else 0.0
    by_kind = lab.assign(found=found).groupby("kind")["found"].mean().round(3).to_dict()
    return {
        "events_flagged": int(len(events)),
        "true_positives": tp,
        "false_positives": int(len(events) - tp),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "recall_by_kind": by_kind,
    }


if __name__ == "__main__":
    from .data import load

    sales, labels = load()
    for name, flags in [("robust (rolling median + MAD)", detect(sales)), ("naive 3-sigma", naive_zscore(sales))]:
        print(f"{name:32s}", score(flags, labels))
