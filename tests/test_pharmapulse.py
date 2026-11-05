import numpy as np
import pandas as pd
import pytest

from pharmapulse.anomaly import detect, naive_zscore, score
from pharmapulse.data import generate
from pharmapulse.forecast import Panel, backtest, seasonal_naive, summarize, wape
from pharmapulse.inventory import plan


def _frames(seed):
    sales, labels = generate(seed)
    sales["week"] = pd.to_datetime(sales["week"])
    labels["start"] = pd.to_datetime(labels["start"])
    labels["end"] = pd.to_datetime(labels["end"])
    return sales, labels


@pytest.fixture(scope="session")
def data():
    return _frames(7)


@pytest.fixture(scope="session")
def small(data):
    sales, labels = data
    keep = sales["product"].isin(["Paracetamol 500mg", "ORS Sachet", "Metformin 500mg"])
    return sales[keep], labels[labels["product"].isin(sales[keep]["product"].unique())]


def test_generator_is_reproducible_and_complete():
    a, _ = generate(7)
    b, _ = generate(7)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby(["product", "region"]).size().eq(156).all()
    assert (a["units"] >= 0).all()


def test_detector_ignores_planned_promotions(data):
    sales, labels = data
    flagged = detect(sales)
    promo_spikes = flagged[(flagged["promo"] == 1) & (flagged["anomaly_type"] == "spike")]
    assert len(promo_spikes) / flagged["promo"].sum() < 0.05


def test_detector_beats_naive_baseline_on_held_out_seed():
    sales, labels = _frames(8)  # threshold was tuned on seed 7
    robust = score(detect(sales), labels)
    base = score(naive_zscore(sales), labels)
    assert robust["f1"] > base["f1"] + 0.1
    assert robust["recall"] > 0.85 and robust["precision"] > 0.8


def test_seasonal_naive_uses_last_year():
    y = np.arange(200, dtype=float)[None, :]
    assert seasonal_naive(y, origin=120, h=3).tolist() == [[69.0, 70.0, 71.0]]


def test_wape():
    assert wape(np.array([10.0, 10.0]), np.array([12.0, 8.0])) == pytest.approx(0.2)


def test_cleaning_only_touches_history(small):
    sales, _ = small
    panel = Panel.from_sales(sales)
    origin = 120
    cleaned = panel.cleaned(origin)
    np.testing.assert_array_equal(cleaned[:, origin + 1:], panel.y[:, origin + 1:])
    assert (cleaned[:, : origin + 1] != panel.y[:, : origin + 1]).any()


def test_gbm_beats_baselines(small):
    sales, labels = small
    preds = backtest(sales, labels, models=["seasonal_naive", "naive", "gbm+clean"], origins_from_end=[17, 9])
    s = summarize(preds)
    assert s.index[0] == "gbm+clean"
    assert s.loc["gbm+clean", "wape_pct"] < 0.8 * s.loc["seasonal_naive", "wape_pct"]
    assert abs(s.loc["gbm+clean", "bias_pct"]) < 5


def test_no_future_leakage(small):
    """Changing data after the origin must not change the forecast."""
    sales, labels = small
    a = backtest(sales, None, models=["gbm"], origins_from_end=[9])
    tampered = sales.copy()
    last = tampered["week"] > tampered["week"].max() - pd.Timedelta(weeks=8)
    tampered.loc[last, "units"] = tampered.loc[last, "units"] * 5
    b = backtest(tampered, None, models=["gbm"], origins_from_end=[9])
    np.testing.assert_allclose(a["forecast"].values, b["forecast"].values)


def test_inventory_plan_scales_with_service_level(small):
    sales, labels = small
    preds = backtest(sales, labels, models=["gbm+clean"], origins_from_end=[9])
    future = preds[preds["origin"] == preds["origin"].max()].rename(columns={"forecast": "forecast"})
    lo = plan(future, preds, service_level=0.90)
    hi = plan(future, preds, service_level=0.99)
    assert (hi["safety_stock"].sum() > lo["safety_stock"].sum())
    assert (hi["order_up_to"] >= hi["demand_over_cover"]).all()
