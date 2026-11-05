# PharmaPulse: Pharma Demand Forecasting & Anomaly Detection

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

PharmaPulse forecasts weekly demand for 12 medicines across 4 regions (48 series, 3 years). It also flags
stockouts, demand spikes and data-entry errors, and turns the forecast into safety-stock and order-up-to
levels for supply planning.

| Result | Value |
|---|---|
| Forecast error, best model (WAPE, 8-week horizon) | **10.9%**, versus 18.4% for seasonal naive: **41% lower error** |
| Forecast bias | **-0.7%** (close to unbiased) |
| Anomaly detection, held-out data (event level) | **89% precision, 92% recall** (F1 0.91), versus F1 0.74 for a 3-sigma rule |
| Stockouts caught | **95%**, versus 51% for the 3-sigma rule |

The synthetic demand has about 10.3% WAPE of pure random noise built in, so the best model is close to the
lowest error possible on this data.

## Backtest

Rolling-origin backtest with 4 forecast origins over the last 32 weeks, an 8-week horizon and all 48 series.
WAPE excludes weeks inside labelled anomalies, because no model can foresee a stockout. The last column
includes them.

| Model | WAPE % | Bias % | WAPE % (all weeks) |
|---|---|---|---|
| **GBM, global model + anomaly cleaning** | **10.93** | -0.65 | 14.77 |
| GBM, global model | 11.34 | -0.35 | 15.20 |
| Holt-Winters ETS + anomaly cleaning | 13.62 | -1.82 | 17.30 |
| Moving average (4 weeks) | 16.41 | 0.80 | 19.78 |
| Holt-Winters ETS | 17.19 | -3.47 | 20.87 |
| Seasonal naive (same week last year) | 18.43 | -4.92 | 21.94 |
| Naive (last week) | 18.45 | -1.33 | 21.68 |

**What the backtest shows**
- **Cleaning anomalies before training matters.** ETS error drops from 17.2% to 13.6%: one stockout in last
  year's history otherwise gets copied into this year's seasonal forecast.
- **One global model beats 48 local ones.** The gradient-boosting model learns seasonality across products in
  the same therapy class and uses the **promotion calendar**, which promotions make known in advance.
- **Gains hold across therapy classes.** GBM + cleaning has 10–12.5% WAPE in every class, against 14–21% for
  seasonal naive (`reports/wape_by_therapy_class.csv`).

## Anomaly detection

The detector compares each week with an expected value: a centred rolling median of non-promotion weeks,
scaled by the series' own promotion uplift. It then scores the log ratio with a **MAD-based robust z-score**.
Planned promotions aren't flagged, and a catch-up week after a stockout counts as part of the stockout, not as
a new spike.

The threshold (4.0) was **tuned on one data seed and evaluated on another**:

| Method (held-out seed) | Precision | Recall | F1 | Stockout recall |
|---|---|---|---|---|
| Rolling median + MAD (this project) | 0.89 | 0.92 | **0.91** | 0.95 |
| Plain 3-sigma per series | 0.90 | 0.62 | 0.74 | 0.51 |

The 3-sigma rule misses half of the stockouts because seasonality hides them. A 40% drop in a peak-season
week can still sit inside the yearly range.

## Stock plan

`pharmapulse/inventory.py` turns the forecast into targets:
safety stock = z(service level) × σ(the model's backtest error for that series) × √(lead time + review period).
Series that are harder to forecast get a bigger buffer. In the dashboard you can change the service level, lead
time and review period.

## Data

The data is synthetic (`pharmapulse/data.py`, fixed seed), modelled on Indian pharma demand patterns:
- monsoon peaks for antibiotics and analgesics, winter peaks for respiratory products, spring allergy season,
  and summer demand for ORS
- growth trends (fastest for chronic therapies and Vitamin D)
- 3–5 promotions per series, plus multiplicative noise
- **124 labelled anomalies** (stockouts, outbreak spikes, data errors) so the detector can be scored

## Quick start (Windows PowerShell)

```powershell
cd "C:\Job\Resume\My Projects\AI Project\PharmaPulse"
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pharmapulse.data        # generate data/ (seconds)
.venv\Scripts\python -m pharmapulse.anomaly     # score detectors
.venv\Scripts\python -m pharmapulse.forecast    # full backtest + next-8-week forecast (~5 min)
.venv\Scripts\streamlit run app.py              # dashboard
.venv\Scripts\python -m pytest                  # 9 tests, ~1 min
```

The tests include a **leakage check**: multiplying sales after the forecast origin by 5 must not change the
forecast.

## Project layout

```
pharmapulse/data.py       synthetic weekly sales + labelled anomalies
pharmapulse/anomaly.py    robust detector, 3-sigma baseline, event-level scoring
pharmapulse/forecast.py   naive / seasonal naive / ETS / global GBM, cleaning, rolling-origin backtest
pharmapulse/inventory.py  safety stock and order-up-to levels
app.py                    Streamlit dashboard: series explorer, model comparison, anomaly log, stock plan
reports/                  backtest outputs and forecast (CSV, Power BI-ready)
tests/                    pytest suite
```
