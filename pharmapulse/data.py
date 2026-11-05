"""Synthetic weekly pharma sales: 12 products x 4 regions x 156 weeks (2023-2025).

Each series = base level x trend x seasonality (by therapy class) x promotion lift x noise.
Labelled anomalies are injected so the detector can be scored:

- stockout:   supply failure, sales fall to 15-65% of normal for 1-3 weeks, then a catch-up week
- spike:      outbreak or tender order, sales rise 1.4-2.4x for 1-3 weeks
- data_error: a single bad week (keyed 10x, or recorded as zero)

    python -m pharmapulse.data      # writes data/sales.csv and data/anomaly_labels.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 7

# product, therapy class, base weekly units, annual trend
PRODUCTS = [
    ("Paracetamol 500mg", "Analgesic", 5200, 0.03),
    ("Ibuprofen 400mg", "Analgesic", 2600, 0.02),
    ("Amoxicillin 250mg", "Antibiotic", 1900, 0.01),
    ("Azithromycin 500mg", "Antibiotic", 1400, 0.04),
    ("Cetirizine 10mg", "Antihistamine", 2200, 0.05),
    ("Salbutamol Inhaler", "Respiratory", 800, 0.06),
    ("ORS Sachet", "Rehydration", 3000, 0.02),
    ("Metformin 500mg", "Chronic", 4100, 0.08),
    ("Atorvastatin 10mg", "Chronic", 2900, 0.09),
    ("Insulin Glargine", "Chronic", 600, 0.12),
    ("Pantoprazole 40mg", "Gastro", 2500, 0.05),
    ("Vitamin D3 60K", "Supplement", 1700, 0.15),
]
REGIONS = {"North": 1.10, "South": 1.25, "East": 0.70, "West": 1.00}

# Seasonal multiplier by month (index 0 = Jan). Monsoon (Jul-Sep) drives infections and
# dehydration, winter drives respiratory, spring drives allergies, summer drives ORS.
SEASONALITY = {
    "Analgesic":     [1.10, 1.05, 0.95, 0.90, 0.90, 0.95, 1.10, 1.15, 1.10, 1.00, 1.00, 1.10],
    "Antibiotic":    [1.15, 1.05, 0.90, 0.85, 0.85, 0.95, 1.20, 1.30, 1.20, 1.00, 0.95, 1.10],
    "Antihistamine": [0.90, 1.10, 1.40, 1.35, 1.10, 0.90, 0.85, 0.85, 0.90, 1.05, 1.15, 0.95],
    "Respiratory":   [1.35, 1.20, 1.00, 0.85, 0.80, 0.80, 0.90, 0.95, 0.95, 1.05, 1.30, 1.45],
    "Rehydration":   [0.70, 0.75, 0.95, 1.30, 1.55, 1.40, 1.25, 1.15, 1.00, 0.85, 0.70, 0.65],
    "Chronic":       [1.00, 0.98, 1.00, 1.00, 1.00, 0.99, 1.00, 1.01, 1.00, 1.01, 1.00, 1.02],
    "Gastro":        [0.95, 0.95, 1.00, 1.05, 1.10, 1.05, 1.05, 1.00, 0.95, 1.05, 1.00, 0.95],
    "Supplement":    [1.20, 1.10, 1.00, 0.90, 0.85, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
}

START = "2023-01-02"  # a Monday
N_WEEKS = 156


def _smooth_seasonal(monthly: list[float], weeks: pd.DatetimeIndex) -> np.ndarray:
    """Interpolate monthly factors to weekly so seasonality has no step changes."""
    m = np.array(monthly + monthly[:1])
    pos = (weeks.day_of_year.values - 1) / 365.25 * 12  # 0..12
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    return m[lo] * (1 - frac) + m[lo + 1] * frac


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    t_years = np.arange(N_WEEKS) / 52.0
    rows, labels = [], []

    for product, therapy, base, trend in PRODUCTS:
        seas = _smooth_seasonal(SEASONALITY[therapy], weeks)
        for region, region_mult in REGIONS.items():
            level = base * region_mult * rng.uniform(0.9, 1.1)
            expected = level * (1 + trend) ** t_years * seas

            # Promotions: planned campaigns, known in advance (usable as a forecast feature).
            promo = np.zeros(N_WEEKS, dtype=int)
            for start in rng.choice(np.arange(4, N_WEEKS - 3), size=rng.integers(3, 6), replace=False):
                promo[start:start + rng.integers(1, 3)] = 1
            lift = rng.uniform(1.15, 1.35)
            expected = expected * np.where(promo == 1, lift, 1.0)

            # Negative-binomial-like noise (variance grows with the mean, as in real demand).
            units = rng.gamma(shape=60, scale=expected / 60)
            units = units.copy()

            # Inject anomalies in non-overlapping windows (avoid the first 8 weeks).
            taken = np.zeros(N_WEEKS, dtype=bool)
            for kind, count in (("stockout", rng.integers(1, 3)), ("spike", rng.integers(0, 2)),
                                ("data_error", rng.integers(0, 2))):
                for _ in range(count):
                    for _try in range(20):
                        length = 1 if kind == "data_error" else int(rng.integers(1, 4))
                        start = int(rng.integers(8, N_WEEKS - length - 1))
                        span = slice(start - 2, start + length + 2)
                        if not taken[span].any() and not promo[start:start + length].any():
                            break
                    else:
                        continue
                    taken[span] = True
                    end = start + length
                    if kind == "stockout":
                        units[start:end] *= rng.uniform(0.15, 0.65, size=length)
                        units[end] *= rng.uniform(1.1, 1.4)  # pent-up demand once stock is back
                    elif kind == "spike":
                        units[start:end] *= rng.uniform(1.4, 2.4, size=length)
                    else:
                        units[start] = units[start] * 10 if rng.random() < 0.5 else 0.0
                    labels.append({"product": product, "region": region, "kind": kind,
                                   "start": weeks[start].date().isoformat(),
                                   "end": weeks[end - 1].date().isoformat(), "weeks": length})

            for i, w in enumerate(weeks):
                rows.append((w.date().isoformat(), product, therapy, region, int(round(units[i])), int(promo[i])))

    sales = pd.DataFrame(rows, columns=["week", "product", "therapy_class", "region", "units", "promo"])
    return sales, pd.DataFrame(labels)


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_path = data_dir / "sales.csv"
    if not sales_path.exists():
        write(data_dir)
    sales = pd.read_csv(sales_path, parse_dates=["week"])
    labels = pd.read_csv(data_dir / "anomaly_labels.csv", parse_dates=["start", "end"])
    return sales, labels


def write(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    sales, labels = generate()
    sales.to_csv(data_dir / "sales.csv", index=False)
    labels.to_csv(data_dir / "anomaly_labels.csv", index=False)


if __name__ == "__main__":
    write()
    s, l = load()
    print(f"sales: {len(s):,} rows, {s.groupby(['product', 'region']).ngroups} series, "
          f"{s.week.min().date()} to {s.week.max().date()}")
    print(f"anomalies injected: {len(l)} ->", l.kind.value_counts().to_dict())
