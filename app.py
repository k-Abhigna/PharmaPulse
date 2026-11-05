"""PharmaPulse dashboard.   streamlit run app.py   (run `python -m pharmapulse.forecast` first)"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pharmapulse.anomaly import detect
from pharmapulse.data import load
from pharmapulse.forecast import REPORTS
from pharmapulse.inventory import plan

st.set_page_config(page_title="PharmaPulse", page_icon="💊", layout="wide")


@st.cache_data
def load_all():
    sales, labels = load()
    flagged = detect(sales)
    preds = pd.read_csv(REPORTS / "backtest_predictions.csv", parse_dates=["week"])
    summary = pd.read_csv(REPORTS / "backtest_summary.csv", index_col=0)
    future = pd.read_csv(REPORTS / "forecast_next_8_weeks.csv", parse_dates=["week"])
    return sales, labels, flagged, preds, summary, future


if not (REPORTS / "backtest_summary.csv").exists():
    st.error("Run `python -m pharmapulse.forecast` first to create the backtest and forecasts.")
    st.stop()

sales, labels, flagged, preds, summary, future = load_all()

st.title("PharmaPulse: Demand Forecasting & Anomaly Detection")
best = summary.index[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", best)
c2.metric("Forecast error (WAPE)", f"{summary.loc[best, 'wape_pct']:.1f}%",
          f"{summary.loc[best, 'wape_pct'] - summary.loc['seasonal_naive', 'wape_pct']:.1f} pts vs seasonal naive",
          delta_color="inverse")
c3.metric("Series", f"{sales.groupby(['product', 'region']).ngroups}")
c4.metric("Anomalies flagged", f"{int(flagged['is_anomaly'].sum())} weeks")

tab1, tab2, tab3, tab4 = st.tabs(["Series explorer", "Model comparison", "Anomaly log", "Stock plan"])

with tab1:
    a, b = st.columns(2)
    product = a.selectbox("Product", sorted(sales["product"].unique()))
    region = b.selectbox("Region", sorted(sales["region"].unique()))
    s = flagged[(flagged["product"] == product) & (flagged["region"] == region)].sort_values("week")
    f = future[(future["product"] == product) & (future["region"] == region)]
    bt = preds[(preds["product"] == product) & (preds["region"] == region) & (preds["model"] == best)]

    fig = go.Figure()
    fig.add_scatter(x=s["week"], y=s["units"], name="Actual units", line=dict(color="#2563eb"))
    fig.add_scatter(x=s["week"], y=s["expected"], name="Expected (detector)",
                    line=dict(color="#94a3b8", dash="dot"))
    fig.add_scatter(x=bt["week"], y=bt["forecast"], name=f"Backtest forecast ({best})", mode="markers",
                    marker=dict(color="#16a34a", size=5))
    fig.add_scatter(x=f["week"], y=f["forecast"], name="Next 8 weeks", line=dict(color="#16a34a", width=3))
    promo = s[s["promo"] == 1]
    fig.add_scatter(x=promo["week"], y=promo["units"], name="Promotion", mode="markers",
                    marker=dict(symbol="star", size=10, color="#f59e0b"))
    an = s[s["is_anomaly"]]
    fig.add_scatter(x=an["week"], y=an["units"], name="Flagged anomaly", mode="markers",
                    marker=dict(symbol="x", size=11, color="#dc2626"), text=an["anomaly_type"],
                    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<br>%{text}<extra></extra>")
    fig.update_layout(height=460, margin=dict(t=10), legend=dict(orientation="h", y=-0.15),
                      yaxis_title="Units / week")
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.caption("Rolling-origin backtest: 4 origins x 8-week horizon x 48 series. "
               "WAPE excludes weeks inside labelled anomalies, which no model can foresee.")
    st.dataframe(summary, width="stretch")
    by_class = pd.read_csv(REPORTS / "wape_by_therapy_class.csv", index_col=0)
    st.subheader("WAPE % by therapy class")
    st.dataframe(by_class, width="stretch")

with tab3:
    log = flagged[flagged["is_anomaly"]][["week", "product", "region", "units", "expected", "anomaly_type",
                                          "robust_z"]].sort_values("week", ascending=False)
    kinds = st.multiselect("Type", sorted(log["anomaly_type"].unique()), default=sorted(log["anomaly_type"].unique()))
    st.dataframe(log[log["anomaly_type"].isin(kinds)].round({"expected": 0, "robust_z": 1}), width="stretch", hide_index=True)

with tab4:
    a, b, c = st.columns(3)
    sl = a.slider("Service level", 0.80, 0.99, 0.95, 0.01)
    lt = b.number_input("Lead time (weeks)", 1, 6, 2)
    rv = c.number_input("Review period (weeks)", 1, 4, 1)
    table = plan(future, preds, best, sl, int(lt), int(rv))
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download stock plan (CSV)", table.to_csv(index=False), "stock_plan.csv", "text/csv")
