"""
Monitoring — data drift detection and rolling model performance tracking.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from finai.config.settings import DEFAULT_TICKERS
from finai.monitoring.drift_detector import DriftDetector
from finai.monitoring.performance_tracker import PerformanceTracker
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="Monitoring · FinAI", layout="wide")
inject_css()
page_header(
    "Model Monitoring",
    "Data drift detection (KS-test + PSI) &nbsp;&middot;&nbsp; Rolling performance tracking",
)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2, _ = st.columns([2, 1, 3])
ticker     = ctrl1.selectbox("Ticker", DEFAULT_TICKERS)
model_type = ctrl2.radio("Model", ["xgb", "lgbm"], horizontal=True)

st.divider()

# ── Drift Detection ───────────────────────────────────────────────────────────
section_label("Data Drift Report")

detector       = DriftDetector(ticker)
existing_report = detector.load_report()

col_btn, _ = st.columns([1, 3])
run_clicked = col_btn.button("Run Drift Analysis", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running KS-test + PSI for all features…"):
        report = detector.run()
    if "error" in report:
        st.error(report["error"])
    else:
        st.success("Drift analysis complete.")
        existing_report = report

if existing_report and "error" not in existing_report:
    report   = existing_report
    n_feat   = report.get("n_features", 0)
    n_drift  = report.get("n_drifted", 0)
    rate     = report.get("drift_rate", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features Tested",  n_feat)
    m2.metric("Drifted Features", n_drift,
              delta=f"{rate:.0%}" if rate > 0.2 else None,
              delta_color="inverse")
    m3.metric("Drift Rate",       f"{rate:.0%}")
    m4.metric("Reference Rows",   report.get("ref_rows", "N/A"))

    if "features" in report:
        rows = []
        for feat, info in report["features"].items():
            rows.append({
                "Feature":    feat,
                "Status":     "DRIFT" if info["drifted"] else "Stable",
                "KS p-value": info["ks"]["p_value"],
                "PSI":        info["psi"]["psi"],
                "PSI Level":  info["psi"]["level"],
                "Ref Mean":   info["ref_mean"],
                "Cur Mean":   info["cur_mean"],
            })
        drift_df = pd.DataFrame(rows).sort_values("PSI", ascending=False)

        st.divider()
        section_label("Feature Drift Table")
        st.dataframe(drift_df, use_container_width=True)

        st.divider()
        section_label("Population Stability Index — Top 20 Features")
        top20 = drift_df.head(20)
        bar = px.bar(
            top20, x="PSI", y="Feature", orientation="h",
            color="PSI", color_continuous_scale=["#1A7F3C", "#D29922", "#F85149"],
            template="plotly_dark",
        )
        bar.add_vline(
            x=0.1, line_dash="dot", line_color="#D29922", line_width=1.5,
            annotation_text="Moderate (0.1)", annotation_position="top right",
            annotation_font=dict(size=10, color="#D29922"),
        )
        bar.add_vline(
            x=0.2, line_dash="dot", line_color="#F85149", line_width=1.5,
            annotation_text="Significant (0.2)", annotation_position="top right",
            annotation_font=dict(size=10, color="#F85149"),
        )
        bar.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            height=480, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            font=dict(size=11, color="#8B949E"),
        )
        bar.update_xaxes(gridcolor="#21262D", gridwidth=0.5)
        bar.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
        st.plotly_chart(bar, use_container_width=True)
else:
    st.info(
        "Click **Run Drift Analysis** to detect feature drift. "
        "Requires the feature pipeline to have been run at least once."
    )

# ── Performance Tracking ───────────────────────────────────────────────────────
st.divider()
section_label("Model Performance History")

tracker = PerformanceTracker(ticker, model_type)
perf_df = tracker.get_history()

if not perf_df.empty:
    stats = tracker.rolling_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Accuracy", f"{stats.get('accuracy_mean', 0):.2%}")
    c2.metric("Avg ROC-AUC",
              f"{stats.get('roc_auc_mean', 0):.4f}" if stats.get("roc_auc_mean") else "N/A")
    c3.metric("Snapshots",    stats.get("n_snapshots", 0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df["timestamp"], y=perf_df["accuracy"],
        name="Accuracy", line=dict(color="#3B82F6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
    ))
    if "roc_auc" in perf_df.columns and perf_df["roc_auc"].notna().any():
        fig.add_trace(go.Scatter(
            x=perf_df["timestamp"], y=perf_df["roc_auc"],
            name="ROC-AUC", line=dict(color="#2EA043", width=2),
            fill="tozeroy", fillcolor="rgba(46,160,67,0.05)",
        ))
    fig.update_layout(
        height=320, template="plotly_dark",
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 1], gridcolor="#21262D", gridwidth=0.5),
        xaxis=dict(gridcolor="#21262D", gridwidth=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11, color="#8B949E"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    section_label("Recent Snapshots")
    st.dataframe(perf_df.tail(10).reset_index(drop=True), use_container_width=True)
else:
    st.info("No performance snapshots recorded yet. Train and evaluate a model first.")

# ── Legend ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| KS p-value | < 0.05 | Statistically significant distribution shift |
| PSI | < 0.1 | Stable — no action required |
| PSI | 0.1 – 0.2 | Moderate drift — monitor closely |
| PSI | > 0.2 | Significant drift — consider retraining |
""")
