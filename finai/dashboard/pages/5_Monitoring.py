"""
Model Monitoring page — data drift detection and performance tracking.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from finai.config.settings import DEFAULT_TICKERS
from finai.monitoring.drift_detector import DriftDetector
from finai.monitoring.performance_tracker import PerformanceTracker

st.set_page_config(page_title="Monitoring · FinAI", layout="wide")
st.title("📉 Model Monitoring")
st.caption("Data drift detection (KS-test + PSI) · Rolling performance tracking")

ticker = st.selectbox("Ticker", DEFAULT_TICKERS)
model_type = st.radio("Model", ["xgb", "lgbm"], horizontal=True)

# ── Drift Detection ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Data Drift Report")

detector = DriftDetector(ticker)
existing_report = detector.load_report()

col_run, col_status = st.columns([1, 3])
if col_run.button("▶ Run Drift Analysis", type="primary", use_container_width=True):
    with st.spinner("Running KS-test + PSI for all features …"):
        report = detector.run()
    if "error" in report:
        st.error(report["error"])
    else:
        st.success("Drift analysis complete!")
        existing_report = report

if existing_report and "error" not in existing_report:
    report = existing_report
    n_feat   = report.get("n_features", 0)
    n_drift  = report.get("n_drifted", 0)
    rate     = report.get("drift_rate", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features Tested",  n_feat)
    m2.metric("Drifted Features", n_drift, delta=f"{rate:.0%}" if rate > 0.2 else None,
              delta_color="inverse")
    m3.metric("Drift Rate",       f"{rate:.0%}")
    m4.metric("Reference Rows",   report.get("ref_rows", "N/A"))

    # Feature drift table
    if "features" in report:
        rows = []
        for feat, info in report["features"].items():
            rows.append({
                "Feature":    feat,
                "Drifted":    "⚠️ YES" if info["drifted"] else "✅ NO",
                "KS p-value": info["ks"]["p_value"],
                "PSI":        info["psi"]["psi"],
                "PSI Level":  info["psi"]["level"],
                "Ref Mean":   info["ref_mean"],
                "Cur Mean":   info["cur_mean"],
            })
        drift_df = pd.DataFrame(rows).sort_values("PSI", ascending=False)
        st.dataframe(drift_df, use_container_width=True)

        # PSI bar chart — top 20
        st.subheader("PSI by Feature (top 20)")
        top20 = drift_df.head(20)
        bar = px.bar(
            top20, x="PSI", y="Feature", orientation="h",
            color="PSI", color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        bar.add_vline(x=0.1, line_dash="dash", line_color="#d29922",
                      annotation_text="Moderate (0.1)", annotation_position="top right")
        bar.add_vline(x=0.2, line_dash="dash", line_color="#f85149",
                      annotation_text="Significant (0.2)", annotation_position="top right")
        bar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bar, use_container_width=True)
else:
    st.info("Click **Run Drift Analysis** to detect feature drift. Requires the feature pipeline to have run first.")

# ── Performance Tracking ──────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Performance History")

tracker = PerformanceTracker(ticker, model_type)
perf_df = tracker.get_history()

if not perf_df.empty:
    stats = tracker.rolling_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Accuracy",  f"{stats.get('accuracy_mean', 0):.2%}")
    c2.metric("Avg ROC-AUC",   f"{stats.get('roc_auc_mean', 0):.4f}" if stats.get("roc_auc_mean") else "N/A")
    c3.metric("Snapshots",     stats.get("n_snapshots", 0))

    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df["timestamp"], y=perf_df["accuracy"],
        name="Accuracy", line=dict(color="#58a6ff", width=2),
    ))
    if "roc_auc" in perf_df.columns:
        fig.add_trace(go.Scatter(
            x=perf_df["timestamp"], y=perf_df["roc_auc"],
            name="ROC-AUC", line=dict(color="#2ea043", width=2),
        ))
    fig.update_layout(
        height=350, template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(perf_df.tail(10).reset_index(drop=True), use_container_width=True)
else:
    st.info("No performance snapshots yet. Train and evaluate a model first.")

st.divider()
st.markdown("""
**How to interpret:**
- **KS p-value < 0.05** → statistically significant distribution shift
- **PSI < 0.1** → Stable (no action needed)
- **PSI 0.1–0.2** → Moderate drift (monitor closely)
- **PSI > 0.2** → Significant drift (consider retraining)
""")
