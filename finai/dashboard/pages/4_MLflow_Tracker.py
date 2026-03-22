"""
Experiment Tracker — MLflow run browser, metric charts, Model Registry.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from finai.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="Experiment Tracker · FinAI", layout="wide")
inject_css()
page_header(
    "Experiment Tracker",
    f"MLflow &nbsp;&middot;&nbsp; Tracking URI: <code>{MLFLOW_TRACKING_URI}</code> "
    f"&nbsp;&middot;&nbsp; Experiment: <code>{MLFLOW_EXPERIMENT_NAME}</code>",
)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        st.info("No experiments found. Train a model on the ML Predictions page first.")
        st.stop()

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )
    if runs.empty:
        st.warning("No completed runs found in this experiment.")
        st.stop()

    # ── Summary stat ──────────────────────────────────────────────────────────
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols  = [c for c in runs.columns if c.startswith("params.")]

    st.metric("Completed Runs", len(runs))
    st.divider()

    # ── Run table ─────────────────────────────────────────────────────────────
    section_label("All Runs")
    display_cols = ["tags.mlflow.runName", "start_time"] + metric_cols[:6]
    display_cols = [c for c in display_cols if c in runs.columns]

    renamed = runs[display_cols].copy()
    renamed.columns = [
        c.replace("metrics.", "").replace("tags.mlflow.", "").replace("params.", "")
        for c in renamed.columns
    ]
    renamed["start_time"] = pd.to_datetime(renamed["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(renamed.reset_index(drop=True), use_container_width=True)

    # ── Metric bar chart ───────────────────────────────────────────────────────
    st.divider()
    section_label("Metric Comparison")

    available_metrics = [c.replace("metrics.", "") for c in metric_cols]
    if available_metrics:
        default_idx = available_metrics.index("roc_auc") if "roc_auc" in available_metrics else 0
        sel_metric  = st.selectbox("Select Metric", available_metrics, index=default_idx)
        col_name    = f"metrics.{sel_metric}"

        chart_df = runs[["tags.mlflow.runName", col_name]].dropna()
        chart_df.columns = ["run", sel_metric]
        chart_df = chart_df.sort_values(sel_metric, ascending=False)

        bar = px.bar(
            chart_df, x="run", y=sel_metric,
            color=sel_metric, color_continuous_scale=["#1E3A5F", "#3B82F6"],
            template="plotly_dark",
        )
        bar.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="", coloraxis_showscale=False,
            font=dict(size=11, color="#8B949E"),
        )
        bar.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
        bar.update_xaxes(gridcolor="#21262D")
        st.plotly_chart(bar, use_container_width=True)

    # ── Scatter: accuracy vs ROC-AUC ──────────────────────────────────────────
    if "metrics.accuracy" in runs.columns and "metrics.roc_auc" in runs.columns:
        st.divider()
        section_label("Accuracy vs ROC-AUC")
        sc_df = runs[
            ["tags.mlflow.runName", "metrics.accuracy", "metrics.roc_auc", "params.model_type"]
        ].dropna()
        scatter = px.scatter(
            sc_df,
            x="metrics.accuracy", y="metrics.roc_auc",
            color="params.model_type", hover_name="tags.mlflow.runName",
            template="plotly_dark",
            color_discrete_map={"xgb": "#3B82F6", "lgbm": "#A371F7"},
        )
        scatter.update_traces(marker=dict(size=12, line=dict(width=1, color="#21262D")))
        scatter.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            font=dict(size=11, color="#8B949E"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        scatter.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
        scatter.update_xaxes(gridcolor="#21262D", gridwidth=0.5)
        st.plotly_chart(scatter, use_container_width=True)

    # ── Model Registry ────────────────────────────────────────────────────────
    st.divider()
    section_label("Model Registry")
    try:
        reg_models = client.search_registered_models()
        if reg_models:
            reg_data = []
            for m in reg_models:
                for v in m.latest_versions:
                    reg_data.append({
                        "Model Name": m.name,
                        "Version":    v.version,
                        "Stage":      v.current_stage,
                        "Created":    str(v.creation_timestamp),
                    })
            st.dataframe(pd.DataFrame(reg_data), use_container_width=True)
        else:
            st.info("No registered models yet.")
    except Exception as e:
        st.warning(f"Model Registry unavailable: {e}")

    st.divider()
    st.caption("For the full MLflow UI run:  `mlflow ui --port 5000`")

except ImportError:
    st.error("MLflow is not installed. Run `pip install mlflow`.")
except Exception as e:
    st.error(f"MLflow error: {e}")
