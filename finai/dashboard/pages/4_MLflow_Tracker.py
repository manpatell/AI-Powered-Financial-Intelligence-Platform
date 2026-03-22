"""
MLflow Experiment Tracker page — browse runs, compare metrics, view artifacts.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from finai.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from finai.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="MLflow Tracker · FinAI", layout="wide")
st.title("🔬 MLflow Experiment Tracker")
st.caption(f"Tracking URI: `{MLFLOW_TRACKING_URI}`  ·  Experiment: `{MLFLOW_EXPERIMENT_NAME}`")

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── Experiment info ───────────────────────────────────────────────────────
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        st.info("No experiments found yet. Train a model on the ML Predictions page first.")
        st.stop()

    # ── Fetch all runs ────────────────────────────────────────────────────────
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )

    if runs.empty:
        st.warning("No completed runs found.")
        st.stop()

    st.metric("Total Runs", len(runs))

    # Clean column names
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols  = [c for c in runs.columns if c.startswith("params.")]

    display_cols = ["tags.mlflow.runName", "start_time"] + metric_cols[:6]
    display_cols = [c for c in display_cols if c in runs.columns]

    renamed = runs[display_cols].copy()
    renamed.columns = [c.replace("metrics.", "").replace("tags.mlflow.", "").replace("params.", "")
                       for c in renamed.columns]
    renamed["start_time"] = pd.to_datetime(renamed["start_time"]).dt.strftime("%Y-%m-%d %H:%M")

    st.subheader("All Runs")
    st.dataframe(renamed.reset_index(drop=True), use_container_width=True)

    # ── Metric comparison chart ───────────────────────────────────────────────
    st.divider()
    st.subheader("Metric Comparison")

    available_metrics = [c.replace("metrics.", "") for c in metric_cols]
    if available_metrics:
        sel_metric = st.selectbox("Metric", available_metrics,
                                  index=available_metrics.index("roc_auc") if "roc_auc" in available_metrics else 0)
        col_name = f"metrics.{sel_metric}"

        chart_df = runs[["tags.mlflow.runName", col_name]].dropna()
        chart_df.columns = ["run", sel_metric]

        bar = px.bar(
            chart_df, x="run", y=sel_metric, color=sel_metric,
            color_continuous_scale="Blues", template="plotly_dark",
            title=f"{sel_metric} across runs",
        )
        bar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=350, margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="", showlegend=False,
        )
        st.plotly_chart(bar, use_container_width=True)

    # ── Scatter: accuracy vs roc_auc ──────────────────────────────────────────
    if "metrics.accuracy" in runs.columns and "metrics.roc_auc" in runs.columns:
        st.divider()
        st.subheader("Accuracy vs ROC-AUC")
        scatter_df = runs[["tags.mlflow.runName", "metrics.accuracy", "metrics.roc_auc",
                            "params.model_type"]].dropna()
        scatter = px.scatter(
            scatter_df,
            x="metrics.accuracy", y="metrics.roc_auc",
            color="params.model_type", hover_name="tags.mlflow.runName",
            template="plotly_dark", size_max=15,
        )
        scatter.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=350, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(scatter, use_container_width=True)

    # ── Model Registry ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Registry")
    try:
        reg_models = client.search_registered_models()
        if reg_models:
            reg_data = []
            for m in reg_models:
                latest = m.latest_versions
                for v in latest:
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

    st.caption("💡 For full experiment UI run: `mlflow ui --port 5000`")

except ImportError:
    st.error("MLflow not installed. Run `pip install mlflow`.")
except Exception as e:
    st.error(f"MLflow error: {e}")
