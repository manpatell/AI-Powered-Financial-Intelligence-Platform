"""
ML Predictions page — train models, view signals, feature importance.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

from finai.config.settings import DEFAULT_TICKERS, MODELS_DIR
from finai.features.feature_pipeline import build_train_test
from finai.models.trainer import train_all_models
from finai.models.predictor import predict, get_latest_signal
from finai.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="ML Predictions · FinAI", layout="wide")
st.title("🤖 ML Predictions")
st.caption("XGBoost & LightGBM — 5-day price direction forecast with MLflow tracking")

# ── Sidebar controls ──────────────────────────────────────────────────────────
ticker = st.selectbox("Select Ticker", DEFAULT_TICKERS)
model_type = st.radio("Model", ["xgb", "lgbm"], horizontal=True)

# ── Train / load ──────────────────────────────────────────────────────────────
model_path = MODELS_DIR / f"{ticker}_{model_type}.joblib"

col_train, col_info = st.columns([1, 2])

with col_train:
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner(f"Building features + training {model_type.upper()} for {ticker} …"):
            try:
                data = build_train_test(ticker, include_sentiment=False)
                results = train_all_models(ticker, data)
                st.success("✅ Training complete! Model saved and logged to MLflow.")
                st.json({k: {m: round(v, 4) for m, v in met.items()}
                         for k, (_, met) in results.items()})
            except Exception as e:
                st.error(f"Training failed: {e}")

with col_info:
    if model_path.exists():
        st.success(f"✅ Saved model found: `{model_path.name}`")
    else:
        st.warning("No saved model found. Click **Train Model** to build one.")

st.divider()

# ── Predictions chart ─────────────────────────────────────────────────────────
if model_path.exists():
    st.subheader("📈 Prediction Timeline")
    with st.spinner("Running inference …"):
        try:
            data = build_train_test(ticker, include_sentiment=False)
            test_df = data["test_df"]
            feat_cols = data["feature_cols"]
            pred_df = predict(ticker, test_df, feat_cols, model_type)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["close"],
                name="Close Price", line=dict(color="#58a6ff", width=2),
            ))

            # Buy / Sell markers
            buys  = pred_df[pred_df["signal"].isin(["BUY", "STRONG BUY"])]
            sells = pred_df[pred_df["signal"].isin(["SELL", "STRONG SELL"])]

            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["close"],
                mode="markers", name="Buy Signal",
                marker=dict(color="#2ea043", size=8, symbol="triangle-up"),
            ))
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["close"],
                mode="markers", name="Sell Signal",
                marker=dict(color="#f85149", size=8, symbol="triangle-down"),
            ))

            fig.update_layout(
                height=400, template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Latest signal card
            st.divider()
            st.subheader("🎯 Latest Signal")
            latest = pred_df.iloc[-1]
            sig_color = {"STRONG BUY": "green", "BUY": "green",
                         "HOLD": "orange", "SELL": "red", "STRONG SELL": "red"}
            color = sig_color.get(str(latest["signal"]), "gray")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal",      str(latest["signal"]))
            c2.metric("Probability", f"{latest['probability']:.2%}")
            c3.metric("Close Price", f"${latest['close']:.2f}")
            c4.metric("Date",        str(pred_df.index[-1].date()))

            # Probability distribution
            st.subheader("Probability Distribution (Test Set)")
            hist_fig = px.histogram(
                pred_df, x="probability", color="prediction",
                nbins=40, template="plotly_dark",
                color_discrete_map={0: "#f85149", 1: "#2ea043"},
                labels={"prediction": "Direction"},
            )
            hist_fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Feature importance
    fi_path = MODELS_DIR / f"{ticker}_{model_type}_feature_importance.csv"
    if fi_path.exists():
        st.divider()
        st.subheader("🔍 Feature Importance")
        fi = pd.read_csv(fi_path).head(20)
        bar_fig = px.bar(
            fi, x="importance", y="feature", orientation="h",
            template="plotly_dark", color="importance",
            color_continuous_scale="Blues",
        )
        bar_fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("Train a model to see predictions and feature importance.")
