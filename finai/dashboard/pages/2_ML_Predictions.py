"""
ML Predictions — train models, view signals, feature importance.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from finai.config.settings import DEFAULT_TICKERS, MODELS_DIR
from finai.features.feature_pipeline import build_train_test
from finai.models.trainer import train_all_models
from finai.models.predictor import predict
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="ML Predictions · FinAI", layout="wide")
inject_css()
page_header(
    "ML Predictions",
    "XGBoost &amp; LightGBM &nbsp;&middot;&nbsp; 5-day directional forecast &nbsp;&middot;&nbsp; MLflow tracking",
)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2, _ = st.columns([2, 1, 3])
ticker     = ctrl1.selectbox("Ticker", DEFAULT_TICKERS)
model_type = ctrl2.radio("Model", ["xgb", "lgbm"], horizontal=True)

model_path = MODELS_DIR / f"{ticker}_{model_type}.joblib"

# ── Train ──────────────────────────────────────────────────────────────────────
section_label("Model Training")
col_btn, col_status = st.columns([1, 3])

with col_btn:
    train_clicked = st.button("Train Model", type="primary", use_container_width=True)

with col_status:
    if model_path.exists():
        st.success(f"Saved model found: `{model_path.name}`")
    else:
        st.warning("No saved model found for this ticker / model combination.")

if train_clicked:
    with st.spinner(f"Building features and training {model_type.upper()} for {ticker}…"):
        try:
            data    = build_train_test(ticker, include_sentiment=False)
            results = train_all_models(ticker, data)
            st.success("Training complete. Model saved and logged to MLflow.")
            metrics_display = {
                k: {m: round(v, 4) for m, v in met.items()}
                for k, (_, met) in results.items()
            }
            st.json(metrics_display)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.divider()

# ── Predictions ────────────────────────────────────────────────────────────────
if not model_path.exists():
    st.info("Train a model to view prediction results and feature importance.")
    st.stop()

section_label("Prediction Timeline — Test Set")
with st.spinner("Running inference…"):
    try:
        data      = build_train_test(ticker, include_sentiment=False)
        test_df   = data["test_df"]
        feat_cols = data["feature_cols"]
        pred_df   = predict(ticker, test_df, feat_cols, model_type)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

# Price + signal chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_df.index, y=pred_df["close"],
    name="Close Price",
    line=dict(color="#3B82F6", width=2),
))

buys  = pred_df[pred_df["signal"].isin(["BUY", "STRONG BUY"])]
sells = pred_df[pred_df["signal"].isin(["SELL", "STRONG SELL"])]

fig.add_trace(go.Scatter(
    x=buys.index, y=buys["close"],
    mode="markers", name="Buy",
    marker=dict(color="#2EA043", size=8, symbol="triangle-up",
                line=dict(color="#fff", width=0.5)),
))
fig.add_trace(go.Scatter(
    x=sells.index, y=sells["close"],
    mode="markers", name="Sell",
    marker=dict(color="#F85149", size=8, symbol="triangle-down",
                line=dict(color="#fff", width=0.5)),
))

fig.update_layout(
    height=400, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.04, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    font=dict(size=11, color="#8B949E"),
    hovermode="x unified",
)
fig.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5)
st.plotly_chart(fig, use_container_width=True)

# ── Latest signal ──────────────────────────────────────────────────────────────
st.divider()
section_label("Latest Signal")
latest = pred_df.iloc[-1]
signal_str = str(latest["signal"])

badge_class = (
    "badge-buy"  if "BUY"  in signal_str else
    "badge-sell" if "SELL" in signal_str else
    "badge-hold"
)
st.markdown(
    f'<span class="badge {badge_class}">{signal_str}</span>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal",      signal_str)
c2.metric("Probability", f"{latest['probability']:.2%}")
c3.metric("Close Price", f"${latest['close']:.2f}")
c4.metric("As of",       str(pred_df.index[-1].date()))

# ── Probability distribution ───────────────────────────────────────────────────
st.divider()
section_label("Prediction Probability Distribution — Test Set")
hist_fig = px.histogram(
    pred_df, x="probability", color="prediction",
    nbins=40, template="plotly_dark",
    color_discrete_map={0: "#F85149", 1: "#2EA043"},
    labels={"prediction": "Direction (0=Down, 1=Up)"},
    barmode="overlay",
    opacity=0.75,
)
hist_fig.update_layout(
    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
    height=280, margin=dict(l=0, r=0, t=10, b=0),
    font=dict(size=11, color="#8B949E"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
hist_fig.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
hist_fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5)
st.plotly_chart(hist_fig, use_container_width=True)

# ── Feature importance ─────────────────────────────────────────────────────────
fi_path = MODELS_DIR / f"{ticker}_{model_type}_feature_importance.csv"
if fi_path.exists():
    st.divider()
    section_label("Feature Importance — Top 20")
    fi = pd.read_csv(fi_path).head(20)
    bar_fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        template="plotly_dark",
        color="importance", color_continuous_scale=["#1E3A5F", "#3B82F6"],
    )
    bar_fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        height=480, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        font=dict(size=11, color="#8B949E"),
    )
    bar_fig.update_yaxes(gridcolor="#21262D")
    bar_fig.update_xaxes(gridcolor="#21262D")
    st.plotly_chart(bar_fig, use_container_width=True)
