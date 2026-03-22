"""
ML Predictions — train models, view signals, feature importance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score, accuracy_score

from finai.config.settings import DEFAULT_TICKERS, MODELS_DIR
from finai.features.feature_pipeline import build_train_test
from finai.models.trainer import train_all_models
from finai.models.predictor import predict
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="ML Predictions · FinAI", layout="wide")
inject_css()
page_header(
    "ML Predictions",
    "XGBoost &amp; LightGBM &nbsp;&middot;&nbsp; Optuna tuning &nbsp;&middot;&nbsp; "
    "TimeSeriesSplit CV &nbsp;&middot;&nbsp; Soft-vote ensemble",
)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3, _ = st.columns([2, 1, 1, 2])
ticker     = ctrl1.selectbox("Ticker", DEFAULT_TICKERS)
model_type = ctrl2.radio("Model", ["xgb", "lgbm", "ensemble"], horizontal=True)
use_tuning = ctrl3.checkbox("Optuna Tuning", value=True,
                             help="30-trial Bayesian search — adds ~2 min per model")

xgb_path  = MODELS_DIR / f"{ticker}_xgb.joblib"
lgbm_path = MODELS_DIR / f"{ticker}_lgbm.joblib"
models_exist = xgb_path.exists() and lgbm_path.exists()

# ── Train ──────────────────────────────────────────────────────────────────────
section_label("Model Training")
col_btn, col_status = st.columns([1, 3])

with col_btn:
    train_clicked = st.button("Train Models", type="primary", use_container_width=True)

with col_status:
    if models_exist:
        st.success(f"Models found for {ticker}  (XGB + LGBM)")
    else:
        st.warning("No saved models found for this ticker. Click Train Models.")

if train_clicked:
    progress = st.progress(0, text="Building feature matrix…")
    try:
        data = build_train_test(ticker, include_sentiment=False)
        n_train = len(data["y_train"])
        n_test  = len(data["y_test"])
        n_feat  = len(data["feature_cols"])
        progress.progress(15, text=f"Feature matrix ready  ({n_train} train / {n_test} test / {n_feat} features)")

        progress.progress(20, text="Training XGBoost with Optuna…" if use_tuning else "Training XGBoost…")
        results = train_all_models(ticker, data, tune=use_tuning)
        progress.progress(100, text="Training complete.")

        st.success("All models trained and logged to MLflow.")

        # Show metrics table
        rows = []
        for mtype, (_, met) in results.items():
            row = {"Model": mtype.upper()}
            row.update({k: f"{v:.4f}" for k, v in met.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    except Exception as e:
        progress.empty()
        st.error(f"Training failed: {e}")

st.divider()

# ── Require models to exist for inference ─────────────────────────────────────
if model_type == "ensemble" and not models_exist:
    st.info("Ensemble requires both XGB and LGBM to be trained first.")
    st.stop()
if not models_exist and model_type != "ensemble":
    model_check = MODELS_DIR / f"{ticker}_{model_type}.joblib"
    if not model_check.exists():
        st.info("Train a model to view predictions.")
        st.stop()

# ── Load features + run inference ────────────────────────────────────────────
section_label("Prediction Results — Test Set")
with st.spinner("Running inference…"):
    try:
        data      = build_train_test(ticker, include_sentiment=False)
        test_df   = data["test_df"]
        feat_cols = data["feature_cols"]
        pred_df   = predict(ticker, test_df, feat_cols, model_type)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

# ── Metrics row ────────────────────────────────────────────────────────────────
try:
    auc = roc_auc_score(data["y_test"], pred_df["probability"].values)
    acc = accuracy_score(data["y_test"], pred_df["prediction"].values)
    pos_rate = data["y_test"].mean()
    baseline_acc = max(pos_rate, 1 - pos_rate)  # naive majority-class baseline

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC",     f"{auc:.4f}")
    m2.metric("Accuracy",    f"{acc:.2%}")
    m3.metric("Baseline",    f"{baseline_acc:.2%}", help="Majority-class accuracy")
    m4.metric("Lift",        f"{acc/baseline_acc:.2f}x",
              delta=f"{(acc-baseline_acc)*100:+.1f}pp vs baseline",
              delta_color="normal")
    m5.metric("Test Samples", len(pred_df))
except Exception:
    pass

# ── Price + signal chart ───────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_df.index, y=pred_df["close"],
    name="Close Price", line=dict(color="#3B82F6", width=2),
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
))

buys  = pred_df[pred_df["signal"].isin(["BUY", "STRONG BUY"])]
sells = pred_df[pred_df["signal"].isin(["SELL", "STRONG SELL"])]

fig.add_trace(go.Scatter(
    x=buys.index, y=buys["close"], mode="markers", name="Buy",
    marker=dict(color="#2EA043", size=9, symbol="triangle-up",
                line=dict(color="#0D1117", width=0.8)),
    hovertemplate="BUY<br>%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=sells.index, y=sells["close"], mode="markers", name="Sell",
    marker=dict(color="#F85149", size=9, symbol="triangle-down",
                line=dict(color="#0D1117", width=0.8)),
    hovertemplate="SELL<br>%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
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
latest     = pred_df.iloc[-1]
signal_str = str(latest["signal"])
badge_cls  = (
    "badge-buy"  if "BUY"  in signal_str else
    "badge-sell" if "SELL" in signal_str else
    "badge-hold"
)
st.markdown(f'<span class="badge {badge_cls}">{signal_str}</span>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal",      signal_str)
c2.metric("Probability", f"{latest['probability']:.2%}")
c3.metric("Close Price", f"${latest['close']:.2f}")
c4.metric("As of",       str(pred_df.index[-1].date()))

# ── Probability distribution ───────────────────────────────────────────────────
st.divider()
section_label("Prediction Probability Distribution")
col_hist, col_cal = st.columns(2)

with col_hist:
    hist_fig = px.histogram(
        pred_df, x="probability", color="prediction", nbins=40,
        template="plotly_dark", barmode="overlay", opacity=0.75,
        color_discrete_map={0: "#F85149", 1: "#2EA043"},
        labels={"prediction": "Direction (0=Down, 1=Up)"},
    )
    hist_fig.add_vline(x=0.5, line_dash="dot", line_color="#8B949E", line_width=1)
    hist_fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(size=11, color="#8B949E"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Predicted Probability (Up)",
    )
    hist_fig.update_yaxes(gridcolor="#21262D", gridwidth=0.5)
    hist_fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5, range=[0, 1])
    st.plotly_chart(hist_fig, use_container_width=True)

# Calibration-style scatter (probability vs rolling actual return)
with col_cal:
    try:
        prob_bins = pd.cut(pred_df["probability"], bins=10)
        pred_df["actual_up"] = data["y_test"]
        cal = pred_df.groupby(prob_bins, observed=True).agg(
            mean_prob=("probability", "mean"),
            actual_rate=("actual_up", "mean"),
            count=("probability", "count"),
        ).reset_index()

        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color="#8B949E", dash="dot", width=1),
        ))
        cal_fig.add_trace(go.Scatter(
            x=cal["mean_prob"], y=cal["actual_rate"],
            mode="markers+lines", name="Model",
            marker=dict(color="#3B82F6", size=cal["count"].clip(5, 20), sizemode="diameter"),
            line=dict(color="#3B82F6", width=1.5),
            hovertemplate="Pred prob: %{x:.2f}<br>Actual rate: %{y:.2f}<extra></extra>",
        ))
        cal_fig.update_layout(
            title=dict(text="Calibration Curve", font=dict(size=12)),
            height=280, template="plotly_dark",
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(size=11, color="#8B949E"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            xaxis=dict(title="Mean Predicted Probability", range=[0, 1],
                       gridcolor="#21262D", gridwidth=0.5),
            yaxis=dict(title="Fraction Positive", range=[0, 1],
                       gridcolor="#21262D", gridwidth=0.5),
        )
        st.plotly_chart(cal_fig, use_container_width=True)
    except Exception:
        st.info("Calibration curve requires sufficient test data.")

# ── Feature importance ─────────────────────────────────────────────────────────
fi_path = MODELS_DIR / f"{ticker}_{model_type if model_type != 'ensemble' else 'xgb'}_feature_importance.csv"
if fi_path.exists():
    st.divider()
    section_label("Feature Importance — Top 25")
    fi = pd.read_csv(fi_path).head(25)

    bar_fig = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"],
        orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale=[[0, "#1E3A5F"], [1, "#3B82F6"]],
            showscale=False,
        ),
    ))
    bar_fig.update_layout(
        height=560, template="plotly_dark",
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed", gridcolor="#21262D"),
        xaxis=dict(gridcolor="#21262D", gridwidth=0.5),
        font=dict(size=11, color="#8B949E"),
    )
    st.plotly_chart(bar_fig, use_container_width=True)
