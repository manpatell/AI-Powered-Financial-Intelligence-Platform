"""
FinAI — Main Streamlit entry point.
Run with:  streamlit run finai/dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="FinAI Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.4rem; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-bottom: 1px solid #21262d;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    .signal-buy   { color: #2ea043; font-weight: 700; }
    .signal-sell  { color: #f85149; font-weight: 700; }
    .signal-hold  { color: #d29922; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("FinAI Platform")
    st.caption("AI-Powered Financial Intelligence")
    st.divider()
    st.markdown("""
    **Navigation**
    - 🏠 Home — Market overview
    - 📊 Stock Analysis — Deep dive
    - 🤖 ML Predictions — Model signals
    - 💬 AI Chatbot — RAG Q&A
    - 🔬 MLflow Tracker — Experiments
    - 📉 Monitoring — Drift & health
    """)
    st.divider()
    st.caption("v1.0.0 · manpatell")

# ── Home Page ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📈 FinAI — Financial Intelligence Platform")
st.markdown("*Real-time stock analysis · ML predictions · RAG-powered insights · MLOps monitoring*")
st.markdown('</div>', unsafe_allow_html=True)

# Quick stats row
from finai.config.settings import DEFAULT_TICKERS
from finai.data.stock_fetcher import fetch_stock_data
import pandas as pd

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers Tracked",  str(len(DEFAULT_TICKERS)), "Active")
col2.metric("ML Models",        "2",   "XGB + LGBM")
col3.metric("RAG Knowledge Base","ChromaDB", "Persistent")
col4.metric("Experiment Tracking","MLflow",  "Auto-logged")

st.divider()

# Market snapshot
st.subheader("⚡ Market Snapshot")
st.caption("Live prices via yfinance — click a ticker in the sidebar pages for deep analysis")

tickers_to_show = DEFAULT_TICKERS[:6]
snapshot_cols = st.columns(len(tickers_to_show))

for col, ticker in zip(snapshot_cols, tickers_to_show):
    try:
        df = fetch_stock_data(ticker, period="5d", use_cache=True)
        latest = float(df["Close"].iloc[-1])
        prev   = float(df["Close"].iloc[-2])
        delta  = (latest - prev) / prev * 100
        col.metric(ticker, f"${latest:.2f}", f"{delta:+.2f}%")
    except Exception:
        col.metric(ticker, "N/A", "")

st.divider()

st.info(
    "👈 **Use the sidebar** to navigate to Stock Analysis, ML Predictions, "
    "the AI Chatbot, MLflow Tracker, or Model Monitoring."
)

st.markdown("""
### About this Platform

| Module | Description |
|--------|-------------|
| **Stock Analysis** | Interactive OHLCV charts, technical indicators, volume profile |
| **ML Predictions** | XGBoost / LightGBM 5-day directional signals with confidence |
| **AI Chatbot** | RAG-powered Q&A grounded in live news and company profiles |
| **MLflow Tracker** | Experiment comparison, metric history, feature importance |
| **Model Monitoring** | Data drift detection (Evidently AI), prediction stability |
""")
