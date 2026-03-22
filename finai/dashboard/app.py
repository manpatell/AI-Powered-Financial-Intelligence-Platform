"""
FinAI — Main Streamlit entry point.
Run with:  streamlit run finai/dashboard/app.py
"""
import streamlit as st
import pandas as pd

from finai.config.settings import DEFAULT_TICKERS
from finai.data.stock_fetcher import fetch_stock_data
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(
    page_title="FinAI Platform",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">FinAI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Financial Intelligence</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem; color:#8B949E; line-height:2;">
    Home &mdash; Market Overview<br>
    Stock Analysis &mdash; Price &amp; Indicators<br>
    ML Predictions &mdash; Model Signals<br>
    AI Assistant &mdash; RAG Q&amp;A<br>
    Experiment Tracker &mdash; MLflow<br>
    Monitoring &mdash; Drift &amp; Health
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#484F58;">v1.0.0 &nbsp;&middot;&nbsp; manpatell</div>',
                unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
page_header(
    "FinAI — Financial Intelligence Platform",
    "Real-time stock analysis &nbsp;&middot;&nbsp; ML predictions &nbsp;&middot;&nbsp; "
    "RAG-powered insights &nbsp;&middot;&nbsp; MLOps monitoring",
)

# ── Platform stats ────────────────────────────────────────────────────────────
section_label("Platform Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers Tracked",     str(len(DEFAULT_TICKERS)))
c2.metric("ML Models",           "XGBoost + LightGBM")
c3.metric("Vector Store",        "ChromaDB")
c4.metric("Experiment Tracking", "MLflow")

st.divider()

# ── Market snapshot ───────────────────────────────────────────────────────────
section_label("Market Snapshot — Live Prices")
st.caption("Prices sourced via yfinance · 4-hour cache")

tickers_to_show = DEFAULT_TICKERS[:6]
cols = st.columns(len(tickers_to_show))
for col, ticker in zip(cols, tickers_to_show):
    try:
        df     = fetch_stock_data(ticker, period="5d", use_cache=True)
        latest = float(df["Close"].iloc[-1])
        prev   = float(df["Close"].iloc[-2])
        delta  = (latest - prev) / prev * 100
        col.metric(ticker, f"${latest:.2f}", f"{delta:+.2f}%")
    except Exception:
        col.metric(ticker, "N/A", "")

st.divider()

# ── Module index ──────────────────────────────────────────────────────────────
section_label("Modules")
st.markdown("""
| Module | Description |
|---|---|
| **Stock Analysis** | Candlestick chart, Bollinger Bands, SMA/EMA, RSI, MACD, volume bars, live news feed |
| **ML Predictions** | Train XGBoost / LightGBM on demand; buy / hold / sell signal overlay; feature importance |
| **AI Assistant** | RAG-powered Q&A backed by ChromaDB; grounded in real news and company profiles |
| **Experiment Tracker** | MLflow run browser; metric comparison charts; Model Registry |
| **Monitoring** | Per-feature KS-test and PSI drift detection; rolling accuracy / ROC-AUC history |
""")

st.divider()
st.markdown(
    '<div style="font-size:0.78rem;color:#8B949E;">Use the sidebar to navigate between modules.</div>',
    unsafe_allow_html=True,
)
