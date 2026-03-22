"""
Stock Analysis page — OHLCV charts, technical indicators, news feed.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finai.config.settings import DEFAULT_TICKERS
from finai.data.stock_fetcher import fetch_stock_data, get_ticker_info
from finai.data.news_fetcher import fetch_all_news
from finai.features.technical_indicators import add_technical_indicators
from finai.features.feature_pipeline import TICKER_NAMES

st.set_page_config(page_title="Stock Analysis · FinAI", layout="wide")
st.title("📊 Stock Analysis")

# ── Controls ──────────────────────────────────────────────────────────────────
col_t, col_p = st.columns([2, 1])
ticker = col_t.selectbox("Ticker", DEFAULT_TICKERS, index=0)
period = col_p.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} …"):
    try:
        raw  = fetch_stock_data(ticker, period=period)
        df   = add_technical_indicators(raw)
        info = get_ticker_info(ticker)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# ── Company Header ────────────────────────────────────────────────────────────
st.subheader(f"{info.get('name', ticker)}  ({ticker})")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Sector",    info.get("sector", "N/A"))
m2.metric("P/E Ratio", f"{info.get('pe_ratio', 'N/A')}")
m3.metric("52W High",  f"${info.get('52w_high', 0):.2f}" if info.get('52w_high') else "N/A")
m4.metric("52W Low",   f"${info.get('52w_low', 0):.2f}"  if info.get('52w_low')  else "N/A")
mkt = info.get("market_cap", 0)
m5.metric("Market Cap", f"${mkt/1e9:.1f}B" if mkt else "N/A")

st.divider()

# ── Main Price Chart ──────────────────────────────────────────────────────────
st.subheader("Price & Volume")
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.03,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="OHLC", increasing_line_color="#2ea043", decreasing_line_color="#f85149",
), row=1, col=1)

# Moving averages
for ma, color in [("sma_20", "#58a6ff"), ("sma_50", "#d29922"), ("ema_10", "#bc8cff")]:
    if ma in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma], name=ma.upper(),
            line=dict(color=color, width=1.2), opacity=0.8,
        ), row=1, col=1)

# Bollinger Bands
if "bb_upper" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], name="BB Lower",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(88,166,255,0.05)",
    ), row=1, col=1)

# RSI
if "rsi_14" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi_14"], name="RSI 14",
        line=dict(color="#d29922", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(248,81,73,0.5)",  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(46,160,67,0.5)",  row=2, col=1)

# Volume bars
colors = ["#2ea043" if c >= o else "#f85149"
          for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(go.Bar(
    x=df.index, y=df["Volume"], name="Volume",
    marker_color=colors, opacity=0.7,
), row=3, col=1)

fig.update_layout(
    height=700,
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
fig.update_yaxes(gridcolor="#21262d")
fig.update_xaxes(gridcolor="#21262d")

st.plotly_chart(fig, use_container_width=True)

# ── MACD Chart ────────────────────────────────────────────────────────────────
st.subheader("MACD")
macd_fig = go.Figure()
if "macd" in df.columns:
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                  line=dict(color="#58a6ff", width=1.5)))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                  line=dict(color="#d29922", width=1.5)))
    macd_fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="Histogram",
                              marker_color=["#2ea043" if v >= 0 else "#f85149"
                                            for v in df["macd_diff"].fillna(0)]))
macd_fig.update_layout(
    height=250, template="plotly_dark",
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(macd_fig, use_container_width=True)

# ── Latest News ───────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"📰 Latest News — {ticker}")
with st.spinner("Fetching news …"):
    news = fetch_all_news(ticker, TICKER_NAMES.get(ticker, ticker))

if not news.empty:
    for _, row in news.head(8).iterrows():
        with st.expander(f"🗞 {row['title']}", expanded=False):
            st.caption(f"**{row.get('source','')}** · {str(row.get('published',''))[:19]}")
            st.write(row.get("summary", ""))
            if row.get("url"):
                st.markdown(f"[Read full article →]({row['url']})")
else:
    st.info("No news articles found.")
