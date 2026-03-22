"""
Stock Analysis — OHLCV charts, technical indicators, news feed.
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
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="Stock Analysis · FinAI", layout="wide")
inject_css()
page_header("Stock Analysis", "Price history · Technical indicators · News feed")

# ── Controls ──────────────────────────────────────────────────────────────────
col_t, col_p, _ = st.columns([2, 1, 3])
ticker = col_t.selectbox("Ticker", DEFAULT_TICKERS, index=0)
period = col_p.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading data for {ticker}…"):
    try:
        raw  = fetch_stock_data(ticker, period=period)
        df   = add_technical_indicators(raw)
        info = get_ticker_info(ticker)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# ── Company header ────────────────────────────────────────────────────────────
company_name = info.get("name", ticker)
st.markdown(
    f'<div style="font-size:1.05rem;font-weight:600;color:#E6EDF3;margin-bottom:0.75rem;">'
    f'{company_name} &nbsp;<span style="color:#8B949E;font-weight:400;">({ticker})</span></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Sector",     info.get("sector", "N/A"))
m2.metric("P/E Ratio",  f"{info.get('pe_ratio', 'N/A')}")
m3.metric("52W High",   f"${info.get('52w_high', 0):.2f}" if info.get("52w_high") else "N/A")
m4.metric("52W Low",    f"${info.get('52w_low', 0):.2f}"  if info.get("52w_low")  else "N/A")
mkt = info.get("market_cap", 0)
m5.metric("Market Cap", f"${mkt/1e9:.1f}B" if mkt else "N/A")

st.divider()

# ── Chart options ─────────────────────────────────────────────────────────────
section_label("Price Chart")
opt1, opt2, opt3 = st.columns(3)
show_ma  = opt1.checkbox("Moving Averages", value=True)
show_bb  = opt2.checkbox("Bollinger Bands", value=True)
show_rsi = opt3.checkbox("RSI Panel", value=True)

# ── Price / RSI / Volume ──────────────────────────────────────────────────────
row_heights = [0.55, 0.22, 0.23] if show_rsi else [0.70, 0.30]
n_rows      = 3 if show_rsi else 2
vol_row     = 3 if show_rsi else 2

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.025,
    subplot_titles=("", "RSI (14)" if show_rsi else "", "Volume"),
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="OHLC",
    increasing_line_color="#2EA043", increasing_fillcolor="#2EA043",
    decreasing_line_color="#F85149", decreasing_fillcolor="#F85149",
), row=1, col=1)

# Moving averages
if show_ma:
    ma_series = [("sma_20", "#3B82F6", "SMA 20"), ("sma_50", "#D29922", "SMA 50"), ("ema_10", "#A371F7", "EMA 10")]
    for col_name, color, label in ma_series:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=label,
                line=dict(color=color, width=1.3), opacity=0.85,
            ), row=1, col=1)

# Bollinger Bands
if show_bb and "bb_upper" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color="rgba(139,148,158,0.4)", width=1, dash="dot"), showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], name="BB Lower",
        line=dict(color="rgba(139,148,158,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(59,130,246,0.04)", showlegend=True,
    ), row=1, col=1)

# RSI panel
rsi_row = 2
if show_rsi and "rsi_14" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi_14"], name="RSI 14",
        line=dict(color="#D29922", width=1.5),
    ), row=rsi_row, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)",
                  line_width=0, row=rsi_row, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(46,160,67,0.05)",
                  line_width=0, row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,81,73,0.45)",
                  line_width=1, row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(46,160,67,0.45)",
                  line_width=1, row=rsi_row, col=1)

# Volume bars
bar_colors = ["#2EA043" if c >= o else "#F85149"
              for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(go.Bar(
    x=df.index, y=df["Volume"], name="Volume",
    marker_color=bar_colors, opacity=0.65,
), row=vol_row, col=1)

fig.update_layout(
    height=680,
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family="sans-serif", size=11, color="#8B949E"),
)
fig.update_yaxes(gridcolor="#21262D", gridwidth=0.5, zeroline=False)
fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5, showspikes=True,
                 spikecolor="#30363D", spikethickness=1)

st.plotly_chart(fig, use_container_width=True)

# ── MACD ──────────────────────────────────────────────────────────────────────
section_label("MACD")
if "macd" in df.columns:
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"], name="MACD",
        line=dict(color="#3B82F6", width=1.5),
    ))
    macd_fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color="#D29922", width=1.5),
    ))
    macd_fig.add_trace(go.Bar(
        x=df.index, y=df["macd_diff"], name="Histogram",
        marker_color=["#2EA043" if v >= 0 else "#F85149"
                      for v in df["macd_diff"].fillna(0)],
        opacity=0.7,
    ))
    macd_fig.update_layout(
        height=220, template="plotly_dark",
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.05, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11, color="#8B949E"),
    )
    macd_fig.update_yaxes(gridcolor="#21262D", gridwidth=0.5, zeroline=True,
                          zerolinecolor="#30363D", zerolinewidth=1)
    macd_fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5)
    st.plotly_chart(macd_fig, use_container_width=True)

# ── News feed ─────────────────────────────────────────────────────────────────
st.divider()
section_label(f"Latest News — {ticker}")
with st.spinner("Fetching news…"):
    news = fetch_all_news(ticker, TICKER_NAMES.get(ticker, ticker))

if not news.empty:
    for _, row in news.head(8).iterrows():
        with st.expander(row["title"], expanded=False):
            st.caption(
                f"**{row.get('source', '')}**"
                f"  ·  {str(row.get('published', ''))[:19]}"
            )
            summary = row.get("summary", "")
            if summary:
                st.write(summary)
            if row.get("url"):
                st.markdown(f"[Read full article]({row['url']})")
else:
    st.info("No news articles found for this ticker.")
