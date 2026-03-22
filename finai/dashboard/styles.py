"""Shared CSS injected into every dashboard page."""

GLOBAL_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #8B949E !important;
    font-size: 0.82rem;
}
[data-testid="stSidebar"] .sidebar-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E6EDF3 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .sidebar-tagline {
    font-size: 0.75rem;
    color: #8B949E !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Page header ──────────────────────────────────────────────────────── */
.page-header {
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 1.5rem;
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
}
.page-header p {
    font-size: 0.82rem;
    color: #8B949E;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262D;
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 6px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.72rem !important;
    font-weight: 600;
    color: #8B949E !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.35rem !important;
    font-weight: 700;
    color: #E6EDF3 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #3B82F6;
    border: none;
    border-radius: 5px;
    color: #fff;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
    padding: 0.45rem 1.1rem;
    transition: background 0.15s;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2563EB;
}
.stButton > button[kind="secondary"] {
    background-color: transparent;
    border: 1px solid #30363D;
    border-radius: 5px;
    color: #E6EDF3;
    font-size: 0.82rem;
}

/* ── Dataframe ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 6px;
    overflow: hidden;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #21262D !important;
    border-radius: 6px !important;
    background: #161B22;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    font-weight: 500;
    color: #C9D1D9 !important;
}

/* ── Info / warning / success banners ────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 6px;
    font-size: 0.83rem;
}

/* ── Selectbox / radio ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8B949E !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr {
    border-color: #21262D !important;
    margin: 1.2rem 0 !important;
}

/* ── Chat ─────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

/* ── Signal badges ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-buy    { background: rgba(46,160,67,0.15);  color: #2EA043; border: 1px solid #2EA043; }
.badge-sell   { background: rgba(248,81,73,0.15);  color: #F85149; border: 1px solid #F85149; }
.badge-hold   { background: rgba(210,153,34,0.15); color: #D29922; border: 1px solid #D29922; }

/* ── Plotly chart container ───────────────────────────────────────────── */
.chart-container {
    border: 1px solid #21262D;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    import streamlit as st
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str):
    import streamlit as st
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)
