"""
AI Assistant — RAG-powered financial Q&A using ChromaDB + LLM.
"""
import streamlit as st

from finai.config.settings import DEFAULT_TICKERS
from finai.rag.rag_chain import FinancialRAG, build_knowledge_base
from finai.rag.document_store import DocumentStore
from finai.dashboard.styles import inject_css, page_header, section_label

st.set_page_config(page_title="AI Assistant · FinAI", layout="wide")
inject_css()
page_header(
    "AI Financial Assistant",
    "Retrieval-Augmented Generation &nbsp;&middot;&nbsp; ChromaDB &nbsp;&middot;&nbsp; FinBERT &nbsp;&middot;&nbsp; LLM",
)

# ── Sidebar — Knowledge base controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)
    store     = DocumentStore()
    doc_count = store.count()
    st.metric("Indexed Documents", doc_count)
    st.write("")

    selected_ticker = st.selectbox("Filter by Ticker", ["All"] + DEFAULT_TICKERS)
    st.write("")

    if st.button("Refresh Knowledge Base", use_container_width=True):
        with st.spinner("Indexing news and company profiles…"):
            build_knowledge_base(DEFAULT_TICKERS)
        st.success("Knowledge base updated.")
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem;color:#8B949E;line-height:1.7;">'
        'Answers are grounded in indexed news articles and company profiles.<br><br>'
        'Add <code>OPENAI_API_KEY</code> or <code>ANTHROPIC_API_KEY</code> '
        'to <code>.env</code> for LLM-generated responses.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello. I am **FinAI**, an AI-powered financial analyst. "
            "I can answer questions about tracked stocks using real-time news and company data.\n\n"
            "**Example queries:**\n"
            "- What is the latest news on NVDA?\n"
            "- Why has TSLA been volatile recently?\n"
            "- What sector does AAPL operate in?"
        ),
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about any stock…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            rag           = FinancialRAG()
            ticker_filter = None if selected_ticker == "All" else selected_ticker
            response      = rag.answer(prompt, ticker=ticker_filter)

        st.markdown(response.answer)

        if response.sources:
            with st.expander(f"Sources  ({len(response.sources)} retrieved)", expanded=False):
                for i, src in enumerate(response.sources, 1):
                    st.markdown(
                        f"**[{i}]** `{src.get('ticker', '?')}` &nbsp;&middot;&nbsp; "
                        f"{src.get('source', 'unknown')} &nbsp;&middot;&nbsp; "
                        f"{str(src.get('published', ''))[:19]}"
                    )
                    if src.get("url"):
                        st.caption(src["url"])
            st.caption(f"Model: `{response.model}`")

    st.session_state.messages.append({"role": "assistant", "content": response.answer})

# ── Clear ──────────────────────────────────────────────────────────────────────
if len(st.session_state.messages) > 1:
    st.write("")
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
