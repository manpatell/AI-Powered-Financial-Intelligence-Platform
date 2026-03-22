"""
AI Chatbot page — RAG-powered financial Q&A using ChromaDB + LLM.
"""
import streamlit as st

from finai.config.settings import DEFAULT_TICKERS
from finai.rag.rag_chain import FinancialRAG, build_knowledge_base
from finai.rag.document_store import DocumentStore

st.set_page_config(page_title="AI Chatbot · FinAI", layout="wide")
st.title("💬 AI Financial Chatbot")
st.caption("Retrieval-Augmented Generation · ChromaDB + FinBERT + LLM")

# ── Knowledge base setup ───────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Knowledge Base")
    store = DocumentStore()
    doc_count = store.count()
    st.metric("Indexed Documents", doc_count)

    selected_ticker = st.selectbox("Filter by Ticker (optional)", ["All"] + DEFAULT_TICKERS)

    if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
        with st.spinner("Indexing news and company profiles …"):
            build_knowledge_base(DEFAULT_TICKERS)
        st.success("Knowledge base updated!")
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I'm **FinAI**, your AI-powered financial analyst. "
            "I can answer questions about any of the tracked stocks using "
            "real-time news and company data.\n\n"
            "Try asking:\n"
            "- *What's the latest news on NVDA?*\n"
            "- *Why has TSLA been volatile recently?*\n"
            "- *What sector does AAPL operate in?*"
        ),
    })

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about any stock …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base …"):
            rag = FinancialRAG()
            ticker_filter = None if selected_ticker == "All" else selected_ticker
            response = rag.answer(prompt, ticker=ticker_filter)

        st.markdown(response.answer)

        # Show sources
        if response.sources:
            with st.expander(f"📚 Sources ({len(response.sources)} chunks retrieved)", expanded=False):
                for i, src in enumerate(response.sources, 1):
                    st.markdown(
                        f"**[{i}]** `{src.get('ticker','?')}` · "
                        f"{src.get('source','unknown')} · "
                        f"{str(src.get('published',''))[:19]}"
                    )
                    if src.get("url"):
                        st.caption(src["url"])
            st.caption(f"Model: `{response.model}`")

    st.session_state.messages.append({"role": "assistant", "content": response.answer})

# Clear chat button
if st.session_state.messages and len(st.session_state.messages) > 1:
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
