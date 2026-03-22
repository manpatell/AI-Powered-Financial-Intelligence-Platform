"""
RAG chain for financial Q&A.
Retrieves relevant context from ChromaDB then passes it to an LLM
(OpenAI GPT-4o-mini or Anthropic Claude; falls back to a template answer
when no API key is configured).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from finai.config.settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, RAG_TOP_K
from finai.rag.document_store import DocumentStore
from finai.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are FinAI, an expert financial analyst assistant.
Answer the user's question using ONLY the context provided below.
Be concise, factual, and cite sources where possible.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

_NO_KEY_MSG = (
    "⚠️ No LLM API key configured. "
    "Add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file to enable AI answers.\n\n"
    "**Retrieved context (top {k} chunks):**\n\n{context}"
)


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    context_used: str
    model: str


class FinancialRAG:
    """End-to-end RAG pipeline over the DocumentStore."""

    def __init__(self, store: Optional[DocumentStore] = None):
        self.store = store or DocumentStore()
        self._llm_backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if OPENAI_API_KEY:
            return "openai"
        if ANTHROPIC_API_KEY:
            return "anthropic"
        return "none"

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def answer(self, question: str, ticker: Optional[str] = None) -> RAGResponse:
        """
        Main entry point.
        1. Retrieve top-k relevant chunks from ChromaDB.
        2. Build context string.
        3. Call LLM (or return raw context if no key).
        """
        chunks = self.store.query(question, ticker=ticker, k=RAG_TOP_K)

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                sources=[],
                context_used="",
                model="none",
            )

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            src  = meta.get("source", "unknown")
            pub  = meta.get("published", "")
            context_parts.append(f"[{i}] ({src}, {pub})\n{chunk['text']}")
        context_str = "\n\n".join(context_parts)

        sources = [c["metadata"] for c in chunks]

        if self._llm_backend == "none":
            return RAGResponse(
                answer=_NO_KEY_MSG.format(k=len(chunks), context=context_str),
                sources=sources,
                context_used=context_str,
                model="none",
            )

        system = _SYSTEM_PROMPT.format(context=context_str)
        try:
            if self._llm_backend == "openai":
                answer = self._call_openai(system, question)
                model  = "gpt-4o-mini"
            else:
                answer = self._call_anthropic(system, question)
                model  = "claude-haiku"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = f"LLM error: {e}\n\nContext retrieved:\n{context_str}"
            model  = "error"

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=context_str,
            model=model,
        )


def build_knowledge_base(tickers: list[str]) -> DocumentStore:
    """
    Populate the vector store with company profiles and latest news
    for each ticker. Safe to call multiple times (deduplicates).
    """
    from finai.data.stock_fetcher import get_ticker_info
    from finai.data.news_fetcher import fetch_all_news
    from finai.features.feature_pipeline import TICKER_NAMES

    store = DocumentStore()
    for ticker in tickers:
        logger.info(f"Indexing knowledge base for {ticker}")
        try:
            info = get_ticker_info(ticker)
            store.add_company_profile(ticker, info)
        except Exception as e:
            logger.warning(f"Profile fetch failed for {ticker}: {e}")
        try:
            news_df = fetch_all_news(ticker, TICKER_NAMES.get(ticker, ticker))
            store.add_news(news_df, ticker)
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")

    logger.info(f"Knowledge base ready. Total docs: {store.count()}")
    return store
