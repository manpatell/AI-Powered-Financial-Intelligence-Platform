"""
ChromaDB document store for financial knowledge.
Ingests news articles, company descriptions, and analyst notes
then exposes a retriever for the RAG chain.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from finai.config.settings import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, RAG_TOP_K
from finai.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """Wraps ChromaDB with sentence-transformer embeddings."""

    def __init__(self, collection_name: str = CHROMA_COLLECTION):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"DocumentStore ready. Collection '{collection_name}' has "
                    f"{self._collection.count()} docs.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_news(self, news_df: pd.DataFrame, ticker: str) -> int:
        """
        Add news articles to the vector store.
        Deduplicates by URL so re-runs are safe.
        """
        if news_df.empty:
            return 0

        existing_ids = set(self._collection.get()["ids"])
        docs, metas, ids = [], [], []

        for _, row in news_df.iterrows():
            doc_id = f"{ticker}_{hash(row.get('url', row['title']))}"
            if doc_id in existing_ids:
                continue
            text = f"{row.get('title', '')}. {row.get('summary', '')}".strip()
            if not text:
                continue
            docs.append(text)
            metas.append({
                "ticker":    ticker,
                "source":    str(row.get("source", "")),
                "published": str(row.get("published", "")),
                "url":       str(row.get("url", "")),
                "type":      "news",
            })
            ids.append(doc_id)

        if not docs:
            return 0

        embeddings = self._embedder.encode(docs, show_progress_bar=False).tolist()
        self._collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        logger.info(f"Added {len(docs)} news docs for {ticker}")
        return len(docs)

    def add_company_profile(self, ticker: str, info: dict) -> None:
        """Add company description as a searchable document."""
        text = (
            f"{info.get('name', ticker)} ({ticker}) operates in the "
            f"{info.get('sector', 'N/A')} sector, specifically {info.get('industry', 'N/A')}. "
            f"{info.get('description', '')}"
        ).strip()
        if not text:
            return
        doc_id = f"profile_{ticker}"
        emb = self._embedder.encode([text], show_progress_bar=False).tolist()
        self._collection.upsert(
            documents=[text],
            embeddings=emb,
            metadatas=[{"ticker": ticker, "type": "profile"}],
            ids=[doc_id],
        )
        logger.debug(f"Upserted company profile for {ticker}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(self, question: str, ticker: Optional[str] = None, k: int = RAG_TOP_K) -> list[dict]:
        """
        Semantic search. If ticker given, filters to that ticker's docs only.
        Returns list of {text, metadata, distance} dicts.
        """
        emb = self._embedder.encode([question], show_progress_bar=False).tolist()
        where = {"ticker": ticker} if ticker else None
        results = self._collection.query(
            query_embeddings=emb,
            n_results=min(k, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        return [{"text": d, "metadata": m, "distance": s}
                for d, m, s in zip(docs, metas, dists)]

    def count(self) -> int:
        return self._collection.count()
