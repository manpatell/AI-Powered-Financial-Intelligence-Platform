"""
Sentiment feature engineering using FinBERT.
Scores financial news headlines and aggregates them to daily signals
that can be joined to the OHLCV feature DataFrame.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from finai.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@lru_cache(maxsize=1)
def _load_finbert():
    """Lazy-load FinBERT pipeline (cached so it only loads once)."""
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT sentiment model …")
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,          # CPU; change to 0 for GPU
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded.")
        return pipe
    except Exception as e:
        logger.warning(f"FinBERT unavailable ({e}), falling back to VADER")
        return None


def _vader_scores(texts: list[str]) -> list[dict]:
    """Fallback: VADER lexicon sentiment (no model download required)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        results = []
        for t in texts:
            s = analyzer.polarity_scores(t)
            results.append({
                "positive": max(s["pos"], 0),
                "negative": max(s["neg"], 0),
                "neutral":  max(s["neu"], 0),
                "compound": s["compound"],
            })
        return results
    except ImportError:
        # Final fallback: neutral scores
        return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}] * len(texts)


def score_texts(texts: list[str]) -> pd.DataFrame:
    """
    Score a list of texts with FinBERT (or VADER fallback).
    Returns DataFrame with columns: positive, negative, neutral, compound, label.
    """
    if not texts:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "compound", "label"])

    pipe = _load_finbert()
    if pipe is not None:
        rows = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                outputs = pipe(batch)
                for out in outputs:
                    scores = {item["label"].lower(): item["score"] for item in out}
                    rows.append({
                        "positive": scores.get("positive", 0.0),
                        "negative": scores.get("negative", 0.0),
                        "neutral":  scores.get("neutral",  0.0),
                        "compound": scores.get("positive", 0.0) - scores.get("negative", 0.0),
                    })
            except Exception as e:
                logger.warning(f"FinBERT batch failed: {e}")
                rows.extend(_vader_scores(batch))
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(_vader_scores(texts))

    df["label"] = df[["positive", "negative", "neutral"]].idxmax(axis=1)
    return df


def build_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a news DataFrame (columns: title, summary, published),
    compute daily aggregated sentiment scores:
      - sentiment_mean, sentiment_std   (compound score)
      - positive_ratio, negative_ratio  (fraction of articles)
      - article_count
    Returns a DataFrame indexed by date.
    """
    if news_df.empty:
        return pd.DataFrame()

    texts = (news_df["title"].fillna("") + ". " + news_df["summary"].fillna("")).tolist()
    scored = score_texts(texts)
    news_df = news_df.reset_index(drop=True)
    combined = pd.concat([news_df, scored], axis=1)

    combined["date"] = pd.to_datetime(combined["published"]).dt.normalize()

    daily = combined.groupby("date").agg(
        sentiment_mean   = ("compound",  "mean"),
        sentiment_std    = ("compound",  "std"),
        positive_ratio   = ("positive",  "mean"),
        negative_ratio   = ("negative",  "mean"),
        article_count    = ("compound",  "count"),
    ).fillna(0)

    return daily


def merge_sentiment_features(ohlcv_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment features into OHLCV feature DataFrame.
    Missing dates get forward-filled (weekend/holiday carry-over).
    """
    daily_sent = build_daily_sentiment(news_df)
    if daily_sent.empty:
        logger.warning("No sentiment data — filling with neutral zeros")
        for col in ["sentiment_mean", "sentiment_std", "positive_ratio", "negative_ratio", "article_count"]:
            ohlcv_df[col] = 0.0
        return ohlcv_df

    ohlcv_df = ohlcv_df.copy()
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index).normalize()

    merged = ohlcv_df.join(daily_sent, how="left")
    sent_cols = ["sentiment_mean", "sentiment_std", "positive_ratio", "negative_ratio", "article_count"]
    merged[sent_cols] = merged[sent_cols].ffill().fillna(0)
    return merged
