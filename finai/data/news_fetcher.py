"""
Financial news fetcher.
Sources: NewsAPI (if key provided) + RSS feeds (always available).
Returns a DataFrame of articles with title, summary, published date, source, ticker.
"""
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd
import requests

from finai.config.settings import CACHE_DIR, NEWS_API_KEY
from finai.utils.logger import get_logger

logger = get_logger(__name__)

# Free RSS feeds per ticker (Yahoo Finance + Finviz)
RSS_FEEDS = {
    "AAPL": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    "MSFT": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    "GOOGL": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOGL&region=US&lang=en-US",
    "AMZN": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMZN&region=US&lang=en-US",
    "TSLA": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    "NVDA": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    "META": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=META&region=US&lang=en-US",
    "JPM":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US",
    "NFLX": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NFLX&region=US&lang=en-US",
    "AMD":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMD&region=US&lang=en-US",
}

# General financial RSS
GENERAL_FEEDS = [
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
]


def _cache_path(ticker: str) -> object:
    return CACHE_DIR / f"news_{ticker}.pkl"


def _is_fresh(path, hours: int = 2) -> bool:
    if not path.exists():
        return False
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) < timedelta(hours=hours)


def _parse_rss(url: str, ticker: str) -> list[dict]:
    """Parse a single RSS feed and return list of article dicts."""
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            pub_dt = datetime(*published[:6]) if published else datetime.now()
            articles.append({
                "ticker": ticker,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", "")),
                "url": entry.get("link", ""),
                "source": feed.feed.get("title", url),
                "published": pub_dt,
            })
    except Exception as e:
        logger.warning(f"RSS parse failed for {url}: {e}")
    return articles


def fetch_news_rss(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch news for a ticker from Yahoo Finance RSS."""
    cache = _cache_path(ticker)
    if use_cache and _is_fresh(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    url = RSS_FEEDS.get(ticker, f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
    articles = _parse_rss(url, ticker)

    df = pd.DataFrame(articles)
    if df.empty:
        df = pd.DataFrame(columns=["ticker", "title", "summary", "url", "source", "published"])
    else:
        df["published"] = pd.to_datetime(df["published"])
        df = df.sort_values("published", ascending=False).reset_index(drop=True)

    with open(cache, "wb") as f:
        pickle.dump(df, f)

    logger.info(f"Fetched {len(df)} news articles for {ticker}")
    return df


def fetch_news_newsapi(ticker: str, company_name: str = "", days_back: int = 7) -> pd.DataFrame:
    """Fetch news via NewsAPI (requires NEWS_API_KEY)."""
    if not NEWS_API_KEY:
        logger.debug("NEWS_API_KEY not set, skipping NewsAPI fetch")
        return pd.DataFrame()

    query = company_name if company_name else ticker
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 30,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = [
            {
                "ticker": ticker,
                "title": a.get("title", ""),
                "summary": a.get("description", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "published": pd.to_datetime(a.get("publishedAt")),
            }
            for a in data.get("articles", [])
        ]
        return pd.DataFrame(articles)
    except Exception as e:
        logger.warning(f"NewsAPI failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_news(ticker: str, company_name: str = "", use_cache: bool = True) -> pd.DataFrame:
    """
    Merge RSS + NewsAPI results (deduplicated by title).
    Falls back gracefully if NewsAPI key is absent.
    """
    rss_df = fetch_news_rss(ticker, use_cache)
    api_df = fetch_news_newsapi(ticker, company_name)

    df = pd.concat([rss_df, api_df], ignore_index=True)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["title"]).sort_values("published", ascending=False)
    df = df[df["title"].str.strip().ne("")].reset_index(drop=True)
    return df
