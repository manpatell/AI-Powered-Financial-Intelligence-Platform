# FinAI — AI-Powered Financial Intelligence Platform

An end-to-end MLOps project covering the full AI/ML lifecycle: real-time data ingestion, feature engineering, model training with experiment tracking, RAG-based Q&A, and a production-ready Streamlit dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Data | yfinance, feedparser, NewsAPI |
| Feature Engineering | pandas, numpy, ta (Technical Analysis) |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Experiment Tracking | MLflow |
| NLP / Sentiment | HuggingFace Transformers (FinBERT), sentence-transformers |
| RAG / LLM | ChromaDB, OpenAI GPT-4o-mini / Anthropic Claude |
| Dashboard | Streamlit, Plotly |
| Model Monitoring | KS-test + PSI drift detection (scipy) |
| Config | python-dotenv, pydantic |
| Logging | loguru |

---

## Project Structure

```
finai/
├── config/              # Settings, paths, and environment variables
├── data/
│   ├── stock_fetcher.py # yfinance OHLCV downloader with disk cache
│   ├── news_fetcher.py  # RSS + NewsAPI merger with deduplication
│   ├── raw/             # Raw parquet files
│   ├── processed/       # Feature-engineered parquet files
│   └── cache/           # Pickle cache for fast reloads
├── features/
│   ├── technical_indicators.py  # 25+ indicators (RSI, MACD, BB, ATR, OBV …)
│   ├── sentiment_features.py    # FinBERT scorer + daily aggregation
│   └── feature_pipeline.py      # Orchestrates fetch → features → target → scale
├── models/
│   ├── trainer.py       # XGBoost + LightGBM with 5-fold CV and MLflow logging
│   ├── predictor.py     # Inference + BUY/HOLD/SELL signal generation
│   ├── saved/           # Serialised models and scalers
│   └── experiments/     # MLflow artifacts
├── rag/
│   ├── document_store.py  # ChromaDB persistent store with sentence-transformer embeddings
│   └── rag_chain.py       # RAG pipeline (OpenAI / Claude / no-key fallback)
├── monitoring/
│   ├── drift_detector.py        # KS-test + PSI per feature
│   └── performance_tracker.py   # Rolling accuracy / AUC snapshots
├── dashboard/
│   ├── app.py                   # Home page + market snapshot
│   └── pages/
│       ├── 1_Stock_Analysis.py  # Candlestick, BB, RSI, MACD, news feed
│       ├── 2_ML_Predictions.py  # Train on demand, signal overlay, feature importance
│       ├── 3_AI_Chatbot.py      # RAG Q&A with source citations
│       ├── 4_MLflow_Tracker.py  # Experiment browser, metric charts, model registry
│       └── 5_Monitoring.py      # Drift report, PSI chart, performance history
└── utils/
    └── logger.py        # loguru structured logger with file rotation
```

---

## Setup

```powershell
# 1. Clone the repo
git clone https://github.com/manpatell/AI-Powered-Financial-Intelligence-Platform.git
cd "AI-Powered-Financial-Intelligence-Platform"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # macOS / Linux

# 3. Install dependencies and the finai package
pip install -r requirements.txt
pip install -e .

# 4. Configure API keys (optional — app works without them)
copy .env.example .env
# Open .env and add your keys

# 5. Run the dashboard
streamlit run finai/dashboard/app.py

# 6. Launch MLflow UI (separate terminal, venv activated)
mlflow ui --port 5000
```

> **PowerShell execution policy** — if step 2 fails, run once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| **Home** | Live price snapshot for 10 tickers |
| **Stock Analysis** | Candlestick + SMA/EMA/BB + RSI + MACD + volume + news feed |
| **ML Predictions** | Train XGBoost/LightGBM on demand, buy/sell signal overlay, probability histogram, feature importance |
| **AI Chatbot** | RAG-powered Q&A grounded in real news and company profiles |
| **MLflow Tracker** | Run table, metric comparison charts, Model Registry browser |
| **Monitoring** | Per-feature KS p-value + PSI drift table, rolling accuracy/AUC history |

---

## Features

- **Real-time Stock Data** — OHLCV data for 10 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, NFLX, AMD) with 4-hour disk cache
- **25+ Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Stochastic, historical volatility, rate-of-change, and more
- **FinBERT Sentiment** — Financial news scored with `ProsusAI/finbert`; VADER fallback when model unavailable; daily aggregation joined to OHLCV features
- **ML Prediction** — XGBoost and LightGBM classifiers predicting 5-day price direction with 5-fold cross-validation
- **MLflow Experiment Tracking** — Full logging of params, metrics, feature importance CSVs, classification reports, and model artifacts; Model Registry integration
- **RAG Chatbot** — ChromaDB vector store with `all-MiniLM-L6-v2` embeddings; supports OpenAI GPT-4o-mini, Anthropic Claude, or no-key context-only mode
- **Data Drift Detection** — Kolmogorov-Smirnov test and Population Stability Index per feature; flags moderate (PSI > 0.1) and significant (PSI > 0.2) drift
- **Model Performance Tracking** — Rolling accuracy and ROC-AUC snapshots to detect degradation over time

---

## API Keys (Optional)

| Key | Used for |
|-----|---------|
| `OPENAI_API_KEY` | GPT-4o-mini answers in the AI Chatbot |
| `ANTHROPIC_API_KEY` | Claude answers in the AI Chatbot (fallback to OpenAI) |
| `NEWS_API_KEY` | Richer news via NewsAPI (app works without it via RSS) |

Without any keys the app runs fully — the chatbot returns retrieved context chunks instead of an LLM-generated answer.
