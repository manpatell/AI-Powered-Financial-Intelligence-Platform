# FinAI — AI-Powered Financial Intelligence Platform

An end-to-end MLOps project that demonstrates the full AI/ML lifecycle: data ingestion, feature engineering, model training with experiment tracking, RAG-based Q&A, and a production-ready Streamlit dashboard.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | yfinance, NewsAPI, feedparser |
| Feature Engineering | pandas, numpy, ta (Technical Analysis) |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Experiment Tracking | MLflow |
| NLP / Sentiment | HuggingFace Transformers, sentence-transformers |
| RAG / LLM | LangChain, ChromaDB, OpenAI / Anthropic |
| Dashboard | Streamlit, Plotly |
| Model Monitoring | Evidently AI |
| Config | python-dotenv, pydantic |
| Logging | loguru |

## Project Structure

```
finai/
├── config/          # Settings and environment variables
├── data/            # Raw, processed, and cached data
├── features/        # Feature engineering pipeline
├── models/          # ML model training + MLflow tracking
├── pipelines/       # End-to-end data + training pipelines
├── rag/             # LangChain RAG + ChromaDB
├── monitoring/      # Evidently drift detection
├── dashboard/       # Streamlit multi-page app
│   └── pages/       # Individual dashboard pages
└── utils/           # Logging, helpers
```

## Setup

```bash
# 1. Clone and create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 4. Run the dashboard
streamlit run finai/dashboard/app.py

# 5. Launch MLflow UI (in a separate terminal)
mlflow ui --port 5000
```

## Features

- **Real-time Stock Data** — Live OHLCV data via yfinance for 10+ tickers
- **Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, and 20+ more
- **Sentiment Analysis** — Financial news sentiment using FinBERT
- **ML Prediction** — XGBoost classifier predicting 5-day price direction
- **MLflow Tracking** — Full experiment history, model registry, artifact storage
- **RAG Chatbot** — Ask natural language questions about any stock using ChromaDB + LLM
- **Model Monitoring** — Data drift and model performance tracking with Evidently
- **Interactive Dashboard** — Multi-page Streamlit app with Plotly charts
