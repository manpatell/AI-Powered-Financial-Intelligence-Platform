"""Central configuration for FinAI platform."""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "finai" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT_DIR / "finai" / "models" / "saved"
CHROMA_DIR = ROOT_DIR / "finai" / "rag" / "chroma_db"
LOGS_DIR = ROOT_DIR / "logs"

# Create dirs if missing
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, CHROMA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
_VALID_ENVS = {"development", "staging", "production"}
APP_ENV = os.getenv("APP_ENV", "development").lower()
if APP_ENV not in _VALID_ENVS:
    warnings.warn(
        f"APP_ENV='{APP_ENV}' is not recognised. Valid values: {sorted(_VALID_ENVS)}. "
        "Defaulting to 'development'.",
        stacklevel=1,
    )
    APP_ENV = "development"

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Warn in production if no LLM key is configured
if APP_ENV == "production" and not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    warnings.warn(
        "Running in production without an LLM API key. "
        "The AI Chatbot will return raw context only. "
        "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable LLM answers.",
        stacklevel=1,
    )

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = "finai-stock-prediction"

# ── Cache ──────────────────────────────────────────────────────────────────────
STOCK_CACHE_TTL_HOURS = int(os.getenv("STOCK_CACHE_TTL_HOURS", "4"))
NEWS_CACHE_TTL_HOURS  = int(os.getenv("NEWS_CACHE_TTL_HOURS",  "2"))


# ── Data ───────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "NFLX", "AMD"]
DEFAULT_PERIOD = "2y"        # yfinance period
DEFAULT_INTERVAL = "1d"      # daily candles
PREDICTION_HORIZON = 5       # days ahead to predict

# ── Features ───────────────────────────────────────────────────────────────────
FEATURE_WINDOW_SHORT = 10
FEATURE_WINDOW_MEDIUM = 20
FEATURE_WINDOW_LONG = 50

# ── Model ──────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "target"     # 1 = price up in N days, 0 = down
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── RAG ────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "financial_docs"
RAG_TOP_K = 5
