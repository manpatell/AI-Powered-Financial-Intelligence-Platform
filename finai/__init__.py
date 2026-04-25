"""FinAI — AI-Powered Financial Intelligence Platform."""
__version__ = "1.0.0"
__author__  = "manpatell"

from finai.config.settings import DEFAULT_TICKERS, PREDICTION_HORIZON
from finai.data.stock_fetcher import fetch_stock_data, fetch_multiple_tickers
from finai.features.feature_pipeline import build_features, build_train_test
from finai.features.technical_indicators import add_technical_indicators, get_feature_columns
from finai.models.trainer import train_model, train_all_models
from finai.models.predictor import predict, get_latest_signal

__all__ = [
    "__version__",
    "__author__",
    # Config
    "DEFAULT_TICKERS",
    "PREDICTION_HORIZON",
    # Data
    "fetch_stock_data",
    "fetch_multiple_tickers",
    # Features
    "build_features",
    "build_train_test",
    "add_technical_indicators",
    "get_feature_columns",
    # Models
    "train_model",
    "train_all_models",
    "predict",
    "get_latest_signal",
]
