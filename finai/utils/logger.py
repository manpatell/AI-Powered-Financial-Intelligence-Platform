"""Centralized logging using loguru."""
import sys
from loguru import logger
from finai.config.settings import LOGS_DIR, ROOT_DIR
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Remove default handler
logger.remove()

# Console handler — color-coded, concise
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    colorize=True,
)

# File handler — full detail, rotation
logger.add(
    LOGS_DIR / "finai_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
)


def get_logger(name: str):
    return logger.bind(name=name)
