"""
Centralized file path configuration.

All runtime file paths are defined here to ensure consistency
across all modules and entry points.
"""

from pathlib import Path

# Project root (directory containing main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output directory for runtime-generated files
OUTPUT_DIR = PROJECT_ROOT / "output"

# Specific file paths
CANDLE_CACHE_FILE = OUTPUT_DIR / "candle_cache.json"
ACTIVE_SYMBOLS_FILE = OUTPUT_DIR / "active_symbols.json"
HISTORICAL_DATA_FILE = OUTPUT_DIR / "historical_data.json"

# Log directory and files
LOG_DIR = PROJECT_ROOT / "logs"
TRADE_JOURNAL_FILE = LOG_DIR / "trade_journal.json"
