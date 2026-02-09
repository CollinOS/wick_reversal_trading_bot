# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python strategy project organized by domain. Key directories:
- `config/` holds strategy configuration dataclasses (`settings.py`) and centralized path constants (`paths.py`).
- `core/` defines shared types used across the strategy.
- `data/` provides market data ingestion adapters and WebSocket candle cache.
- `signals/` generates wick-reversal signals.
- `risk/` enforces sizing, limits, and drawdown rules.
- `execution/` handles order submission and tracking.
- `backtest/` contains the backtesting engine.
- `trading/` shared trading infrastructure: `BaseTradingManager`, `LiveTradingManager`, `PaperTradingManager`, `SymbolWatcher`.
- `scanner/` market scanning: `HyperliquidScanner`, `LiveMarketMonitor`, `MarketScore`.
- `utils/` contains logging helpers; `logs/` stores runtime output.
- `output/` stores runtime-generated data files (candle cache, active symbols, historical data).

Top-level entry points: `main.py` (orchestrator), `run_live.py` (live trading), `run_paper_trade.py` (paper trading), `run_backtest.py` (backtesting), `scan_markets.py` (market scanner), `live_monitor.py` (live monitor).

## Build, Test, and Development Commands
This project is run directly with Python. Common commands:
- `pip install aiohttp pandas numpy` installs core dependencies.
- `pip install eth-account` enables optional Hyperliquid integration.
- `python run_backtest.py --input output/historical_data.json --analyze` runs backtesting.
- `python scan_markets.py --days 30 --top 20` scans markets for opportunities.
- `python live_monitor.py --auto-update` starts the live market monitor.
- `python verify_setup.py --private-key KEY` runs pre-flight checks.
- `python run_live.py --private-key KEY --watch-symbols output/active_symbols.json` starts live trading.
- `python run_paper_trade.py --watch-symbols output/active_symbols.json` starts paper trading.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use type hints and dataclasses for configuration and data models (see `config/` and `core/`).
- Name modules by responsibility (`signals/detection.py`, `risk/management.py`).
- Keep domain logic in its corresponding package; avoid cross-cutting helpers in `main.py`.
No formatter or linter is configured in this repo.

## Testing Guidelines
No automated test framework or `tests/` directory is currently present. If you add tests:
- Prefer `pytest` and name files `test_*.py`.
- Keep tests close to the module they cover or in a top-level `tests/` folder.
- Document how to run them (e.g., `pytest -q`) in this file and `README.md`.

## Commit & Pull Request Guidelines
No Git history is available in this checkout, so follow a clear, imperative convention:
- Example: `signals: add wick ATR filter` or `risk: tighten drawdown stop`.
Pull requests should include:
- A short summary and rationale.
- Noted config changes (e.g., defaults in `config/settings.py`).
- Backtest evidence or sample output when behavior changes.

## Security & Configuration Tips
- Treat API keys and exchange credentials as secrets; never commit them.
- Keep environment-specific settings out of source; prefer local overrides or `.env` files (if added).
- Paper-trade first when adjusting risk or execution settings.
