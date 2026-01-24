# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python strategy project organized by domain. Key directories:
- `config/` holds strategy configuration dataclasses (see `config/settings.py`).
- `core/` defines shared types used across the strategy.
- `data/` provides market data ingestion adapters.
- `signals/` generates wick-reversal signals.
- `risk/` enforces sizing, limits, and drawdown rules.
- `execution/` handles order submission and tracking.
- `backtest/` contains the backtesting engine.
- `utils/` contains logging helpers; `logs/` stores runtime output.
Top-level entry points: `main.py` (live/paper orchestration) and `run_test_data.py` (demo/backtest usage).

## Build, Test, and Development Commands
This project is run directly with Python. Common commands:
- `pip install aiohttp pandas numpy` installs core dependencies.
- `pip install eth-account` enables optional Hyperliquid integration.
- `python run_test_data.py` runs the example workflow.
- `python main.py` starts the main strategy runner (configure in `config/settings.py` first).

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
