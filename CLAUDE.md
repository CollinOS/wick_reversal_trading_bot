# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based crypto trading strategy that exploits exaggerated price wicks in low-liquidity perpetual futures markets using mean-reversion. The strategy detects wick rejection patterns on closed candles and trades the subsequent price reversion.

## Commands

```bash
# Install dependencies
pip install aiohttp pandas numpy
pip install eth-account  # Optional: Hyperliquid integration

# Run backtest with synthetic data
python run_test_data.py

# Fetch real historical data from Hyperliquid
python fetch_hyperliquid_history.py --symbols DOGE-PERP kSHIB-PERP --timeframe 5m --days 90 --out historical_data.json

# Run backtest with real historical data
python run_real_data.py --input historical_data.json

# Run main strategy (requires data provider configuration)
python main.py
```

No linter or test framework is configured. If adding tests, use `pytest` with `test_*.py` naming.

## Architecture

### Data Flow
1. **Data ingestion** (`data/ingestion.py`) → Candles from exchange APIs (Hyperliquid, Bybit, or simulated)
2. **Signal detection** (`signals/detection.py`) → `WickAnalyzer` checks wick criteria, `MarketFilter` applies regime filters
3. **Risk assessment** (`risk/management.py`) → Position sizing, exposure limits, drawdown checks
4. **Order execution** (`execution/orders.py`) → `OrderManager` submits via `ExecutionHandler` implementations
5. **Main orchestrator** (`main.py`) → `WickReversalStrategy` coordinates all components

### Key Classes
- `WickReversalStrategy` (main.py) - Main orchestrator handling candle processing, position lifecycle, and live streaming
- `SignalGenerator` (signals/detection.py) - Combines `WickAnalyzer` + `MarketFilter` to produce `Signal` objects
- `RiskManager` (risk/management.py) - Position sizing, circuit breakers, cooldown enforcement
- `BacktestEngine` (backtest/engine.py) - Historical simulation with Monte Carlo analysis

### Configuration System
All parameters are centralized in `config/settings.py` using nested dataclasses:
- `StrategyConfig` (master) contains `SignalConfig`, `EntryConfig`, `ExitConfig`, `RiskConfig`, `FilterConfig`, `ExecutionConfig`, `BacktestConfig`
- `SymbolConfig` for per-symbol overrides
- `DEFAULT_CONFIG` provides sensible defaults

### Core Types (core/types.py)
- `Candle` - OHLCV with computed properties (upper_wick, lower_wick, wick_to_body_ratio)
- `Signal` - Generated signals with entry/stop/target levels and filter status
- `Position` - Active position tracking with trailing stop logic
- `TradeResult` - Completed trade records for analysis

## Signal Detection Logic

Signals require:
1. **Minimum wick size** (% of price)
2. **Rejection confirmation** (close must be away from wick extreme)
3. **One or more** (configurable AND/OR):
   - Wick-to-body ratio threshold
   - Wick > k × ATR
   - Distance from VWAP > threshold ATR

Upper wick → SHORT signal; Lower wick → LONG signal

Market filters disable trading during: volume spikes, ATR expansion, large BTC moves, low volume, wide spreads, thin orderbooks.

## Coding Conventions

- Use type hints and dataclasses for data models
- Follow PEP 8 with 4-space indentation
- Module names reflect responsibility (e.g., `signals/detection.py`, `risk/management.py`)
- Keep domain logic in its corresponding package; `main.py` is the orchestrator only
- Async/await for I/O operations (exchange connections, candle streaming)
