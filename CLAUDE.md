# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based crypto trading strategy that exploits exaggerated price wicks in low-liquidity perpetual futures markets using mean-reversion. The strategy detects wick rejection patterns on closed candles and trades the subsequent price reversion. Supports live trading on Hyperliquid (mainnet/testnet) with dynamic symbol selection.

## Commands

```bash
# Install dependencies
pip install aiohttp pandas numpy
pip install eth-account  # Required for Hyperliquid trading

# Scan markets to find best symbols for trading
python scan_markets.py --days 30 --top 20

# Fetch historical data for backtesting
python fetch_hyperliquid_history.py --symbols TAO-PERP AAVE-PERP --timeframe 5m --days 90 --out historical_data.json

# Run backtest with analysis
python run_real_data.py --input historical_data.json --analyze

# Paper trading on testnet
python run_paper_trade.py --private-key $HL_PRIVATE_KEY --capital 1000

# LIVE trading on mainnet (real money!)
python run_live.py --private-key $HL_PRIVATE_KEY --capital 200

# Live market monitor (finds best symbols in real-time)
python live_monitor.py --current TAO-PERP AAVE-PERP --auto-update --output active_symbols.json

# Combined: monitor + live trading with dynamic symbols
python live_monitor.py --auto-update --output active_symbols.json &
python run_live.py --watch-symbols active_symbols.json --capital 200
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
- `HyperliquidScanner` (scan_markets.py) - Scans all markets and ranks by strategy profitability
- `LiveMarketMonitor` (live_monitor.py) - Real-time monitoring with alerts and auto-symbol updates
- `LiveTradingManager` / `DynamicTradingManager` (run_live.py, run_paper_trade.py) - Manages live trading with dynamic symbol updates via `SymbolWatcher`

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

## Live Trading Integration

### Environment Variables
- `HL_PRIVATE_KEY` - Ethereum private key for Hyperliquid authentication

### Dynamic Symbol Selection
The system supports real-time symbol rotation:
1. `live_monitor.py` scans all markets and writes best symbols to `active_symbols.json`
2. `run_live.py` / `run_paper_trade.py` watch this file via `SymbolWatcher` and add/remove symbols dynamically
3. Pre-existing manual positions are detected and ignored by the bot

### Market Scoring (scan_markets.py, live_monitor.py)
Markets are ranked by weighted criteria:
- Signal frequency (35%) - Most important for wick strategy
- Win rate from simulated trades (25%)
- Profit factor (20%)
- ATR volatility (15%)
- Signal strength (5%)

## Coding Conventions

- Use type hints and dataclasses for data models
- Follow PEP 8 with 4-space indentation
- Module names reflect responsibility (e.g., `signals/detection.py`, `risk/management.py`)
- Keep domain logic in its corresponding package; `main.py` is the orchestrator only
- Async/await for I/O operations (exchange connections, candle streaming)
