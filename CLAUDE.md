# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based crypto trading bot that exploits exaggerated price wicks in low-liquidity perpetual futures markets using mean-reversion. Supports live trading on Hyperliquid (mainnet/testnet), automated market scanning via WebSocket, backtesting with Monte Carlo analysis, and dynamic symbol rotation.

## Commands

```bash
# Install dependencies
pip install aiohttp pandas numpy
pip install eth-account hyperliquid-python-sdk  # Required for Hyperliquid trading

# Scan markets to find best symbols
python scan_markets.py --days 30 --top 20

# Fetch historical data for backtesting
python fetch_hyperliquid_history.py --symbols TAO-PERP AAVE-PERP --timeframe 5m --days 90

# Run backtest with analysis
python run_backtest.py --input output/historical_data.json --analyze

# Paper trading on testnet
python run_paper_trade.py --private-key $HL_PRIVATE_KEY --capital 1000

# Verify setup before going live
python verify_setup.py --private-key $HL_PRIVATE_KEY

# Live market monitor (Terminal 1 — finds best symbols in real-time)
python live_monitor.py --auto-update --output output/active_symbols.json

# Live trading (Terminal 2 — executes trades)
python run_live.py --private-key $HL_PRIVATE_KEY --watch-symbols output/active_symbols.json --capital 200
```

No linter or test framework is configured. If adding tests, use `pytest` with `test_*.py` naming.

## Architecture

### Data Flow
1. **Data ingestion** (`data/ingestion.py`) → Candles from Hyperliquid/Bybit APIs or simulated data
2. **Candle caching** (`data/candle_cache.py`) → WebSocket candle cache with disk persistence (200 candles/symbol rolling buffer)
3. **Signal detection** (`signals/detection.py`) → `WickAnalyzer` checks wick criteria with weighted scoring, `MarketFilter` applies regime filters
4. **Risk assessment** (`risk/management.py`) → Position sizing with dynamic leverage (3-5x based on signal confidence)
5. **Order execution** (`execution/orders.py`) → `HyperliquidExecutor` places orders via SDK, including exchange-based SL/TP trigger orders
6. **Main orchestrator** (`main.py`) → `WickReversalStrategy` coordinates candle processing, position lifecycle, trailing stops, partial TP

### Key Classes
- `WickReversalStrategy` (main.py) — Main orchestrator: candle processing, signal evaluation, position entry/exit, exchange SL/TP placement, trailing stops, partial take profit
- `SignalGenerator` (signals/detection.py) — Combines `WickAnalyzer` + `MarketFilter` to produce weighted `Signal` objects
- `RiskManager` (risk/management.py) — Position sizing, circuit breakers (5% daily / 15% max drawdown), cooldown enforcement
- `BacktestEngine` (backtest/engine.py) — Historical simulation with Monte Carlo analysis and per-dimension trade breakdown
- `HyperliquidScanner` (scanner/market_scanner.py) — Scans all perp markets and ranks by strategy profitability (signal freq 35%, win rate 25%, profit factor 20%, ATR 15%, strength 5%)
- `LiveMarketMonitor` (scanner/live_monitor.py) — Real-time WebSocket monitoring with alerts, auto-symbol updates, Discord webhook support
- `LiveTradingManager` (trading/live_manager.py) — Manages live trading with session P&L tracking, pre-existing position detection, periodic cache refresh
- `PaperTradingManager` (trading/paper_manager.py) — Paper trading manager (testnet)
- `BaseTradingManager` (trading/base_manager.py) — Shared infrastructure: stream management, candle cache loading, symbol subscription/unsubscription
- `SymbolWatcher` (trading/symbol_watcher.py) — Watches JSON file for dynamic symbol rotation with callbacks
- `CandleCache` (data/candle_cache.py) — WebSocket streaming with persistent disk storage, atomic writes, background save every 60s
- `HyperliquidExecutor` (execution/orders.py) — Hyperliquid SDK integration: orders, trigger orders (SL/TP), balance/position queries, 62+ symbol asset index

### Configuration System
All parameters in `config/settings.py` using nested dataclasses:
- `StrategyConfig` (master) contains `SignalConfig`, `EntryConfig`, `ExitConfig`, `RiskConfig`, `DynamicLeverageConfig`, `FilterConfig`, `ExecutionConfig`, `BacktestConfig`
- `SymbolConfig` for per-symbol overrides (risk multiplier, base position size $200-$500)
- `DEFAULT_CONFIG` with defaults: TAO-PERP, AAVE-PERP, ZRO-PERP

File paths centralized in `config/paths.py`:
- `CANDLE_CACHE_FILE`, `ACTIVE_SYMBOLS_FILE`, `HISTORICAL_DATA_FILE`, `TRADE_JOURNAL_FILE`
- All runtime data goes to `output/`, logs to `logs/`

### Core Types (core/types.py)
- `Candle` — OHLCV with computed properties (upper_wick, lower_wick, wick_to_body_ratio, range, midpoint)
- `Signal` — Signal with strength (0-1), criteria met, entry/stop/target levels, filter status
- `Position` — Active position with trailing stop logic, exchange order IDs for SL/TP, partial TP tracking
- `TradeResult` — Completed trade records with commission and exit reason
- `MarketData` — Aggregated candle + indicators (ATR, VWAP, volume metrics, volatility, orderbook, BTC ref)
- `Order` — Order with fill tracking, commission, slippage, exchange order ID
- Enums: `Side`, `OrderType`, `OrderStatus`, `PositionStatus`, `SignalType`, `FilterResult`

## Signal Detection Logic

Signals require:
1. **Minimum wick size**: 0.8% of price
2. **Rejection confirmation**: Close must be >= 42% of wick away from extreme
3. **One or more** (OR logic by default, configurable to AND):
   - Wick-to-body ratio >= 2.3x (weight 0.95, 67.4% win rate)
   - Wick > 1.5x ATR (weight 1.3, 81% win rate — most predictive)
   - Distance from VWAP > 1.0 ATR (weight 1.0, 68.7% win rate)
   - Volume bonus for high volume at wick extreme (weight 0.8)

Upper wick -> SHORT signal; Lower wick -> LONG signal

Signal strength is 0-1 based on weighted sum of criteria met.

### Market Filters
Filters disable trading when:
- Volume spike > 3.5x baseline
- ATR expansion > 2.2x baseline
- BTC move > 3%
- Volume < $1,000 USD
- Spread > 1%
- Orderbook depth < $4,000 USD at 0.5% from mid
- Momentum > 3% move over 12 candles

## Live Trading Integration

### Dynamic Symbol Selection
1. `live_monitor.py` scans all markets via WebSocket and writes best symbols to `output/active_symbols.json`
2. `run_live.py` / `run_paper_trade.py` watch this file via `SymbolWatcher` and add/remove symbols dynamically
3. Pre-existing manual positions are detected and ignored by the bot
4. Pinned symbols can be set with `--pinned` to prevent auto-removal

### Exchange-Based Protection
- SL/TP trigger orders placed on Hyperliquid for each position
- Partial TP: 50% of position closes at 0.5x ATR, remainder at full target
- Positions are protected even if bot goes offline
- Bot reconciles with exchange fills on restart

### Key Defaults
- Risk per trade: 2% of account equity
- Dynamic leverage: 3-5x based on signal confidence (+0.25x per criteria met, capped at 2.5x multiplier)
- Max positions: 10 concurrent
- Stop loss: Beyond wick extreme + 0.4x ATR buffer (max 2.5%)
- Take profit: 0.9x ATR from entry
- Trailing stop: Activates at 75% of target, trails at 0.5x ATR
- Max hold: 12 candles
- Daily loss limit: 5%, max drawdown: 15%

## Coding Conventions

- Use type hints and dataclasses for data models
- Follow PEP 8 with 4-space indentation
- Module names reflect responsibility (e.g., `signals/detection.py`, `risk/management.py`)
- Keep domain logic in its corresponding package; `main.py` is the orchestrator only
- Async/await for I/O operations (exchange connections, candle streaming)
- Entry-point scripts stay at project root for direct execution (no `setup.py` needed)
- All file paths use `config/paths.py` constants — never hardcode paths
