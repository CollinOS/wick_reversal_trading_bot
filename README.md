# Wick Reversal Trading Bot

A Python-based crypto trading bot that exploits exaggerated price wicks in low-liquidity perpetual futures markets using mean-reversion. Supports live trading on Hyperliquid, paper trading on testnet, automated market scanning, and historical backtesting with Monte Carlo analysis.

Quick Note: This bot performs quite well when backtesting on old data, but has failed to maintain profitability when trading on live markets. Making the repo public in case any friends want to toy around with it and see what they can get working. The biggest struggle points were losing profits due to slippage and racking up large losses on low cap alts making large moves quickly. Good luck!

## Strategy Overview

### Concept
This strategy targets low-to-medium liquidity crypto perpetual futures where exaggerated wicks frequently occur due to thin order books and stop-loss sweeps. The core assumption is that extreme wicks represent temporary price dislocations — not new information — and price will revert to fair value shortly after.

### Key Features
- **Live trading** on Hyperliquid mainnet with exchange-based stop loss/take profit orders
- **Paper trading** on Hyperliquid testnet with identical logic
- **Automated market scanning** — ranks all perpetual markets by strategy profitability
- **Real-time market monitor** — WebSocket-based opportunity detection with dynamic symbol rotation
- **Backtesting engine** with Monte Carlo analysis (1000 simulations)
- **Dynamic leverage** — scales 3-5x based on signal confidence
- **Partial take profit** — closes 50% at first target, lets the rest ride
- **Multi-condition signal detection** (wick-to-body ratio, ATR-normalized wick size, VWAP distance)
- **Market regime filters** (volume spikes, volatility expansion, BTC correlation, momentum)

## Project Structure

```
wick_reversal_trading_bot/
├── main.py                          # Strategy orchestrator (WickReversalStrategy)
├── run_live.py                      # Live trading CLI
├── run_paper_trade.py               # Paper trading CLI
├── run_backtest.py                  # Backtest with historical data
├── scan_markets.py                  # Market scanner CLI
├── live_monitor.py                  # Live market monitor CLI
├── fetch_hyperliquid_history.py     # Fetch candle history from Hyperliquid
├── fetch_bybit_history.py           # Fetch candle history from Bybit
├── verify_setup.py                  # Pre-flight checks for live trading
├── test_candle_fetch.py             # Candle fetch smoke test
│
├── config/
│   ├── settings.py                  # All configurable parameters (nested dataclasses)
│   └── paths.py                     # Centralized path constants
├── core/
│   └── types.py                     # Data structures (Candle, Signal, Position, Order)
├── data/
│   ├── ingestion.py                 # Data providers (Hyperliquid, Bybit, Simulated)
│   └── candle_cache.py              # WebSocket candle cache with disk persistence
├── signals/
│   └── detection.py                 # Wick analysis, market filters, signal generation
├── risk/
│   └── management.py                # Position sizing, exposure limits, circuit breakers
├── execution/
│   └── orders.py                    # Order management and Hyperliquid SDK integration
├── backtest/
│   └── engine.py                    # Backtesting engine with Monte Carlo analysis
├── utils/
│   └── logger.py                    # Structured logging and trade journal
├── trading/                         # Shared live/paper trading infrastructure
│   ├── symbol_watcher.py            # Dynamic symbol rotation from JSON file
│   ├── base_manager.py              # Common trading manager logic
│   ├── live_manager.py              # LiveTradingManager (mainnet)
│   └── paper_manager.py             # PaperTradingManager (testnet)
├── scanner/                         # Market scanning modules
│   ├── market_scanner.py            # HyperliquidScanner + MarketScore
│   └── live_monitor.py              # LiveMarketMonitor + WebSocket alerts
├── output/                          # Runtime-generated data (gitignored)
└── logs/                            # Trade journals and logs (gitignored)
```

## Installation

```bash
# Install dependencies
pip install aiohttp pandas numpy

# For Hyperliquid live trading
pip install eth-account hyperliquid-python-sdk
```

## Quick Start

### 1. Scan Markets for Opportunities

```bash
# Scan all Hyperliquid perps and rank by strategy profitability
python scan_markets.py --days 30 --top 20
```

### 2. Fetch Historical Data and Backtest

```bash
# Fetch 90 days of 5m candles for top symbols
python fetch_hyperliquid_history.py --symbols TAO-PERP AAVE-PERP ZRO-PERP --timeframe 5m --days 90

# Run backtest with analysis
python run_backtest.py --input output/historical_data.json --analyze

# Export trades to CSV
python run_backtest.py --input output/historical_data.json --export-csv trades.csv
```

### 3. Paper Trade on Testnet

```bash
python run_paper_trade.py --private-key YOUR_PRIVATE_KEY --capital 1000
```

### 4. Go Live

```bash
# Step 1: Verify your setup
python verify_setup.py --private-key YOUR_PRIVATE_KEY

# Step 2: Start the market monitor (Terminal 1)
python live_monitor.py --auto-update --output output/active_symbols.json

# Step 3: Start the trading bot (Terminal 2)
python run_live.py --private-key YOUR_PRIVATE_KEY --watch-symbols output/active_symbols.json --capital 200
```

The monitor continuously scans all markets and writes the best symbols to `output/active_symbols.json`. The trading bot watches this file and dynamically adds/removes symbols.

## Signal Detection

Signals are generated when a closed candle exhibits an exaggerated wick. One or more conditions must be met (OR logic by default):

| Condition | Default Threshold | Weight | Historical Win Rate |
|-----------|------------------|--------|---------------------|
| Wick-to-body ratio | >= 2.3x | 0.95 | 67.4% |
| Wick ATR ratio | >= 1.5x ATR | 1.3 | 81.0% (most predictive) |
| VWAP distance | >= 1.0 ATR | 1.0 | 68.7% |
| Volume bonus | High volume at wick | 0.8 | — |

**Additional requirements:**
- Minimum wick size: 0.8% of price
- Rejection confirmation: Close must be >= 42% of wick away from extreme

**Signal types:**
- Upper wick -> SHORT (price rejected from highs)
- Lower wick -> LONG (price rejected from lows)

## Entry Rules

| Rule | Description |
|------|-------------|
| Entry timing | After candle close confirms rejection |
| Entry mode | Retrace to 60% of wick midpoint (default) or at candle close |
| Max wait | 2 candles for retrace entry |
| Order type | Market orders (for reliability) |

## Exit Rules

### Take Profit
- **Primary target**: ATR-based (0.9x ATR from entry)
- **Partial TP**: 50% of position closes at 0.5x ATR, remainder rides to full target

### Stop Loss
- Placed beyond wick extreme + 0.4x ATR buffer
- Maximum: 2.5% of entry price
- **Exchange-based**: SL/TP orders placed directly on Hyperliquid for protection if bot goes offline

### Additional Exits
- **Trailing stop**: Activates at 75% of target, trails at 0.5x ATR
- **Time exit**: Maximum 12 candles held

## Risk Management

### Position Sizing
```
Position Size = (Account Equity x Risk %) / (Entry - Stop Loss)
```
Default: 2% risk per trade

### Dynamic Leverage
Leverage scales with signal confidence:
- Base: 3x on Hyperliquid
- +0.25x per additional criteria met
- Capped at 5x maximum
- Signal strength multiplier: 0.5x to 2.5x

### Exposure Limits
| Parameter | Default | Description |
|-----------|---------|-------------|
| Max positions | 10 | Total simultaneous positions |
| Max per symbol | 1 | Positions per trading pair |
| Max leverage | 5.0x | Maximum effective leverage |
| Base position size | $200-$500 | Per-symbol configurable |

### Circuit Breakers
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Daily loss | -5% | Stop trading for the day |
| Max drawdown | -15% | Stop all trading |

## Market Filters

Trading is **disabled** when:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Volume spike | > 3.5x baseline | Abnormal activity (news, manipulation) |
| ATR expansion | > 2.2x baseline | Regime change, breakout |
| BTC move | > 3% in recent candles | Market-wide risk-off |
| Low volume | < $1,000 USD | Insufficient liquidity |
| Wide spread | > 1% | Poor execution conditions |
| Thin orderbook | < $4,000 depth | Slippage risk |
| Momentum | > 3% move in 12 candles | Counter-trend too risky |

## Market Scanning

The scanner ranks all Hyperliquid perpetual markets using weighted criteria:

| Criteria | Weight | Description |
|----------|--------|-------------|
| Signal frequency | 35% | Signals per day — most important |
| Win rate | 25% | Simulated trade win percentage |
| Profit factor | 20% | Gross profit / gross loss |
| Volatility (ATR) | 15% | Higher ATR = more opportunities |
| Signal strength | 5% | Average signal quality |

```bash
# Full market scan
python scan_markets.py --days 30 --top 20

# Filter for volatile assets only
python scan_markets.py --days 14 --volatile-only --min-signals 50

# Export results
python scan_markets.py --days 30 --export results.csv
```

## Live Trading

### Two-Terminal Setup

**Terminal 1** — Market monitor (detects best symbols):
```bash
python live_monitor.py --auto-update --output output/active_symbols.json
```

**Terminal 2** — Trading bot (executes trades):
```bash
python run_live.py --private-key YOUR_PRIVATE_KEY --watch-symbols output/active_symbols.json
```

### Features
- **Dynamic symbol rotation**: Monitor updates `active_symbols.json`, bot subscribes/unsubscribes automatically
- **Pinned symbols**: `--pinned TAO-PERP` to always keep a symbol active
- **Pre-existing position detection**: Bot ignores positions opened manually
- **Session P&L tracking**: Running profit/loss displayed throughout session
- **Candle caching**: Monitor writes cached candle data for the bot to preload (faster startup)
- **Exchange-based stops**: SL/TP orders on Hyperliquid protect positions even if bot is offline
- **Graceful shutdown**: Ctrl+C cleanly disconnects streams

### Pre-Flight Verification

```bash
python verify_setup.py --private-key YOUR_PRIVATE_KEY
```

Checks: private key validity, wallet address, exchange connection, account balance, asset index mapping, existing positions.

## Backtesting

```bash
# Fetch data
python fetch_hyperliquid_history.py --symbols TAO-PERP AAVE-PERP --days 90

# Run backtest with Monte Carlo analysis
python run_backtest.py --input output/historical_data.json --analyze

# Detailed trade log
python run_backtest.py --input output/historical_data.json --trade-details

# Export trades to CSV
python run_backtest.py --input output/historical_data.json --export-csv trades.csv
```

### Metrics Reported
- Win rate, profit factor, expectancy
- Sharpe, Sortino, and Calmar ratios
- Max drawdown (depth and duration)
- Exit reason breakdown (TP, SL, trailing, time)
- Long vs short performance
- Monte Carlo 5th/95th percentile outcomes

## Configuration

All parameters are in `config/settings.py` using nested dataclasses. Key sections:

| Config | Controls |
|--------|----------|
| `SignalConfig` | Wick detection thresholds |
| `EntryConfig` | Entry mode, retrace settings |
| `ExitConfig` | TP targets, SL buffer, trailing stop, partial TP |
| `RiskConfig` | Risk per trade, max positions, drawdown limits |
| `DynamicLeverageConfig` | Signal-based leverage scaling |
| `FilterConfig` | Market regime filters |
| `ExecutionConfig` | Slippage, order timeouts, leverage range |
| `BacktestConfig` | Commission, funding rates, Monte Carlo settings |
| `SymbolConfig` | Per-symbol overrides (risk multiplier, base size) |

Default symbols: TAO-PERP, AAVE-PERP, ZRO-PERP

## Target Markets

**Ideal characteristics:**
- Mid-cap altcoins with perpetual futures
- Enough liquidity for execution ($1K+ daily volume)
- Frequent wick patterns (thinner order books)
- Active retail participation (stop hunts common)

**Avoid:**
- BTC/ETH (too efficient, wicks = real information)
- Very low-cap tokens (execution impossible)
- New listings (unstable behavior)

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always:
- Trade only with capital you can afford to lose
- Understand the risks of leveraged trading
- Do your own research and testing
- Start with small positions and scale gradually

## License

MIT License - See LICENSE file for details.
