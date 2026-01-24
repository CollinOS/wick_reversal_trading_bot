# Wick Reversal Strategy

A systematic crypto trading strategy that exploits exaggerated price wicks in low-liquidity perpetual futures markets using a mean-reversion approach.

## Strategy Overview

### Concept
This strategy targets low-to-medium liquidity crypto perpetual futures markets where exaggerated wicks frequently occur due to thin order books and stop-loss sweeps. The core assumption is that extreme wicks represent temporary price dislocations—not new information—and price will revert to fair value shortly after.

### Key Features
- **Mean-reversion approach** targeting wick rejection patterns
- **Multi-condition signal detection** (wick-to-body ratio, ATR-normalized wick size, VWAP distance)
- **Comprehensive risk management** (fixed fractional sizing, hard stops, position limits)
- **Market regime filters** (volume spikes, volatility expansion, BTC correlation)
- **Modular architecture** for easy customization and backtesting

## Project Structure

```
wick_reversal_strategy/
├── config/
│   └── settings.py          # All configurable parameters
├── core/
│   └── types.py              # Data structures (Candle, Signal, Position, Order)
├── data/
│   └── ingestion.py          # Data providers (Hyperliquid, Bybit, Simulated)
├── signals/
│   └── detection.py          # Wick analysis and signal generation
├── risk/
│   └── management.py         # Position sizing, exposure, drawdown control
├── execution/
│   └── orders.py             # Order management and execution handlers
├── backtest/
│   └── engine.py             # Backtesting engine with Monte Carlo analysis
├── utils/
│   └── logging.py            # Structured logging and trade journal
├── main.py                   # Main strategy orchestrator
├── run_test_data.py            # Example usage script
└── README.md
```

## Installation

```bash
# Clone or copy the strategy directory
cd wick_reversal_strategy

# Install dependencies
pip install aiohttp pandas numpy

# For Hyperliquid integration (optional)
pip install eth-account
```

### Key Commands

  # Run backtest with detailed analysis
  python run_real_data.py --input historical_data.json --analyze

  # Export trades to CSV for external analysis
  python run_real_data.py --input historical_data.json --export-csv trades.csv

  # Both
  python run_real_data.py --input historical_data.json --analyze --export-csv trades.csv
  

## Quick Start

### 1. Run Backtest Example

```python
from run_test_data import run_backtest_example

engine, metrics = run_backtest_example()
print(metrics.to_dict())
```

### 2. Customize Configuration

```python
from config.settings import StrategyConfig, SignalConfig

config = StrategyConfig()

# Adjust signal detection
config.signal.wick_to_body_ratio = 2.5
config.signal.wick_atr_multiplier = 1.5

# Adjust risk parameters
config.risk.risk_per_trade_pct = 0.005  # 0.5% risk per trade
config.risk.max_positions = 3
config.risk.max_leverage = 2.0
```

### 3. Run Paper Trading

```python
import asyncio
from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider

async def paper_trade():
    config = StrategyConfig()
    config.paper_trading = True
    
    provider = HyperliquidProvider(testnet=True)
    strategy = WickReversalStrategy(config=config, data_provider=provider)
    
    await strategy.initialize(initial_capital=10000)
    await strategy.run_live(["DOGE-PERP", "SHIB-PERP"])

asyncio.run(paper_trade())
```

## Signal Detection

### Entry Criteria
Signals are generated when a closed candle exhibits an exaggerated wick meeting one or more conditions:

| Condition | Default Threshold | Description |
|-----------|------------------|-------------|
| Wick-to-body ratio | ≥ 2.0 | Wick must be 2x the body size |
| Wick ATR ratio | ≥ 1.5 | Wick must exceed 1.5x ATR |
| VWAP distance | ≥ 1.0 ATR | Wick extreme must be 1 ATR from VWAP |

**Additional Requirements:**
- Minimum wick size: 0.1% of price
- Rejection confirmation: Close must be ≥30% of wick away from extreme

### Signal Types
- **Upper Wick → SHORT**: Price rejected from highs
- **Lower Wick → LONG**: Price rejected from lows

## Entry Rules

| Rule | Description |
|------|-------------|
| Entry timing | After candle close confirms rejection |
| Entry price | Candle close OR retrace to wick midpoint |
| Order type | Limit preferred (configurable) |
| No wick entries | Never enter during the wick formation |

## Exit Rules

### Take Profit Targets (Configurable)
1. **VWAP** (default) - Mean-reversion target
2. **Candle Open** - Prior equilibrium
3. **Wick Midpoint** - Partial reversion
4. **ATR-based** - Fixed ATR multiple

### Stop Loss
- Placed just beyond wick extreme
- Buffer: 0.2 × ATR (configurable)
- Maximum: 2% of entry price

### Additional Exits
- **Trailing stop**: Activates at 70% of target, trails at 0.3 ATR
- **Time exit**: Maximum 20 candles held

## Risk Management

### Position Sizing
```
Position Size = (Account Equity × Risk %) / (Entry - Stop Loss)
```

Default: 0.5% risk per trade

### Exposure Limits
| Parameter | Default | Description |
|-----------|---------|-------------|
| Max positions | 3 | Total simultaneous positions |
| Max per symbol | 1 | Positions per trading pair |
| Max leverage | 3.0x | Maximum effective leverage |
| Cooldown | 5 candles | Wait period after each trade |

### Circuit Breakers
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Daily loss | -2% | Stop trading for the day |
| Max drawdown | -10% | Stop all trading |

## Market Filters

Trading is **disabled** when:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Volume spike | > 3x baseline | Abnormal activity (news, manipulation) |
| ATR expansion | > 2x baseline | Regime change, breakout |
| BTC move | > 3% in 5 candles | Market-wide risk-off |
| Low volume | < $100K USD | Insufficient liquidity |
| Wide spread | > 0.1% | Poor execution conditions |
| Thin orderbook | < $50K depth | Slippage risk |

## Recommended Exchanges

### 1. Hyperliquid (Primary Recommendation)
**Pros:**
- Decentralized perpetual DEX on Arbitrum
- No KYC required
- Very low fees (0.02% maker / 0.05% taker)
- Transparent on-chain order book
- Good altcoin perp liquidity
- API-friendly with WebSocket support

**Cons:**
- Newer platform (less battle-tested)
- Limited fiat on-ramp

### 2. Bybit
**Pros:**
- Wide selection of altcoin perpetuals
- Robust API and WebSocket
- Testnet available
- Higher overall volume

**Cons:**
- Centralized (counterparty risk)
- KYC required for larger accounts

### 3. dYdX v4
**Pros:**
- Fully decentralized (sovereign chain)
- No KYC
- Good for larger positions

**Cons:**
- Lower altcoin selection
- More complex integration

## Target Markets

**Ideal characteristics:**
- Mid-cap altcoins with perpetual futures
- Enough liquidity for execution ($100K+ daily volume)
- Frequent wick patterns (thinner order books)
- Active retail participation (stop hunts common)

**Example symbols:**
- DOGE-PERP
- SHIB-PERP
- PEPE-PERP
- FLOKI-PERP
- BONK-PERP
- WIF-PERP

**Avoid:**
- BTC/ETH (too efficient, wicks = information)
- Very low-cap tokens (execution impossible)
- New listings (unstable behavior)

## Backtesting

### Running a Backtest

```python
from backtest.engine import BacktestEngine
from config.settings import StrategyConfig

config = StrategyConfig()
engine = BacktestEngine(config)

# historical_data: Dict[symbol, List[Candle]]
metrics = engine.run(historical_data, btc_data)

print(f"Net Profit: ${metrics.net_profit:.2f}")
print(f"Win Rate: {metrics.win_rate*100:.1f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown_pct*100:.1f}%")
```

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Win Rate | > 55% | Percentage of profitable trades |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| Sharpe Ratio | > 1.5 | Risk-adjusted returns |
| Max Drawdown | < 15% | Peak-to-trough decline |
| Expectancy | > 0 | Average profit per trade |

### Monte Carlo Analysis

```python
from backtest.engine import run_monte_carlo_analysis

results = run_monte_carlo_analysis(
    trades=engine.trades,
    initial_capital=10000,
    num_simulations=1000
)

print(f"5th percentile final equity: ${results['final_equity']['percentile_5']:,.2f}")
print(f"95th percentile max drawdown: {results['max_drawdown']['percentile_95']*100:.1f}%")
```

## Live Trading Checklist

### Before Going Live
- [ ] Backtest on 6+ months of historical data
- [ ] Test across different market regimes (bull, bear, sideways)
- [ ] Monte Carlo analysis shows acceptable worst-case scenarios
- [ ] Paper trade for minimum 2 weeks
- [ ] Verify API connectivity and order execution
- [ ] Set up monitoring and alerting
- [ ] Document your risk tolerance and stop-loss rules

### Go-Live Process
1. Start with 25% of intended capital
2. Trade minimum position sizes for first week
3. Scale up gradually if results match backtest
4. Monitor for execution slippage vs. backtest assumptions
5. Review and adjust parameters monthly

## Configuration Reference

See `config/settings.py` for complete parameter documentation:

```python
@dataclass
class StrategyConfig:
    strategy_name: str = "WickReversal_v1"
    timeframe: TimeFrame = TimeFrame.M5
    
    signal: SignalConfig      # Wick detection parameters
    entry: EntryConfig        # Entry execution settings
    exit: ExitConfig          # Exit targets and stops
    risk: RiskConfig          # Position sizing and limits
    filters: FilterConfig     # Market condition filters
    execution: ExecutionConfig # Order execution settings
    backtest: BacktestConfig  # Backtesting parameters
    
    symbols: List[SymbolConfig]  # Trading pairs
```

## Logging

All signals, orders, fills, and positions are logged to:
- Console (human-readable)
- `logs/wick_reversal_YYYYMMDD.log` (file)
- `logs/wick_reversal_YYYYMMDD_structured.jsonl` (JSON for analysis)

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always:
- Trade only with capital you can afford to lose
- Understand the risks of leveraged trading
- Do your own research and testing
- Start with small positions

## License

MIT License - See LICENSE file for details.
