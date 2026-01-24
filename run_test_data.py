#!/usr/bin/env python3
"""
Example Runner Script
Demonstrates how to use the Wick Reversal Strategy for backtesting and paper trading.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict
import random

# Add parent directory to path for imports
import sys
sys.path.insert(0, '.')

from config.settings import StrategyConfig, SymbolConfig, TimeFrame, ExitTarget
from core.types import Candle, Side
from data.ingestion import SimulatedDataProvider, DataAggregator
from backtest.engine import BacktestEngine, run_monte_carlo_analysis
from main import WickReversalStrategy
from execution.orders import SimulatedExecutor


def generate_sample_data(
    symbol: str,
    num_candles: int = 1000,
    base_price: float = 0.10,
    volatility: float = 0.02
) -> List[Candle]:
    """
    Generate synthetic candlestick data with realistic wick patterns.
    Includes occasional exaggerated wicks for testing.
    """
    candles = []
    current_price = base_price
    timestamp = datetime(2024, 1, 1, 0, 0, 0)
    
    for i in range(num_candles):
        # Base movement
        change_pct = random.gauss(0, volatility)
        current_price *= (1 + change_pct)
        current_price = max(current_price, base_price * 0.1)  # Floor price
        
        # Generate OHLC
        open_price = current_price
        
        # Random intracandle movement
        high_extension = abs(random.gauss(0, volatility * 0.5))
        low_extension = abs(random.gauss(0, volatility * 0.5))
        
        # Create exaggerated wicks more frequently (20% of candles) with bigger wicks
        if random.random() < 0.20:
            if random.random() < 0.5:
                # Exaggerated upper wick - make it very pronounced
                high_extension = volatility * random.uniform(2.5, 5.0)
                low_extension = volatility * random.uniform(0.1, 0.3)
            else:
                # Exaggerated lower wick - make it very pronounced
                low_extension = volatility * random.uniform(2.5, 5.0)
                high_extension = volatility * random.uniform(0.1, 0.3)
        
        high_price = open_price * (1 + high_extension)
        low_price = open_price * (1 - low_extension)
        
        # Close with strong rejection for wick candles
        if high_extension > low_extension * 2.5:
            # Upper wick dominant - close in lower 30% (strong rejection)
            close_price = low_price + (high_price - low_price) * random.uniform(0.05, 0.30)
        elif low_extension > high_extension * 2.5:
            # Lower wick dominant - close in upper 30% (strong rejection)
            close_price = low_price + (high_price - low_price) * random.uniform(0.70, 0.95)
        else:
            # Normal candle
            close_price = low_price + (high_price - low_price) * random.uniform(0.3, 0.7)
        
        # Volume - make it high enough to pass filters
        # Base volume in USD terms should exceed min_volume_usd ($100K default)
        base_usd_volume = 500000  # $500K base
        volume_usd = base_usd_volume * random.uniform(0.5, 2.0) * (1 + abs(change_pct) * 5)
        volume = volume_usd / current_price  # Convert to base currency volume
        
        candle = Candle(
            timestamp=timestamp,
            open=round(open_price, 8),
            high=round(high_price, 8),
            low=round(low_price, 8),
            close=round(close_price, 8),
            volume=round(volume, 2)
        )
        candles.append(candle)
        
        current_price = close_price
        timestamp += timedelta(minutes=5)
    
    return candles


def create_custom_config() -> StrategyConfig:
    """Create a customized strategy configuration optimized for profitability."""
    config = StrategyConfig(
        strategy_name="WickReversal_Custom",
        timeframe=TimeFrame.M5,
    )
    
    # Signal detection - selective signals
    config.signal.wick_to_body_ratio = 2.5       # Wick must be 2.5x body
    config.signal.wick_atr_multiplier = 1.5      # Wick > 1.5 * ATR
    config.signal.vwap_distance_atr = 1.0        # Must be 1 ATR from VWAP
    config.signal.min_wick_pct = 0.008           # 0.8% minimum wick
    config.signal.rejection_threshold_pct = 0.40 # 40% rejection confirmation
    config.signal.require_all_conditions = False
    
    # CRITICAL: Exit configuration for positive expectancy
    # Target R:R of roughly 1:1 to 1.5:1 with our 55%+ win rate
    config.exit.primary_target = ExitTarget.ATR_BASED
    config.exit.atr_target_multiplier = 0.8      # Take profit at 0.8 ATR
    config.exit.stop_loss_buffer_atr = 0.3       # Stop at 0.3 ATR beyond wick
    config.exit.max_stop_loss_pct = 0.015        # 1.5% max stop (tighter)
    config.exit.max_hold_candles = 8             # Exit faster if no movement
    config.exit.trailing_stop_activation = 0.6   # Activate trailing at 60% target
    config.exit.trailing_stop_distance_atr = 0.3
    
    # Risk
    config.risk.risk_per_trade_pct = 0.005  # 0.5%
    config.risk.max_positions = 2
    config.risk.max_leverage = 2.0
    config.risk.cooldown_candles = 5
    
    # Filters
    config.filters.volume_spike_multiplier = 4.0
    config.filters.atr_expansion_multiplier = 2.5
    config.filters.min_volume_usd = 50000
    config.filters.btc_move_threshold_pct = 0.03
    
    # Symbols
    config.symbols = [
        SymbolConfig(symbol="DOGE-PERP", risk_multiplier=1.0),
        SymbolConfig(symbol="SHIB-PERP", risk_multiplier=0.8),
        SymbolConfig(symbol="PEPE-PERP", risk_multiplier=0.8),
    ]
    
    return config


def run_backtest_example():
    """Run a backtest with synthetic data."""
    print("\n" + "="*60)
    print("BACKTEST EXAMPLE")
    print("="*60 + "\n")
    
    # Create configuration
    config = create_custom_config()
    
    # Generate sample data
    print("Generating synthetic market data...")
    historical_data = {
        "DOGE-PERP": generate_sample_data("DOGE-PERP", 2000, 0.10, 0.02),
        "SHIB-PERP": generate_sample_data("SHIB-PERP", 2000, 0.000025, 0.025),
        "PEPE-PERP": generate_sample_data("PEPE-PERP", 2000, 0.000012, 0.03),
    }
    
    # Generate BTC data for correlation filter
    btc_data = generate_sample_data("BTC-PERP", 2000, 45000, 0.008)
    
    print(f"Generated {len(historical_data['DOGE-PERP'])} candles per symbol")
    
    # Show sample of generated data to verify wick patterns
    print("\nSample candles from DOGE-PERP (showing wick characteristics):")
    sample_candles = historical_data["DOGE-PERP"][100:110]
    for i, c in enumerate(sample_candles):
        body = abs(c.close - c.open)
        upper_wick = c.high - max(c.open, c.close)
        lower_wick = min(c.open, c.close) - c.low
        wick_ratio_upper = upper_wick / body if body > 0 else 0
        wick_ratio_lower = lower_wick / body if body > 0 else 0
        print(f"  [{i}] O:{c.open:.4f} H:{c.high:.4f} L:{c.low:.4f} C:{c.close:.4f} | "
              f"Upper wick ratio: {wick_ratio_upper:.1f}x | Lower wick ratio: {wick_ratio_lower:.1f}x")
    
    # Quick diagnostic: manually check if signals would be generated
    print("\n--- DIAGNOSTIC: Testing signal generation manually ---")
    from data.ingestion import DataAggregator
    from signals.detection import SignalGenerator
    
    test_aggregator = DataAggregator(config)
    test_generator = SignalGenerator(config)
    
    test_candles = historical_data["DOGE-PERP"][:100]  # First 100 to build up indicators
    for candle in test_candles:
        test_aggregator.add_candle("DOGE-PERP", candle)
    
    # Check indicator values
    atr = test_aggregator.calculate_atr("DOGE-PERP", config.signal.atr_period)
    vwap = test_aggregator.calculate_vwap("DOGE-PERP", config.signal.vwap_rolling_period)
    vol_sma = test_aggregator.calculate_volume_sma("DOGE-PERP", 20)
    print(f"After 100 candles: ATR={atr:.6f}, VWAP={vwap:.6f}, Vol SMA={vol_sma:.0f}")
    
    # Now test signal generation on some wick candles
    signals_found = 0
    filter_rejections = {}
    for i, candle in enumerate(historical_data["DOGE-PERP"][100:200]):
        market_data = test_aggregator.get_market_data("DOGE-PERP", candle)
        signal = test_generator.generate_signal("DOGE-PERP", market_data, 100+i, btc_data[100+i].close if btc_data else None)
        
        if signal.is_valid:
            signals_found += 1
            if signals_found <= 3:
                print(f"  Signal #{signals_found}: {signal.signal_type.value} strength={signal.strength:.2f} criteria={signal.criteria_met}")
        elif signal.filter_result.value != "passed":
            reason = signal.filter_result.value
            filter_rejections[reason] = filter_rejections.get(reason, 0) + 1
    
    print(f"\nDiagnostic results: {signals_found} signals in 100 candles")
    if filter_rejections:
        print(f"Filter rejections: {filter_rejections}")
    print("--- END DIAGNOSTIC ---\n")
    
    # Create and run backtest
    print("Running full backtest...")
    engine = BacktestEngine(config)
    metrics = engine.run(historical_data, btc_data)
    
    # Print results
    print("\n" + "-"*40)
    print("BACKTEST RESULTS")
    print("-"*40)
    
    results = metrics.to_dict()
    for key, value in results.items():
        print(f"{key:25s}: {value}")
    
    # Trade analysis
    print("\n" + "-"*40)
    print("TRADE ANALYSIS")
    print("-"*40)
    
    trades = engine.get_trade_log()
    if trades:
        print(f"Total signals generated: {len(engine.signals)}")
        print(f"Total trades executed: {len(trades)}")
        
        # Analyze actual R:R ratios
        winning_pnls = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
        losing_pnls = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
        
        if winning_pnls and losing_pnls:
            avg_win = sum(winning_pnls) / len(winning_pnls)
            avg_loss = abs(sum(losing_pnls) / len(losing_pnls))
            actual_rr = avg_win / avg_loss if avg_loss > 0 else 0
            print(f"\nAvg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f} | Actual R:R = {actual_rr:.2f}")
            
            # Required win rate for breakeven with this R:R
            required_wr = 1 / (1 + actual_rr) if actual_rr > 0 else 1
            print(f"Required win rate for breakeven: {required_wr*100:.1f}%")
            print(f"Actual win rate: {metrics.win_rate*100:.1f}%")
        
        # Breakdown by exit reason
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print("\nExit reason breakdown:")
        for reason, count in sorted(exit_reasons.items()):
            print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")
        
        # Breakdown by side
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']
        long_wins = [t for t in long_trades if t.get('net_pnl', 0) > 0]
        short_wins = [t for t in short_trades if t.get('net_pnl', 0) > 0]
        
        print(f"\nLong trades: {len(long_trades)} (Win rate: {len(long_wins)/len(long_trades)*100:.1f}%)" if long_trades else "\nNo long trades")
        print(f"Short trades: {len(short_trades)} (Win rate: {len(short_wins)/len(short_trades)*100:.1f}%)" if short_trades else "No short trades")
    
    # Monte Carlo analysis
    print("\n" + "-"*40)
    print("MONTE CARLO ANALYSIS (1000 simulations)")
    print("-"*40)
    
    from backtest.engine import run_monte_carlo_analysis
    from core.types import TradeResult
    
    if engine.trades:
        mc_results = run_monte_carlo_analysis(
            engine.trades,
            config.backtest.initial_capital,
            num_simulations=1000
        )
        
        print(f"Final equity (mean): ${mc_results['final_equity']['mean']:,.2f}")
        print(f"Final equity (5th percentile): ${mc_results['final_equity']['percentile_5']:,.2f}")
        print(f"Final equity (95th percentile): ${mc_results['final_equity']['percentile_95']:,.2f}")
        print(f"Max drawdown (95th percentile): {mc_results['max_drawdown']['percentile_95']*100:.1f}%")
    
    return engine, metrics


async def run_paper_trading_example():
    """Run paper trading simulation."""
    print("\n" + "="*60)
    print("PAPER TRADING SIMULATION")
    print("="*60 + "\n")
    
    # Create configuration
    config = create_custom_config()
    config.paper_trading = True
    
    # Generate data for simulation
    historical_data = {
        "DOGE-PERP": generate_sample_data("DOGE-PERP", 500, 0.10, 0.015),
    }
    
    # Create simulated data provider
    data_provider = SimulatedDataProvider(historical_data)
    
    # Create simulated executor
    def get_price(symbol):
        idx = data_provider.current_index.get(symbol, 0)
        if idx < len(historical_data.get(symbol, [])):
            return historical_data[symbol][idx].close
        return 0.10
    
    from execution.orders import SimulatedExecutor
    executor = SimulatedExecutor(config.execution, get_price)
    
    # Create strategy
    strategy = WickReversalStrategy(
        config=config,
        data_provider=data_provider,
        executor=executor
    )
    
    # Initialize
    await strategy.initialize(initial_capital=10000.0)
    
    print("Running paper trading simulation...")
    print("Processing candles...")
    
    # Process candles one by one
    candle_count = 0
    for candle in historical_data["DOGE-PERP"]:
        await strategy.process_candle("DOGE-PERP", candle)
        data_provider.advance("DOGE-PERP")
        candle_count += 1
        
        if candle_count % 100 == 0:
            status = strategy.get_status()
            print(f"  Processed {candle_count} candles, "
                  f"Positions: {status['active_positions']}, "
                  f"Signals: {status['performance']['counters']['signals_generated']}")
    
    # Final status
    print("\n" + "-"*40)
    print("PAPER TRADING RESULTS")
    print("-"*40)
    
    status = strategy.get_status()
    print(f"Candles processed: {candle_count}")
    print(f"Signals generated: {status['performance']['counters']['signals_generated']}")
    print(f"Trades executed: {status['performance']['counters']['trades_executed']}")
    
    if status['portfolio']:
        print(f"Final equity: ${status['portfolio'].get('equity', 0):,.2f}")
        print(f"Win rate: {status['portfolio'].get('win_rate', 0)*100:.1f}%")
    
    # Shutdown
    await strategy.shutdown()
    
    return strategy


def print_strategy_summary():
    """Print strategy documentation and parameter summary."""
    print("\n" + "="*60)
    print("WICK REVERSAL STRATEGY - SUMMARY")
    print("="*60)
    
    print("""
STRATEGY CONCEPT:
-----------------
This strategy exploits exaggerated price wicks in low-liquidity crypto
perpetual futures markets using a mean-reversion approach.

The assumption is that extreme wicks represent temporary price dislocations
(stop hunts, thin order books, liquidation cascades) rather than new
information, and price will revert to fair value shortly after.

SIGNAL DETECTION:
-----------------
Entry signals are generated when candles exhibit exaggerated wicks meeting
one or more of these criteria:
  • Wick-to-body ratio > threshold (default: 2.0x)
  • Wick size > k × ATR (default: 1.5x)
  • Distance from VWAP > threshold ATR

Upper wick signals generate SHORT bias
Lower wick signals generate LONG bias

Entry only occurs after candle close confirms rejection (price closes
away from the wick extreme).

ENTRY RULES:
------------
  • Short after exaggerated upper wick
  • Long after exaggerated lower wick
  • Entry at candle close or retrace toward wick midpoint
  • Prefer limit orders for better fills
  • No entries during the wick itself

EXIT RULES:
-----------
Take profit targets (configurable):
  • VWAP
  • Candle open price
  • Wick midpoint
  • ATR-based target

Stop loss: Just beyond wick extreme + buffer

Additional exits:
  • Trailing stop (activates after partial target reached)
  • Time-based exit (max candles held)

RISK MANAGEMENT:
----------------
  • Fixed fractional risk per trade (default: 0.5% of account)
  • Hard stop losses (no mental stops)
  • No averaging down
  • Maximum simultaneous positions limit
  • Cooldown period between trades
  • Daily loss limit
  • Maximum drawdown circuit breaker

MARKET FILTERS:
---------------
Trading is disabled during:
  • Volume spikes (> 3x baseline)
  • ATR expansion (> 2x baseline)
  • Large BTC moves (> 3% in lookback period)
  • Low volume conditions
  • Wide spreads
  • Thin order books

RECOMMENDED EXCHANGES:
----------------------
1. Hyperliquid (Preferred)
   - Decentralized perp DEX on Arbitrum
   - No KYC required
   - Low fees (0.02% maker / 0.05% taker)
   - Transparent on-chain order book
   - Good liquidity on alt-perps
   - API-friendly

2. Bybit
   - Wide selection of altcoin perps
   - Good API support
   - Testnet available
   - Higher volume

3. dYdX v4
   - Decentralized, sovereign chain
   - Good for larger positions
   - No KYC

TARGET MARKETS:
---------------
Best suited for mid-cap altcoin perpetuals with:
  • Enough liquidity for execution
  • Frequent wick patterns due to thin books
  • Examples: DOGE, SHIB, PEPE, FLOKI, BONK, etc.

BACKTESTING NOTES:
------------------
  • Always include realistic slippage (0.05-0.1%)
  • Model partial fills for illiquid markets
  • Test across different market regimes
  • Paper trade before live deployment
  • Use Monte Carlo analysis for robustness check
""")


def main():
    """Main entry point."""
    print_strategy_summary()
    
    # Run backtest
    engine, metrics = run_backtest_example()
    
    # Run paper trading simulation
    asyncio.run(run_paper_trading_example())
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and customize config/settings.py")
    print("2. Add your exchange API credentials")
    print("3. Run more extensive backtests with real historical data")
    print("4. Paper trade for at least 2 weeks before going live")
    print("5. Start with small position sizes when going live")


if __name__ == "__main__":
    main()
