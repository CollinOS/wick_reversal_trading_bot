#!/usr/bin/env python3
"""
Run a backtest using real historical data exported by fetch_hyperliquid_history.py.
"""

import argparse
import json
from datetime import datetime
from typing import Dict, List

from backtest.engine import (
    BacktestEngine,
    run_monte_carlo_analysis,
    analyze_trades_by_dimension,
    print_trade_analysis,
    export_trades_to_csv,
)
from config.settings import StrategyConfig, SymbolConfig, TimeFrame
from core.types import Candle


def load_candles(raw: List[dict]) -> List[Candle]:
    candles: List[Candle] = []
    for item in raw:
        candles.append(
            Candle(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
            )
        )
    candles.sort(key=lambda c: c.timestamp)
    return candles


def map_timeframe(value: str) -> TimeFrame:
    tf_map = {
        "1m": TimeFrame.M1,
        "5m": TimeFrame.M5,
        "15m": TimeFrame.M15,
        "30m": TimeFrame.M30,
        "1h": TimeFrame.H1,
        "4h": TimeFrame.H4,
    }
    return tf_map.get(value, TimeFrame.M5)


def create_real_data_config(symbols: List[str], timeframe: str) -> StrategyConfig:
    config = StrategyConfig(
        strategy_name="WickReversal_RealData",
        timeframe=map_timeframe(timeframe),
    )
    config.paper_trading = False
    config.symbols = [SymbolConfig(symbol=s) for s in symbols]
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest with real historical data")
    parser.add_argument(
        "--input",
        default="historical_data.json",
        help="Path to JSON file created by fetch_hyperliquid_history.py",
    )
    parser.add_argument("--symbols", nargs="*", help="Subset of symbols to include")
    parser.add_argument("--timeframe", default=None, help="Override timeframe (e.g., 5m)")
    parser.add_argument("--no-btc", action="store_true", help="Skip BTC correlation data")
    parser.add_argument("--no-monte", action="store_true", help="Skip Monte Carlo analysis")
    parser.add_argument("--analyze", action="store_true", help="Run detailed trade analysis")
    parser.add_argument("--export-csv", type=str, help="Export trades to CSV file (e.g., trades.csv)")
    return parser.parse_args()


def print_metrics(engine: BacktestEngine, metrics) -> None:
    print("\n" + "-" * 40)
    print("BACKTEST RESULTS")
    print("-" * 40)
    results = metrics.to_dict()
    for key, value in results.items():
        print(f"{key:25s}: {value}")

    print("\n" + "-" * 40)
    print("TRADE ANALYSIS")
    print("-" * 40)

    trades = engine.get_trade_log()
    if not trades:
        print("No trades executed.")
        return

    print(f"Total signals generated: {len(engine.signals)}")
    print(f"Total trades executed: {len(trades)}")

    winning_pnls = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losing_pnls = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    if winning_pnls and losing_pnls:
        avg_win = sum(winning_pnls) / len(winning_pnls)
        avg_loss = abs(sum(losing_pnls) / len(losing_pnls))
        actual_rr = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"\nAvg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f} | Actual R:R = {actual_rr:.2f}")
        required_wr = 1 / (1 + actual_rr) if actual_rr > 0 else 1
        print(f"Required win rate for breakeven: {required_wr*100:.1f}%")
        print(f"Actual win rate: {metrics.win_rate*100:.1f}%")

    exit_reasons: Dict[str, int] = {}
    for trade in trades:
        reason = trade.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print("\nExit reason breakdown:")
    for reason, count in sorted(exit_reasons.items()):
        print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")

    long_trades = [t for t in trades if t.get("side") == "long"]
    short_trades = [t for t in trades if t.get("side") == "short"]
    long_wins = [t for t in long_trades if t.get("net_pnl", 0) > 0]
    short_wins = [t for t in short_trades if t.get("net_pnl", 0) > 0]

    if long_trades:
        print(f"\nLong trades: {len(long_trades)} (Win rate: {len(long_wins)/len(long_trades)*100:.1f}%)")
    else:
        print("\nNo long trades")
    if short_trades:
        print(f"Short trades: {len(short_trades)} (Win rate: {len(short_wins)/len(short_trades)*100:.1f}%)")
    else:
        print("No short trades")


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_symbols = payload.get("symbols", {})
    if not raw_symbols:
        raise ValueError("Input file missing 'symbols' data.")

    symbols = args.symbols if args.symbols else list(raw_symbols.keys())
    missing = [s for s in symbols if s not in raw_symbols]
    if missing:
        raise ValueError(f"Symbols not found in input: {', '.join(missing)}")

    timeframe = args.timeframe or payload.get("meta", {}).get("timeframe", "5m")

    historical_data = {s: load_candles(raw_symbols[s]) for s in symbols}
    btc_data = None
    if not args.no_btc and payload.get("btc"):
        btc_data = load_candles(payload["btc"])

    config = create_real_data_config(symbols, timeframe)

    engine = BacktestEngine(config)
    metrics = engine.run(historical_data, btc_data)
    print_metrics(engine, metrics)

    if not args.no_monte and engine.trades:
        print("\n" + "-" * 40)
        print("MONTE CARLO ANALYSIS (1000 simulations)")
        print("-" * 40)
        mc_results = run_monte_carlo_analysis(
            engine.trades,
            config.backtest.initial_capital,
            num_simulations=1000,
        )
        print(f"Final equity (mean): ${mc_results['final_equity']['mean']:,.2f}")
        print(f"Final equity (5th percentile): ${mc_results['final_equity']['percentile_5']:,.2f}")
        print(f"Final equity (95th percentile): ${mc_results['final_equity']['percentile_95']:,.2f}")
        print(f"Max drawdown (95th percentile): {mc_results['max_drawdown']['percentile_95']*100:.1f}%")

    # Detailed trade analysis
    if args.analyze and engine.trades:
        analysis = analyze_trades_by_dimension(engine.trades)
        print_trade_analysis(analysis)

    # Export trades to CSV
    if args.export_csv and engine.trades:
        export_trades_to_csv(engine.trades, args.export_csv)
        print(f"\nTrades exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
