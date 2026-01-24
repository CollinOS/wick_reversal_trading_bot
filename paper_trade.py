#!/usr/bin/env python3
"""
Paper trading script for Hyperliquid.

Connects to real market data but simulates order execution.
Run for 2+ weeks before going live.

Usage:
    python paper_trade.py
    python paper_trade.py --symbols DOGE-PERP PEPE-PERP
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime

from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider
from execution.orders import SimulatedExecutor
from config.settings import StrategyConfig


# Global for clean shutdown
strategy = None


def handle_shutdown(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutting down paper trading...")
    if strategy:
        asyncio.create_task(strategy.shutdown())
    sys.exit(0)


async def run_paper_trading(symbols: list[str], initial_capital: float):
    global strategy

    config = StrategyConfig()
    config.paper_trading = True

    # Connect to real Hyperliquid market data
    provider = HyperliquidProvider(testnet=False)

    # Simulated executor - no real orders
    def get_price(symbol):
        return 0  # Price comes from live candles
    executor = SimulatedExecutor(config.execution, get_price)

    strategy = WickReversalStrategy(config, provider, executor)

    print("=" * 50)
    print("PAPER TRADING - Wick Reversal Strategy")
    print("=" * 50)
    print(f"Capital: ${initial_capital:,.2f}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {config.timeframe.value}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    print("Connecting to Hyperliquid...")

    await strategy.initialize(initial_capital=initial_capital)

    print("Connected. Watching for signals...")
    print("Press Ctrl+C to stop\n")

    try:
        await strategy.run_live(symbols)
    except asyncio.CancelledError:
        pass
    finally:
        await strategy.shutdown()
        print_final_summary()


def print_final_summary():
    """Print final paper trading summary."""
    if not strategy:
        return

    status = strategy.get_status()
    print("\n" + "=" * 50)
    print("PAPER TRADING SESSION ENDED")
    print("=" * 50)

    portfolio = status.get("portfolio", {})
    perf = status.get("performance", {})

    print(f"Candles processed: {status.get('candle_count', 0)}")
    print(f"Signals generated: {perf.get('counters', {}).get('signals_generated', 0)}")
    print(f"Trades executed: {perf.get('counters', {}).get('trades_executed', 0)}")

    if portfolio:
        print(f"Final equity: ${portfolio.get('equity', 0):,.2f}")
        print(f"Win rate: {portfolio.get('win_rate', 0)*100:.1f}%")

    print("\nCheck logs/ directory for detailed trade journal.")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper trade on Hyperliquid")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["DOGE-PERP"],
        help="Symbols to trade (default: DOGE-PERP)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital for simulation (default: 1000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Register shutdown handler
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    asyncio.run(run_paper_trading(args.symbols, args.capital))


if __name__ == "__main__":
    main()
