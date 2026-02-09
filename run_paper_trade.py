#!/usr/bin/env python3
"""
Run paper trading on Hyperliquid testnet.

Usage:
    python run_paper_trade.py --private-key YOUR_PRIVATE_KEY

    Or set environment variable:
    export HL_PRIVATE_KEY=your_private_key
    python run_paper_trade.py

    With auto-updating symbols from live monitor:
    python run_paper_trade.py --watch-symbols output/active_symbols.json
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from config.settings import StrategyConfig, SymbolConfig, DEFAULT_CONFIG
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor
from trading import PaperTradingManager, SymbolWatcher, quiet_console_logging


# Configure logging - quiet mode for trading (only warnings/errors to console)
# All INFO logs still go to file via StructuredLogger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trade on Hyperliquid testnet")
    parser.add_argument(
        "--private-key",
        default=os.environ.get("HL_PRIVATE_KEY"),
        help="Ethereum private key (or set HL_PRIVATE_KEY env var)"
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use mainnet instead of testnet (CAUTION: real money!)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital for position sizing (default: 1000)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols to trade (e.g., TAO-PERP AAVE-PERP)"
    )
    parser.add_argument(
        "--watch-symbols",
        type=str,
        default=None,
        help="JSON file to watch for symbol updates (from live_monitor.py)"
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=30,
        help="How often to check for symbol updates in seconds (default: 30)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if not args.private_key:
        print("Error: Private key required.")
        print("Either pass --private-key or set HL_PRIVATE_KEY environment variable")
        print("\nTo generate a new wallet:")
        print('  python -c "from eth_account import Account; a = Account.create(); print(f\'Address: {a.address}\\nPrivate Key: {a.key.hex()}\')"')
        print("\nThen get testnet funds from: https://app.hyperliquid-testnet.xyz")
        sys.exit(1)

    testnet = not args.mainnet

    if not testnet:
        print("\n" + "="*60)
        print("WARNING: MAINNET MODE - REAL MONEY AT RISK!")
        print("="*60)
        confirm = input("Type 'YES' to confirm mainnet trading: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    # Load config
    config = DEFAULT_CONFIG

    # Determine initial symbols
    if args.watch_symbols and Path(args.watch_symbols).exists():
        # Load from watch file
        with open(args.watch_symbols, 'r') as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        if not symbols:
            symbols = [s.symbol for s in config.symbols]
        logger.info(f"Loaded symbols from {args.watch_symbols}: {symbols}")
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = [s.symbol for s in config.symbols]

    # Update config with symbols
    config.symbols = [SymbolConfig(symbol=s) for s in symbols]

    print(f"\n{'='*60}")
    print(f"WICK REVERSAL STRATEGY - {'TESTNET' if testnet else 'MAINNET'} PAPER TRADING")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {config.timeframe.value}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Base Position: ${config.symbols[0].base_position_usd}")
    print(f"Max Position: ${config.symbols[0].max_position_usd}")
    print(f"Max Leverage: {config.risk.max_leverage}x")
    if args.watch_symbols:
        print(f"Watching: {args.watch_symbols} (every {args.watch_interval}s)")
    print(f"{'='*60}\n")

    # Initialize data provider
    data_provider = HyperliquidProvider(testnet=testnet)

    # Initialize executor
    executor = HyperliquidExecutor(
        private_key=args.private_key,
        testnet=testnet
    )

    # Set execution config for dynamic leverage
    executor.set_execution_config(config.execution)

    # Create dynamic trading manager
    manager = PaperTradingManager(
        config=config,
        data_provider=data_provider,
        executor=executor,
        initial_capital=args.capital
    )

    # Setup symbol watcher if enabled
    symbol_watcher = None
    if args.watch_symbols:
        symbol_watcher = SymbolWatcher(
            filepath=args.watch_symbols,
            check_interval=args.watch_interval
        )
        symbol_watcher.on_symbols_changed = manager.update_symbols

    # Handle graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load candle cache from file (written by live_monitor)
        print("\nLoading candle cache from live_monitor...")
        if not manager.load_candle_cache():
            print("\nERROR: Cannot start without fresh candle data!")
            print("Please ensure live_monitor.py is running first.")
            sys.exit(1)

        # Initialize connections with small delays to avoid rate limits
        print("\nConnecting to Hyperliquid...")
        await data_provider.connect()
        await asyncio.sleep(1)  # Small delay between API clients
        await executor.connect()

        # Start trading
        await manager.start(symbols)

        # Now quiet console logging - file logging continues
        quiet_console_logging()

        print("Strategy initialized. Press Ctrl+C to stop\n")

        # Create tasks
        tasks = [asyncio.create_task(shutdown_event.wait())]

        # Add symbol watcher task if enabled
        if symbol_watcher:
            tasks.append(asyncio.create_task(symbol_watcher.watch()))

        # Wait for shutdown
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.exception(f"Error during trading: {e}")
    finally:
        await manager.stop()
        await data_provider.disconnect()
        await executor.disconnect()

        # Print final summary
        status = manager.get_status()
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Candles Processed: {status.get('candle_count', 0)}")
        print(f"Active Positions: {status.get('active_positions', 0)}")
        if status.get('portfolio'):
            portfolio = status['portfolio']
            print(f"Total Trades: {portfolio.get('total_trades', 0)}")
            print(f"Win Rate: {portfolio.get('win_rate', 0)*100:.1f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
