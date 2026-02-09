#!/usr/bin/env python3
"""
Run LIVE trading on Hyperliquid MAINNET.

WARNING: This uses REAL money. Make sure you understand the risks.

Usage:
    python run_live.py --private-key YOUR_PRIVATE_KEY --capital 200

    Or set environment variable:
    export HL_PRIVATE_KEY=your_private_key
    python run_live.py --capital 200
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
from trading import LiveTradingManager, SymbolWatcher, quiet_console_logging


# Configure logging - quiet mode for live trading (only warnings/errors to console)
# All INFO logs still go to file via StructuredLogger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIVE trading on Hyperliquid MAINNET")
    parser.add_argument(
        "--private-key",
        default=os.environ.get("HL_PRIVATE_KEY"),
        help="Ethereum private key (or set HL_PRIVATE_KEY env var)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=200.0,
        help="Trading capital in USD (default: 200)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to trade (e.g., TAO-PERP AAVE-PERP)"
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
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip the confirmation prompt (use with caution!)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if not args.private_key:
        print("Error: Private key required.")
        print("Either pass --private-key or set HL_PRIVATE_KEY environment variable")
        sys.exit(1)

    # Load config first so we can show actual values
    config = DEFAULT_CONFIG

    # Safety confirmation
    print("\n" + "="*70)
    print(" WARNING: LIVE TRADING ON MAINNET - REAL MONEY AT RISK!")
    print("="*70)
    print(f"\n  Capital: ${args.capital:,.2f}")
    print(f"  Base position: ${config.symbols[0].base_position_usd}")
    print(f"  Max position: ${config.symbols[0].max_position_usd}")
    print(f"  Symbols: {args.symbols or 'from watch file'}")
    print("\n" + "="*70)

    if not args.skip_confirmation:
        confirm = input("\nType 'YES' to confirm LIVE trading: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    # Determine initial symbols
    if args.watch_symbols and Path(args.watch_symbols).exists():
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
    print(f"WICK REVERSAL STRATEGY - LIVE MAINNET TRADING")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {config.timeframe.value}")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Base Position: ${config.symbols[0].base_position_usd}")
    print(f"Max Position: ${config.symbols[0].max_position_usd}")
    if config.execution.dynamic_leverage_enabled:
        print(f"Dynamic Leverage: {config.execution.min_leverage}x-{config.execution.max_leverage}x (based on confidence)")
    else:
        print(f"Leverage: Using Hyperliquid defaults")
    if args.watch_symbols:
        print(f"Watching: {args.watch_symbols} (every {args.watch_interval}s)")
    print(f"{'='*60}\n")

    # Initialize data provider (uses mainnet for data)
    data_provider = HyperliquidProvider(testnet=False, use_mainnet_data=True)

    # Initialize executor (MAINNET - real money!)
    executor = HyperliquidExecutor(
        private_key=args.private_key,
        testnet=False  # MAINNET
    )

    # Set execution config for dynamic leverage
    executor.set_execution_config(config.execution)

    # Create trading manager
    manager = LiveTradingManager(
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

        # Verify connection and authentication before trading
        print("\nVerifying exchange connection...")
        if not await executor.verify_connection():
            print("ERROR: Failed to verify exchange connection!")
            print("Please check your private key and try again.")
            sys.exit(1)

        # Check for any existing positions
        existing_positions = await executor.get_all_positions()
        if existing_positions:
            print(f"\nWARNING: Found {len(existing_positions)} existing position(s):")
            for pos in existing_positions:
                side = "LONG" if pos["size"] > 0 else "SHORT"
                print(f"  {pos['symbol']}: {side} {abs(pos['size']):.4f} @ ${pos['entry_price']:.2f} "
                      f"(PnL: ${pos['unrealized_pnl']:.2f})")
            print()

        # Start trading
        await manager.start(symbols)

        # Now quiet console logging - file logging continues
        quiet_console_logging()

        print(f"\n{'='*60}")
        print("LIVE TRADING ACTIVE")
        print(f"{'='*60}")
        print(f"Wallet: {executor.wallet_address}")
        print("Press Ctrl+C to stop\n")

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
        print(f"Session Start: {status.get('session_start', 'N/A')}")
        print(f"Candles Processed: {status.get('candle_count', 0)}")
        print(f"Active Positions (bot): {status.get('active_positions', 0)}")
        if status.get('ignored_positions'):
            print(f"Ignored Positions (manual): {', '.join(status['ignored_positions'])}")
        print(f"{'='*60}")
        print("BOT TRADING RESULTS (this session only):")
        print(f"  Trades Executed: {status.get('session_trades', 0)}")
        print(f"  Win Rate: {status.get('session_win_rate', 0)*100:.1f}%")
        print(f"  Session P&L: ${status.get('session_pnl', 0):+.2f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
