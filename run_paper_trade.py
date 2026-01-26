#!/usr/bin/env python3
"""
Run paper trading on Hyperliquid testnet.

Usage:
    python run_paper_trade.py --private-key YOUR_PRIVATE_KEY

    Or set environment variable:
    export HL_PRIVATE_KEY=your_private_key
    python run_paper_trade.py

    With auto-updating symbols from live monitor:
    python run_paper_trade.py --watch-symbols active_symbols.json
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Set, Optional

from config.settings import StrategyConfig, SymbolConfig, DEFAULT_CONFIG
from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SymbolWatcher:
    """Watches a JSON file for symbol updates from live_monitor.py"""

    def __init__(self, filepath: str, check_interval: int = 30):
        self.filepath = Path(filepath)
        self.check_interval = check_interval
        self.last_modified: Optional[float] = None
        self.current_symbols: Set[str] = set()
        self.on_symbols_changed: Optional[callable] = None

    def read_symbols(self) -> Set[str]:
        """Read symbols from the JSON file."""
        try:
            if not self.filepath.exists():
                return set()

            with open(self.filepath, 'r') as f:
                data = json.load(f)

            symbols = set(data.get("symbols", []))
            return symbols
        except Exception as e:
            logger.warning(f"Failed to read symbols file: {e}")
            return self.current_symbols

    def check_for_updates(self) -> Optional[Set[str]]:
        """Check if file has been updated and return new symbols if changed."""
        try:
            if not self.filepath.exists():
                return None

            mtime = self.filepath.stat().st_mtime

            if self.last_modified is None:
                self.last_modified = mtime
                self.current_symbols = self.read_symbols()
                return None

            if mtime > self.last_modified:
                self.last_modified = mtime
                new_symbols = self.read_symbols()

                if new_symbols != self.current_symbols:
                    old_symbols = self.current_symbols
                    self.current_symbols = new_symbols

                    added = new_symbols - old_symbols
                    removed = old_symbols - new_symbols

                    if added:
                        logger.info(f"📈 New symbols detected: {added}")
                    if removed:
                        logger.info(f"📉 Symbols removed: {removed}")

                    return new_symbols

            return None
        except Exception as e:
            logger.warning(f"Error checking for symbol updates: {e}")
            return None

    async def watch(self):
        """Continuously watch for symbol changes."""
        logger.info(f"Watching {self.filepath} for symbol updates...")

        while True:
            new_symbols = self.check_for_updates()

            if new_symbols is not None and self.on_symbols_changed:
                await self.on_symbols_changed(new_symbols)

            await asyncio.sleep(self.check_interval)


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


class DynamicTradingManager:
    """Manages trading with dynamic symbol updates."""

    def __init__(
        self,
        config: StrategyConfig,
        data_provider: HyperliquidProvider,
        executor: HyperliquidExecutor,
        initial_capital: float
    ):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.initial_capital = initial_capital

        self.strategy: Optional[WickReversalStrategy] = None
        self.current_symbols: Set[str] = set()
        self.symbol_tasks: dict = {}
        self.is_running = False

    async def start(self, symbols: list):
        """Start trading with initial symbols."""
        self.current_symbols = set(symbols)

        # Create strategy
        self.strategy = WickReversalStrategy(
            config=self.config,
            data_provider=self.data_provider,
            executor=self.executor
        )

        await self.strategy.initialize(initial_capital=self.initial_capital)
        self.is_running = True

        logger.info(f"Started trading: {', '.join(symbols)}")

        # Start streaming for each symbol
        for symbol in symbols:
            await self._start_symbol_stream(symbol)

        # Also stream BTC for correlation
        await self._start_btc_stream()

    async def _start_symbol_stream(self, symbol: str):
        """Start candle stream for a symbol."""
        if symbol in self.symbol_tasks:
            return

        async def stream_symbol():
            try:
                async for candle in self.data_provider.subscribe_candles(
                    symbol, self.config.timeframe.value
                ):
                    if not self.is_running:
                        break
                    if symbol not in self.current_symbols:
                        break

                    orderbook = await self.data_provider.get_orderbook_snapshot(symbol)
                    await self.strategy.process_candle(symbol, candle, orderbook)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error streaming {symbol}: {e}")

        task = asyncio.create_task(stream_symbol())
        self.symbol_tasks[symbol] = task
        logger.info(f"📡 Started stream: {symbol}")

    async def _start_btc_stream(self):
        """Start BTC stream for correlation filter."""
        if "BTC-PERP" in self.symbol_tasks:
            return

        async def stream_btc():
            try:
                async for candle in self.data_provider.subscribe_candles(
                    "BTC-PERP", self.config.timeframe.value
                ):
                    if not self.is_running:
                        break
                    self.strategy.btc_price_history.append(candle.close)
                    if len(self.strategy.btc_price_history) > 100:
                        self.strategy.btc_price_history = self.strategy.btc_price_history[-100:]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error streaming BTC: {e}")

        task = asyncio.create_task(stream_btc())
        self.symbol_tasks["BTC-PERP"] = task

    async def _stop_symbol_stream(self, symbol: str):
        """Stop candle stream for a symbol."""
        if symbol not in self.symbol_tasks:
            return

        task = self.symbol_tasks[symbol]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.symbol_tasks[symbol]
        logger.info(f"🛑 Stopped stream: {symbol}")

    async def update_symbols(self, new_symbols: Set[str]):
        """Update trading symbols dynamically."""
        if new_symbols == self.current_symbols:
            return

        added = new_symbols - self.current_symbols
        removed = self.current_symbols - new_symbols

        # Log the change
        print(f"\n{'='*60}")
        print("🔄 SYMBOL UPDATE DETECTED")
        print(f"{'='*60}")
        if added:
            print(f"   ➕ Adding: {', '.join(added)}")
        if removed:
            print(f"   ➖ Removing: {', '.join(removed)}")
        print(f"   📊 New symbols: {', '.join(new_symbols)}")
        print(f"{'='*60}\n")

        # Stop removed symbols (but don't close positions immediately)
        for symbol in removed:
            await self._stop_symbol_stream(symbol)

        # Update config
        self.config.symbols = [SymbolConfig(symbol=s) for s in new_symbols]

        # Start new symbols
        for symbol in added:
            await self._start_symbol_stream(symbol)

        self.current_symbols = new_symbols

    async def stop(self):
        """Stop all trading."""
        self.is_running = False

        # Cancel all symbol tasks
        for symbol, task in list(self.symbol_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.symbol_tasks.clear()

        if self.strategy:
            await self.strategy.shutdown()

    def get_status(self) -> dict:
        """Get current trading status."""
        if self.strategy:
            return self.strategy.get_status()
        return {}


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

    # Create dynamic trading manager
    manager = DynamicTradingManager(
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
        logger.info("Shutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize connections
        await data_provider.connect()
        await executor.connect()

        # Start trading
        await manager.start(symbols)

        logger.info("Strategy initialized. Starting live trading loop...")
        logger.info("Press Ctrl+C to stop")

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
