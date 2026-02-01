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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, Optional, Dict

from config.settings import StrategyConfig, SymbolConfig, DEFAULT_CONFIG
from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor
from core.types import Candle


# Configure logging - quiet mode for trading (only warnings/errors to console)
# All INFO logs still go to file via StructuredLogger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quiet_console_logging():
    """
    Silence all console handlers so only banners (print statements) show.
    File logging continues for analysis.
    """
    for name in logging.root.manager.loggerDict:
        log = logging.getLogger(name)
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                handler.setLevel(logging.ERROR)

    # Also silence root logger's console handlers
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)


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

    # Maximum age (in minutes) for candle data before symbol is considered "not ready"
    MAX_DATA_AGE_MINUTES = 30  # 30 minutes - cache should be fresh from live_monitor

    # Path to cache file written by live_monitor
    CACHE_FILE = Path("data/candle_cache.json")

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

        # Cached candles loaded from file (written by live_monitor)
        self.cached_candles: Dict[str, list] = {}
        self.cache_loaded_at: Optional[datetime] = None

        self.strategy: Optional[WickReversalStrategy] = None
        self.current_symbols: Set[str] = set()
        self.symbol_tasks: dict = {}
        self.is_running = False

        # Track symbols that don't have fresh enough data
        self.symbols_not_ready: Set[str] = set()

    def load_candle_cache(self) -> bool:
        """Load candles from cache file written by live_monitor.

        Returns True if cache is fresh and valid, False otherwise.
        """
        if not self.CACHE_FILE.exists():
            print(f"  WARNING: Cache file not found: {self.CACHE_FILE}")
            print(f"  Make sure live_monitor.py is running!")
            return False

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            # Check cache age
            saved_at = datetime.fromisoformat(data.get("saved_at", "2000-01-01"))
            cache_age_minutes = (datetime.utcnow() - saved_at).total_seconds() / 60

            if cache_age_minutes > self.MAX_DATA_AGE_MINUTES:
                print(f"  WARNING: Cache is {cache_age_minutes:.0f} minutes old (max: {self.MAX_DATA_AGE_MINUTES})")
                print(f"  Make sure live_monitor.py is running!")
                return False

            # Load candles
            candle_data = data.get("candles", {})
            total_candles = 0

            for symbol, candle_list in candle_data.items():
                candles = []
                for c in candle_list:
                    candles.append(Candle(
                        timestamp=datetime.fromisoformat(c["timestamp"]),
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"]
                    ))
                self.cached_candles[symbol] = candles
                total_candles += len(candles)

            self.cache_loaded_at = datetime.utcnow()
            print(f"  Loaded {total_candles} candles for {len(self.cached_candles)} symbols from cache")
            print(f"  Cache age: {cache_age_minutes:.1f} minutes")
            return True

        except Exception as e:
            print(f"  ERROR loading cache: {e}")
            return False

    def get_cached_candles(self, symbol: str, count: int = 100) -> list:
        """Get candles for a symbol from the loaded cache."""
        if symbol not in self.cached_candles:
            return []
        return self.cached_candles[symbol][-count:]

    def reload_symbol_from_cache(self, symbol: str) -> bool:
        """Reload a single symbol's candles from the cache file.

        Used when a new symbol is added - live_monitor may have fresh data for it.
        """
        if not self.CACHE_FILE.exists():
            return False

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            candle_data = data.get("candles", {})
            if symbol not in candle_data:
                return False

            candles = []
            for c in candle_data[symbol]:
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(c["timestamp"]),
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"]
                ))

            self.cached_candles[symbol] = candles
            return len(candles) > 0

        except Exception as e:
            logger.warning(f"Failed to reload {symbol} from cache: {e}")
            return False

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

        # Pre-load historical candles to seed ATR and other indicators
        data_ready = await self._preload_historical_data(symbol)

        # If data is not ready, add to ignored positions (will be removed when fresh data arrives)
        if not data_ready:
            self.strategy.ignored_positions.add(symbol)

        # Track candles received to know when symbol becomes ready
        candles_received = [0]  # Use list to allow modification in nested function

        async def stream_symbol():
            try:
                async for candle in self.data_provider.subscribe_candles(
                    symbol, self.config.timeframe.value
                ):
                    if not self.is_running:
                        break
                    if symbol not in self.current_symbols:
                        break

                    candles_received[0] += 1

                    # Check if symbol was not ready but now has fresh data
                    if symbol in self.symbols_not_ready:
                        # After receiving enough fresh candles for ATR (15+), enable trading
                        if candles_received[0] >= 15:
                            self.symbols_not_ready.discard(symbol)
                            self.strategy.ignored_positions.discard(symbol)
                            print(f"\n  {symbol}: Fresh data received ({candles_received[0]} candles) - TRADING ENABLED\n")

                    orderbook = await self.data_provider.get_orderbook_snapshot(symbol)
                    await self.strategy.process_candle(symbol, candle, orderbook)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error streaming {symbol}: {e}")

        task = asyncio.create_task(stream_symbol())
        self.symbol_tasks[symbol] = task
        logger.debug(f"Started stream: {symbol}")

    async def _preload_historical_data(self, symbol: str, num_candles: int = 100):
        """Pre-load historical candles to seed indicators (ATR, VWAP, etc.).

        Uses candles from cache file (written by live_monitor).
        Returns True if data is fresh enough for trading, False otherwise.
        """
        try:
            # Get candles from loaded cache
            candles = self.get_cached_candles(symbol, num_candles)

            # If not in cache, try reloading from file (live_monitor may have added it)
            if not candles:
                if self.reload_symbol_from_cache(symbol):
                    candles = self.get_cached_candles(symbol, num_candles)

            if candles:
                newest = candles[-1]
                data_age_minutes = (datetime.utcnow() - newest.timestamp).total_seconds() / 60

                # Add candles to the data aggregator
                for candle in candles:
                    self.strategy.data_aggregator.add_candle(symbol, candle)

                if data_age_minutes > self.MAX_DATA_AGE_MINUTES:
                    print(f"  {symbol}: {len(candles)} candles "
                          f"(⚠️ {data_age_minutes:.0f}min old - TRADING DISABLED until fresh data)")
                    self.symbols_not_ready.add(symbol)
                    return False
                elif data_age_minutes > 10:
                    print(f"  {symbol}: {len(candles)} candles "
                          f"(newest is {data_age_minutes:.0f}min old)")
                else:
                    print(f"  {symbol}: {len(candles)} candles (current)")
                return True
            else:
                print(f"  {symbol}: No cached data - TRADING DISABLED until fresh data")
                self.symbols_not_ready.add(symbol)
                return False

        except Exception as e:
            logger.warning(f"Failed to preload historical data for {symbol}: {e}")
            self.symbols_not_ready.add(symbol)
            return False

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

    # Set execution config for dynamic leverage
    executor.set_execution_config(config.execution)

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
