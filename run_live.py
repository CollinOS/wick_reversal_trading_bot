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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, Optional, Dict

from config.settings import StrategyConfig, SymbolConfig, DEFAULT_CONFIG
from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor
from core.types import Candle


# Configure logging - quiet mode for live trading (only warnings/errors to console)
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
                        logger.info(f"New symbols detected: {added}")
                    if removed:
                        logger.info(f"Symbols removed: {removed}")

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


class LiveTradingManager:
    """Manages live trading with dynamic symbol updates."""

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

        # Track pre-existing positions to ignore (manual trades)
        self.pre_existing_positions: Set[str] = set()

        # Track symbols that don't have fresh enough data
        self.symbols_not_ready: Set[str] = set()

        # Session P&L tracking
        self.session_trades: list = []
        self.session_pnl: float = 0.0
        self.session_start_time: Optional[datetime] = None

        # Background tasks
        self._cache_refresh_task = None

    def load_candle_cache(self, max_retries: int = 3) -> bool:
        """Load candles from cache file written by live_monitor.

        Returns True if cache is fresh and valid, False otherwise.
        Retries on JSON errors (may occur if live_monitor is mid-write).
        """
        if not self.CACHE_FILE.exists():
            print(f"  WARNING: Cache file not found: {self.CACHE_FILE}")
            print(f"  Make sure live_monitor.py is running!")
            return False

        for attempt in range(max_retries):
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

            except json.JSONDecodeError as e:
                # Cache file may be mid-write, wait and retry
                if attempt < max_retries - 1:
                    print(f"  Cache file corrupted (attempt {attempt + 1}/{max_retries}), retrying in 2s...")
                    import time
                    time.sleep(2)
                else:
                    print(f"  ERROR: Cache file corrupted after {max_retries} attempts: {e}")
                    print(f"  Try restarting live_monitor.py to regenerate the cache.")
                    return False

            except Exception as e:
                print(f"  ERROR loading cache: {e}")
                return False

        return False

    def get_cached_candles(self, symbol: str, count: int = 100) -> list:
        """Get candles for a symbol from the loaded cache."""
        if symbol not in self.cached_candles:
            return []
        return self.cached_candles[symbol][-count:]

    def _refresh_cache_from_file(self) -> bool:
        """Silently refresh all cached candles from the cache file.

        Called periodically to pick up fresh data from live_monitor.
        """
        if not self.CACHE_FILE.exists():
            return False

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            candle_data = data.get("candles", {})
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

            self.cache_loaded_at = datetime.utcnow()
            return True

        except Exception as e:
            logger.debug(f"Cache refresh failed: {e}")
            return False

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
        self.session_start_time = datetime.utcnow()

        # Record pre-existing positions to ignore
        existing_positions = await self.executor.get_all_positions()
        for pos in existing_positions:
            self.pre_existing_positions.add(pos["symbol"])
            logger.info(f"Ignoring pre-existing position: {pos['symbol']} "
                       f"({'LONG' if pos['size'] > 0 else 'SHORT'} {abs(pos['size']):.4f})")

        # Create strategy with trade callbacks
        self.strategy = WickReversalStrategy(
            config=self.config,
            data_provider=self.data_provider,
            executor=self.executor
        )

        # Set up trade callbacks for logging
        self.strategy.on_trade_entry = self._on_trade_entry
        self.strategy.on_trade_exit = self._on_trade_exit

        # Pass ignored positions to strategy
        self.strategy.ignored_positions = self.pre_existing_positions

        await self.strategy.initialize(initial_capital=self.initial_capital)
        self.is_running = True

        logger.info(f"Started LIVE trading: {', '.join(symbols)}")
        if self.pre_existing_positions:
            logger.info(f"Ignoring {len(self.pre_existing_positions)} pre-existing position(s)")

        # Start streaming for each symbol
        for symbol in symbols:
            await self._start_symbol_stream(symbol)

        # Also stream BTC for correlation
        await self._start_btc_stream()

        # Start background cache refresh task
        self._cache_refresh_task = asyncio.create_task(self._periodic_cache_refresh())

    async def _periodic_cache_refresh(self):
        """Periodically refresh cache from live_monitor's file."""
        while self.is_running:
            await asyncio.sleep(60)  # Refresh every 60 seconds
            if self._refresh_cache_from_file():
                # Check if any not-ready symbols now have fresh data
                for symbol in list(self.symbols_not_ready):
                    candles = self.get_cached_candles(symbol, 100)
                    if candles:
                        newest = candles[-1]
                        age_min = (datetime.utcnow() - newest.timestamp).total_seconds() / 60
                        if age_min < self.MAX_DATA_AGE_MINUTES:
                            # Re-seed the data aggregator with fresh data
                            for candle in candles:
                                self.strategy.data_aggregator.add_candle(symbol, candle)
                            self.symbols_not_ready.discard(symbol)
                            self.strategy.ignored_positions.discard(symbol)
                            print(f"\n  {symbol}: Fresh cache data - TRADING ENABLED\n")

    def _on_trade_entry(self, symbol: str, side: str, quantity: float, price: float,
                        stop_loss: float = None, take_profit: float = None,
                        leverage: float = None, risk_amount: float = None,
                        criteria_met: list = None, signal_strength: float = None,
                        leverage_breakdown: dict = None,
                        base_position_size: float = None, dynamic_multiplier: float = None):
        """Callback when a trade is entered."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Calculate USD values
        position_value = quantity * price  # Total position value on exchange
        base_usd = (base_position_size * price) if base_position_size else position_value
        multiplier = dynamic_multiplier if dynamic_multiplier else 1.0

        # Calculate what leverage was set based on confidence
        # Maps multiplier 0.5-2.5 to leverage 3-5x
        min_lev, max_lev = 3, 5
        normalized = max(0.0, min(1.0, (multiplier - 0.5) / 2.0))
        exchange_leverage = int(round(min_lev + normalized * (max_lev - min_lev)))
        margin_used = position_value / exchange_leverage

        print(f"\n{'='*60}")
        print(f"TRADE ENTRY - {timestamp}")
        print(f"{'='*60}")
        print(f"  Symbol: {symbol}")
        print(f"  Signal Strength: {signal_strength:.2f}" if signal_strength else "")
        if criteria_met:
            print(f"  Criteria Met: {', '.join(criteria_met)}")
        print(f"  Side: {side.upper()}")
        print(f"  Entry: ${price:.4f}  |  Stop: ${stop_loss:.4f}  |  Target: ${take_profit:.4f}")
        print(f"{'─'*60}")
        print(f"  Base Position:     ${base_usd:.2f}")
        print(f"  Dynamic Multiplier: {multiplier:.2f}x")
        print(f"  Position Value:    ${position_value:.2f}")
        print(f"  Exchange Leverage: {exchange_leverage}x (margin: ${margin_used:.2f})")
        print(f"  Risk Amount:       ${risk_amount:.2f}")
        print(f"{'='*60}\n")

    def _on_trade_exit(self, symbol: str, side: str, quantity: float,
                       entry_price: float, exit_price: float, pnl: float, reason: str):
        """Callback when a trade is exited."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Update session P&L
        self.session_pnl += pnl
        self.session_trades.append({
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "timestamp": timestamp
        })

        # Calculate win/loss
        result = "WIN" if pnl > 0 else "LOSS"
        pnl_pct = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0

        # Calculate session statistics
        wins = [t['pnl'] for t in self.session_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.session_trades if t['pnl'] <= 0]
        total_trades = len(self.session_trades)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        print(f"\n{'='*60}")
        print(f"TRADE EXIT - {result} - {timestamp}")
        print(f"{'='*60}")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side}")
        print(f"  Entry Price: ${entry_price:.4f}")
        print(f"  Exit Price: ${exit_price:.4f}")
        print(f"  Exit Reason: {reason}")
        print(f"  Trade P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"{'─'*60}")
        print(f"  SESSION STATS:")
        print(f"  Total Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%")
        print(f"  Wins: {len(wins)}  |  Losses: {len(losses)}")
        print(f"  Avg Win: ${avg_win:+.2f}  |  Avg Loss: ${avg_loss:+.2f}")
        print(f"  SESSION P&L: ${self.session_pnl:+.2f}")
        print(f"{'='*60}\n")

    async def _start_symbol_stream(self, symbol: str):
        """Start candle stream for a symbol."""
        if symbol in self.symbol_tasks:
            return

        # Reload cache from file to get latest data from live_monitor
        self._refresh_cache_from_file()

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

                # Seed momentum filter with historical prices
                self.strategy.signal_generator.seed_momentum_data(symbol, candles)

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
        logger.debug(f"Stopped stream: {symbol}")

    async def update_symbols(self, new_symbols: Set[str]):
        """Update trading symbols dynamically."""
        if new_symbols == self.current_symbols:
            return

        added = new_symbols - self.current_symbols
        removed = self.current_symbols - new_symbols

        # Log the change
        # print(f"\n{'='*60}")
        # print("SYMBOL UPDATE DETECTED")
        # print(f"{'='*60}")
        # if added:
        #     print(f"   Adding: {', '.join(added)}")
        # if removed:
        #     print(f"   Removing: {', '.join(removed)}")
        # print(f"   New symbols: {', '.join(new_symbols)}")
        # print(f"{'='*60}\n")

        # Stop removed symbols
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

        # Cancel cache refresh task
        if hasattr(self, '_cache_refresh_task') and self._cache_refresh_task:
            self._cache_refresh_task.cancel()
            try:
                await self._cache_refresh_task
            except asyncio.CancelledError:
                pass

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
        status = {}
        if self.strategy:
            status = self.strategy.get_status()

        # Add session-specific stats
        status["session_trades"] = len(self.session_trades)
        status["session_pnl"] = self.session_pnl
        status["session_start"] = self.session_start_time.isoformat() if self.session_start_time else None
        status["ignored_positions"] = list(self.pre_existing_positions)

        # Calculate session win rate
        if self.session_trades:
            wins = len([t for t in self.session_trades if t['pnl'] > 0])
            status["session_win_rate"] = wins / len(self.session_trades)
        else:
            status["session_win_rate"] = 0

        return status


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
