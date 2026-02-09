"""
Live trading manager for Hyperliquid mainnet.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Set, Optional

from config.settings import StrategyConfig
from core.types import Candle
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor
from trading.base_manager import BaseTradingManager

logger = logging.getLogger(__name__)


class LiveTradingManager(BaseTradingManager):
    """Manages live trading with session tracking and cache refresh."""

    def __init__(
        self,
        config: StrategyConfig,
        data_provider: HyperliquidProvider,
        executor: HyperliquidExecutor,
        initial_capital: float
    ):
        super().__init__(config, data_provider, executor, initial_capital)

        # Track pre-existing positions to ignore (manual trades)
        self.pre_existing_positions: Set[str] = set()

        # Session P&L tracking
        self.session_trades: list = []
        self.session_pnl: float = 0.0
        self.session_start_time: Optional[datetime] = None

        # Background tasks
        self._cache_refresh_task = None

    def load_candle_cache(self, max_retries: int = 3) -> bool:
        """Load candles from cache file with retry logic for mid-write errors."""
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
                    time.sleep(2)
                else:
                    print(f"  ERROR: Cache file corrupted after {max_retries} attempts: {e}")
                    print(f"  Try restarting live_monitor.py to regenerate the cache.")
                    return False

            except Exception as e:
                print(f"  ERROR loading cache: {e}")
                return False

        return False

    async def _pre_start(self, symbols: list):
        """Record pre-existing positions before starting."""
        self.session_start_time = datetime.utcnow()

        # Record pre-existing positions to ignore
        existing_positions = await self.executor.get_all_positions()
        for pos in existing_positions:
            self.pre_existing_positions.add(pos["symbol"])
            logger.info(f"Ignoring pre-existing position: {pos['symbol']} "
                       f"({'LONG' if pos['size'] > 0 else 'SHORT'} {abs(pos['size']):.4f})")

        if self.pre_existing_positions:
            logger.info(f"Ignoring {len(self.pre_existing_positions)} pre-existing position(s)")

    async def _configure_strategy(self):
        """Set up trade callbacks and ignored positions on the strategy."""
        self.strategy.on_trade_entry = self._on_trade_entry
        self.strategy.on_trade_exit = self._on_trade_exit
        self.strategy.ignored_positions = self.pre_existing_positions

    async def _post_start(self):
        """Start background cache refresh task."""
        self._cache_refresh_task = asyncio.create_task(self._periodic_cache_refresh())

    def _on_pre_stream(self, symbol: str):
        """Refresh cache from file before starting a symbol stream."""
        self._refresh_cache_from_file()

    def _on_data_preloaded(self, symbol: str, candles: list):
        """Seed momentum filter with historical prices."""
        self.strategy.signal_generator.seed_momentum_data(symbol, candles)

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
        print(f"{'---'*20}")
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
        print(f"{'---'*20}")
        print(f"  SESSION STATS:")
        print(f"  Total Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%")
        print(f"  Wins: {len(wins)}  |  Losses: {len(losses)}")
        print(f"  Avg Win: ${avg_win:+.2f}  |  Avg Loss: ${avg_loss:+.2f}")
        print(f"  SESSION P&L: ${self.session_pnl:+.2f}")
        print(f"{'='*60}\n")

    async def stop(self):
        """Stop all trading including cache refresh."""
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
        """Get current trading status with session-specific stats."""
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
