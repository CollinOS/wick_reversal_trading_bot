#!/usr/bin/env python3
"""
Live Market Monitor for Wick Reversal Strategy

Continuously monitors all Hyperliquid markets and alerts when better
opportunities are detected than the currently traded symbols.

Features:
- Real-time volatility monitoring
- Alert when new high-volatility assets emerge
- Optional auto-update of trading symbols
- Discord/console notifications

Usage:
    # Console alerts only
    python live_monitor.py --current TAO-PERP AAVE-PERP ZRO-PERP

    # With Discord webhook
    python live_monitor.py --current TAO-PERP --discord-webhook https://discord.com/api/webhooks/...

    # Auto-update mode (writes to symbols.json for bot to read)
    python live_monitor.py --auto-update --output active_symbols.json
"""

import argparse
import asyncio
import aiohttp
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from collections import deque
import time

from config.settings import StrategyConfig
from core.types import Candle
from data.candle_cache import CandleCache


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Real-time snapshot of a market's trading potential."""
    symbol: str
    timestamp: datetime

    # Current metrics
    current_price: float = 0.0
    volume_24h: float = 0.0

    # Volatility metrics (from recent candles)
    atr_pct: float = 0.0
    recent_wick_count: int = 0  # Wick signals in last N candles
    avg_wick_size_pct: float = 0.0

    # Scoring
    opportunity_score: float = 0.0

    # Change tracking
    score_change: float = 0.0  # vs last check
    is_spiking: bool = False   # Sudden volatility increase


@dataclass
class Alert:
    """Alert for user notification."""
    timestamp: datetime
    alert_type: str  # "new_opportunity", "volatility_spike", "symbol_swap"
    symbol: str
    message: str
    score: float
    priority: str = "normal"  # "low", "normal", "high", "urgent"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.alert_type,
            "symbol": self.symbol,
            "message": self.message,
            "score": self.score,
            "priority": self.priority
        }


class LiveMarketMonitor:
    """Monitors markets in real-time for trading opportunities."""

    BASE_URL = "https://api.hyperliquid.xyz"

    def __init__(
        self,
        current_symbols: List[str],
        config: StrategyConfig,
        check_interval: int = 300,  # 5 minutes - now uses cached data, not API
        lookback_candles: int = 50,
        alert_threshold: float = 20.0,  # Score difference to trigger alert
    ):
        self.current_symbols = set(current_symbols)
        self.config = config
        self.check_interval = check_interval
        self.lookback_candles = lookback_candles
        self.alert_threshold = alert_threshold

        self._session: Optional[aiohttp.ClientSession] = None

        # Candle cache - owns WebSocket streams for all symbols
        self.candle_cache: Optional[CandleCache] = None

        # State tracking
        self.market_snapshots: Dict[str, MarketSnapshot] = {}
        self.previous_scores: Dict[str, float] = {}
        self.alerts: deque = deque(maxlen=100)

        # Callbacks
        self.on_alert: Optional[callable] = None
        self.on_symbol_change: Optional[callable] = None

        # Auto-update settings
        self.auto_update = False
        self.output_file: Optional[str] = None
        self.min_symbols = 3
        self.max_symbols = 5
        self.pinned_symbols: Set[str] = set()  # Symbols that are always included

    async def connect(self):
        """Initialize HTTP session and start candle cache."""
        self._session = aiohttp.ClientSession()

        # Initialize candle cache - this owns WebSocket streams for all symbols
        self.candle_cache = CandleCache(
            cache_dir="data",
            timeframe="5m",
            testnet=False
        )

        print("Starting candle cache (WebSocket streams for all symbols)...")
        await self.candle_cache.start()
        print(f"Candle cache active - streaming {len(self.candle_cache.subscribed_symbols)} symbols")

        logger.info("Live monitor connected to Hyperliquid")

    async def disconnect(self):
        """Close HTTP session and stop candle cache."""
        if self.candle_cache:
            await self.candle_cache.stop()
            print("Candle cache stopped and saved to disk")

        if self._session:
            await self._session.close()

    def get_all_markets(self) -> List[str]:
        """Get list of all perpetual markets from candle cache."""
        if self.candle_cache:
            return list(self.candle_cache.subscribed_symbols)
        return []

    async def get_market_stats(self) -> Dict[str, Dict]:
        """Get current stats for all markets."""
        # Get mid prices
        async with self._session.post(
            f"{self.BASE_URL}/info",
            json={"type": "allMids"}
        ) as resp:
            mids = await resp.json()

        # Get 24h stats
        async with self._session.post(
            f"{self.BASE_URL}/info",
            json={"type": "metaAndAssetCtxs"}
        ) as resp:
            data = await resp.json()

        stats = {}
        universe = data[0].get("universe", []) if data else []
        asset_ctxs = data[1] if len(data) > 1 else []

        for i, asset in enumerate(universe):
            name = asset["name"]
            ctx = asset_ctxs[i] if i < len(asset_ctxs) else {}
            stats[f"{name}-PERP"] = {
                "mid_price": float(mids.get(name, 0)),
                "volume_24h": float(ctx.get("dayNtlVlm", 0)),
                "open_interest": float(ctx.get("openInterest", 0)),
            }

        return stats

    def get_recent_candles(self, symbol: str, count: int = 50) -> List[Candle]:
        """Get recent candles from cache (real-time WebSocket data)."""
        if not self.candle_cache:
            return []

        candles = self.candle_cache.get_candles(symbol, count)
        return candles

    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate ATR from candles."""
        if len(candles) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i-1].close),
                abs(candles[i].low - candles[i-1].close)
            )
            true_ranges.append(tr)

        # Use available candles if less than period
        use_period = min(period, len(true_ranges))
        return sum(true_ranges[-use_period:]) / use_period if use_period > 0 else 0.0

    def count_wick_signals(self, candles: List[Candle]) -> tuple:
        """Count potential wick signals and average wick size."""
        if not candles:
            return 0, 0.0

        signal_count = 0
        wick_sizes = []

        min_wick_pct = self.config.signal.min_wick_pct

        for candle in candles:
            if candle.close <= 0:
                continue

            upper_wick_pct = candle.upper_wick / candle.close
            lower_wick_pct = candle.lower_wick / candle.close

            max_wick = max(upper_wick_pct, lower_wick_pct)
            wick_sizes.append(max_wick)

            # Count as potential signal if wick exceeds threshold
            if max_wick >= min_wick_pct:
                signal_count += 1

        avg_wick = sum(wick_sizes) / len(wick_sizes) if wick_sizes else 0
        return signal_count, avg_wick

    def calculate_opportunity_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate opportunity score for a market."""
        # Weights focused on real-time opportunity
        # - Recent wick signals: 40% (immediate opportunities)
        # - ATR volatility: 35% (ongoing volatility)
        # - Volume: 15% (liquidity for execution)
        # - Wick size: 10% (quality of signals)

        # Wick signal score (target: 5-15 signals in last 50 candles)
        wick_score = min(100, snapshot.recent_wick_count * 8)

        # ATR score (target: 0.3%+ ATR)
        atr_score = min(100, snapshot.atr_pct * 100 * 30)  # 0.33% -> 100

        # Volume score (target: $100k+)
        vol_score = min(100, snapshot.volume_24h / 5000)  # $500k -> 100

        # Wick size score
        wick_size_score = min(100, snapshot.avg_wick_size_pct * 100 * 15)  # 0.67% -> 100

        score = (
            wick_score * 0.40 +
            atr_score * 0.35 +
            vol_score * 0.15 +
            wick_size_score * 0.10
        )

        return score

    def scan_market(self, symbol: str, stats: Dict) -> Optional[MarketSnapshot]:
        """Scan a single market and create snapshot."""
        market_stats = stats.get(symbol, {})

        if market_stats.get("mid_price", 0) == 0:
            return None

        # Get recent candles from cache (real-time data)
        candles = self.get_recent_candles(symbol, self.lookback_candles)

        if len(candles) < 10:  # Need at least 10 candles
            return None

        # Calculate metrics
        atr = self.calculate_atr(candles, 14)
        atr_pct = atr / candles[-1].close if candles[-1].close > 0 else 0

        wick_count, avg_wick = self.count_wick_signals(candles)

        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=market_stats.get("mid_price", 0),
            volume_24h=market_stats.get("volume_24h", 0),
            atr_pct=atr_pct,
            recent_wick_count=wick_count,
            avg_wick_size_pct=avg_wick,
        )

        # Calculate score
        snapshot.opportunity_score = self.calculate_opportunity_score(snapshot)

        # Check for score change
        prev_score = self.previous_scores.get(symbol, 0)
        snapshot.score_change = snapshot.opportunity_score - prev_score

        # Detect volatility spike (score increased by 20+ points)
        snapshot.is_spiking = snapshot.score_change >= 20

        return snapshot

    async def run_scan(self) -> List[MarketSnapshot]:
        """Run a full market scan using cached candle data."""
        all_symbols = self.get_all_markets()

        # Check cache health
        cache_stats = self.candle_cache.get_cache_stats() if self.candle_cache else {}
        symbols_with_data = len(cache_stats.get("symbols_with_data", {}))

        if symbols_with_data == 0:
            logger.warning("Candle cache is empty - waiting for WebSocket data...")
            return []

        logger.info(f"Scanning {symbols_with_data} symbols with cached data")

        stats = await self.get_market_stats()

        # Filter out stables and low volume
        stables = {"USDC", "USDT", "DAI", "FRAX", "LUSD"}
        filtered = [
            s for s in all_symbols
            if s.replace("-PERP", "") not in stables
            and stats.get(s, {}).get("volume_24h", 0) >= 1000
        ]

        # Scan markets (no longer async - just reading from cache)
        snapshots = []
        for symbol in filtered:
            result = self.scan_market(symbol, stats)
            if result is not None:
                snapshots.append(result)

        logger.info(f"Got {len(snapshots)} valid snapshots")

        # Update state
        for snap in snapshots:
            self.previous_scores[snap.symbol] = snap.opportunity_score
            self.market_snapshots[snap.symbol] = snap

        # Sort by score
        snapshots.sort(key=lambda x: x.opportunity_score, reverse=True)

        return snapshots

    def check_for_alerts(self, snapshots: List[MarketSnapshot]) -> List[Alert]:
        """Check if any alerts should be triggered."""
        alerts = []

        # Get scores of current symbols
        current_scores = []
        for sym in self.current_symbols:
            snap = self.market_snapshots.get(sym)
            if snap:
                current_scores.append(snap.opportunity_score)

        min_current_score = min(current_scores) if current_scores else 0
        avg_current_score = sum(current_scores) / len(current_scores) if current_scores else 0

        # Check top opportunities
        for snap in snapshots[:10]:
            # Skip if already trading
            if snap.symbol in self.current_symbols:
                continue

            # Alert if significantly better than worst current symbol
            score_diff = snap.opportunity_score - min_current_score

            if score_diff >= self.alert_threshold:
                priority = "high" if score_diff >= 40 else "normal"

                alert = Alert(
                    timestamp=datetime.utcnow(),
                    alert_type="new_opportunity",
                    symbol=snap.symbol,
                    message=f"{snap.symbol} score {snap.opportunity_score:.1f} is {score_diff:.1f} points better than your worst symbol",
                    score=snap.opportunity_score,
                    priority=priority
                )
                alerts.append(alert)

            # Alert on volatility spikes
            if snap.is_spiking:
                alert = Alert(
                    timestamp=datetime.utcnow(),
                    alert_type="volatility_spike",
                    symbol=snap.symbol,
                    message=f"{snap.symbol} volatility spiking! Score jumped {snap.score_change:.1f} points",
                    score=snap.opportunity_score,
                    priority="high"
                )
                alerts.append(alert)

        return alerts

    def get_recommended_symbols(self, snapshots: List[MarketSnapshot], count: int = 3) -> List[str]:
        """Get recommended symbols based on current scan."""
        # Start with pinned symbols (always included)
        result = list(self.pinned_symbols)

        # Filter for minimum quality (lowered thresholds)
        viable = [s for s in snapshots if s.opportunity_score >= 15 and s.recent_wick_count >= 1]

        # If still nothing, just return top by score
        if not viable and snapshots:
            viable = snapshots

        # Add top recommendations that aren't already pinned
        for snap in viable:
            if snap.symbol not in self.pinned_symbols:
                result.append(snap.symbol)
            if len(result) >= count:
                break

        return result[:count]

    async def update_symbol_file(self, symbols: List[str]):
        """Write current recommended symbols to file for bot to read."""
        if not self.output_file:
            return

        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "symbols": symbols,
            "snapshots": {
                sym: asdict(self.market_snapshots[sym])
                for sym in symbols
                if sym in self.market_snapshots
            }
        }

        # Convert datetime objects to strings
        for sym, snap in data["snapshots"].items():
            snap["timestamp"] = snap["timestamp"].isoformat() if isinstance(snap["timestamp"], datetime) else snap["timestamp"]

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Updated {self.output_file} with symbols: {symbols}")

    async def send_discord_alert(self, webhook_url: str, alert: Alert):
        """Send alert to Discord webhook."""
        color = {
            "low": 0x808080,
            "normal": 0x00ff00,
            "high": 0xff9900,
            "urgent": 0xff0000
        }.get(alert.priority, 0x00ff00)

        embed = {
            "title": f"🚨 {alert.alert_type.replace('_', ' ').title()}",
            "description": alert.message,
            "color": color,
            "fields": [
                {"name": "Symbol", "value": alert.symbol, "inline": True},
                {"name": "Score", "value": f"{alert.score:.1f}", "inline": True},
                {"name": "Priority", "value": alert.priority.upper(), "inline": True},
            ],
            "timestamp": alert.timestamp.isoformat()
        }

        payload = {"embeds": [embed]}

        try:
            async with self._session.post(webhook_url, json=payload) as resp:
                if resp.status != 204:
                    logger.warning(f"Discord webhook returned {resp.status}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def print_status(self, snapshots: List[MarketSnapshot]):
        """Print current market status to console."""
        now = datetime.utcnow().strftime("%H:%M:%S")

        # Get cache stats
        cache_stats = self.candle_cache.get_cache_stats() if self.candle_cache else {}
        total_candles = cache_stats.get("total_candles", 0)
        symbols_with_data = len(cache_stats.get("symbols_with_data", {}))

        # Get freshness report
        freshness = self.candle_cache.get_freshness_report() if self.candle_cache else {}
        fresh = freshness.get("fresh", 0)
        stale = freshness.get("stale", 0)
        very_stale = freshness.get("very_stale", 0)
        received = freshness.get("candles_received", 0)

        print(f"\n{'='*80}")
        print(f"LIVE MARKET MONITOR - {now} UTC")
        print(f"Cache: {total_candles} candles | Fresh: {fresh} | Stale: {stale} | Old: {very_stale} | WS received: {received}")
        print(f"{'='*80}")

        # Show pinned symbols if any
        if self.pinned_symbols:
            print(f"\n📌 PINNED (always active): {', '.join(self.pinned_symbols)}")

        # Current symbols status
        print(f"\n📊 CURRENTLY TRADING:")
        for sym in self.current_symbols:
            snap = self.market_snapshots.get(sym)
            pinned = "📌" if sym in self.pinned_symbols else "  "
            if snap:
                trend = "📈" if snap.score_change > 0 else "📉" if snap.score_change < 0 else "➡️"
                print(f" {pinned} {sym:<15} Score: {snap.opportunity_score:>5.1f} {trend} ({snap.score_change:+.1f})  "
                      f"Wicks: {snap.recent_wick_count}  ATR: {snap.atr_pct*100:.2f}%")
            else:
                print(f" {pinned} {sym:<15} (no data yet)")

        # Top opportunities
        print(f"\n🔥 TOP OPPORTUNITIES:")
        for i, snap in enumerate(snapshots[:8], 1):
            current = "✅" if snap.symbol in self.current_symbols else "  "
            spike = "🚀" if snap.is_spiking else "  "
            print(f" {i}. {current} {snap.symbol:<15} Score: {snap.opportunity_score:>5.1f} {spike}  "
                  f"Wicks: {snap.recent_wick_count}  ATR: {snap.atr_pct*100:.2f}%  Vol: ${snap.volume_24h/1000:.0f}k")

        # Recommendations
        recommended = self.get_recommended_symbols(snapshots, 3)
        new_recs = [r for r in recommended if r not in self.current_symbols]
        if new_recs:
            print(f"\n💡 CONSIDER SWITCHING TO: {', '.join(new_recs)}")

        print(f"\n{'='*80}")

    async def run(
        self,
        discord_webhook: Optional[str] = None,
        console_alerts: bool = True
    ):
        """Main monitoring loop."""
        logger.info(f"Starting live monitor. Checking every {self.check_interval}s")
        logger.info(f"Currently trading: {', '.join(self.current_symbols)}")

        if self.auto_update:
            logger.info(f"Auto-update enabled. Writing to: {self.output_file}")

        # Wait for WebSocket to collect some fresh candles before first scan
        print("\nWaiting 30 seconds for WebSocket to collect fresh candles...")
        await asyncio.sleep(30)

        # Check if we have fresh data
        if self.candle_cache:
            freshness = self.candle_cache.get_freshness_report()
            received = freshness.get("candles_received", 0)
            print(f"WebSocket has received {received} candle updates. Starting scans.\n")

        while True:
            try:
                # Run scan
                snapshots = await self.run_scan()

                # Check for alerts
                alerts = self.check_for_alerts(snapshots)

                # Process alerts
                for alert in alerts:
                    self.alerts.append(alert)

                    if console_alerts:
                        priority_icon = {"low": "ℹ️", "normal": "📢", "high": "⚠️", "urgent": "🚨"}.get(alert.priority, "📢")
                        print(f"\n{priority_icon} ALERT: {alert.message}")

                    if discord_webhook:
                        await self.send_discord_alert(discord_webhook, alert)

                    if self.on_alert:
                        await self.on_alert(alert)

                # Auto-update symbols if enabled
                if self.auto_update:
                    recommended = self.get_recommended_symbols(snapshots, self.max_symbols)
                    if recommended:
                        # Check if significant change
                        current_set = self.current_symbols
                        new_set = set(recommended)

                        if new_set != current_set:
                            logger.info(f"Updating symbols: {current_set} -> {new_set}")
                            self.current_symbols = new_set
                            await self.update_symbol_file(recommended)

                            if self.on_symbol_change:
                                await self.on_symbol_change(recommended)

                # Print status
                self.print_status(snapshots)

                # Wait for next scan
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Monitor stopped")
                break
            except Exception as e:
                logger.exception(f"Error in monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retry


async def main():
    parser = argparse.ArgumentParser(description="Live market monitor for wick reversal strategy")
    parser.add_argument(
        "--current",
        nargs="+",
        default=["TAO-PERP", "AAVE-PERP", "ZRO-PERP"],
        help="Currently traded symbols"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 min, uses cached WebSocket data)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Score difference to trigger alert (default: 20)"
    )
    parser.add_argument(
        "--discord-webhook",
        type=str,
        help="Discord webhook URL for alerts"
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Automatically update recommended symbols"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="active_symbols.json",
        help="Output file for auto-update mode"
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console output"
    )
    parser.add_argument(
        "--pinned",
        nargs="+",
        default=[],
        help="Symbols to always include (cannot be removed by auto-update)"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=5,
        help="Maximum number of symbols to track (default: 5)"
    )
    args = parser.parse_args()

    config = StrategyConfig()

    monitor = LiveMarketMonitor(
        current_symbols=args.current,
        config=config,
        check_interval=args.interval,
        alert_threshold=args.threshold
    )

    monitor.auto_update = args.auto_update
    monitor.output_file = args.output
    monitor.pinned_symbols = set(args.pinned)
    monitor.max_symbols = args.max_symbols

    # Ensure pinned symbols are always in current symbols
    if args.pinned:
        monitor.current_symbols.update(args.pinned)

    try:
        await monitor.connect()

        # Format interval nicely
        if args.interval >= 3600:
            interval_str = f"{args.interval / 3600:.1f} hours"
        elif args.interval >= 60:
            interval_str = f"{args.interval / 60:.0f} minutes"
        else:
            interval_str = f"{args.interval}s"

        print(f"\n{'='*60}")
        print("WICK REVERSAL - LIVE MARKET MONITOR")
        print(f"{'='*60}")
        print(f"Data source: Real-time WebSocket (CandleCache)")
        print(f"Currently trading: {', '.join(args.current)}")
        print(f"Scan interval: {interval_str}")
        print(f"Alert threshold: {args.threshold} points")
        print(f"Max symbols: {args.max_symbols}")
        if args.pinned:
            print(f"Pinned symbols: {', '.join(args.pinned)}")
        if args.discord_webhook:
            print(f"Discord alerts: Enabled")
        if args.auto_update:
            print(f"Auto-update: Enabled -> {args.output}")
        print(f"Cache file: data/candle_cache.json")
        print(f"{'='*60}")
        print("\nPress Ctrl+C to stop\n")

        await monitor.run(
            discord_webhook=args.discord_webhook,
            console_alerts=not args.no_console
        )

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await monitor.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
