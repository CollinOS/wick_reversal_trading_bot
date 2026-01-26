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
        check_interval: int = 300,  # 5 minutes
        lookback_candles: int = 50,
        alert_threshold: float = 20.0,  # Score difference to trigger alert
    ):
        self.current_symbols = set(current_symbols)
        self.config = config
        self.check_interval = check_interval
        self.lookback_candles = lookback_candles
        self.alert_threshold = alert_threshold

        self._session: Optional[aiohttp.ClientSession] = None

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

    async def connect(self):
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession()
        logger.info("Live monitor connected to Hyperliquid")

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()

    async def get_all_markets(self) -> List[str]:
        """Get list of all perpetual markets."""
        payload = {"type": "meta"}
        async with self._session.post(f"{self.BASE_URL}/info", json=payload) as resp:
            data = await resp.json()

        symbols = []
        for asset in data.get("universe", []):
            symbols.append(f"{asset['name']}-PERP")
        return symbols

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

    async def get_recent_candles(self, symbol: str, count: int = 50) -> List[Candle]:
        """Fetch recent candles for volatility analysis."""
        coin = symbol.replace("-PERP", "")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=count * 5 / 60 + 1)  # 5m candles

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "5m",
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        try:
            async with self._session.post(f"{self.BASE_URL}/info", json=payload) as resp:
                data = await resp.json()

            if not isinstance(data, list):
                return []

            candles = []
            for c in data[-count:]:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(c['t'] / 1000),
                    open=float(c['o']),
                    high=float(c['h']),
                    low=float(c['l']),
                    close=float(c['c']),
                    volume=float(c['v'])
                ))
            return candles
        except Exception as e:
            logger.debug(f"Failed to get candles for {symbol}: {e}")
            return []

    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate ATR from candles."""
        if len(candles) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i-1].close),
                abs(candles[i].low - candles[i-1].close)
            )
            true_ranges.append(tr)

        return sum(true_ranges[-period:]) / period

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

    async def scan_market(self, symbol: str, stats: Dict) -> Optional[MarketSnapshot]:
        """Scan a single market and create snapshot."""
        market_stats = stats.get(symbol, {})

        if market_stats.get("mid_price", 0) == 0:
            return None

        # Get recent candles
        candles = await self.get_recent_candles(symbol, self.lookback_candles)

        if len(candles) < 20:
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
        """Run a full market scan."""
        all_symbols = await self.get_all_markets()
        stats = await self.get_market_stats()

        # Filter out stables and low volume
        stables = {"USDC", "USDT", "DAI", "FRAX", "LUSD"}
        filtered = [
            s for s in all_symbols
            if s.replace("-PERP", "") not in stables
            and stats.get(s, {}).get("volume_24h", 0) >= 10000
        ]

        # Scan markets concurrently (with limit)
        semaphore = asyncio.Semaphore(10)

        async def scan_with_limit(symbol):
            async with semaphore:
                return await self.scan_market(symbol, stats)

        tasks = [scan_with_limit(s) for s in filtered]
        results = await asyncio.gather(*tasks)

        snapshots = [r for r in results if r is not None]

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
        # Filter for minimum quality
        viable = [s for s in snapshots if s.opportunity_score >= 30 and s.recent_wick_count >= 3]

        # Return top N
        return [s.symbol for s in viable[:count]]

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

        print(f"\n{'='*80}")
        print(f"LIVE MARKET MONITOR - {now} UTC")
        print(f"{'='*80}")

        # Current symbols status
        print(f"\n📊 CURRENTLY TRADING:")
        for sym in self.current_symbols:
            snap = self.market_snapshots.get(sym)
            if snap:
                trend = "📈" if snap.score_change > 0 else "📉" if snap.score_change < 0 else "➡️"
                print(f"   {sym:<15} Score: {snap.opportunity_score:>5.1f} {trend} ({snap.score_change:+.1f})  "
                      f"Wicks: {snap.recent_wick_count}  ATR: {snap.atr_pct*100:.2f}%")

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
        help="Check interval in seconds (default: 300 = 5 min)"
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

    try:
        await monitor.connect()

        print(f"\n{'='*60}")
        print("WICK REVERSAL - LIVE MARKET MONITOR")
        print(f"{'='*60}")
        print(f"Currently trading: {', '.join(args.current)}")
        print(f"Check interval: {args.interval}s")
        print(f"Alert threshold: {args.threshold} points")
        if args.discord_webhook:
            print(f"Discord alerts: Enabled")
        if args.auto_update:
            print(f"Auto-update: Enabled -> {args.output}")
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
