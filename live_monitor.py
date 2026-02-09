#!/usr/bin/env python3
"""
Live Market Monitor for Wick Reversal Strategy

Continuously monitors all Hyperliquid markets and alerts when better
opportunities are detected than the currently traded symbols.

Usage:
    # Console alerts only
    python live_monitor.py --current TAO-PERP AAVE-PERP ZRO-PERP

    # With Discord webhook
    python live_monitor.py --current TAO-PERP --discord-webhook https://discord.com/api/webhooks/...

    # Auto-update mode (writes to symbols.json for bot to read)
    python live_monitor.py --auto-update --output output/active_symbols.json
"""

import argparse
import asyncio
import logging

from config.settings import StrategyConfig
from config.paths import ACTIVE_SYMBOLS_FILE, CANDLE_CACHE_FILE
from scanner import LiveMarketMonitor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
        default=str(ACTIVE_SYMBOLS_FILE),
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
        default=10,
        help="Maximum number of symbols to track (default: 10)"
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
        print(f"Cache file: {CANDLE_CACHE_FILE}")
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
