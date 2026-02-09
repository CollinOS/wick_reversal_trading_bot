#!/usr/bin/env python3
"""
Market Scanner for Wick Reversal Strategy

Scans all Hyperliquid perpetual markets and ranks them by profitability
potential for the wick reversal strategy.

Usage:
    python scan_markets.py --days 30 --top 20
    python scan_markets.py --days 14 --min-signals 50 --export results.csv
"""

import argparse
import asyncio
import logging

from config.settings import StrategyConfig
from scanner import HyperliquidScanner, print_results, export_results


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def main():
    parser = argparse.ArgumentParser(description="Scan Hyperliquid markets for wick reversal opportunities")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze (default: 30)")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--top", type=int, default=20, help="Show top N results (default: 20)")
    parser.add_argument("--min-signals", type=int, default=10, help="Minimum signals required (default: 10)")
    parser.add_argument("--min-volume", type=float, default=10000, help="Minimum 24h volume USD (default: 10000)")
    parser.add_argument("--export", type=str, help="Export results to CSV file")
    parser.add_argument("--include-stables", action="store_true", help="Include stablecoins")
    parser.add_argument("--volatile-only", action="store_true", help="Only show high-volatility assets (ATR > 0.3%)")
    args = parser.parse_args()

    config = StrategyConfig()
    scanner = HyperliquidScanner(config)

    try:
        await scanner.connect()

        print(f"\nScanning Hyperliquid markets...")
        print(f"  - Analyzing {args.days} days of {args.timeframe} candles")
        print(f"  - Minimum {args.min_signals} signals required")
        print(f"  - Minimum ${args.min_volume:,.0f} daily volume\n")

        results = await scanner.scan_all_markets(
            days=args.days,
            timeframe=args.timeframe,
            min_signals=args.min_signals,
            min_volume=args.min_volume,
            exclude_stables=not args.include_stables
        )

        # Filter for volatile assets if requested
        if args.volatile_only:
            results = [r for r in results if r.avg_atr_pct >= 0.003]  # 0.3%+ ATR
            print(f"Filtered to {len(results)} volatile assets (ATR >= 0.3%)")

        print_results(results, args.top)

        if args.export:
            export_results(results, args.export)

        # Print recommended symbols for config
        if results:
            print("\n" + "=" * 60)
            print("RECOMMENDED SYMBOLS FOR CONFIG")
            print("=" * 60)
            top_symbols = [r.symbol for r in results[:5]]
            print(f"\nTop 5 by overall score: {', '.join(top_symbols)}")

            # By signal frequency (most important for this strategy!)
            by_freq = sorted(results, key=lambda x: x.signals_per_day, reverse=True)[:5]
            print(f"Top 5 by signal frequency: {', '.join(f'{r.symbol}({r.signals_per_day:.1f}/day)' for r in by_freq)}")

            # By volatility
            by_vol = sorted(results, key=lambda x: x.avg_atr_pct, reverse=True)[:5]
            print(f"Top 5 by volatility: {', '.join(f'{r.symbol}({r.avg_atr_pct*100:.2f}%)' for r in by_vol)}")

            # By win rate (but only if enough signals)
            viable = [r for r in results if r.signals_per_day >= 1.0]
            if viable:
                by_winrate = sorted(viable, key=lambda x: x.win_rate, reverse=True)[:5]
                print(f"Top 5 by win rate (1+ sig/day): {', '.join(f'{r.symbol}({r.win_rate*100:.0f}%)' for r in by_winrate)}")

            print("\n>>> For wick reversal, prioritize SIGNAL FREQUENCY over win rate! <<<")
            print(">>> More volatile assets = more trading opportunities <<<")
            print("\nNext step: Run full backtest on top picks:")
            if top_symbols:
                syms = ' '.join(top_symbols[:3])
                print(f"  python fetch_hyperliquid_history.py --symbols {syms} --days 90")
                print(f"  python run_backtest.py --analyze")

    finally:
        await scanner.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
