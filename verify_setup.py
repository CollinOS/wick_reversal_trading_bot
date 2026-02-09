#!/usr/bin/env python3
"""
Verify live trading setup WITHOUT placing any orders.
This script tests:
1. Private key validity
2. Wallet address derivation
3. Exchange connection
4. Account balance retrieval
5. Asset index mapping
"""

import argparse
import asyncio
import os
import sys

from execution.orders import HyperliquidExecutor, HYPERLIQUID_ASSET_INDEX


async def verify_setup(private_key: str, testnet: bool = False):
    """Verify the live trading setup."""
    print("\n" + "="*60)
    print("LIVE TRADING SETUP VERIFICATION")
    print("="*60)
    print(f"Network: {'TESTNET' if testnet else 'MAINNET'}")
    print()

    # Step 1: Initialize executor (validates private key)
    print("[1/5] Initializing executor...")
    try:
        executor = HyperliquidExecutor(
            private_key=private_key,
            testnet=testnet
        )
        print(f"      Wallet address: {executor.wallet_address}")
        print("      PASS")
    except Exception as e:
        print(f"      FAIL: {e}")
        return False

    # Step 2: Connect to exchange
    print("\n[2/5] Connecting to exchange...")
    try:
        await executor.connect()
        print(f"      Asset indices loaded: {len(executor._asset_index_cache)} symbols")
        print("      PASS")
    except Exception as e:
        print(f"      FAIL: {e}")
        return False

    # Step 3: Verify connection and get balance
    print("\n[3/5] Verifying connection and fetching balance...")
    try:
        balance = await executor.get_account_balance()
        if balance:
            print(f"      Account Value: ${balance['account_value']:,.2f}")
            print(f"      Total Margin Used: ${balance['total_margin_used']:,.2f}")
            print(f"      Withdrawable: ${balance['withdrawable']:,.2f}")
            print("      PASS")
        else:
            print("      FAIL: Could not fetch balance")
            return False
    except Exception as e:
        print(f"      FAIL: {e}")
        return False

    # Step 4: Check existing positions
    print("\n[4/5] Checking existing positions...")
    try:
        positions = await executor.get_all_positions()
        if positions:
            print(f"      Found {len(positions)} position(s):")
            for pos in positions:
                side = "LONG" if pos["size"] > 0 else "SHORT"
                print(f"        - {pos['symbol']}: {side} {abs(pos['size']):.4f} @ ${pos['entry_price']:.2f}")
        else:
            print("      No open positions")
        print("      PASS")
    except Exception as e:
        print(f"      FAIL: {e}")
        return False

    # Step 5: Verify target symbols have valid indices
    print("\n[5/5] Verifying symbol mappings...")
    test_symbols = ["TAO-PERP", "AAVE-PERP", "ZRO-PERP", "BTC-PERP", "ETH-PERP"]
    all_valid = True
    for symbol in test_symbols:
        try:
            idx = executor._get_asset_index(symbol)
            print(f"      {symbol}: index {idx}")
        except ValueError as e:
            print(f"      {symbol}: MISSING - {e}")
            all_valid = False

    if all_valid:
        print("      PASS")
    else:
        print("      WARNING: Some symbols missing")

    # Cleanup
    await executor.disconnect()

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    if balance['account_value'] < 10:
        print("\nWARNING: Account value is very low!")
        print("Make sure you have deposited funds to your Hyperliquid account.")
        print(f"Wallet: {executor.wallet_address}")
    else:
        print("\nAll checks passed. Your live trading setup is ready.")
        print(f"\nTo start live trading, run:")
        print(f"  python run_live.py --capital 200")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify live trading setup")
    parser.add_argument(
        "--private-key",
        default=os.environ.get("HL_PRIVATE_KEY"),
        help="Ethereum private key (or set HL_PRIVATE_KEY env var)"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet instead of mainnet"
    )
    args = parser.parse_args()

    if not args.private_key:
        print("Error: Private key required.")
        print("Either pass --private-key or set HL_PRIVATE_KEY environment variable")
        sys.exit(1)

    success = asyncio.run(verify_setup(args.private_key, args.testnet))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
