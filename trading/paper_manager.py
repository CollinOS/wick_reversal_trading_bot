"""
Paper trading manager for Hyperliquid testnet.
"""

import logging
from typing import Set

from config.settings import SymbolConfig
from trading.base_manager import BaseTradingManager

logger = logging.getLogger(__name__)


class PaperTradingManager(BaseTradingManager):
    """Manages paper trading with dynamic symbol updates."""

    async def update_symbols(self, new_symbols: Set[str]):
        """Update trading symbols dynamically with verbose logging."""
        if new_symbols == self.current_symbols:
            return

        added = new_symbols - self.current_symbols
        removed = self.current_symbols - new_symbols

        # Log the change
        print(f"\n{'='*60}")
        print("SYMBOL UPDATE DETECTED")
        print(f"{'='*60}")
        if added:
            print(f"   Adding: {', '.join(added)}")
        if removed:
            print(f"   Removing: {', '.join(removed)}")
        print(f"   New symbols: {', '.join(new_symbols)}")
        print(f"{'='*60}\n")

        # Stop removed symbols
        for symbol in removed:
            await self._stop_symbol_stream(symbol)

        # Update config
        self.config.symbols = [SymbolConfig(symbol=s) for s in new_symbols]

        # Start new symbols
        for symbol in added:
            await self._start_symbol_stream(symbol)

        self.current_symbols = new_symbols
