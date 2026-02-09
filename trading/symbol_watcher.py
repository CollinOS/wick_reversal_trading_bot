"""
Symbol watcher for dynamic symbol updates from live_monitor.py.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Set, Optional

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
