"""
Logging and Monitoring Module
Comprehensive logging for signals, trades, and system events.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import sys


class StructuredLogger:
    """
    Structured logger for trading system events.
    Outputs both human-readable and JSON-formatted logs.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        log_level: str = "INFO",
        json_output: bool = True
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_output = json_output
        
        # Create loggers
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_level)
    
    def _setup_handlers(self, log_level: str):
        """Setup console and file handlers."""
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level))
        self.logger.addHandler(console_handler)
        
        # File handler - human readable
        date_str = datetime.utcnow().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{date_str}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # JSON file handler for structured logs
        if self.json_output:
            self.json_file = self.log_dir / f"{self.name}_{date_str}_structured.jsonl"
    
    def _log_json(self, event_type: str, data: dict):
        """Write structured JSON log entry."""
        if not self.json_output:
            return
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **data
        }
        
        with open(self.json_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_signal(self, signal: Any):
        """Log a trading signal."""
        signal_dict = signal.to_dict() if hasattr(signal, 'to_dict') else str(signal)
        
        self.logger.info(
            f"SIGNAL | {signal_dict.get('symbol', 'N/A')} | "
            f"{signal_dict.get('signal_type', 'N/A')} | "
            f"strength={signal_dict.get('strength', 0):.2f}"
        )
        
        self._log_json("signal", signal_dict)
    
    def log_order(self, order: Any, action: str = "submitted"):
        """Log an order event."""
        order_dict = order.to_dict() if hasattr(order, 'to_dict') else str(order)
        
        self.logger.info(
            f"ORDER {action.upper()} | {order_dict.get('symbol', 'N/A')} | "
            f"{order_dict.get('side', 'N/A')} | "
            f"qty={order_dict.get('quantity', 0):.6f} | "
            f"price={order_dict.get('price', 'N/A')}"
        )
        
        self._log_json(f"order_{action}", order_dict)
    
    def log_fill(self, order: Any, latency_ms: float = 0):
        """Log an order fill."""
        order_dict = order.to_dict() if hasattr(order, 'to_dict') else str(order)
        
        self.logger.info(
            f"FILL | {order_dict.get('symbol', 'N/A')} | "
            f"qty={order_dict.get('filled_quantity', 0):.6f} | "
            f"price={order_dict.get('average_fill_price', 'N/A')} | "
            f"slippage={order_dict.get('slippage', 0)*100:.3f}% | "
            f"latency={latency_ms:.1f}ms"
        )
        
        self._log_json("fill", {**order_dict, "latency_ms": latency_ms})
    
    def log_position_open(self, position: Any):
        """Log position opening."""
        pos_dict = position.to_dict() if hasattr(position, 'to_dict') else str(position)
        
        self.logger.info(
            f"POSITION OPEN | {pos_dict.get('symbol', 'N/A')} | "
            f"{pos_dict.get('side', 'N/A')} | "
            f"entry={pos_dict.get('entry_price', 0):.6f} | "
            f"stop={pos_dict.get('stop_loss', 0):.6f} | "
            f"target={pos_dict.get('take_profit', 0):.6f}"
        )
        
        self._log_json("position_open", pos_dict)
    
    def log_position_close(self, position: Any, exit_reason: str):
        """Log position closing."""
        pos_dict = position.to_dict() if hasattr(position, 'to_dict') else str(position)
        
        self.logger.info(
            f"POSITION CLOSE | {pos_dict.get('symbol', 'N/A')} | "
            f"{exit_reason} | "
            f"PnL=${pos_dict.get('net_pnl', 0):.2f} | "
            f"duration={pos_dict.get('candles_held', 0)} candles"
        )
        
        self._log_json("position_close", {**pos_dict, "exit_reason": exit_reason})
    
    def log_risk_event(self, event_type: str, details: dict):
        """Log risk management events."""
        self.logger.warning(
            f"RISK | {event_type} | {json.dumps(details)}"
        )
        
        self._log_json("risk_event", {"event_type": event_type, **details})
    
    def log_filter_rejection(self, symbol: str, filter_result: str, details: str):
        """Log when a signal is rejected by filters."""
        self.logger.debug(
            f"FILTER | {symbol} | {filter_result} | {details}"
        )
        
        self._log_json("filter_rejection", {
            "symbol": symbol,
            "filter_result": filter_result,
            "details": details
        })
    
    def log_account_update(self, account_state: dict):
        """Log account state update."""
        self.logger.info(
            f"ACCOUNT | equity=${account_state.get('total_equity', 0):.2f} | "
            f"positions={account_state.get('open_positions', 0)} | "
            f"drawdown={account_state.get('current_drawdown', 0)*100:.2f}%"
        )
        
        self._log_json("account_update", account_state)
    
    def log_error(self, error_type: str, message: str, details: Optional[dict] = None):
        """Log errors."""
        self.logger.error(f"ERROR | {error_type} | {message}")
        
        self._log_json("error", {
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })
    
    def log_system_event(self, event: str, details: Optional[dict] = None):
        """Log system events."""
        self.logger.info(f"SYSTEM | {event}")
        
        self._log_json("system", {"event": event, **(details or {})})


class TradeJournal:
    """
    Maintains a detailed trade journal for analysis.
    """
    
    def __init__(self, journal_path: str = "logs/trade_journal.json"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades: List[dict] = []
        self._load_existing()
    
    def _load_existing(self):
        """Load existing journal entries."""
        if self.journal_path.exists():
            try:
                with open(self.journal_path, 'r') as f:
                    self.trades = json.load(f)
            except json.JSONDecodeError:
                self.trades = []
    
    def _save(self):
        """Save journal to file."""
        with open(self.journal_path, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def record_trade(
        self,
        trade_result: Any,
        signal_details: Optional[dict] = None,
        market_context: Optional[dict] = None,
        notes: str = ""
    ):
        """Record a completed trade with context."""
        trade_dict = trade_result.to_dict() if hasattr(trade_result, 'to_dict') else trade_result
        
        entry = {
            "recorded_at": datetime.utcnow().isoformat(),
            "trade": trade_dict,
            "signal_details": signal_details or {},
            "market_context": market_context or {},
            "notes": notes
        }
        
        self.trades.append(entry)
        self._save()
    
    def add_note_to_trade(self, trade_id: str, note: str):
        """Add a note to an existing trade."""
        for trade in self.trades:
            if trade.get("trade", {}).get("position_id") == trade_id:
                if "notes" not in trade:
                    trade["notes"] = ""
                trade["notes"] += f"\n[{datetime.utcnow().isoformat()}] {note}"
                self._save()
                return True
        return False
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[dict]:
        """Get filtered trade list."""
        filtered = self.trades
        
        if symbol:
            filtered = [t for t in filtered if t.get("trade", {}).get("symbol") == symbol]
        
        if start_date:
            filtered = [
                t for t in filtered 
                if datetime.fromisoformat(t.get("trade", {}).get("entry_time", "1970-01-01")) >= start_date
            ]
        
        if end_date:
            filtered = [
                t for t in filtered
                if datetime.fromisoformat(t.get("trade", {}).get("exit_time", "2100-01-01")) <= end_date
            ]
        
        return filtered
    
    def generate_summary(self) -> dict:
        """Generate summary statistics from journal."""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        winning = [t for t in self.trades if t.get("trade", {}).get("net_pnl", 0) > 0]
        losing = [t for t in self.trades if t.get("trade", {}).get("net_pnl", 0) <= 0]
        
        total_pnl = sum(t.get("trade", {}).get("net_pnl", 0) for t in self.trades)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring.
    """
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.metrics: Dict[str, List[float]] = {
            "execution_latency": [],
            "signal_processing_time": [],
            "data_fetch_time": [],
            "pnl_updates": []
        }
        self.counters: Dict[str, int] = {
            "signals_generated": 0,
            "trades_executed": 0,
            "orders_submitted": 0,
            "errors": 0
        }
    
    def record_latency(self, metric_name: str, value_ms: float):
        """Record a latency measurement."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value_ms)
        
        # Keep only last 1000 values
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def increment_counter(self, counter_name: str):
        """Increment a counter."""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += 1
    
    def get_summary(self) -> dict:
        """Get performance summary."""
        import statistics
        
        summary = {
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "counters": self.counters.copy(),
            "latencies": {}
        }
        
        for metric_name, values in self.metrics.items():
            if values:
                summary["latencies"][metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 20 else max(values),
                    "max": max(values)
                }
        
        return summary


def setup_logging(config) -> StructuredLogger:
    """Setup logging for the trading system."""
    return StructuredLogger(
        name="wick_reversal",
        log_dir="logs",
        log_level=config.log_level,
        json_output=True
    )
