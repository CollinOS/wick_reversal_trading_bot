"""
Backtesting Engine
Simulates strategy performance on historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import statistics
import math

from core.types import (
    Candle, MarketData, Signal, Position, TradeResult, 
    Side, PositionStatus, SignalType
)
from config.settings import StrategyConfig, BacktestConfig
from data.ingestion import DataAggregator, SimulatedDataProvider
from signals.detection import SignalGenerator
from risk.management import RiskManager


logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commission: float = 0.0
    
    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade stats
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_trade_duration_candles: float = 0.0
    
    # Per-side stats
    long_trades: int = 0
    long_wins: int = 0
    short_trades: int = 0
    short_wins: int = 0
    
    # Risk-adjusted
    avg_risk_reward_ratio: float = 0.0
    realized_rr_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "net_profit": round(self.net_profit, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "annualized_return_pct": round(self.annualized_return_pct * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "win_rate": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 4),
            "avg_trade_duration": round(self.avg_trade_duration_candles, 1),
        }


@dataclass
class EquityCurve:
    """Tracks equity over time."""
    timestamps: List[datetime] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    drawdown: List[float] = field(default_factory=list)
    
    def add_point(self, timestamp: datetime, equity_value: float, peak: float):
        self.timestamps.append(timestamp)
        self.equity.append(equity_value)
        dd = (peak - equity_value) / peak if peak > 0 else 0
        self.drawdown.append(dd)


class BacktestEngine:
    """
    Main backtesting engine.
    Simulates strategy execution on historical data.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_aggregator = DataAggregator(config)
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        
        # Results storage
        self.trades: List[TradeResult] = []
        self.signals: List[Signal] = []
        self.equity_curve = EquityCurve()

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.position_metadata: Dict[str, dict] = {}  # Store signal/market data per position
        self.current_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.candle_index: int = 0
    
    def run(
        self,
        historical_data: Dict[str, List[Candle]],
        btc_data: Optional[List[Candle]] = None
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.
        
        Args:
            historical_data: Dict mapping symbol to list of candles
            btc_data: Optional BTC candles for correlation filter
        
        Returns:
            BacktestMetrics with performance results
        """
        # Initialize
        initial_capital = self.config.backtest.initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.risk_manager.initialize(initial_capital)
        
        # Align data by timestamp
        all_timestamps = self._get_aligned_timestamps(historical_data)
        
        logger.info(f"Starting backtest with {len(all_timestamps)} candles")
        logger.info(f"Symbols: {list(historical_data.keys())}")
        
        # Main simulation loop
        for idx, timestamp in enumerate(all_timestamps):
            self.candle_index = idx
            
            # Get BTC price for filter
            btc_price = None
            if btc_data:
                btc_candle = self._get_candle_at_time(btc_data, timestamp)
                if btc_candle:
                    btc_price = btc_candle.close
            
            # Process each symbol
            for symbol, candles in historical_data.items():
                candle = self._get_candle_at_time(candles, timestamp)
                if not candle:
                    continue
                
                # Update existing positions
                self._update_positions(symbol, candle)
                
                # Check exits for open positions
                self._check_exits(symbol, candle, timestamp)
                
                # Generate market data
                market_data = self.data_aggregator.get_market_data(symbol, candle)
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    symbol, market_data, idx, btc_price
                )
                
                if signal.is_valid:
                    self.signals.append(signal)
                    self._process_signal(signal, candle, timestamp, market_data)
            
            # Update equity curve
            self._update_equity(timestamp)
        
        # Close any remaining positions at last price
        self._close_all_positions(all_timestamps[-1] if all_timestamps else datetime.utcnow())
        
        # Calculate metrics
        metrics = self._calculate_metrics(initial_capital, all_timestamps)
        
        return metrics
    
    def _get_aligned_timestamps(
        self,
        historical_data: Dict[str, List[Candle]]
    ) -> List[datetime]:
        """Get sorted unique timestamps across all symbols."""
        timestamps = set()
        for candles in historical_data.values():
            for candle in candles:
                timestamps.add(candle.timestamp)
        return sorted(timestamps)
    
    def _get_candle_at_time(
        self,
        candles: List[Candle],
        timestamp: datetime
    ) -> Optional[Candle]:
        """Get candle at specific timestamp."""
        # Binary search would be more efficient for large datasets
        for candle in candles:
            if candle.timestamp == timestamp:
                return candle
        return None
    
    def _update_positions(self, symbol: str, candle: Candle):
        """Update position metrics with current price."""
        if symbol in self.positions:
            position = self.positions[symbol]
            if position.is_open:
                position.update_unrealized_pnl(candle.close)
                position.current_candle_count = self.candle_index
                
                # Update trailing stop
                atr = self.data_aggregator.calculate_atr(symbol)
                if atr > 0:
                    position.update_trailing_stop(candle.close, atr, self.config.exit)
    
    def _check_exits(self, symbol: str, candle: Candle, timestamp: datetime):
        """Check and execute exits for open positions."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if not position.is_open:
            return
        
        exit_price = None
        exit_reason = None
        
        # Check stop loss (use low/high depending on side)
        if position.side == Side.LONG:
            # Stop hit if low <= stop price
            effective_stop = position.trailing_stop_price if position.trailing_stop_active else position.stop_loss
            if candle.low <= effective_stop:
                exit_price = effective_stop  # Assume stop is hit at stop price
                exit_reason = "trailing_stop" if position.trailing_stop_active else "stop_loss"
        else:
            # Stop hit if high >= stop price
            effective_stop = position.trailing_stop_price if position.trailing_stop_active else position.stop_loss
            if candle.high >= effective_stop:
                exit_price = effective_stop
                exit_reason = "trailing_stop" if position.trailing_stop_active else "stop_loss"
        
        # Check take profit
        if exit_price is None:
            if position.side == Side.LONG:
                if candle.high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:
                if candle.low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
        
        # Check time-based exit
        if exit_price is None and position.candles_held >= self.config.exit.max_hold_candles:
            exit_price = candle.close
            exit_reason = "time_exit"
        
        # Execute exit if triggered
        if exit_price is not None:
            self._close_position(position, exit_price, timestamp, exit_reason)
    
    def _process_signal(self, signal: Signal, candle: Candle, timestamp: datetime, market_data: Optional[MarketData] = None):
        """Process a valid trading signal."""
        # Check if we already have a position in this symbol
        if signal.symbol in self.positions and self.positions[signal.symbol].is_open:
            return  # Skip, already have position

        # Get symbol config
        symbol_config = None
        for sc in self.config.symbols:
            if sc.symbol == signal.symbol:
                symbol_config = {
                    'risk_multiplier': sc.risk_multiplier,
                    'min_position_size': sc.min_position_size,
                    'max_position_usd': sc.max_position_usd
                }
                break

        # Assess risk
        assessment = self.risk_manager.assess_trade(signal, candle.close, symbol_config)

        if not assessment.approved:
            logger.debug(f"Trade rejected: {assessment.reason}")
            return

        # Create position
        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            status=PositionStatus.OPEN,
            quantity=assessment.position_size,
            entry_price=signal.suggested_entry or candle.close,
            stop_loss=signal.suggested_stop,
            take_profit=signal.suggested_target,
            risk_amount=assessment.risk_amount,
            opened_at=timestamp,
            entry_candle_count=self.candle_index,
            current_candle_count=self.candle_index,
            signal_id=signal.id
        )

        # Calculate risk/reward ratio
        if signal.side == Side.LONG:
            risk = position.entry_price - position.stop_loss
            reward = position.take_profit - position.entry_price
        else:
            risk = position.stop_loss - position.entry_price
            reward = position.entry_price - position.take_profit

        position.risk_reward_ratio = reward / risk if risk > 0 else 0

        # Apply commission (entry)
        commission = position.entry_price * position.quantity * self.config.backtest.commission_pct
        position.total_commission = commission
        self.current_equity -= commission

        # Store position metadata for analysis
        leverage_mult, _ = self.risk_manager.calculate_leverage_multiplier(signal)

        # Parse criteria to identify which were met
        criteria_list = signal.criteria_met
        has_wtb = any('wtb_ratio' in c for c in criteria_list)
        has_atr = any('atr_ratio' in c for c in criteria_list)
        has_vwap = any('vwap_dist' in c for c in criteria_list)

        # Calculate wick size
        if signal.side == Side.LONG:
            wick_size = candle.lower_wick
        else:
            wick_size = candle.upper_wick
        wick_size_pct = (wick_size / candle.close * 100) if candle.close > 0 else 0

        # Price vs VWAP
        price_vs_vwap = 0.0
        if market_data and market_data.vwap > 0:
            price_vs_vwap = ((candle.close - market_data.vwap) / market_data.vwap) * 100

        self.position_metadata[position.id] = {
            'signal_strength': signal.strength,
            'signal_criteria': criteria_list,
            'entry_hour': timestamp.hour,
            'entry_day_of_week': timestamp.weekday(),
            'leverage_multiplier': leverage_mult,
            'num_criteria_met': len(criteria_list),
            'has_wtb_ratio': has_wtb,
            'has_atr_ratio': has_atr,
            'has_vwap_dist': has_vwap,
            'atr_at_entry': market_data.atr if market_data else 0.0,
            'vwap_at_entry': market_data.vwap if market_data else 0.0,
            'atr_ratio_at_entry': market_data.atr_ratio if market_data else 0.0,
            'volume_ratio_at_entry': market_data.volume_ratio if market_data else 0.0,
            'price_vs_vwap_pct': price_vs_vwap,
            'wick_size_pct': wick_size_pct,
        }

        # Store position
        self.positions[signal.symbol] = position
        self.risk_manager.register_position(position)

        # Record cooldown
        self.signal_generator.record_signal(signal.symbol, self.candle_index)

        logger.debug(
            f"Position opened: {signal.symbol} {signal.side.value} "
            f"@ {position.entry_price:.6f} qty={position.quantity:.6f} "
            f"leverage_mult={leverage_mult:.2f}"
        )
    
    def _close_position(
        self,
        position: Position,
        exit_price: float,
        timestamp: datetime,
        reason: str
    ):
        """Close a position and record the trade."""
        # Calculate P&L
        if position.side == Side.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity

        # Exit commission
        exit_commission = exit_price * position.quantity * self.config.backtest.commission_pct
        position.total_commission += exit_commission

        net_pnl = gross_pnl - position.total_commission

        # Update equity
        self.current_equity += gross_pnl - exit_commission

        # Get stored metadata
        metadata = self.position_metadata.get(position.id, {})

        # Create trade result
        duration = (timestamp - position.opened_at).total_seconds() if position.opened_at else 0

        trade = TradeResult(
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            entry_time=position.opened_at,
            exit_price=exit_price,
            exit_time=timestamp,
            exit_reason=reason,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            commission=position.total_commission,
            net_pnl=net_pnl,
            pnl_pct=net_pnl / (position.entry_price * position.quantity) if position.entry_price * position.quantity > 0 else 0,
            risk_amount=position.risk_amount,
            reward_risk_ratio=position.risk_reward_ratio,
            candles_held=position.candles_held,
            duration_seconds=int(duration),
            # Signal info from metadata
            signal_strength=metadata.get('signal_strength', 0.0),
            signal_criteria=metadata.get('signal_criteria', []),
            atr_at_entry=metadata.get('atr_at_entry', 0.0),
            vwap_at_entry=metadata.get('vwap_at_entry', 0.0),
            # Extended analysis fields
            entry_hour=metadata.get('entry_hour', 0),
            entry_day_of_week=metadata.get('entry_day_of_week', 0),
            leverage_multiplier=metadata.get('leverage_multiplier', 1.0),
            num_criteria_met=metadata.get('num_criteria_met', 0),
            has_wtb_ratio=metadata.get('has_wtb_ratio', False),
            has_atr_ratio=metadata.get('has_atr_ratio', False),
            has_vwap_dist=metadata.get('has_vwap_dist', False),
            atr_ratio_at_entry=metadata.get('atr_ratio_at_entry', 0.0),
            volume_ratio_at_entry=metadata.get('volume_ratio_at_entry', 0.0),
            price_vs_vwap_pct=metadata.get('price_vs_vwap_pct', 0.0),
            wick_size_pct=metadata.get('wick_size_pct', 0.0),
        )

        self.trades.append(trade)

        # Clean up metadata
        if position.id in self.position_metadata:
            del self.position_metadata[position.id]

        # Update position status
        position.status = PositionStatus.CLOSED
        position.realized_pnl = gross_pnl
        position.closed_at = timestamp

        # Update risk manager
        self.risk_manager.close_position(position, exit_price, timestamp)

        logger.debug(
            f"Position closed: {position.symbol} {reason} "
            f"@ {exit_price:.6f} PnL=${net_pnl:.2f}"
        )
    
    def _close_all_positions(self, timestamp: datetime):
        """Close all remaining positions at end of backtest."""
        for symbol, position in self.positions.items():
            if position.is_open:
                # Use last available price
                candles = self.data_aggregator.candle_buffers.get(symbol, [])
                if candles:
                    last_price = list(candles)[-1].close
                    self._close_position(position, last_price, timestamp, "end_of_test")
    
    def _update_equity(self, timestamp: datetime):
        """Update equity curve."""
        # Calculate total equity including unrealized P&L
        unrealized = sum(
            p.unrealized_pnl for p in self.positions.values() if p.is_open
        )
        total_equity = self.current_equity + unrealized
        
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        self.equity_curve.add_point(timestamp, total_equity, self.peak_equity)
    
    def _calculate_metrics(
        self,
        initial_capital: float,
        timestamps: List[datetime]
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()
        
        if not self.trades:
            return metrics
        
        # Basic counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.net_pnl <= 0)
        
        # P&L
        winning_pnls = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losing_pnls = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        
        metrics.gross_profit = sum(winning_pnls) if winning_pnls else 0
        metrics.gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        metrics.net_profit = self.current_equity - initial_capital
        metrics.total_commission = sum(t.commission for t in self.trades)
        
        # Returns
        metrics.total_return_pct = metrics.net_profit / initial_capital
        
        if timestamps and len(timestamps) > 1:
            days = (timestamps[-1] - timestamps[0]).days
            if days > 0:
                years = days / 365.25
                if metrics.total_return_pct > -1:
                    metrics.annualized_return_pct = (1 + metrics.total_return_pct) ** (1/years) - 1
        
        # Drawdown
        if self.equity_curve.drawdown:
            metrics.max_drawdown_pct = max(self.equity_curve.drawdown)
        
        # Win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # Average win/loss
        metrics.avg_win = statistics.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = statistics.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win + 
            (1 - metrics.win_rate) * metrics.avg_loss
        )
        
        # Average duration
        durations = [t.candles_held for t in self.trades]
        metrics.avg_trade_duration_candles = statistics.mean(durations)
        
        # Per-side stats
        long_trades = [t for t in self.trades if t.side == Side.LONG]
        short_trades = [t for t in self.trades if t.side == Side.SHORT]
        
        metrics.long_trades = len(long_trades)
        metrics.long_wins = sum(1 for t in long_trades if t.net_pnl > 0)
        metrics.short_trades = len(short_trades)
        metrics.short_wins = sum(1 for t in short_trades if t.net_pnl > 0)
        
        # Risk metrics
        metrics.avg_risk_reward_ratio = statistics.mean([t.reward_risk_ratio for t in self.trades])
        
        # Calculate Sharpe ratio
        if len(self.equity_curve.equity) > 1:
            returns = []
            for i in range(1, len(self.equity_curve.equity)):
                ret = (self.equity_curve.equity[i] - self.equity_curve.equity[i-1]) / self.equity_curve.equity[i-1]
                returns.append(ret)
            
            if returns and statistics.stdev(returns) > 0:
                # Annualize (assuming 5-minute candles, ~105,120 candles/year)
                annual_factor = math.sqrt(105120)
                metrics.sharpe_ratio = (statistics.mean(returns) / statistics.stdev(returns)) * annual_factor
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            if len(negative_returns) >= 2 and statistics.stdev(negative_returns) > 0:
                metrics.sortino_ratio = (statistics.mean(returns) / statistics.stdev(negative_returns)) * annual_factor
        
        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct
        
        return metrics
    
    def get_trade_log(self) -> List[dict]:
        """Get detailed trade log."""
        return [t.to_dict() for t in self.trades]
    
    def get_signal_log(self) -> List[dict]:
        """Get signal log."""
        return [s.to_dict() for s in self.signals]
    
    def get_equity_curve_data(self) -> dict:
        """Get equity curve data for plotting."""
        return {
            "timestamps": [t.isoformat() for t in self.equity_curve.timestamps],
            "equity": self.equity_curve.equity,
            "drawdown": self.equity_curve.drawdown
        }


def export_trades_to_csv(trades: List[TradeResult], filepath: str):
    """Export trades to CSV for external analysis."""
    import csv

    if not trades:
        logger.warning("No trades to export")
        return

    fieldnames = list(trades[0].to_dict().keys())

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade.to_dict())

    logger.info(f"Exported {len(trades)} trades to {filepath}")


def analyze_trades_by_dimension(trades: List[TradeResult]) -> dict:
    """
    Analyze trade performance across multiple dimensions.
    Returns insights about what factors correlate with winning trades.
    """
    if not trades:
        return {}

    results = {
        'by_symbol': {},
        'by_hour': {},
        'by_day_of_week': {},
        'by_criteria_count': {},
        'by_criteria_type': {
            'wtb_ratio': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
            'atr_ratio': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
            'vwap_dist': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        },
        'by_leverage_multiplier': {},
        'by_exit_reason': {},
        'by_side': {'long': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                    'short': {'wins': 0, 'losses': 0, 'total_pnl': 0.0}},
    }

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for trade in trades:
        is_win = trade.is_winner
        pnl = trade.net_pnl

        # By symbol
        if trade.symbol not in results['by_symbol']:
            results['by_symbol'][trade.symbol] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_symbol'][trade.symbol]['wins' if is_win else 'losses'] += 1
        results['by_symbol'][trade.symbol]['total_pnl'] += pnl

        # By hour
        hour = trade.entry_hour
        if hour not in results['by_hour']:
            results['by_hour'][hour] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_hour'][hour]['wins' if is_win else 'losses'] += 1
        results['by_hour'][hour]['total_pnl'] += pnl

        # By day of week
        day = day_names[trade.entry_day_of_week]
        if day not in results['by_day_of_week']:
            results['by_day_of_week'][day] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_day_of_week'][day]['wins' if is_win else 'losses'] += 1
        results['by_day_of_week'][day]['total_pnl'] += pnl

        # By criteria count
        count = trade.num_criteria_met
        if count not in results['by_criteria_count']:
            results['by_criteria_count'][count] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_criteria_count'][count]['wins' if is_win else 'losses'] += 1
        results['by_criteria_count'][count]['total_pnl'] += pnl

        # By criteria type
        if trade.has_wtb_ratio:
            results['by_criteria_type']['wtb_ratio']['wins' if is_win else 'losses'] += 1
            results['by_criteria_type']['wtb_ratio']['total_pnl'] += pnl
        if trade.has_atr_ratio:
            results['by_criteria_type']['atr_ratio']['wins' if is_win else 'losses'] += 1
            results['by_criteria_type']['atr_ratio']['total_pnl'] += pnl
        if trade.has_vwap_dist:
            results['by_criteria_type']['vwap_dist']['wins' if is_win else 'losses'] += 1
            results['by_criteria_type']['vwap_dist']['total_pnl'] += pnl

        # By leverage multiplier (bucketed)
        lev_bucket = round(trade.leverage_multiplier, 1)
        if lev_bucket not in results['by_leverage_multiplier']:
            results['by_leverage_multiplier'][lev_bucket] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_leverage_multiplier'][lev_bucket]['wins' if is_win else 'losses'] += 1
        results['by_leverage_multiplier'][lev_bucket]['total_pnl'] += pnl

        # By exit reason
        reason = trade.exit_reason
        if reason not in results['by_exit_reason']:
            results['by_exit_reason'][reason] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        results['by_exit_reason'][reason]['wins' if is_win else 'losses'] += 1
        results['by_exit_reason'][reason]['total_pnl'] += pnl

        # By side
        side = trade.side.value
        results['by_side'][side]['wins' if is_win else 'losses'] += 1
        results['by_side'][side]['total_pnl'] += pnl

    # Calculate win rates for each dimension
    for dimension in results.values():
        if isinstance(dimension, dict):
            for key, stats in dimension.items():
                if isinstance(stats, dict) and 'wins' in stats and 'losses' in stats:
                    total = stats['wins'] + stats['losses']
                    stats['total_trades'] = total
                    stats['win_rate'] = (stats['wins'] / total * 100) if total > 0 else 0
                    stats['avg_pnl'] = stats['total_pnl'] / total if total > 0 else 0

    return results


def print_trade_analysis(analysis: dict):
    """Print formatted trade analysis results."""
    print("\n" + "=" * 60)
    print("DETAILED TRADE ANALYSIS")
    print("=" * 60)

    # By Symbol
    print("\n--- Performance by Symbol ---")
    for symbol, stats in sorted(analysis.get('by_symbol', {}).items()):
        print(f"  {symbol}: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    # By Day of Week
    print("\n--- Performance by Day of Week ---")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in day_order:
        if day in analysis.get('by_day_of_week', {}):
            stats = analysis['by_day_of_week'][day]
            print(f"  {day}: {stats['total_trades']} trades, "
                  f"{stats['win_rate']:.1f}% win rate, "
                  f"${stats['total_pnl']:.2f} total PnL")

    # By Hour (top 5 best and worst)
    print("\n--- Best Hours (by win rate, min 10 trades) ---")
    hours = [(h, s) for h, s in analysis.get('by_hour', {}).items() if s['total_trades'] >= 10]
    hours_sorted = sorted(hours, key=lambda x: x[1]['win_rate'], reverse=True)
    for hour, stats in hours_sorted[:5]:
        print(f"  Hour {hour:02d}: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    print("\n--- Worst Hours (by win rate, min 10 trades) ---")
    for hour, stats in hours_sorted[-5:]:
        print(f"  Hour {hour:02d}: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    # By Criteria Count
    print("\n--- Performance by Number of Criteria Met ---")
    for count, stats in sorted(analysis.get('by_criteria_count', {}).items()):
        print(f"  {count} criteria: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    # By Criteria Type
    print("\n--- Performance by Criteria Type ---")
    for criteria, stats in analysis.get('by_criteria_type', {}).items():
        if stats['total_trades'] > 0:
            print(f"  {criteria}: {stats['total_trades']} trades, "
                  f"{stats['win_rate']:.1f}% win rate, "
                  f"${stats['total_pnl']:.2f} total PnL")

    # By Leverage Multiplier
    print("\n--- Performance by Leverage Multiplier ---")
    for mult, stats in sorted(analysis.get('by_leverage_multiplier', {}).items()):
        print(f"  {mult:.1f}x: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    # By Exit Reason
    print("\n--- Performance by Exit Reason ---")
    for reason, stats in sorted(analysis.get('by_exit_reason', {}).items()):
        print(f"  {reason}: {stats['total_trades']} trades, "
              f"{stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:.2f} total PnL")

    # By Side
    print("\n--- Performance by Side ---")
    for side, stats in analysis.get('by_side', {}).items():
        if stats['total_trades'] > 0:
            print(f"  {side.upper()}: {stats['total_trades']} trades, "
                  f"{stats['win_rate']:.1f}% win rate, "
                  f"${stats['total_pnl']:.2f} total PnL")

    print("\n" + "=" * 60)


def print_trade_details(trades: List[TradeResult], output_file: Optional[str] = None):
    """Print detailed information for each individual trade.

    Args:
        trades: List of TradeResult objects
        output_file: Optional file path to write output (prints to console if None)
    """
    import sys

    if not trades:
        print("\nNo trades to display.")
        return

    # Use file or stdout
    if output_file:
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = sys.stdout

    def out(text=""):
        print(text, file=f)

    out("=" * 80)
    out("INDIVIDUAL TRADE DETAILS")
    out("=" * 80)

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for i, trade in enumerate(trades, 1):
        result = "WIN" if trade.is_winner else "LOSS"

        out(f"\n{'-' * 80}")
        out(f"Trade #{i} | {trade.symbol} | {trade.side.value.upper()} | [{result}]")
        out(f"{'-' * 80}")

        # Timing
        entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M") if trade.entry_time else "N/A"
        exit_time = trade.exit_time.strftime("%Y-%m-%d %H:%M") if trade.exit_time else "N/A"
        day_name = day_names[trade.entry_day_of_week]
        out(f"  Entry: {entry_time} ({day_name}, Hour {trade.entry_hour:02d})")
        out(f"  Exit:  {exit_time} ({trade.exit_reason})")
        out(f"  Duration: {trade.candles_held} candles ({trade.duration_seconds // 60} minutes)")

        # Price levels
        out(f"\n  Prices:")
        out(f"    Entry:       ${trade.entry_price:.6f}")
        out(f"    Exit:        ${trade.exit_price:.6f}")
        pct_move = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
        if trade.side == Side.SHORT:
            pct_move = -pct_move
        out(f"    Move:        {pct_move:+.3f}%")

        # P&L
        out(f"\n  P&L:")
        out(f"    Gross:       ${trade.gross_pnl:+.2f}")
        out(f"    Commission:  ${trade.commission:.2f}")
        out(f"    Net:         ${trade.net_pnl:+.2f} ({trade.pnl_pct*100:+.2f}%)")
        out(f"    Position:    {trade.quantity:.6f} units")

        # Risk metrics
        out(f"\n  Risk Metrics:")
        out(f"    Risk Amount: ${trade.risk_amount:.2f}")
        out(f"    R:R Ratio:   {trade.reward_risk_ratio:.2f}")
        r_multiple = trade.net_pnl / trade.risk_amount if trade.risk_amount > 0 else 0
        out(f"    R-Multiple:  {r_multiple:+.2f}R")

        # Signal details
        out(f"\n  Signal Details:")
        out(f"    Strength:    {trade.signal_strength:.3f}")
        out(f"    Leverage:    {trade.leverage_multiplier:.2f}x")
        out(f"    Criteria ({trade.num_criteria_met}):")
        for criterion in trade.signal_criteria:
            out(f"      - {criterion}")

        # Criteria flags
        criteria_flags = []
        if trade.has_wtb_ratio:
            criteria_flags.append("WTB")
        if trade.has_atr_ratio:
            criteria_flags.append("ATR")
        if trade.has_vwap_dist:
            criteria_flags.append("VWAP")
        out(f"    Flags:       [{', '.join(criteria_flags) if criteria_flags else 'None'}]")

        # Market context
        out(f"\n  Market Context at Entry:")
        out(f"    ATR:         {trade.atr_at_entry:.6f}")
        out(f"    ATR Ratio:   {trade.atr_ratio_at_entry:.2f}x (vs baseline)")
        out(f"    VWAP:        ${trade.vwap_at_entry:.6f}")
        out(f"    Price/VWAP:  {trade.price_vs_vwap_pct:+.2f}%")
        out(f"    Vol Ratio:   {trade.volume_ratio_at_entry:.2f}x")
        out(f"    Wick Size:   {trade.wick_size_pct:.3f}%")

    # Summary footer
    out(f"\n{'=' * 80}")
    total_pnl = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.is_winner)
    out(f"Total: {len(trades)} trades | {wins} wins ({wins/len(trades)*100:.1f}%) | Net P&L: ${total_pnl:+.2f}")
    out("=" * 80)

    if output_file:
        f.close()


def run_monte_carlo_analysis(
    trades: List[TradeResult],
    initial_capital: float,
    num_simulations: int = 1000
) -> dict:
    """
    Run Monte Carlo simulation to assess strategy robustness.
    Shuffles trade order to see distribution of outcomes.
    """
    import random
    
    if not trades:
        return {}
    
    final_equities = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
        # Shuffle trades
        shuffled = trades.copy()
        random.shuffle(shuffled)
        
        # Simulate equity curve
        equity = initial_capital
        peak = initial_capital
        max_dd = 0
        
        for trade in shuffled:
            equity += trade.net_pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd)
    
    return {
        "final_equity": {
            "mean": statistics.mean(final_equities),
            "median": statistics.median(final_equities),
            "std": statistics.stdev(final_equities),
            "percentile_5": sorted(final_equities)[int(0.05 * len(final_equities))],
            "percentile_95": sorted(final_equities)[int(0.95 * len(final_equities))]
        },
        "max_drawdown": {
            "mean": statistics.mean(max_drawdowns),
            "median": statistics.median(max_drawdowns),
            "percentile_95": sorted(max_drawdowns)[int(0.95 * len(max_drawdowns))]
        }
    }
