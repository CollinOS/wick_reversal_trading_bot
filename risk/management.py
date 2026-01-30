"""
Risk Management Module
Handles position sizing, exposure limits, and risk controls.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.types import (
    Signal, Position, Order, AccountState, Side, PositionStatus
)
from config.settings import StrategyConfig, RiskConfig, DynamicLeverageConfig


logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Result of risk assessment for a potential trade."""
    approved: bool
    reason: str = ""
    position_size: float = 0.0
    risk_amount: float = 0.0
    leverage_used: float = 0.0
    base_position_size: float = 0.0  # Position size before dynamic multiplier
    dynamic_multiplier: float = 1.0  # The dynamic leverage multiplier applied

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "position_size": self.position_size,
            "risk_amount": self.risk_amount,
            "leverage_used": self.leverage_used,
            "base_position_size": self.base_position_size,
            "dynamic_multiplier": self.dynamic_multiplier
        }


class PositionSizer:
    """Calculates appropriate position sizes based on risk parameters."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        side: Side,
        symbol_config: Optional[dict] = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on fixed fractional risk.
        
        Returns: (position_size, risk_amount)
        """
        risk_pct = self.config.risk_per_trade_pct
        if symbol_config and 'risk_multiplier' in symbol_config:
            risk_pct *= symbol_config['risk_multiplier']
        
        risk_amount = account_equity * risk_pct
        
        if side == Side.LONG:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss placement")
            return 0.0, 0.0
        
        position_size = risk_amount / risk_per_unit
        
        # Apply leverage constraint
        max_position_value = account_equity * self.config.max_leverage
        position_value = position_size * entry_price
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            risk_amount = position_size * risk_per_unit
        
        # Apply symbol constraints
        if symbol_config:
            if 'min_position_size' in symbol_config:
                if position_size < symbol_config['min_position_size']:
                    return 0.0, 0.0
            # Cap at base_position_usd (baseline for normal confidence trades)
            # The leverage multiplier will scale this up for confident trades
            if 'base_position_usd' in symbol_config:
                base_size = symbol_config['base_position_usd'] / entry_price
                if position_size > base_size:
                    position_size = base_size
                    risk_amount = position_size * risk_per_unit

        return position_size, risk_amount
    
    def calculate_leverage(
        self,
        position_size: float,
        entry_price: float,
        account_equity: float
    ) -> float:
        """Calculate effective leverage used."""
        position_value = position_size * entry_price
        if account_equity <= 0:
            return 0.0
        return position_value / account_equity


class ExposureManager:
    """Manages overall portfolio exposure and position limits."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
    
    def add_position(self, position: Position):
        self.positions[position.id] = position
    
    def remove_position(self, position_id: str):
        if position_id in self.positions:
            del self.positions[position_id]
    
    def update_position(self, position: Position):
        self.positions[position.id] = position
    
    def get_open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.is_open]
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        return [p for p in self.positions.values() if p.symbol == symbol and p.is_open]
    
    def count_open_positions(self) -> int:
        return len(self.get_open_positions())
    
    def count_symbol_positions(self, symbol: str) -> int:
        return len(self.get_positions_by_symbol(symbol))
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        if self.count_open_positions() >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        if self.count_symbol_positions(symbol) >= self.config.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached"
        return True, ""
    
    def get_total_exposure(self, current_prices: Dict[str, float]) -> float:
        total = 0.0
        for position in self.get_open_positions():
            price = current_prices.get(position.symbol, position.entry_price)
            total += position.quantity * price
        return total
    
    def get_net_exposure(self, current_prices: Dict[str, float]) -> float:
        net = 0.0
        for position in self.get_open_positions():
            price = current_prices.get(position.symbol, position.entry_price)
            value = position.quantity * price
            net += value if position.side == Side.LONG else -value
        return net


class DrawdownMonitor:
    """Monitors and controls drawdown levels."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.daily_start_date: Optional[datetime] = None
    
    def update(self, current_equity: float, timestamp: datetime):
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.daily_start_date is None or timestamp.date() > self.daily_start_date.date():
            self.daily_start_equity = current_equity
            self.daily_start_date = timestamp
    
    def get_current_drawdown(self, current_equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - current_equity) / self.peak_equity
    
    def get_daily_pnl(self, current_equity: float) -> float:
        return current_equity - self.daily_start_equity
    
    def get_daily_pnl_pct(self, current_equity: float) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return self.get_daily_pnl(current_equity) / self.daily_start_equity
    
    def check_limits(self, current_equity: float) -> Tuple[bool, str]:
        current_dd = self.get_current_drawdown(current_equity)
        if current_dd >= self.config.max_drawdown_pct:
            return False, f"Max drawdown reached: {current_dd*100:.1f}%"
        daily_pnl_pct = self.get_daily_pnl_pct(current_equity)
        if daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            return False, f"Daily loss limit reached: {daily_pnl_pct*100:.1f}%"
        return True, ""


class RiskManager:
    """Main risk management engine."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position_sizer = PositionSizer(config.risk)
        self.exposure_manager = ExposureManager(config.risk)
        self.drawdown_monitor = DrawdownMonitor(config.risk)
        self.recent_trades: Dict[str, datetime] = {}
        self.account_state = AccountState()

    def calculate_leverage_multiplier(self, signal: Signal) -> Tuple[float, dict]:
        """
        Calculate dynamic leverage multiplier based on signal quality.

        Higher confidence signals (more criteria met, higher strength) get more leverage.
        ATR ratio criterion gets extra bonus due to 81% historical win rate.

        Returns: (multiplier, breakdown_dict)
        """
        dl_config = self.config.dynamic_leverage

        breakdown = {
            "base": dl_config.base_multiplier,
            "criteria_bonus": 0.0,
            "strength_bonus": 0.0,
            "atr_ratio_bonus": 0.0,
            "criteria_met": signal.criteria_met,
            "signal_strength": signal.strength,
            "num_criteria": len(signal.criteria_met),
        }

        if not dl_config.enabled:
            breakdown["final"] = 1.0
            return 1.0, breakdown

        multiplier = dl_config.base_multiplier

        # Check if ATR ratio criterion is met (high-value signal)
        has_atr_ratio = any('atr_ratio' in c for c in signal.criteria_met)
        breakdown["has_atr_ratio"] = has_atr_ratio

        # Only apply bonuses if signal meets minimum strength threshold
        if signal.strength >= dl_config.min_strength_for_bonus:
            # Bonus for multiple criteria met (beyond the first)
            # criteria_met includes rejection + optional conditions
            num_criteria = len(signal.criteria_met)
            if num_criteria > 1:
                criteria_bonus = (num_criteria - 1) * dl_config.per_criteria_bonus
                multiplier += criteria_bonus
                breakdown["criteria_bonus"] = criteria_bonus

            # Bonus for high-strength signals
            if signal.strength > 0.6:
                multiplier += dl_config.high_strength_bonus
                breakdown["strength_bonus"] = dl_config.high_strength_bonus

            # Bonus for ATR ratio criterion (historically 81% win rate)
            if has_atr_ratio:
                multiplier += dl_config.atr_ratio_bonus
                breakdown["atr_ratio_bonus"] = dl_config.atr_ratio_bonus

        # Clamp to min/max bounds
        pre_clamp = multiplier
        multiplier = max(dl_config.min_multiplier, min(multiplier, dl_config.max_multiplier))
        breakdown["pre_clamp"] = pre_clamp
        breakdown["final"] = multiplier

        return multiplier, breakdown
    
    def initialize(self, initial_equity: float):
        self.account_state.total_equity = initial_equity
        self.account_state.available_balance = initial_equity
        self.account_state.starting_equity = initial_equity
        self.account_state.peak_equity = initial_equity
        self.account_state.daily_starting_equity = initial_equity
        self.drawdown_monitor.peak_equity = initial_equity
        self.drawdown_monitor.daily_start_equity = initial_equity
        self.drawdown_monitor.daily_start_date = datetime.utcnow()
    
    def update_account(self, equity: float, available_balance: float, timestamp: datetime):
        self.account_state.total_equity = equity
        self.account_state.available_balance = available_balance
        self.account_state.timestamp = timestamp
        self.account_state.update_drawdown()
        self.drawdown_monitor.update(equity, timestamp)
        self.account_state.open_positions = self.exposure_manager.count_open_positions()
    
    def assess_trade(
        self,
        signal: Signal,
        current_price: float,
        symbol_config: Optional[dict] = None
    ) -> RiskAssessment:
        assessment = RiskAssessment(approved=False)
        
        if not signal.is_valid:
            assessment.reason = "Invalid signal"
            return assessment
        
        within_limits, reason = self.drawdown_monitor.check_limits(self.account_state.total_equity)
        if not within_limits:
            assessment.reason = reason
            return assessment
        
        can_open, reason = self.exposure_manager.can_open_position(signal.symbol)
        if not can_open:
            assessment.reason = reason
            return assessment
        
        existing_positions = self.exposure_manager.get_positions_by_symbol(signal.symbol)
        if existing_positions and not self.config.risk.allow_averaging:
            for pos in existing_positions:
                if pos.side == signal.side:
                    assessment.reason = "No averaging down allowed"
                    return assessment
        
        entry_price = signal.suggested_entry or current_price
        stop_loss = signal.suggested_stop

        if stop_loss is None:
            assessment.reason = "No stop loss defined"
            return assessment

        # Calculate dynamic leverage multiplier based on signal quality
        leverage_multiplier, leverage_breakdown = self.calculate_leverage_multiplier(signal)

        base_position_size, base_risk_amount = self.position_sizer.calculate_position_size(
            self.account_state.total_equity, entry_price, stop_loss, signal.side, symbol_config
        )

        # Apply dynamic leverage multiplier to position size
        position_size = base_position_size * leverage_multiplier
        risk_amount = base_risk_amount * leverage_multiplier

        # Track if position was capped
        was_capped = False
        cap_reason = ""

        # Cap at max_position_usd (absolute maximum for highest confidence trades)
        if symbol_config and 'max_position_usd' in symbol_config:
            max_size = symbol_config['max_position_usd'] / entry_price
            if position_size > max_size:
                was_capped = True
                cap_reason = f"max_position_usd (${symbol_config['max_position_usd']})"
                position_size = max_size
                # Recalculate risk amount based on capped position
                if signal.side == Side.LONG:
                    risk_per_unit = entry_price - stop_loss
                else:
                    risk_per_unit = stop_loss - entry_price
                risk_amount = position_size * risk_per_unit

        if position_size <= 0:
            assessment.reason = "Position size calculation failed"
            return assessment

        leverage = self.position_sizer.calculate_leverage(
            position_size, entry_price, self.account_state.total_equity
        )

        if leverage > self.config.risk.max_leverage:
            assessment.reason = f"Leverage {leverage:.1f}x exceeds max"
            return assessment

        assessment.approved = True
        assessment.position_size = position_size
        assessment.risk_amount = risk_amount
        assessment.leverage_used = leverage
        assessment.base_position_size = base_position_size
        assessment.dynamic_multiplier = leverage_multiplier

        # Store breakdown for external logging
        assessment.leverage_breakdown = leverage_breakdown
        assessment.was_capped = was_capped
        assessment.cap_reason = cap_reason

        # Log to file (concise format)
        position_value = position_size * entry_price
        base_value = base_position_size * entry_price
        logger.info(
            f"Trade approved: {signal.symbol} {signal.side.value.upper()} | "
            f"base=${base_value:.2f} x{leverage_multiplier:.2f} = ${position_value:.2f} | "
            f"leverage={leverage:.2f}x | risk=${risk_amount:.2f} | "
            f"criteria={','.join(leverage_breakdown['criteria_met'])}"
        )
        return assessment
    
    def register_position(self, position: Position):
        self.exposure_manager.add_position(position)
        self.recent_trades[position.symbol] = position.opened_at or datetime.utcnow()
        self.account_state.margin_used += position.quantity * position.entry_price
        self.account_state.open_positions = self.exposure_manager.count_open_positions()
    
    def close_position(self, position: Position, exit_price: float, exit_time: datetime):
        if position.side == Side.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        position.realized_pnl = gross_pnl
        position.closed_at = exit_time
        position.status = PositionStatus.CLOSED
        
        self.account_state.margin_used -= position.quantity * position.entry_price
        self.account_state.total_trades += 1
        self.account_state.daily_trades += 1
        
        if gross_pnl > 0:
            self.account_state.winning_trades += 1
        else:
            self.account_state.losing_trades += 1
        
        self.exposure_manager.remove_position(position.id)
        self.account_state.open_positions = self.exposure_manager.count_open_positions()
        
        logger.info(f"Position closed: {position.symbol} PnL=${gross_pnl:.2f}")
    
    def update_positions(self, current_prices: Dict[str, float], current_atr: Dict[str, float]):
        for position in self.exposure_manager.get_open_positions():
            if position.symbol in current_prices:
                price = current_prices[position.symbol]
                atr = current_atr.get(position.symbol, 0)
                position.update_unrealized_pnl(price)
                if atr > 0:
                    position.update_trailing_stop(price, atr, self.config.exit)
    
    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        if position.trailing_stop_active and position.trailing_stop_price:
            stop = position.trailing_stop_price
        else:
            stop = position.stop_loss
        
        if position.side == Side.LONG:
            return current_price <= stop
        return current_price >= stop
    
    def check_take_profit(self, position: Position, current_price: float) -> bool:
        if position.side == Side.LONG:
            return current_price >= position.take_profit
        return current_price <= position.take_profit
    
    def check_time_exit(self, position: Position) -> bool:
        return position.candles_held >= self.config.exit.max_hold_candles
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> dict:
        positions = self.exposure_manager.get_open_positions()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": self.account_state.total_equity,
            "available_balance": self.account_state.available_balance,
            "open_positions": len(positions),
            "total_exposure": self.exposure_manager.get_total_exposure(current_prices),
            "net_exposure": self.exposure_manager.get_net_exposure(current_prices),
            "unrealized_pnl": sum(p.unrealized_pnl for p in positions),
            "current_drawdown": self.drawdown_monitor.get_current_drawdown(self.account_state.total_equity),
            "daily_pnl": self.drawdown_monitor.get_daily_pnl(self.account_state.total_equity),
            "win_rate": self.account_state.win_rate,
            "total_trades": self.account_state.total_trades,
        }
