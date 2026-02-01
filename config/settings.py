"""
Wick Reversal Strategy - Configuration Settings
All strategy parameters are centralized here for easy tuning and backtesting.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class TimeFrame(Enum):
    """Supported timeframes for candlestick data."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"


class ExitTarget(Enum):
    """Available take-profit target types."""
    VWAP = "vwap"
    CANDLE_OPEN = "candle_open"
    WICK_MIDPOINT = "wick_midpoint"
    ATR_BASED = "atr_based"


class EntryMode(Enum):
    """Entry execution modes."""
    CANDLE_CLOSE = "candle_close"
    RETRACE_TO_MIDPOINT = "retrace_to_midpoint"


@dataclass
class SignalConfig:
    """Configuration for wick signal detection."""
    # Wick-to-body ratio threshold (e.g., 2.0 means wick >= 2x body)
    wick_to_body_ratio: float = 2.3  # Middle ground (was 2.2 -> tested up to 2.5)

    # Wick size relative to ATR (e.g., 1.5 means wick > 1.5 * ATR)
    wick_atr_multiplier: float = 1.5

    # Distance from VWAP threshold (in ATR units)
    vwap_distance_atr: float = 1.0

    # Minimum wick size in absolute terms (as % of price)
    min_wick_pct: float = 0.008  # 0.8% - filters out small noise wicks

    # Body confirmation: close must be at least this % away from wick extreme
    rejection_threshold_pct: float = 0.42  # Middle ground (was 0.35)

    # Use all conditions (AND) or any condition (OR)
    require_all_conditions: bool = False  # OR logic - allows more trades

    # ATR period for calculations
    atr_period: int = 14

    # VWAP type: 'session' or 'rolling'
    vwap_type: str = "rolling"
    vwap_rolling_period: int = 20


@dataclass
class EntryConfig:
    """Configuration for trade entries."""
    mode: EntryMode = EntryMode.RETRACE_TO_MIDPOINT
    
    # For retrace entry: wait for price to retrace this % toward wick midpoint
    retrace_pct: float = 0.6
    
    # Maximum time to wait for retrace entry (in candles)
    max_retrace_wait_candles: int = 2
    
    # Use limit orders for entry (False = market orders for reliable fills)
    use_limit_orders: bool = False
    
    # Limit order offset from target price (as % of ATR)
    limit_order_offset_atr: float = 0.03


@dataclass
class ExitConfig:
    """Configuration for trade exits."""
    # Primary take-profit target
    primary_target: ExitTarget = ExitTarget.ATR_BASED

    # Stop loss buffer beyond wick extreme (as % of ATR)
    stop_loss_buffer_atr: float = 0.4

    # Maximum stop loss as % of entry price
    max_stop_loss_pct: float = 0.025  # 2.5%

    # For ATR-based targets: multiplier
    atr_target_multiplier: float = 0.9  # Middle ground (was 1.2)

    # Time-based exit: max candles to hold position
    max_hold_candles: int = 12

    # Trailing stop activation (% of target reached)
    trailing_stop_activation: float = 0.75  # Middle ground (was 0.90)
    trailing_stop_distance_atr: float = 0.5  # Tighter trailing stop

    # Partial take profit settings
    partial_tp_enabled: bool = True
    partial_tp_percent: float = 0.50  # Take 50% of position at first target
    partial_tp_atr_multiplier: float = 0.5  # First target at 0.5x ATR (closer than full TP)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Fixed fractional risk per trade (as % of account)
    risk_per_trade_pct: float = 0.02  # 1.5% per trade (reduced for more positions)

    # Maximum simultaneous positions
    max_positions: int = 10  # Increased to capitalize on high volatility periods

    # Maximum positions per symbol
    max_positions_per_symbol: int = 1

    # Cooldown period after trade (in candles)
    cooldown_candles: int = 0

    # Daily loss limit (as % of starting daily equity)
    daily_loss_limit_pct: float = 0.05  # 5% daily loss limit (increased for more positions)

    # Maximum drawdown before stopping (as % of peak equity)
    max_drawdown_pct: float = 0.15  # 15% max drawdown

    # No averaging down
    allow_averaging: bool = False
    
    # Maximum leverage
    max_leverage: float = 5.0


@dataclass
class DynamicLeverageConfig:
    """Configuration for signal-based leverage adjustment."""
    enabled: bool = True

    # Base leverage multiplier (applied to all trades)
    base_multiplier: float = 1.0

    # Bonus multiplier per criteria met beyond the first
    # e.g., if 3 criteria met and bonus=0.25, multiplier = 1.0 + (2 * 0.25) = 1.5
    per_criteria_bonus: float = 0.25

    # Minimum signal strength to receive any bonus
    min_strength_for_bonus: float = 0.4

    # Bonus multiplier for high-strength signals (strength > 0.6)
    high_strength_bonus: float = 0.3

    # Bonus multiplier when ATR ratio criterion is met (81% win rate in backtest)
    atr_ratio_bonus: float = 0.4

    # Maximum total multiplier (caps the leverage boost)
    max_multiplier: float = 2.5

    # Minimum multiplier (floor)
    min_multiplier: float = 0.5


@dataclass
class FilterConfig:
    """Market condition filters."""
    # Volume spike threshold (multiple of baseline volume)
    volume_spike_multiplier: float = 3.5
    volume_baseline_period: int = 20
    
    # ATR expansion threshold (multiple of baseline ATR)
    atr_expansion_multiplier: float = 2.2
    atr_baseline_period: int = 50
    
    # BTC correlation filter: pause if BTC moves more than this %
    btc_move_threshold_pct: float = 0.03  # 3%
    btc_lookback_candles: int = 5
    
    # Minimum volume filter (in USD per candle)
    min_volume_usd: float = 1000  # Lowered for altcoins on 5m timeframe

    # Spread filter (as % of price)
    max_spread_pct: float = 0.01  # 1% - more realistic for altcoins
    
    # Order book depth filter (minimum depth in USD at X% from mid)
    min_orderbook_depth_usd: float = 2500
    orderbook_depth_pct: float = 0.005  # 0.5% from mid
    
    # Time-of-day filters (UTC hours to avoid)
    avoid_hours: List[int] = field(default_factory=lambda: [])
    
    # Days to avoid (0=Monday, 6=Sunday)
    avoid_days: List[int] = field(default_factory=lambda: [])


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    # Slippage model (as % of price)
    expected_slippage_pct: float = 0.0005  # 0.05%

    # Maximum acceptable slippage for market orders
    max_slippage_pct: float = 0.002  # 0.2%

    # Partial fill handling: minimum fill % to accept
    min_fill_pct: float = 0.5

    # Order timeout (seconds)
    order_timeout_seconds: int = 30

    # Retry attempts for failed orders
    max_retries: int = 3

    # Use reduce-only for exits
    reduce_only_exits: bool = True

    # Dynamic leverage settings (Hyperliquid)
    # Leverage is scaled based on trade confidence (dynamic multiplier)
    dynamic_leverage_enabled: bool = True
    min_leverage: int = 3  # Minimum leverage for low confidence trades
    max_leverage: int = 5  # Maximum leverage for high confidence trades


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Initial capital
    initial_capital: float = 1000.0
    
    # Commission per trade (as %)
    commission_pct: float = 0.0006  # 0.06% taker fee
    
    # Funding rate assumption (per 8 hours, as %)
    funding_rate_pct: float = 0.0001  # 0.01%
    
    # Slippage model for backtesting
    backtest_slippage_pct: float = 0.0005
    
    # Monte Carlo simulations for robustness
    monte_carlo_runs: int = 1000


@dataclass
class SymbolConfig:
    """Per-symbol configuration overrides."""
    symbol: str
    enabled: bool = True

    # Override risk parameters
    risk_multiplier: float = 1.0

    # Override signal parameters
    wick_atr_multiplier_override: Optional[float] = None

    # Minimum position size (in base currency)
    min_position_size: float = 0.001

    # Base position size (in USD) - used for normal confidence trades (1 signal)
    base_position_usd: float = 200

    # Maximum position size (in USD) - used for highest confidence trades
    max_position_usd: float = 500


@dataclass
class StrategyConfig:
    """Master configuration combining all components."""
    # Strategy identification
    strategy_name: str = "WickReversal_v1"
    strategy_version: str = "1.9.1"
    
    # Primary timeframe
    timeframe: TimeFrame = TimeFrame.M5
    
    # Component configurations
    signal: SignalConfig = field(default_factory=SignalConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    dynamic_leverage: DynamicLeverageConfig = field(default_factory=DynamicLeverageConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Symbols to trade
    symbols: List[SymbolConfig] = field(default_factory=lambda: [
        SymbolConfig(symbol="TAO-PERP"),
        SymbolConfig(symbol="AAVE-PERP"),
        SymbolConfig(symbol="ZRO-PERP"),
    ])
    
    # Logging level
    log_level: str = "INFO"
    
    # Paper trading mode
    paper_trading: bool = True


# Default configuration instance
DEFAULT_CONFIG = StrategyConfig()


def load_config_from_dict(config_dict: dict) -> StrategyConfig:
    """Load configuration from a dictionary (e.g., from JSON/YAML file)."""
    # Implementation for loading from external config files
    pass


def save_config_to_dict(config: StrategyConfig) -> dict:
    """Export configuration to dictionary for serialization."""
    # Implementation for saving to external config files
    pass
