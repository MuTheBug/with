"""
Configuration for Binance Futures Trading Bot.
All parameters are tuned for small-balance aggressive compounding.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND_UP = "weak_trend_up"
    WEAK_TREND_DOWN = "weak_trend_down"
    RANGE_LOW_VOL = "range_low_vol"
    RANGE_HIGH_VOL = "range_high_vol"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    CHOPPY = "choppy"


class StrategyName(Enum):
    TREND_MOMENTUM = "trend_momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALP = "scalp"
    MOMENTUM_SURF = "momentum_surf"


# Which strategy dominates in each regime
REGIME_STRATEGY_MAP: Dict[MarketRegime, StrategyName] = {
    MarketRegime.STRONG_TREND_UP: StrategyName.MOMENTUM_SURF,
    MarketRegime.STRONG_TREND_DOWN: StrategyName.MOMENTUM_SURF,
    MarketRegime.WEAK_TREND_UP: StrategyName.TREND_MOMENTUM,
    MarketRegime.WEAK_TREND_DOWN: StrategyName.TREND_MOMENTUM,
    MarketRegime.RANGE_LOW_VOL: StrategyName.MEAN_REVERSION,
    MarketRegime.RANGE_HIGH_VOL: StrategyName.BREAKOUT,
    MarketRegime.BREAKOUT_UP: StrategyName.BREAKOUT,
    MarketRegime.BREAKOUT_DOWN: StrategyName.BREAKOUT,
    MarketRegime.CHOPPY: StrategyName.SCALP,
}


@dataclass
class BinanceConfig:
    api_key: str = os.environ.get("BINANCE_API_KEY", "")
    api_secret: str = os.environ.get("BINANCE_API_SECRET", "")
    testnet: bool = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"
    base_url: str = ""

    def __post_init__(self):
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"


@dataclass
class TradingPairs:
    """Pairs ranked by suitability for small-balance trading."""
    primary: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    ])
    secondary: List[str] = field(default_factory=lambda: [
        "BNBUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT",
    ])


@dataclass
class TimeframeConfig:
    """Multi-timeframe analysis configuration."""
    execution_tf: str = "5m"       # Execution timeframe
    signal_tf: str = "15m"         # Signal generation
    trend_tf: str = "1h"           # Trend identification
    macro_tf: str = "4h"           # Macro regime
    timeframes: List[str] = field(default_factory=lambda: [
        "5m", "15m", "1h", "4h"
    ])
    lookback_bars: Dict[str, int] = field(default_factory=lambda: {
        "5m": 200, "15m": 200, "1h": 200, "4h": 100,
    })


@dataclass
class RegimeConfig:
    """Regime detection thresholds."""
    adx_strong_trend: float = 30.0
    adx_weak_trend: float = 20.0
    adx_period: int = 14
    bb_width_high_vol: float = 0.06
    bb_width_low_vol: float = 0.02
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    volume_ma_period: int = 20
    hurst_period: int = 100
    regime_lookback: int = 50
    # Minimum confidence to act on a regime detection
    min_regime_confidence: float = 0.35


@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Per-trade risk as fraction of balance
    max_risk_per_trade: float = 0.02          # 2% per trade
    max_risk_per_trade_aggressive: float = 0.03  # 3% when high confidence
    # Maximum concurrent positions
    max_positions: int = 1                     # 1 for balance < $50
    max_positions_medium: int = 2              # 2 for balance $50-$500
    max_positions_large: int = 3               # 3 for balance > $500
    # Drawdown circuit breakers
    max_drawdown_soft: float = 0.10            # 10% - reduce size
    max_drawdown_hard: float = 0.15            # 15% - stop trading
    max_drawdown_absolute: float = 0.20        # 20% - emergency stop
    # Leverage limits by regime confidence
    max_leverage: int = 20
    min_leverage: int = 3
    default_leverage: int = 10
    # Consecutive loss protection
    max_consecutive_losses: int = 5
    cooldown_after_max_losses: int = 4         # hours (or bars in backtest)
    # Minimum balance to trade
    min_balance_to_trade: float = 3.0          # USDT
    # Binance futures minimum notional
    min_notional: float = 5.0                  # USDT minimum order
    # Fee structure
    maker_fee: float = 0.0002                  # 0.02%
    taker_fee: float = 0.0004                  # 0.04%
    # Slippage estimate
    slippage_pct: float = 0.0005               # 0.05%
    # Kelly criterion fraction (use fractional Kelly)
    kelly_fraction: float = 0.25               # Quarter Kelly


@dataclass
class StrategyParams:
    """Strategy-specific parameters, optimized via backtesting."""

    # Trend Momentum Strategy
    trend_ema_fast: int = 8
    trend_ema_slow: int = 21
    trend_ema_signal: int = 50
    trend_rsi_entry_long: float = 45.0
    trend_rsi_entry_short: float = 55.0
    trend_macd_fast: int = 12
    trend_macd_slow: int = 26
    trend_macd_signal: int = 9
    trend_adx_min: float = 20.0
    trend_atr_sl_mult: float = 2.0
    trend_atr_tp_mult: float = 4.0
    trend_min_rr: float = 2.0

    # Mean Reversion Strategy
    mr_bb_period: int = 20
    mr_bb_std: float = 2.0
    mr_rsi_oversold: float = 25.0
    mr_rsi_overbought: float = 75.0
    mr_rsi_period: int = 14
    mr_atr_sl_mult: float = 1.5
    mr_atr_tp_mult: float = 2.5
    mr_min_bb_width: float = 0.015
    mr_volume_confirm: float = 0.8

    # Breakout Strategy
    bo_lookback: int = 20
    bo_volume_mult: float = 1.5
    bo_atr_sl_mult: float = 1.5
    bo_atr_tp_mult: float = 4.0
    bo_consolidation_bars: int = 10
    bo_bb_squeeze_threshold: float = 0.015
    bo_min_range_pct: float = 0.005

    # Scalp Strategy
    scalp_rsi_period: int = 7
    scalp_rsi_oversold: float = 20.0
    scalp_rsi_overbought: float = 80.0
    scalp_ema_fast: int = 5
    scalp_ema_slow: int = 13
    scalp_atr_sl_mult: float = 1.5
    scalp_atr_tp_mult: float = 2.0
    scalp_max_hold_bars: int = 12

    # Momentum Surf Strategy
    mom_ema_fast: int = 5
    mom_ema_slow: int = 13
    mom_ema_trend: int = 50
    mom_rsi_period: int = 14
    mom_rsi_min_long: float = 50.0
    mom_rsi_max_short: float = 50.0
    mom_adx_min: float = 25.0
    mom_atr_sl_mult: float = 2.5
    mom_trail_atr_mult: float = 3.5
    mom_volume_mult: float = 1.2


@dataclass
class BotConfig:
    """Master configuration."""
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    pairs: TradingPairs = field(default_factory=TradingPairs)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    # State persistence
    state_file: str = "bot_state.json"
    # Update interval in seconds
    loop_interval: int = 10
