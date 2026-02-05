"""
Trading strategy implementations.

Each strategy:
  - Takes a DataFrame with computed indicators
  - Returns a Signal (LONG/SHORT/NONE) with entry, stop-loss, and take-profit levels
  - Is designed for a specific market regime where it has the highest edge
  - Uses a combination of leading and lagging indicators, with recent price
    action as the primary direction filter to avoid lagging indicator traps
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from config import BotConfig, StrategyParams, StrategyName
import indicators as ind


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class TradeSignal:
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: StrategyName
    confidence: float        # 0-1, how confident the strategy is
    leverage_hint: int       # Suggested leverage based on setup quality
    reason: str = ""

    @property
    def risk_reward(self) -> float:
        if self.direction == SignalDirection.NONE:
            return 0.0
        if self.direction == SignalDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        if risk <= 0:
            return 0.0
        return reward / risk

    @property
    def risk_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.LONG:
            return (self.entry_price - self.stop_loss) / self.entry_price
        elif self.direction == SignalDirection.SHORT:
            return (self.stop_loss - self.entry_price) / self.entry_price
        return 0.0


NO_SIGNAL = TradeSignal(
    direction=SignalDirection.NONE, entry_price=0, stop_loss=0,
    take_profit=0, strategy=StrategyName.TREND_MOMENTUM,
    confidence=0, leverage_hint=1,
)


class BaseStrategy:
    """Base class for all strategies."""

    name: StrategyName = StrategyName.TREND_MOMENTUM

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.params = self.config.strategy

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        raise NotImplementedError

    def _no_signal(self) -> TradeSignal:
        return TradeSignal(
            direction=SignalDirection.NONE, entry_price=0, stop_loss=0,
            take_profit=0, strategy=self.name, confidence=0, leverage_hint=1,
        )

    def _safe_val(self, df: pd.DataFrame, col: str, offset: int = -1):
        """Safely get a value from a DataFrame column."""
        if col not in df.columns:
            return np.nan
        idx = offset
        if abs(idx) > len(df):
            return np.nan
        val = df[col].iloc[idx]
        return val

    def _recent_momentum(self, df: pd.DataFrame, bars: int = 5) -> float:
        """
        Returns recent price momentum: +1 strongly bullish, -1 strongly bearish, 0 neutral.
        This is a LEADING indicator based on recent price action.
        """
        if len(df) < bars + 1:
            return 0.0
        closes = df["close"].iloc[-(bars + 1):]
        changes = closes.pct_change().iloc[1:]
        # Weighted: more recent bars matter more
        weights = np.arange(1, bars + 1, dtype=float)
        weighted_change = np.average(changes.values, weights=weights)
        # Normalize to -1 to +1 range (0.5% per bar = max signal)
        return float(np.clip(weighted_change * 200, -1, 1))

    def _recent_higher_lows(self, df: pd.DataFrame, bars: int = 5) -> int:
        """Count higher lows in recent bars. Positive = bullish structure."""
        if len(df) < bars:
            return 0
        lows = df["low"].iloc[-bars:]
        count = sum(1 for i in range(1, len(lows)) if lows.iloc[i] > lows.iloc[i - 1])
        return count

    def _recent_lower_highs(self, df: pd.DataFrame, bars: int = 5) -> int:
        """Count lower highs in recent bars. Positive = bearish structure."""
        if len(df) < bars:
            return 0
        highs = df["high"].iloc[-bars:]
        count = sum(1 for i in range(1, len(highs)) if highs.iloc[i] < highs.iloc[i - 1])
        return count


class TrendMomentumStrategy(BaseStrategy):
    """
    For trending regimes. Enters in the direction of the trend when
    momentum is confirmed by multiple indicators AND recent price action.
    """

    name = StrategyName.TREND_MOMENTUM

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 60:
            return self._no_signal()

        p = self.params
        close = df["close"].iloc[-1]

        ema_fast = self._safe_val(df, "ema_8")
        ema_slow = self._safe_val(df, "ema_21")
        ema_trend = self._safe_val(df, "ema_50")
        rsi_val = self._safe_val(df, "rsi_14")
        adx_val = self._safe_val(df, "adx")
        plus_di = self._safe_val(df, "plus_di")
        minus_di = self._safe_val(df, "minus_di")
        macd_hist = self._safe_val(df, "macd_hist")
        macd_hist_prev = self._safe_val(df, "macd_hist", -2)
        atr_val = self._safe_val(df, "atr")
        vol_ratio = self._safe_val(df, "volume_ratio")

        vals = [ema_fast, ema_slow, ema_trend, rsi_val, adx_val, atr_val]
        if any(np.isnan(v) for v in vals):
            return self._no_signal()

        # Recent price momentum (LEADING signal - most important)
        momentum = self._recent_momentum(df, 5)

        long_score = 0
        short_score = 0

        # Recent momentum (strongest weight - 3 points)
        if momentum > 0.3:
            long_score += 3
        elif momentum > 0.1:
            long_score += 2
        elif momentum > 0:
            long_score += 1

        if momentum < -0.3:
            short_score += 3
        elif momentum < -0.1:
            short_score += 2
        elif momentum < 0:
            short_score += 1

        # EMA alignment
        if ema_fast > ema_slow:
            long_score += 1
        else:
            short_score += 1

        if close > ema_trend:
            long_score += 1
        else:
            short_score += 1

        # DI direction
        if not np.isnan(plus_di) and not np.isnan(minus_di):
            if plus_di > minus_di:
                long_score += 1
            else:
                short_score += 1

        # RSI
        if rsi_val > 55:
            long_score += 1
        elif rsi_val < 45:
            short_score += 1

        # MACD
        if not np.isnan(macd_hist):
            if macd_hist > 0:
                long_score += 1
            elif macd_hist < 0:
                short_score += 1
            if not np.isnan(macd_hist_prev):
                if macd_hist > macd_hist_prev:
                    long_score += 1
                elif macd_hist < macd_hist_prev:
                    short_score += 1

        # Price structure
        if self._recent_higher_lows(df, 4) >= 2:
            long_score += 1
        if self._recent_lower_highs(df, 4) >= 2:
            short_score += 1

        direction = SignalDirection.NONE

        # Need strong bias: >= 6 points AND momentum agrees
        if long_score >= 6 and long_score > short_score + 2 and momentum > 0:
            direction = SignalDirection.LONG
            confidence = min(long_score / 10, 1.0)
        elif short_score >= 6 and short_score > long_score + 2 and momentum < 0:
            direction = SignalDirection.SHORT
            confidence = min(short_score / 10, 1.0)

        if direction == SignalDirection.NONE:
            return self._no_signal()

        # Volume boost
        if not np.isnan(vol_ratio) and vol_ratio > 1.2:
            confidence = min(confidence + 0.1, 1.0)

        # ADX boost
        if adx_val > 30:
            confidence = min(confidence + 0.05, 1.0)

        sl_dist = atr_val * p.trend_atr_sl_mult
        tp_dist = atr_val * p.trend_atr_tp_mult

        if direction == SignalDirection.LONG:
            stop_loss = close - sl_dist
            take_profit = close + tp_dist
        else:
            stop_loss = close + sl_dist
            take_profit = close - tp_dist

        leverage = 12 if confidence > 0.7 else 10

        return TradeSignal(
            direction=direction, entry_price=close,
            stop_loss=stop_loss, take_profit=take_profit,
            strategy=self.name, confidence=confidence,
            leverage_hint=leverage,
            reason=f"Trend, ADX={adx_val:.1f}, RSI={rsi_val:.1f}, mom={momentum:.2f}",
        )


class MeanReversionStrategy(BaseStrategy):
    """
    For range-bound regimes. Enters when price hits Bollinger Band extremes
    with RSI confirmation. Targets the mean (middle BB).
    """

    name = StrategyName.MEAN_REVERSION

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._no_signal()

        p = self.params
        close = df["close"].iloc[-1]
        open_price = df["open"].iloc[-1]

        bb_upper = self._safe_val(df, "bb_upper")
        bb_lower = self._safe_val(df, "bb_lower")
        bb_mid = self._safe_val(df, "bb_mid")
        bb_pct = self._safe_val(df, "bb_pct")
        rsi_val = self._safe_val(df, "rsi_14")
        rsi_7 = self._safe_val(df, "rsi_7")
        atr_val = self._safe_val(df, "atr")
        stoch_k = self._safe_val(df, "stoch_k")
        williams = self._safe_val(df, "williams_r")

        vals = [bb_upper, bb_lower, bb_mid, rsi_val, atr_val]
        if any(np.isnan(v) for v in vals):
            return self._no_signal()

        direction = SignalDirection.NONE
        confidence = 0.0

        # === LONG (oversold bounce) ===
        long_score = 0

        if not np.isnan(bb_pct):
            if bb_pct < 0.05:
                long_score += 3  # Very oversold
            elif bb_pct < 0.15:
                long_score += 2
            elif bb_pct < 0.25:
                long_score += 1

        if rsi_val < 25:
            long_score += 2
        elif rsi_val < 35:
            long_score += 1

        if not np.isnan(rsi_7) and rsi_7 < 20:
            long_score += 1
        elif not np.isnan(rsi_7) and rsi_7 < 30:
            long_score += 1

        if not np.isnan(stoch_k) and stoch_k < 20:
            long_score += 1

        if not np.isnan(williams) and williams < -80:
            long_score += 1

        # Bullish candle (closing above open = reversal candle)
        if close > open_price:
            long_score += 1

        # === SHORT (overbought reversal) ===
        short_score = 0

        if not np.isnan(bb_pct):
            if bb_pct > 0.95:
                short_score += 3
            elif bb_pct > 0.85:
                short_score += 2
            elif bb_pct > 0.75:
                short_score += 1

        if rsi_val > 75:
            short_score += 2
        elif rsi_val > 65:
            short_score += 1

        if not np.isnan(rsi_7) and rsi_7 > 80:
            short_score += 1
        elif not np.isnan(rsi_7) and rsi_7 > 70:
            short_score += 1

        if not np.isnan(stoch_k) and stoch_k > 80:
            short_score += 1

        if not np.isnan(williams) and williams > -20:
            short_score += 1

        if close < open_price:
            short_score += 1

        # Require minimum 4 points
        if long_score >= 4:
            direction = SignalDirection.LONG
            confidence = min(long_score / 8, 0.95)
        elif short_score >= 4:
            direction = SignalDirection.SHORT
            confidence = min(short_score / 8, 0.95)

        if direction == SignalDirection.NONE:
            return self._no_signal()

        # Stops and targets
        sl_dist = atr_val * p.mr_atr_sl_mult
        # Target: BB midline (the mean)
        if direction == SignalDirection.LONG:
            stop_loss = close - sl_dist
            take_profit = bb_mid
            # Ensure minimum R:R of 1.2
            reward = take_profit - close
            risk = close - stop_loss
            if risk > 0 and reward / risk < 1.2:
                take_profit = close + sl_dist * 1.5
        else:
            stop_loss = close + sl_dist
            take_profit = bb_mid
            reward = close - take_profit
            risk = stop_loss - close
            if risk > 0 and reward / risk < 1.2:
                take_profit = close - sl_dist * 1.5

        leverage = 10 if confidence > 0.6 else 8

        return TradeSignal(
            direction=direction, entry_price=close,
            stop_loss=stop_loss, take_profit=take_profit,
            strategy=self.name, confidence=confidence,
            leverage_hint=leverage,
            reason=f"MR: BB%={bb_pct:.2f}, RSI={rsi_val:.1f}",
        )


class BreakoutStrategy(BaseStrategy):
    """
    For range-bound with high volatility or squeeze regimes.
    Detects consolidation -> expansion breakouts with volume + momentum.
    """

    name = StrategyName.BREAKOUT

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return self._no_signal()

        p = self.params
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        atr_val = self._safe_val(df, "atr")
        vol_ratio = self._safe_val(df, "volume_ratio")
        squeeze = self._safe_val(df, "squeeze")
        squeeze_mom = self._safe_val(df, "squeeze_mom")
        bbw = self._safe_val(df, "bb_width")
        macd_hist = self._safe_val(df, "macd_hist")
        rsi_val = self._safe_val(df, "rsi_14")

        if any(np.isnan(v) for v in [atr_val, bbw]):
            return self._no_signal()

        # Recent momentum
        momentum = self._recent_momentum(df, 3)

        # Find consolidation range
        lookback = p.bo_lookback
        recent = df.tail(lookback)
        highest = recent["high"].max()
        lowest = recent["low"].min()

        long_score = 0
        short_score = 0

        # Range breakout
        if close > highest * 0.998:
            long_score += 3
        if close < lowest * 1.002:
            short_score += 3

        # Squeeze release
        if not np.isnan(squeeze):
            recent_squeeze = df["squeeze"].iloc[-min(10, len(df)):]
            if recent_squeeze.iloc[:-1].sum() >= 2 and squeeze == 0:
                if not np.isnan(squeeze_mom) and squeeze_mom > 0:
                    long_score += 2
                elif not np.isnan(squeeze_mom) and squeeze_mom < 0:
                    short_score += 2

        # Volume surge
        if not np.isnan(vol_ratio) and vol_ratio > 1.5:
            if momentum > 0:
                long_score += 2
            elif momentum < 0:
                short_score += 2
        elif not np.isnan(vol_ratio) and vol_ratio > 1.0:
            if momentum > 0:
                long_score += 1
            elif momentum < 0:
                short_score += 1

        # Momentum direction
        if momentum > 0.2:
            long_score += 2
        elif momentum > 0:
            long_score += 1
        if momentum < -0.2:
            short_score += 2
        elif momentum < 0:
            short_score += 1

        # MACD
        if not np.isnan(macd_hist):
            if macd_hist > 0:
                long_score += 1
            elif macd_hist < 0:
                short_score += 1

        # RSI
        if not np.isnan(rsi_val):
            if rsi_val > 55:
                long_score += 1
            elif rsi_val < 45:
                short_score += 1

        direction = SignalDirection.NONE

        if long_score >= 5 and long_score > short_score + 2 and momentum > 0:
            direction = SignalDirection.LONG
            confidence = min(long_score / 10, 1.0)
        elif short_score >= 5 and short_score > long_score + 2 and momentum < 0:
            direction = SignalDirection.SHORT
            confidence = min(short_score / 10, 1.0)

        if direction == SignalDirection.NONE:
            return self._no_signal()

        sl_dist = atr_val * p.bo_atr_sl_mult
        tp_dist = atr_val * p.bo_atr_tp_mult

        if direction == SignalDirection.LONG:
            stop_loss = close - sl_dist
            take_profit = close + tp_dist
        else:
            stop_loss = close + sl_dist
            take_profit = close - tp_dist

        leverage = 12 if confidence > 0.7 else 8

        return TradeSignal(
            direction=direction, entry_price=close,
            stop_loss=stop_loss, take_profit=take_profit,
            strategy=self.name, confidence=confidence,
            leverage_hint=leverage,
            reason=f"Breakout: vol={vol_ratio:.1f}, mom={momentum:.2f}",
        )


class ScalpStrategy(BaseStrategy):
    """
    For choppy, low-conviction regimes.
    Quick mean-reversion scalps with tight stops.
    Uses fast oscillators and recent price action.
    """

    name = StrategyName.SCALP

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 20:
            return self._no_signal()

        p = self.params
        close = df["close"].iloc[-1]
        open_price = df["open"].iloc[-1]

        rsi_7 = self._safe_val(df, "rsi_7")
        ema_fast = self._safe_val(df, "ema_5")
        ema_slow = self._safe_val(df, "ema_13")
        atr_val = self._safe_val(df, "atr")
        bb_pct = self._safe_val(df, "bb_pct")
        stoch_k = self._safe_val(df, "stoch_k")
        stoch_d = self._safe_val(df, "stoch_d")

        if any(np.isnan(v) for v in [rsi_7, atr_val]):
            return self._no_signal()

        direction = SignalDirection.NONE
        confidence = 0.0

        long_score = 0
        short_score = 0

        # RSI 7 extremes (primary signal)
        if rsi_7 < 20:
            long_score += 2
        elif rsi_7 < 30:
            long_score += 1

        if rsi_7 > 80:
            short_score += 2
        elif rsi_7 > 70:
            short_score += 1

        # Stochastic RSI cross
        if not np.isnan(stoch_k) and not np.isnan(stoch_d):
            if stoch_k < 25 and stoch_k > stoch_d:
                long_score += 2
            elif stoch_k < 30:
                long_score += 1
            if stoch_k > 75 and stoch_k < stoch_d:
                short_score += 2
            elif stoch_k > 70:
                short_score += 1

        # BB position
        if not np.isnan(bb_pct):
            if bb_pct < 0.1:
                long_score += 1
            if bb_pct > 0.9:
                short_score += 1

        # Reversal candle
        if close > open_price and close > df["close"].iloc[-2]:
            long_score += 1
        if close < open_price and close < df["close"].iloc[-2]:
            short_score += 1

        # EMA proximity bounce
        if not np.isnan(ema_fast):
            if close > ema_fast and df["close"].iloc[-2] <= ema_fast:
                long_score += 1
            if close < ema_fast and df["close"].iloc[-2] >= ema_fast:
                short_score += 1

        if long_score >= 3 and long_score > short_score:
            direction = SignalDirection.LONG
            confidence = min(long_score / 6, 0.85)
        elif short_score >= 3 and short_score > long_score:
            direction = SignalDirection.SHORT
            confidence = min(short_score / 6, 0.85)

        if direction == SignalDirection.NONE:
            return self._no_signal()

        # Tight stops for scalps
        sl_dist = atr_val * p.scalp_atr_sl_mult
        tp_dist = atr_val * p.scalp_atr_tp_mult

        if direction == SignalDirection.LONG:
            stop_loss = close - sl_dist
            take_profit = close + tp_dist
        else:
            stop_loss = close + sl_dist
            take_profit = close - tp_dist

        leverage = 10

        return TradeSignal(
            direction=direction, entry_price=close,
            stop_loss=stop_loss, take_profit=take_profit,
            strategy=self.name, confidence=confidence,
            leverage_hint=leverage,
            reason=f"Scalp: RSI7={rsi_7:.1f}",
        )


class MomentumSurfStrategy(BaseStrategy):
    """
    For strong trending regimes.
    Rides strong momentum with wider stops and trailing exits.
    Only enters when trend is clearly confirmed by BOTH indicators AND price action.
    """

    name = StrategyName.MOMENTUM_SURF

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 60:
            return self._no_signal()

        p = self.params
        close = df["close"].iloc[-1]

        ema_fast = self._safe_val(df, "ema_5")
        ema_slow = self._safe_val(df, "ema_13")
        ema_trend = self._safe_val(df, "ema_50")
        rsi_val = self._safe_val(df, "rsi_14")
        adx_val = self._safe_val(df, "adx")
        plus_di = self._safe_val(df, "plus_di")
        minus_di = self._safe_val(df, "minus_di")
        macd_hist = self._safe_val(df, "macd_hist")
        atr_val = self._safe_val(df, "atr")
        vol_ratio = self._safe_val(df, "volume_ratio")

        vals = [ema_fast, ema_slow, ema_trend, rsi_val, adx_val, atr_val]
        if any(np.isnan(v) for v in vals):
            return self._no_signal()

        # Strong momentum required
        momentum = self._recent_momentum(df, 5)

        # ADX must show strong trend
        if adx_val < 25:
            return self._no_signal()

        long_score = 0
        short_score = 0

        # Strong recent momentum (PRIMARY)
        if momentum > 0.4:
            long_score += 3
        elif momentum > 0.2:
            long_score += 2
        elif momentum > 0.1:
            long_score += 1

        if momentum < -0.4:
            short_score += 3
        elif momentum < -0.2:
            short_score += 2
        elif momentum < -0.1:
            short_score += 1

        # EMA stack
        if ema_fast > ema_slow > ema_trend:
            long_score += 2
        elif ema_fast > ema_slow:
            long_score += 1

        if ema_fast < ema_slow < ema_trend:
            short_score += 2
        elif ema_fast < ema_slow:
            short_score += 1

        # Price above/below all EMAs
        if close > ema_fast and close > ema_slow and close > ema_trend:
            long_score += 1
        if close < ema_fast and close < ema_slow and close < ema_trend:
            short_score += 1

        # DI
        if not np.isnan(plus_di) and not np.isnan(minus_di):
            if plus_di > minus_di and adx_val > 30:
                long_score += 2
            elif plus_di > minus_di:
                long_score += 1
            if minus_di > plus_di and adx_val > 30:
                short_score += 2
            elif minus_di > plus_di:
                short_score += 1

        # RSI momentum zone
        if rsi_val > 55 and rsi_val < 80:
            long_score += 1
        if rsi_val < 45 and rsi_val > 20:
            short_score += 1

        # MACD
        if not np.isnan(macd_hist) and macd_hist > 0:
            long_score += 1
        elif not np.isnan(macd_hist) and macd_hist < 0:
            short_score += 1

        # Volume above average
        if not np.isnan(vol_ratio) and vol_ratio > 1.2:
            if momentum > 0:
                long_score += 1
            else:
                short_score += 1

        # Higher lows / lower highs structure
        if self._recent_higher_lows(df, 5) >= 3:
            long_score += 1
        if self._recent_lower_highs(df, 5) >= 3:
            short_score += 1

        direction = SignalDirection.NONE

        # Very high bar - need strong alignment
        if long_score >= 7 and long_score > short_score + 3 and momentum > 0.1:
            direction = SignalDirection.LONG
            confidence = min(long_score / 12, 1.0)
        elif short_score >= 7 and short_score > long_score + 3 and momentum < -0.1:
            direction = SignalDirection.SHORT
            confidence = min(short_score / 12, 1.0)

        if direction == SignalDirection.NONE:
            return self._no_signal()

        # Wide stops for momentum trades
        sl_dist = atr_val * p.mom_atr_sl_mult
        tp_mult = p.mom_trail_atr_mult + max((adx_val - 25) / 25, 0)
        tp_dist = atr_val * max(tp_mult, 2.5)

        if direction == SignalDirection.LONG:
            stop_loss = close - sl_dist
            take_profit = close + tp_dist
        else:
            stop_loss = close + sl_dist
            take_profit = close - tp_dist

        leverage = 15 if confidence > 0.8 else 12

        return TradeSignal(
            direction=direction, entry_price=close,
            stop_loss=stop_loss, take_profit=take_profit,
            strategy=self.name, confidence=confidence,
            leverage_hint=leverage,
            reason=f"Momentum: ADX={adx_val:.1f}, RSI={rsi_val:.1f}, mom={momentum:.2f}",
        )


# Strategy registry
STRATEGY_REGISTRY = {
    StrategyName.TREND_MOMENTUM: TrendMomentumStrategy,
    StrategyName.MEAN_REVERSION: MeanReversionStrategy,
    StrategyName.BREAKOUT: BreakoutStrategy,
    StrategyName.SCALP: ScalpStrategy,
    StrategyName.MOMENTUM_SURF: MomentumSurfStrategy,
}


def get_strategy(name: StrategyName, config: Optional[BotConfig] = None) -> BaseStrategy:
    """Factory function to get a strategy instance."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return cls(config)
