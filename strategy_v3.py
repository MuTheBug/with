"""
High Win-Rate Scalping Strategy v3
Focus on smaller, more consistent profits with higher win rate

Key changes:
1. RSI extremes + quick reversal = high probability
2. Smaller targets (1:1 to 1.5:1 R:R) but higher win rate
3. Quick exits to lock in profits
4. Trade both trends and ranges
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class StrategyType(Enum):
    SCALP = "scalp"


@dataclass
class TradeSignal:
    signal_type: SignalType
    strategy: StrategyType
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_2: float
    position_size_pct: float
    confidence: float
    reason: str
    timestamp: pd.Timestamp = None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 0.0001)
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    middle = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class ScalpingStrategy:
    """
    High win-rate scalping strategy

    Entry conditions:
    1. RSI extreme (< 25 or > 75)
    2. Price at Bollinger Band extreme
    3. Momentum showing reversal
    4. Quick mean reversion expected

    Exit: Quick targets, tight stops
    """

    def __init__(self):
        self.rsi_period = 7  # Fast RSI for quick signals
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.bb_period = 20
        self.bb_std = 2.0

        # Scalping targets - smaller but higher probability
        self.target_atr_mult = 1.2  # 1.2x ATR target
        self.stop_atr_mult = 1.0  # 1x ATR stop

        # Confirmation candles
        self.min_reversal_candles = 1

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs for trend
        df['ema_10'] = ema(df['close'], 10)
        df['ema_20'] = ema(df['close'], 20)
        df['ema_50'] = ema(df['close'], 50)

        # Fast RSI
        df['rsi'] = rsi(df['close'], self.rsi_period)
        df['rsi_14'] = rsi(df['close'], 14)  # Standard RSI for confirmation

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(
            df['close'], self.bb_period, self.bb_std
        )

        # BB position (0 = at lower, 1 = at upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)

        # Candle analysis
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['close'] * 100
        df['bullish'] = df['close'] > df['open']
        df['bearish'] = df['close'] < df['open']

        # Momentum
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)

        # Range position
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['range_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 0.0001)

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 100:
            return None

        df = self.calculate_indicators(df)

        c = df.iloc[-1]  # Current
        p1 = df.iloc[-2]  # Previous
        p2 = df.iloc[-3]

        if pd.isna(c['rsi']) or pd.isna(c['atr']) or c['atr'] == 0:
            return None

        signal = None

        # ===== LONG SCALP =====
        # Conditions:
        # 1. RSI was extremely oversold (< 25) and turning up
        # 2. Price near lower Bollinger Band
        # 3. Bullish candle confirmation
        # 4. Not in extreme downtrend

        rsi_oversold_reversal = (
            p1['rsi'] < self.rsi_oversold and
            c['rsi'] > p1['rsi'] and
            c['rsi'] < 50  # Still has room to go up
        )

        bb_oversold = c['bb_position'] < 0.2  # Near lower band

        bullish_reversal = (
            c['bullish'] and
            c['body_pct'] > 0.1  # Decent sized candle
        )

        not_crash = c['momentum_5'] > -5  # Not in a crash

        if rsi_oversold_reversal and bb_oversold and bullish_reversal and not_crash:
            entry_price = c['close']
            atr_val = c['atr']

            # Tight stop below recent low
            recent_low = df['low'].iloc[-5:].min()
            stop_loss = min(recent_low - atr_val * 0.3, entry_price - atr_val * self.stop_atr_mult)

            # Quick target
            take_profit = entry_price + atr_val * self.target_atr_mult
            take_profit_2 = entry_price + atr_val * 2

            # Ensure minimum stop distance (0.3%)
            if entry_price - stop_loss < entry_price * 0.003:
                stop_loss = entry_price * 0.997

            confidence = 0.75
            reasons = [f"RSI oversold reversal ({p1['rsi']:.1f}->{c['rsi']:.1f})"]

            if c['volume_ratio'] > 1.2:
                confidence += 0.05
                reasons.append("High volume")

            if c['ema_10'] > c['ema_20']:
                confidence += 0.05
                reasons.append("EMA bullish")

            signal = TradeSignal(
                signal_type=SignalType.LONG,
                strategy=StrategyType.SCALP,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=min(confidence, 1.0),
                reason=" | ".join(reasons)
            )

        # ===== SHORT SCALP =====
        rsi_overbought_reversal = (
            p1['rsi'] > self.rsi_overbought and
            c['rsi'] < p1['rsi'] and
            c['rsi'] > 50
        )

        bb_overbought = c['bb_position'] > 0.8

        bearish_reversal = (
            c['bearish'] and
            c['body_pct'] < -0.1
        )

        not_pump = c['momentum_5'] < 5

        if rsi_overbought_reversal and bb_overbought and bearish_reversal and not_pump:
            entry_price = c['close']
            atr_val = c['atr']

            recent_high = df['high'].iloc[-5:].max()
            stop_loss = max(recent_high + atr_val * 0.3, entry_price + atr_val * self.stop_atr_mult)

            take_profit = entry_price - atr_val * self.target_atr_mult
            take_profit_2 = entry_price - atr_val * 2

            if stop_loss - entry_price < entry_price * 0.003:
                stop_loss = entry_price * 1.003

            confidence = 0.75
            reasons = [f"RSI overbought reversal ({p1['rsi']:.1f}->{c['rsi']:.1f})"]

            if c['volume_ratio'] > 1.2:
                confidence += 0.05
                reasons.append("High volume")

            if c['ema_10'] < c['ema_20']:
                confidence += 0.05
                reasons.append("EMA bearish")

            signal = TradeSignal(
                signal_type=SignalType.SHORT,
                strategy=StrategyType.SCALP,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=min(confidence, 1.0),
                reason=" | ".join(reasons)
            )

        # Validate
        if signal and not self._validate(signal):
            return None

        return signal

    def _validate(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        # R:R at least 1:1
        if reward / risk < 1.0:
            return False

        # Max stop distance 2%
        stop_pct = risk / entry
        if stop_pct > 0.02:
            return False

        # Min confidence
        if signal.confidence < 0.70:
            return False

        return True


class EMAMomentumStrategy:
    """
    Simple EMA + Momentum Strategy
    Trades strong momentum with trend confirmation
    """

    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df['ema_8'] = ema(df['close'], 8)
        df['ema_21'] = ema(df['close'], 21)
        df['ema_55'] = ema(df['close'], 55)

        # RSI
        df['rsi'] = rsi(df['close'], 14)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)

        # Momentum
        df['mom_1'] = df['close'].pct_change(1) * 100
        df['mom_3'] = df['close'].pct_change(3) * 100

        # Candle
        df['bullish'] = df['close'] > df['open']
        df['bearish'] = df['close'] < df['open']

        # Crossover detection
        df['ema_cross_up'] = (df['ema_8'] > df['ema_21']) & (df['ema_8'].shift() <= df['ema_21'].shift())
        df['ema_cross_down'] = (df['ema_8'] < df['ema_21']) & (df['ema_8'].shift() >= df['ema_21'].shift())

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 100:
            return None

        df = self.calculate_indicators(df)

        c = df.iloc[-1]
        p1 = df.iloc[-2]

        if pd.isna(c['ema_55']) or pd.isna(c['atr']) or c['atr'] == 0:
            return None

        signal = None

        # ===== LONG: EMA crossover with momentum =====
        # Recent bullish EMA cross (within last 3 bars)
        recent_cross_up = df['ema_cross_up'].iloc[-3:].any()

        uptrend = c['ema_21'] > c['ema_55']
        strong_momentum = c['mom_3'] > 1  # 1% up in 3 bars
        rsi_ok = 40 < c['rsi'] < 70

        if recent_cross_up and uptrend and strong_momentum and rsi_ok and c['bullish']:
            entry = c['close']
            atr_val = c['atr']

            stop_loss = entry - atr_val * 1.5
            take_profit = entry + atr_val * 2
            take_profit_2 = entry + atr_val * 3

            signal = TradeSignal(
                signal_type=SignalType.LONG,
                strategy=StrategyType.SCALP,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=0.72,
                reason="EMA cross up | Uptrend | Momentum"
            )

        # ===== SHORT: EMA crossover down =====
        recent_cross_down = df['ema_cross_down'].iloc[-3:].any()

        downtrend = c['ema_21'] < c['ema_55']
        weak_momentum = c['mom_3'] < -1
        rsi_ok_short = 30 < c['rsi'] < 60

        if recent_cross_down and downtrend and weak_momentum and rsi_ok_short and c['bearish']:
            entry = c['close']
            atr_val = c['atr']

            stop_loss = entry + atr_val * 1.5
            take_profit = entry - atr_val * 2
            take_profit_2 = entry - atr_val * 3

            signal = TradeSignal(
                signal_type=SignalType.SHORT,
                strategy=StrategyType.SCALP,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=0.72,
                reason="EMA cross down | Downtrend | Momentum"
            )

        if signal and not self._validate(signal):
            return None

        return signal

    def _validate(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        if reward / risk < 1.2:
            return False

        stop_pct = risk / entry
        if stop_pct > 0.025:
            return False

        if signal.confidence < 0.70:
            return False

        return True


class CombinedStrategy:
    """Combines multiple strategies and takes best signals"""

    def __init__(self):
        self.scalping = ScalpingStrategy()
        self.ema_momentum = EMAMomentumStrategy()

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        signals = []

        # Get signals from each strategy
        scalp_signal = self.scalping.analyze(df)
        if scalp_signal:
            signals.append(scalp_signal)

        ema_signal = self.ema_momentum.analyze(df)
        if ema_signal:
            signals.append(ema_signal)

        if not signals:
            return None

        # Return highest confidence signal
        return max(signals, key=lambda s: s.confidence)


# Default export
SimpleStrategy = CombinedStrategy
