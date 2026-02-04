"""
Technical Indicators for Trading Strategy
Optimized for small account scalping
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SignalResult:
    """Result of signal calculation"""
    signal: int  # 1 = long, -1 = short, 0 = no signal
    strength: float  # 0-1 signal strength
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator"""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()

    return k, d


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = atr(high, low, close, 1)
    atr_val = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()

    return adx_val, plus_di, minus_di


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average"""
    return sma(volume, period)


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Supertrend indicator"""
    atr_val = atr(high, low, close, period)

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)

    supertrend_line = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend_line.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend_line.iloc[i-1]:
            supertrend_line.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend_line.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1

    return supertrend_line, direction


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Momentum indicator"""
    return series.diff(period)


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC)"""
    return ((series - series.shift(period)) / series.shift(period)) * 100


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation)


def heikin_ashi(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Heikin-Ashi candles"""
    ha_close = (open_ + high + low + close) / 4

    ha_open = pd.Series(index=close.index, dtype=float)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2

    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

    return ha_open, ha_high, ha_low, ha_close


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators for a dataframe"""

    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # EMA
    df['ema_8'] = ema(df['close'], 8)
    df['ema_21'] = ema(df['close'], 21)
    df['ema_50'] = ema(df['close'], 50)
    df['ema_200'] = ema(df['close'], 200)

    # SMA
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)

    # RSI
    df['rsi'] = rsi(df['close'], 14)
    df['rsi_6'] = rsi(df['close'], 6)  # Fast RSI for scalping

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage

    # Stochastic
    df['stoch_k'], df['stoch_d'] = stochastic(df['high'], df['low'], df['close'])

    # ADX
    df['adx'], df['plus_di'], df['minus_di'] = adx(df['high'], df['low'], df['close'])

    # Volume
    df['volume_sma'] = volume_sma(df['volume'])
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Momentum
    df['momentum'] = momentum(df['close'])
    df['roc'] = rate_of_change(df['close'])

    # Williams %R
    df['williams_r'] = williams_r(df['high'], df['low'], df['close'])

    # CCI
    df['cci'] = cci(df['high'], df['low'], df['close'])

    # Supertrend
    df['supertrend'], df['supertrend_dir'] = supertrend(df['high'], df['low'], df['close'])

    # Price action features
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']

    # Trend strength
    df['trend_strength'] = abs(df['ema_8'] - df['ema_21']) / df['atr']

    # Volatility regime
    df['volatility_regime'] = df['atr_pct'].rolling(20).apply(
        lambda x: 1 if x.iloc[-1] > x.mean() else 0
    )

    return df


def detect_support_resistance(
    df: pd.DataFrame,
    lookback: int = 50,
    tolerance: float = 0.02
) -> Tuple[list, list]:
    """Detect support and resistance levels"""
    supports = []
    resistances = []

    recent_data = df.tail(lookback)

    # Find local minima (supports)
    for i in range(2, len(recent_data) - 2):
        if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
            recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
            recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
            recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
            supports.append(recent_data['low'].iloc[i])

    # Find local maxima (resistances)
    for i in range(2, len(recent_data) - 2):
        if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
            recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
            recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
            recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
            resistances.append(recent_data['high'].iloc[i])

    # Cluster similar levels
    supports = _cluster_levels(supports, tolerance)
    resistances = _cluster_levels(resistances, tolerance)

    return supports, resistances


def _cluster_levels(levels: list, tolerance: float) -> list:
    """Cluster similar price levels"""
    if not levels:
        return []

    levels = sorted(levels)
    clusters = [[levels[0]]]

    for level in levels[1:]:
        if abs(level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [np.mean(cluster) for cluster in clusters]
