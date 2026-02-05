"""
High-performance technical indicators computed with NumPy/Pandas.
All functions operate on pandas DataFrames with OHLCV columns.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average - faster response, less lag."""
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_period)


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, middle, lower."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bb_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band Width (normalized)."""
    upper, middle, lower = bollinger_bands(series, period, std_dev)
    return (upper - lower) / middle


def keltner_channels(
    df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: upper, middle, lower."""
    middle = ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, middle, lower


def squeeze_momentum(
    df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
    kc_period: int = 20, kc_atr: int = 14, kc_mult: float = 1.5
) -> Tuple[pd.Series, pd.Series]:
    """
    TTM Squeeze: detects low-volatility compression.
    Returns (is_squeeze: bool series, momentum: float series).
    """
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"], bb_period, bb_std)
    kc_upper, kc_mid, kc_lower = keltner_channels(df, kc_period, kc_atr, kc_mult)
    is_squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Momentum: linear regression of (close - midline) over lookback
    midline = (df["high"].rolling(kc_period).max() + df["low"].rolling(kc_period).min()) / 2
    midline = (midline + ema(df["close"], kc_period)) / 2
    momentum = df["close"] - midline
    return is_squeeze, momentum


# ---------------------------------------------------------------------------
# Trend / Momentum Indicators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: macd_line, signal_line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index.
    Returns (ADX, +DI, -DI).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(df, period)

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return adx_val, plus_di, minus_di


def stochastic_rsi(
    series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
    k_period: int = 3, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI: %K, %D."""
    rsi_val = rsi(series, rsi_period)
    rsi_min = rsi_val.rolling(window=stoch_period).min()
    rsi_max = rsi_val.rolling(window=stoch_period).max()
    stoch_rsi = (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch_rsi.rolling(window=k_period).mean() * 100
    d = k.rolling(window=d_period).mean()
    return k, d


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_sma = sma(tp, period)
    mean_dev = tp.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - tp_sma) / (0.015 * mean_dev)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = df["high"].rolling(window=period).max()
    lowest = df["low"].rolling(window=period).min()
    return -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    return (df["volume"] * direction).cumsum()


def vwap_rolling(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP approximation."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    return (tp * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to its moving average."""
    vol_ma = sma(df["volume"], period)
    return df["volume"] / vol_ma.replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = tp * df["volume"]
    tp_diff = tp.diff()

    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(period).sum()
    neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(period).sum()

    mf_ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mf_ratio))


# ---------------------------------------------------------------------------
# Statistical / Regime Helpers
# ---------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """
    Estimate Hurst exponent using R/S analysis.
    H < 0.5 => mean-reverting
    H ~ 0.5 => random walk
    H > 0.5 => trending
    """
    series = series.dropna().values
    if len(series) < max_lag + 10:
        return 0.5

    lags = range(2, min(max_lag, len(series) // 4))
    rs_values = []

    for lag in lags:
        n_chunks = len(series) // lag
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            mean_val = chunk.mean()
            deviations = chunk - mean_val
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = chunk.std()
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _ = np.polyfit(x, y, 1)
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(series: pd.Series, window: int = 100, max_lag: int = 40) -> pd.Series:
    """Rolling Hurst exponent."""
    result = pd.Series(index=series.index, dtype=float)
    values = series.values
    for i in range(window, len(values)):
        chunk = values[i - window:i]
        result.iloc[i] = hurst_exponent(pd.Series(chunk), max_lag)
    return result


def autocorrelation(series: pd.Series, lag: int = 1) -> pd.Series:
    """Rolling autocorrelation."""
    return series.rolling(50).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )


def realized_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility from log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=period).std() * np.sqrt(365 * 24)  # hourly to annual


def support_resistance_levels(
    df: pd.DataFrame, lookback: int = 50, num_levels: int = 3
) -> Tuple[list, list]:
    """
    Identify support and resistance levels using pivot points.
    Returns (support_levels, resistance_levels).
    """
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    close_price = df["close"].iloc[-1]

    # Find local maxima and minima
    resistance = []
    support = []

    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
           highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            if highs[i] > close_price:
                resistance.append(highs[i])
            else:
                support.append(highs[i])

        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
           lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            if lows[i] < close_price:
                support.append(lows[i])
            else:
                resistance.append(lows[i])

    # Sort by proximity to current price
    support = sorted(support, key=lambda x: abs(x - close_price))[:num_levels]
    resistance = sorted(resistance, key=lambda x: abs(x - close_price))[:num_levels]

    return sorted(support), sorted(resistance, reverse=True)


def compute_all_indicators(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Compute all indicators and attach them to the DataFrame.
    This is the main entry point for indicator computation.
    """
    from config import StrategyParams, RegimeConfig
    if cfg is None:
        sp = StrategyParams()
        rc = RegimeConfig()
    else:
        sp = cfg.strategy
        rc = cfg.regime

    out = df.copy()

    # EMAs
    out["ema_5"] = ema(df["close"], 5)
    out["ema_8"] = ema(df["close"], 8)
    out["ema_13"] = ema(df["close"], 13)
    out["ema_21"] = ema(df["close"], 21)
    out["ema_50"] = ema(df["close"], 50)
    out["ema_100"] = ema(df["close"], 100)
    out["ema_200"] = ema(df["close"], 200)

    # RSI
    out["rsi_7"] = rsi(df["close"], 7)
    out["rsi_14"] = rsi(df["close"], 14)

    # MACD
    macd_line, signal_line, histogram = macd(
        df["close"], sp.trend_macd_fast, sp.trend_macd_slow, sp.trend_macd_signal
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = histogram

    # ADX
    adx_val, plus_di, minus_di = adx(df, rc.adx_period)
    out["adx"] = adx_val
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    # ATR
    out["atr"] = atr(df, rc.atr_period)
    out["atr_pct"] = out["atr"] / df["close"] * 100

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"], rc.bb_period, rc.bb_std)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width(df["close"], rc.bb_period, rc.bb_std)
    out["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Squeeze
    is_sq, sq_mom = squeeze_momentum(df)
    out["squeeze"] = is_sq.astype(int)
    out["squeeze_mom"] = sq_mom

    # Stochastic RSI
    stoch_k, stoch_d = stochastic_rsi(df["close"])
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    # Williams %R
    out["williams_r"] = williams_r(df)

    # CCI
    out["cci"] = cci(df)

    # Volume
    out["obv"] = obv(df)
    out["volume_ratio"] = volume_ratio(df, rc.volume_ma_period)
    out["vwap"] = vwap_rolling(df)
    out["mfi"] = mfi(df)

    # Realized vol
    out["realized_vol"] = realized_volatility(df["close"])

    return out
