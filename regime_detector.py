"""
Multi-timeframe market regime detection engine.

Classifies the current market into one of several regimes using a weighted
scoring system across trend, volatility, momentum, and mean-reversion signals.
Each timeframe contributes to the final regime classification with different
weights (higher TFs get more weight for trend, lower TFs for execution timing).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config import MarketRegime, RegimeConfig, BotConfig
import indicators as ind


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float          # 0-1 confidence in the classification
    trend_score: float         # -1 (bearish) to +1 (bullish)
    volatility_score: float    # 0 (low) to 1 (high)
    momentum_score: float      # -1 to +1
    mean_reversion_score: float  # 0 (not MR) to 1 (strongly MR)
    hurst: float               # Hurst exponent
    details: Dict[str, float] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class RegimeDetector:
    """
    Multi-timeframe market regime detector.
    Combines signals from multiple indicators and timeframes to classify
    the current market regime with a confidence score.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.rc = self.config.regime

    def detect(
        self,
        data_by_tf: Dict[str, pd.DataFrame],
        primary_tf: str = "1h",
    ) -> RegimeResult:
        """
        Detect market regime from multi-timeframe data.

        Args:
            data_by_tf: Dict mapping timeframe string to OHLCV DataFrame
                        with indicators already computed.
            primary_tf: The primary timeframe for regime detection.

        Returns:
            RegimeResult with regime classification and confidence.
        """
        # Weights for each timeframe contribution
        tf_weights = {"5m": 0.10, "15m": 0.20, "1h": 0.40, "4h": 0.30}

        trend_scores = []
        vol_scores = []
        mom_scores = []
        mr_scores = []
        weights = []

        for tf, df in data_by_tf.items():
            if df is None or len(df) < 50:
                continue
            w = tf_weights.get(tf, 0.2)
            weights.append(w)

            ts = self._trend_score(df)
            vs = self._volatility_score(df)
            ms = self._momentum_score(df)
            mrs = self._mean_reversion_score(df)

            trend_scores.append(ts * w)
            vol_scores.append(vs * w)
            mom_scores.append(ms * w)
            mr_scores.append(mrs * w)

        if not weights:
            # Fallback: use whatever data we have
            for tf, df in data_by_tf.items():
                if df is not None and len(df) >= 30:
                    return self._single_tf_detect(df)
            return RegimeResult(
                regime=MarketRegime.CHOPPY, confidence=0.3,
                trend_score=0, volatility_score=0.5,
                momentum_score=0, mean_reversion_score=0, hurst=0.5,
            )

        total_w = sum(weights)
        trend = sum(trend_scores) / total_w
        vol = sum(vol_scores) / total_w
        mom = sum(mom_scores) / total_w
        mr = sum(mr_scores) / total_w

        # Hurst from primary TF
        h = 0.5
        primary_df = data_by_tf.get(primary_tf)
        if primary_df is not None and len(primary_df) >= 100:
            h = ind.hurst_exponent(primary_df["close"], max_lag=40)

        regime, confidence = self._classify(trend, vol, mom, mr, h)

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            trend_score=trend,
            volatility_score=vol,
            momentum_score=mom,
            mean_reversion_score=mr,
            hurst=h,
            details={
                "trend": trend, "vol": vol, "mom": mom,
                "mr": mr, "hurst": h,
            },
        )

    def _single_tf_detect(self, df: pd.DataFrame) -> RegimeResult:
        """Fallback: detect regime from a single timeframe."""
        trend = self._trend_score(df)
        vol = self._volatility_score(df)
        mom = self._momentum_score(df)
        mr = self._mean_reversion_score(df)
        h = ind.hurst_exponent(df["close"], max_lag=40) if len(df) >= 100 else 0.5

        regime, confidence = self._classify(trend, vol, mom, mr, h)
        return RegimeResult(
            regime=regime, confidence=confidence,
            trend_score=trend, volatility_score=vol,
            momentum_score=mom, mean_reversion_score=mr, hurst=h,
        )

    def _trend_score(self, df: pd.DataFrame) -> float:
        """
        Compute trend score from -1 (strong bearish) to +1 (strong bullish).
        Uses ADX, EMA alignment, price vs EMAs.
        """
        scores = []

        # ADX + DI direction
        if "adx" in df.columns and "plus_di" in df.columns:
            adx_val = df["adx"].iloc[-1]
            plus_di = df["plus_di"].iloc[-1]
            minus_di = df["minus_di"].iloc[-1]

            if not np.isnan(adx_val):
                adx_strength = min(adx_val / 50.0, 1.0)  # Normalize to 0-1
                direction = 1.0 if plus_di > minus_di else -1.0
                scores.append(adx_strength * direction)

        # EMA alignment (5 > 21 > 50 > 200 = bullish, reverse = bearish)
        ema_cols = ["ema_5", "ema_21", "ema_50", "ema_200"]
        available = [c for c in ema_cols if c in df.columns]
        if len(available) >= 3:
            vals = [df[c].iloc[-1] for c in available if not np.isnan(df[c].iloc[-1])]
            if len(vals) >= 3:
                # Count how many are in descending order (bullish)
                bullish_count = sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1])
                bearish_count = sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1])
                total = len(vals) - 1
                alignment = (bullish_count - bearish_count) / total
                scores.append(alignment)

        # Price vs EMA 50
        if "ema_50" in df.columns:
            price = df["close"].iloc[-1]
            ema50 = df["ema_50"].iloc[-1]
            if not np.isnan(ema50) and ema50 > 0:
                pct_above = (price - ema50) / ema50
                scores.append(np.clip(pct_above * 10, -1, 1))

        # EMA slope (21-period)
        if "ema_21" in df.columns:
            ema21 = df["ema_21"]
            if len(ema21) >= 10:
                slope = (ema21.iloc[-1] - ema21.iloc[-10]) / ema21.iloc[-10]
                scores.append(np.clip(slope * 20, -1, 1))

        return float(np.mean(scores)) if scores else 0.0

    def _volatility_score(self, df: pd.DataFrame) -> float:
        """
        Compute volatility score from 0 (very low) to 1 (very high).
        Uses BB width, ATR%, realized vol.
        """
        scores = []

        # BB width normalized
        if "bb_width" in df.columns:
            bbw = df["bb_width"].iloc[-1]
            bbw_mean = df["bb_width"].rolling(50).mean().iloc[-1]
            if not np.isnan(bbw) and not np.isnan(bbw_mean) and bbw_mean > 0:
                relative_vol = bbw / bbw_mean
                scores.append(np.clip((relative_vol - 0.5) / 1.5, 0, 1))

        # ATR as % of price
        if "atr_pct" in df.columns:
            atr_pct = df["atr_pct"].iloc[-1]
            if not np.isnan(atr_pct):
                # Typical crypto ATR% ranges: 0.5% (low) to 5%+ (high)
                scores.append(np.clip(atr_pct / 5.0, 0, 1))

        # Realized vol
        if "realized_vol" in df.columns:
            rv = df["realized_vol"].iloc[-1]
            if not np.isnan(rv):
                scores.append(np.clip(rv / 2.0, 0, 1))

        # Squeeze detection (inverse: squeeze = low vol)
        if "squeeze" in df.columns:
            sq = df["squeeze"].iloc[-1]
            if sq == 1:
                scores.append(0.1)  # Very low vol during squeeze
            else:
                scores.append(0.5)

        return float(np.mean(scores)) if scores else 0.5

    def _momentum_score(self, df: pd.DataFrame) -> float:
        """
        Compute momentum score from -1 (strong bearish momentum) to +1 (strong bullish).
        Uses RSI, MACD, Stochastic RSI, volume.
        """
        scores = []

        # RSI centered around 50
        if "rsi_14" in df.columns:
            rsi_val = df["rsi_14"].iloc[-1]
            if not np.isnan(rsi_val):
                scores.append((rsi_val - 50) / 50)

        # MACD histogram direction and magnitude
        if "macd_hist" in df.columns:
            hist = df["macd_hist"].iloc[-1]
            price = df["close"].iloc[-1]
            if not np.isnan(hist) and price > 0:
                norm_hist = hist / price * 1000  # Normalize
                scores.append(np.clip(norm_hist, -1, 1))

        # MACD crossover (recent)
        if "macd" in df.columns and "macd_signal" in df.columns:
            m = df["macd"].iloc[-1]
            s = df["macd_signal"].iloc[-1]
            if not np.isnan(m) and not np.isnan(s):
                scores.append(np.clip((m - s) / abs(s) if s != 0 else 0, -1, 1))

        # Stochastic RSI
        if "stoch_k" in df.columns:
            sk = df["stoch_k"].iloc[-1]
            if not np.isnan(sk):
                scores.append((sk - 50) / 50)

        # Volume momentum
        if "volume_ratio" in df.columns:
            vr = df["volume_ratio"].iloc[-1]
            price_change = df["close"].pct_change().iloc[-1]
            if not np.isnan(vr) and not np.isnan(price_change):
                vol_mom = np.sign(price_change) * min(vr / 2.0, 1.0)
                scores.append(vol_mom)

        return float(np.mean(scores)) if scores else 0.0

    def _mean_reversion_score(self, df: pd.DataFrame) -> float:
        """
        Score from 0 (not mean-reverting) to 1 (strongly mean-reverting).
        Uses BB %B, autocorrelation, distance from VWAP, Hurst.
        """
        scores = []

        # BB %B extremes (price near bands = potential MR opportunity)
        if "bb_pct" in df.columns:
            bb_pct_val = df["bb_pct"].iloc[-1]
            if not np.isnan(bb_pct_val):
                # Distance from 0.5 (center of bands)
                dist = abs(bb_pct_val - 0.5) * 2  # 0 at center, 1 at bands
                scores.append(min(dist, 1.0))

        # RSI at extremes
        if "rsi_14" in df.columns:
            rsi_val = df["rsi_14"].iloc[-1]
            if not np.isnan(rsi_val):
                # How far from 50 (neutral)
                rsi_extreme = abs(rsi_val - 50) / 50
                scores.append(rsi_extreme)

        # Price distance from VWAP
        if "vwap" in df.columns:
            price = df["close"].iloc[-1]
            vwap_val = df["vwap"].iloc[-1]
            if not np.isnan(vwap_val) and vwap_val > 0:
                dist = abs(price - vwap_val) / vwap_val
                scores.append(min(dist * 20, 1.0))

        # Negative autocorrelation suggests mean reversion
        close_returns = df["close"].pct_change()
        if len(close_returns) >= 50:
            autocorr = close_returns.iloc[-50:].autocorr(lag=1)
            if not np.isnan(autocorr):
                # Negative autocorr => mean-reverting
                mr_signal = max(-autocorr, 0)
                scores.append(mr_signal)

        return float(np.mean(scores)) if scores else 0.0

    def _classify(
        self, trend: float, vol: float, mom: float, mr: float, hurst: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Classify into a regime based on aggregate scores.
        Returns (regime, confidence).
        """
        abs_trend = abs(trend)
        trend_dir = 1 if trend > 0 else -1

        # Strong trending
        if abs_trend > 0.5 and hurst > 0.55:
            confidence = min(abs_trend * 0.6 + (hurst - 0.5) * 2 * 0.4, 1.0)
            if abs_trend > 0.7:
                regime = MarketRegime.STRONG_TREND_UP if trend_dir > 0 else MarketRegime.STRONG_TREND_DOWN
                return regime, confidence
            else:
                regime = MarketRegime.WEAK_TREND_UP if trend_dir > 0 else MarketRegime.WEAK_TREND_DOWN
                return regime, confidence * 0.85

        # Breakout detection: low vol transitioning to high vol, with directional momentum
        if vol > 0.55 and abs(mom) > 0.4 and abs_trend > 0.3:
            confidence = min((vol * 0.3 + abs(mom) * 0.4 + abs_trend * 0.3), 1.0)
            regime = MarketRegime.BREAKOUT_UP if mom > 0 else MarketRegime.BREAKOUT_DOWN
            return regime, confidence

        # Range-bound / mean-reverting
        if mr > 0.5 and abs_trend < 0.3 and hurst < 0.5:
            confidence = min(mr * 0.5 + (0.5 - hurst) * 2 * 0.3 + (0.3 - abs_trend) / 0.3 * 0.2, 1.0)
            if vol < 0.35:
                return MarketRegime.RANGE_LOW_VOL, confidence
            else:
                return MarketRegime.RANGE_HIGH_VOL, confidence

        # Low volatility range (even without strong MR signal)
        if vol < 0.3 and abs_trend < 0.25:
            confidence = min((0.3 - vol) / 0.3 * 0.5 + (0.25 - abs_trend) / 0.25 * 0.5, 1.0)
            return MarketRegime.RANGE_LOW_VOL, confidence * 0.7

        # High vol range
        if vol > 0.6 and abs_trend < 0.3:
            confidence = min(vol * 0.5 + (0.3 - abs_trend) / 0.3 * 0.5, 1.0)
            return MarketRegime.RANGE_HIGH_VOL, confidence * 0.7

        # Weak trend
        if abs_trend > 0.2:
            confidence = abs_trend * 0.7
            regime = MarketRegime.WEAK_TREND_UP if trend_dir > 0 else MarketRegime.WEAK_TREND_DOWN
            return regime, confidence

        # Default: choppy
        confidence = max(0.3, 1.0 - abs_trend - abs(mom) - abs(mr - 0.5))
        return MarketRegime.CHOPPY, min(confidence, 0.6)
