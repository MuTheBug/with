"""
Configuration for Small Account Trading Bot
Optimized for accounts starting with ~$4.5 USDT
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    """Trading configuration parameters"""

    # API Configuration
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")

    # Use testnet for safety (set to False for live trading)
    use_testnet: bool = True

    # Testnet endpoints
    testnet_base_url: str = "https://testnet.binancefuture.com"
    testnet_ws_url: str = "wss://stream.binancefuture.com"

    # Production endpoints
    prod_base_url: str = "https://fapi.binance.com"
    prod_ws_url: str = "wss://fstream.binance.com"

    # Algo orders endpoint (new)
    algo_order_endpoint: str = "/fapi/v1/algo/order"

    @property
    def base_url(self) -> str:
        return self.testnet_base_url if self.use_testnet else self.prod_base_url

    @property
    def ws_url(self) -> str:
        return self.testnet_ws_url if self.use_testnet else self.prod_ws_url


@dataclass
class StrategyConfig:
    """Strategy parameters optimized for small accounts"""

    # Account settings
    initial_balance: float = 4.5  # Starting balance in USDT
    leverage: int = 10  # Moderate leverage for safety
    margin_type: str = "ISOLATED"  # Isolated margin for safety

    # Trading pairs (sorted by volatility potential)
    trading_pairs: tuple = ("SOLUSDT", "XRPUSDT", "BTCUSDT")
    primary_pair: str = "SOLUSDT"  # SOL has good volatility

    # Timeframes
    primary_timeframe: str = "5m"  # 5-minute for entries
    trend_timeframe: str = "1h"  # 1-hour for trend

    # Risk Management - CRITICAL for small accounts
    max_risk_per_trade: float = 0.02  # 2% of account risked per trade
    max_daily_loss: float = 0.10  # 10% max daily drawdown
    max_open_positions: int = 1  # Only 1 position for small account

    # Position sizing - Use risk-based sizing, not percentage of account
    # Position is calculated based on stop distance, not account percentage
    position_size_pct: float = 0.50  # Max 50% of leveraged account
    min_notional: float = 5.0  # Binance minimum

    # Strategy Parameters - Momentum Scalping
    fast_ema: int = 9
    slow_ema: int = 21
    signal_ema: int = 5
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR for volatility-based stops - Tighter for 1h data
    atr_period: int = 14
    atr_multiplier_sl: float = 1.0  # Tighter stop loss
    atr_multiplier_tp: float = 2.0  # 2:1 R:R minimum

    # Entry filters
    min_volume_ratio: float = 1.0  # Lowered for more signals
    trend_filter_enabled: bool = True

    # Exit settings
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.01  # 1% trailing stop
    partial_take_profit: bool = True
    partial_tp_pct: float = 0.5  # Close 50% at first TP

    # Time filters
    avoid_low_volume_hours: bool = True
    low_volume_hours: tuple = (0, 1, 2, 3, 4, 5)  # UTC hours to avoid


@dataclass
class AlgoOrderConfig:
    """Configuration for Binance Algo Orders (TWAP/VP)"""

    # TWAP (Time-Weighted Average Price) settings
    twap_enabled: bool = False  # Enable for larger positions
    twap_duration: int = 300  # 5 minutes

    # VP (Volume Participation) settings
    vp_enabled: bool = False
    vp_rate: float = 0.1  # 10% volume participation

    # For small accounts, we use regular limit orders
    # Algo orders are useful when scaling up


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_balance: float = 4.5
    commission_rate: float = 0.0004  # 0.04% taker fee
    slippage_pct: float = 0.0005  # 0.05% slippage assumption

    # Data paths
    data_dir: str = "/home/user/with"
    btc_data: str = "BTCUSDT_1h_1year.csv"
    sol_data: str = "SOLUSDT_1h_1year.csv"
    xrp_data: str = "XRPUSDT_1h_1year.csv"

    # Backtest settings
    start_date: Optional[str] = None  # Use all data if None
    end_date: Optional[str] = None


# Global configuration instances
trading_config = TradingConfig()
strategy_config = StrategyConfig()
algo_config = AlgoOrderConfig()
backtest_config = BacktestConfig()
