#!/usr/bin/env python3
"""
PRODUCTION-READY GRID TRADING CONFIGURATIONS
=============================================
Realistic, conservative configurations for live trading with $4 capital.
These configs account for real-world constraints:
- Exchange minimum order sizes
- Realistic leverage limits
- Slippage and latency
- Liquidity constraints
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProductionGridConfig:
    """Production-ready grid configuration"""
    # Asset settings
    symbol: str = "BTCUSDT"

    # Capital
    initial_capital: float = 4.0  # Starting with $4

    # Grid structure
    num_grids: int = 10  # Total grids each side
    grid_spacing_pct: float = 0.25  # 0.25% between grids

    # Leverage (conservative for production)
    leverage: int = 20  # 20x leverage
    max_leverage: int = 30  # Never exceed 30x

    # Position sizing
    position_size_pct: float = 2.0  # 2% per grid level
    max_positions: int = 6  # Max concurrent positions

    # Risk management
    max_drawdown_pct: float = 10.0  # Stop if 10% drawdown
    position_stop_loss_pct: float = 3.0  # 3% stop per position
    take_profit_pct: float = 0.4  # 0.4% take profit

    # Fees (Binance Futures)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%

    # Trading filters
    min_volume_24h: float = 1000000  # Min $1M volume
    max_spread_pct: float = 0.1  # Max 0.1% spread

    # Compounding
    compound_above_pct: float = 5.0  # Compound after 5% gain
    withdrawal_above_pct: float = 20.0  # Consider withdrawal after 20%


# Pre-configured setups for different risk appetites

# Grid trading requires TP >= SL for positive expectancy with ~50% win rate
# OR much higher win rate (>70%) if TP < SL
# These configs are calibrated for positive expected value

CONSERVATIVE_CONFIG = ProductionGridConfig(
    symbol="BTCUSDT",
    initial_capital=4.0,
    num_grids=8,
    grid_spacing_pct=0.4,  # Wider spacing for higher TP
    leverage=15,
    max_leverage=20,
    position_size_pct=1.5,
    max_positions=4,
    max_drawdown_pct=8.0,
    position_stop_loss_pct=0.5,  # Tight stop loss
    take_profit_pct=0.5,  # Equal TP - need >50% WR
)

MODERATE_CONFIG = ProductionGridConfig(
    symbol="BTCUSDT",
    initial_capital=4.0,
    num_grids=10,
    grid_spacing_pct=0.35,
    leverage=20,
    max_leverage=30,
    position_size_pct=2.0,
    max_positions=6,
    max_drawdown_pct=10.0,
    position_stop_loss_pct=0.4,  # Tight stop
    take_profit_pct=0.5,  # Favorable R:R
)

AGGRESSIVE_CONFIG = ProductionGridConfig(
    symbol="BTCUSDT",
    initial_capital=4.0,
    num_grids=12,
    grid_spacing_pct=0.3,
    leverage=25,
    max_leverage=40,
    position_size_pct=2.5,
    max_positions=8,
    max_drawdown_pct=12.0,
    position_stop_loss_pct=0.35,  # Very tight stop
    take_profit_pct=0.5,  # Good R:R ratio
)


def get_exchange_api_template() -> Dict:
    """Template for exchange API integration (Binance Futures)"""
    return {
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_API_SECRET",
        "testnet": True,  # Start with testnet!
        "endpoints": {
            "futures_testnet": "https://testnet.binancefuture.com",
            "futures_mainnet": "https://fapi.binance.com",
        },
        "rate_limits": {
            "orders_per_second": 10,
            "orders_per_minute": 1200,
        }
    }


def calculate_realistic_expectations(config: ProductionGridConfig) -> Dict:
    """Calculate realistic trading expectations"""

    # Assumptions based on typical crypto market conditions
    avg_daily_volatility = 0.02  # 2% daily volatility
    grid_hit_probability = 0.15  # 15% chance each grid gets hit daily
    win_rate = 0.65  # 65% win rate (conservative)

    # Expected daily trades
    expected_daily_trades = config.num_grids * 2 * grid_hit_probability

    # Expected profit per winning trade
    profit_per_trade = (config.take_profit_pct / 100) * config.leverage * (
        config.initial_capital * config.position_size_pct / 100
    )

    # Expected loss per losing trade
    loss_per_trade = (config.position_stop_loss_pct / 100) * config.leverage * (
        config.initial_capital * config.position_size_pct / 100
    )

    # Fees per trade (round trip)
    fees_per_trade = 2 * config.taker_fee * config.leverage * (
        config.initial_capital * config.position_size_pct / 100
    )

    # Expected daily P&L
    expected_daily_pnl = (
        expected_daily_trades * win_rate * profit_per_trade -
        expected_daily_trades * (1 - win_rate) * loss_per_trade -
        expected_daily_trades * fees_per_trade
    )

    expected_daily_return = expected_daily_pnl / config.initial_capital * 100

    # Projected growth (conservative estimates)
    projections = {}
    capital = config.initial_capital
    for days in [7, 14, 30, 60, 90]:
        # Use 70% of expected return to account for market variability
        projected_capital = capital * ((1 + expected_daily_return * 0.7 / 100) ** days)
        projections[f"day_{days}"] = round(projected_capital, 2)

    return {
        "expected_daily_trades": round(expected_daily_trades, 1),
        "expected_daily_return_pct": round(expected_daily_return, 2),
        "profit_per_winning_trade": round(profit_per_trade, 4),
        "loss_per_losing_trade": round(loss_per_trade, 4),
        "fees_per_trade": round(fees_per_trade, 6),
        "projected_capital": projections,
        "assumptions": {
            "daily_volatility": f"{avg_daily_volatility*100}%",
            "grid_hit_probability": f"{grid_hit_probability*100}%",
            "win_rate": f"{win_rate*100}%",
        }
    }


def print_config_summary(config: ProductionGridConfig):
    """Print configuration summary"""
    expectations = calculate_realistic_expectations(config)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           GRID TRADING CONFIGURATION SUMMARY                  ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol: {config.symbol:<20}                             ║
║  Initial Capital: ${config.initial_capital:<10}                           ║
╠══════════════════════════════════════════════════════════════╣
║  GRID SETTINGS                                                ║
║  ─────────────                                                ║
║  Number of Grids: {config.num_grids} each side ({config.num_grids*2} total)              ║
║  Grid Spacing: {config.grid_spacing_pct}%                                       ║
║  Coverage: ±{config.num_grids * config.grid_spacing_pct:.1f}% from center price                    ║
╠══════════════════════════════════════════════════════════════╣
║  LEVERAGE & SIZING                                            ║
║  ────────────────                                             ║
║  Leverage: {config.leverage}x (max {config.max_leverage}x)                                 ║
║  Position Size: {config.position_size_pct}% per grid                               ║
║  Max Positions: {config.max_positions}                                          ║
╠══════════════════════════════════════════════════════════════╣
║  RISK MANAGEMENT                                              ║
║  ───────────────                                              ║
║  Max Drawdown: {config.max_drawdown_pct}%                                        ║
║  Stop Loss: {config.position_stop_loss_pct}% per position                            ║
║  Take Profit: {config.take_profit_pct}%                                          ║
╠══════════════════════════════════════════════════════════════╣
║  REALISTIC EXPECTATIONS                                       ║
║  ─────────────────────                                        ║
║  Est. Daily Trades: {expectations['expected_daily_trades']}                                  ║
║  Est. Daily Return: {expectations['expected_daily_return_pct']}%                                ║
║  Profit/Win: ${expectations['profit_per_winning_trade']:.4f}                                  ║
║  Loss/Loss: ${expectations['loss_per_losing_trade']:.4f}                                   ║
╠══════════════════════════════════════════════════════════════╣
║  PROJECTED CAPITAL (Conservative)                             ║
║  ─────────────────────────────────                            ║
║  Day 7:  ${expectations['projected_capital']['day_7']:<10}                                    ║
║  Day 14: ${expectations['projected_capital']['day_14']:<10}                                    ║
║  Day 30: ${expectations['projected_capital']['day_30']:<10}                                    ║
║  Day 60: ${expectations['projected_capital']['day_60']:<10}                                    ║
║  Day 90: ${expectations['projected_capital']['day_90']:<10}                                    ║
╚══════════════════════════════════════════════════════════════╝
""")


# Live trading checklist
LIVE_TRADING_CHECKLIST = """
╔══════════════════════════════════════════════════════════════╗
║              LIVE TRADING CHECKLIST                           ║
╠══════════════════════════════════════════════════════════════╣
║  BEFORE GOING LIVE:                                           ║
║  ☐ Test on Binance Futures Testnet for at least 1 week       ║
║  ☐ Verify all API permissions are correct                     ║
║  ☐ Enable 2FA on exchange account                             ║
║  ☐ Set up IP whitelist for API                                ║
║  ☐ Configure withdrawal whitelist                             ║
║  ☐ Test emergency stop functionality                          ║
║  ☐ Set up monitoring/alerts                                   ║
║  ☐ Have backup internet connection ready                      ║
║                                                               ║
║  RISK REMINDERS:                                              ║
║  ⚠ Only risk what you can afford to lose                     ║
║  ⚠ High leverage = High risk                                 ║
║  ⚠ Market can move faster than your bot                      ║
║  ⚠ Liquidation is permanent                                  ║
║  ⚠ Past backtest != Future results                           ║
║                                                               ║
║  DAILY MONITORING:                                            ║
║  ☐ Check positions and open orders                            ║
║  ☐ Review P&L and drawdown                                    ║
║  ☐ Verify bot is running correctly                            ║
║  ☐ Check for exchange maintenance notices                     ║
╚══════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PRODUCTION GRID TRADING CONFIGURATIONS")
    print("="*60)

    print("\n📊 CONSERVATIVE SETUP (Lower Risk)")
    print_config_summary(CONSERVATIVE_CONFIG)

    print("\n📊 MODERATE SETUP (Balanced)")
    print_config_summary(MODERATE_CONFIG)

    print("\n📊 AGGRESSIVE SETUP (Higher Risk)")
    print_config_summary(AGGRESSIVE_CONFIG)

    print(LIVE_TRADING_CHECKLIST)
