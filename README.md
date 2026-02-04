# Small Account Trading Bot

Autonomous trading bot for Binance USDT-M Futures, optimized for small accounts starting with ~$4.5 USDT.

## Strategy Overview

The bot uses an **Optimized Trend-Following Strategy** that was backtested on 1 year of historical data during a severe bear market:

| Asset | Buy & Hold | Strategy |
|-------|------------|----------|
| SOL   | -55%       | +4.80%   |
| XRP   | -47%       | -5.14%   |
| BTC   | -23%       | -0.21%   |

### Key Features

- **Strict Trend Detection**: Only trades when ALL EMAs (10, 20, 50, 100) are aligned
- **High Confidence Entries**: Requires strong trend strength (>2x ATR separation)
- **Risk Management**: 1.5% risk per trade, tight stop losses
- **Both Directions**: Can go LONG in uptrends and SHORT in downtrends
- **Anti-Overtrading**: Minimum 12 bars between trades

## Files

```
├── bot.py              # Main autonomous trading bot
├── binance_client.py   # Binance API client (supports Algo Orders)
├── config.py           # Configuration settings
├── strategy_optimized.py  # The profitable strategy
├── backtest_v2.py      # Backtesting framework
├── requirements.txt    # Python dependencies
└── *.csv               # Historical data for backtesting
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables:

```bash
# For testnet (recommended for testing)
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_API_SECRET="your_testnet_api_secret"

# For live trading
export BINANCE_API_KEY="your_live_api_key"
export BINANCE_API_SECRET="your_live_api_secret"
export TRADING_MODE="live"
```

Get testnet keys from: https://testnet.binancefuture.com/

## Running the Bot

### Backtest First (Recommended)

```bash
python backtest_v2.py
```

### Run Live Bot

```bash
python bot.py
```

## Strategy Logic

### Entry Conditions

**SHORT (in downtrends):**
- EMA 10 < EMA 20 < EMA 50 < EMA 100
- Price below EMA 20
- Trend strength > 2
- RSI > 45 (pullback) OR price near EMA 20
- Bearish candle confirmation

**LONG (in uptrends):**
- EMA 10 > EMA 20 > EMA 50 > EMA 100
- Price above EMA 20
- Trend strength > 2
- RSI < 55 (pullback) OR price near EMA 20
- Bullish candle confirmation

### Exit Conditions

- Stop Loss: 1.2x ATR from entry
- Take Profit: 3x ATR from entry (2.5:1 R:R)
- Trailing Stop: Activated when profit > 2%

### Risk Management

- Max 1.5% account risk per trade
- Max 10% daily loss limit
- Isolated margin mode
- 10x leverage (configurable)

## Binance Algo Orders Support

The bot supports Binance's new Algo Orders endpoints:
- TWAP (Time-Weighted Average Price)
- VP (Volume Participation)

For small accounts, regular limit/market orders are used. Algo orders become useful when scaling up.

## Disclaimer

**WARNING**: This bot trades with real money. Cryptocurrency trading involves substantial risk of loss.

- Always test on testnet first
- Start with small amounts
- Past performance does not guarantee future results
- The author is not responsible for any losses

## Backtest Results

Full backtest on 1-year hourly data (bear market period):

```
SOL: $4.50 -> $4.72 (+4.80%) | 25 trades | 36% win rate
XRP: $4.50 -> $4.27 (-5.14%) | 17 trades | 29% win rate
BTC: $4.50 -> $4.49 (-0.21%) | 34 trades | 38% win rate

Combined: Near breakeven in a market that dropped 23-55%
```
