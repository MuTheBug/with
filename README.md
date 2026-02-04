# Futures Neutral Grid Trading System

A comprehensive, fully autonomous grid trading system for Binance Futures with backtesting, optimization, and **live trading** capabilities. Designed for small capital accounts ($4+) with low drawdown.

## Features

- **Fully Autonomous Live Trading**: Auto-places orders on Binance Futures
- **Multi-Layer Grid System**: Inner scalping grids + outer swing grids
- **Dynamic Leverage**: Automatically adjusts based on volatility and market conditions
- **Advanced Risk Management**: Multiple safeguards including max drawdown, position stops, and trailing stops
- **ML-Based Signals**: Market regime detection and signal generation
- **Capital Compounding**: Automatic profit compounding for rapid growth
- **Anti-Liquidation System**: Proactive risk management to prevent liquidation
- **Real-time Monitoring**: Dashboard with alerts (Telegram/Discord)
- **Binance Algo API Support**: Uses Binance's native grid trading endpoints

## System Components

### Backtesting & Analysis
| File | Description |
|------|-------------|
| `grid_trading_system.py` | Core grid trading engine with backtesting |
| `ultra_optimized_grid.py` | Multi-layer ultra-optimized strategy |
| `advanced_grid_features.py` | ML signals, regime detection |
| `production_config.py` | Production configurations |
| `run_grid_analysis.py` | Main analysis runner |

### Live Trading
| File | Description |
|------|-------------|
| `binance_connector.py` | Binance Futures API connector (REST + WebSocket) |
| `live_grid_bot.py` | Autonomous trading bot |
| `monitoring.py` | Real-time monitoring & alerts |
| `start_bot.py` | Easy launcher script |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Backtests (Recommended First)

```bash
# Complete analysis with optimization
python3 run_grid_analysis.py

# View production configurations
python3 production_config.py
```

### 3. Setup Live Trading

```bash
# Create configuration file
python3 start_bot.py --mode setup

# Edit config.json with your API keys
nano config.json

# Start on TESTNET first!
python3 start_bot.py
```

### 4. Or Use Environment Variables

```bash
export BINANCE_API_KEY='your_api_key'
export BINANCE_API_SECRET='your_api_secret'
export BINANCE_TESTNET='true'  # Use testnet first!
export TRADING_SYMBOL='BTCUSDT'
export LEVERAGE='20'
export TOTAL_INVESTMENT='4.0'

python3 start_bot.py
```

## Configuration

### config.json Example

```json
{
  "binance": {
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "testnet": true
  },
  "trading": {
    "symbol": "BTCUSDT",
    "total_investment": 4.0,
    "leverage": 20,
    "num_grids": 10,
    "grid_spacing_pct": 0.3,
    "max_drawdown_pct": 10.0,
    "position_stop_loss_pct": 3.0,
    "take_profit_pct": 0.5,
    "use_trailing_stop": true
  },
  "alerts": {
    "telegram_enabled": false,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "discord_enabled": false,
    "discord_webhook_url": ""
  }
}
```

### Grid Configuration Options

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Leverage | 15x | 20x | 25x |
| Max Leverage | 20x | 30x | 40x |
| Grid Spacing | 0.4% | 0.35% | 0.3% |
| Stop Loss | 0.5% | 0.4% | 0.35% |
| Take Profit | 0.5% | 0.5% | 0.5% |
| Max Drawdown | 8% | 10% | 12% |

## Live Trading Modes

### Custom Grid Bot (Recommended)
Our implementation with full control over strategy:

```bash
python3 start_bot.py --mode custom
```

### Binance Native Grid Algo
Uses Binance's built-in grid trading API:

```bash
python3 start_bot.py --mode native
```

## Monitoring & Alerts

### Telegram Setup
1. Create a bot with @BotFather
2. Get your chat ID from @userinfobot
3. Add to config:
```json
{
  "alerts": {
    "telegram_enabled": true,
    "telegram_bot_token": "YOUR_BOT_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID"
  }
}
```

### Discord Setup
1. Create a webhook in your Discord server
2. Add to config:
```json
{
  "alerts": {
    "discord_enabled": true,
    "discord_webhook_url": "YOUR_WEBHOOK_URL"
  }
}
```

## Risk Management

The system includes multiple layers of protection:

1. **Position-Level**: Individual stop-loss and take-profit per trade
2. **Portfolio-Level**: Maximum drawdown circuit breaker
3. **Daily Limits**: Max daily loss protection
4. **Dynamic Sizing**: Reduces position size during drawdowns
5. **Trailing Stops**: Lock in profits on winning trades
6. **Leverage Adjustment**: Lower leverage in high volatility
7. **Emergency Close**: Auto-close on critical loss threshold

## API Endpoints Used

### REST API
- Account: `/fapi/v2/account`, `/fapi/v2/balance`
- Orders: `/fapi/v1/order`, `/fapi/v1/allOpenOrders`
- Leverage: `/fapi/v1/leverage`
- Market Data: `/fapi/v1/ticker/price`, `/fapi/v1/klines`

### Algo API (Grid Trading)
- Place Grid: `/fapi/v1/algo/futures/grid`
- Modify Grid: `PUT /fapi/v1/algo/futures/grid`
- Cancel Grid: `DELETE /fapi/v1/algo/futures/grid`
- Get Orders: `/fapi/v1/algo/futures/grid/openOrders`

### WebSocket Streams
- Price: `<symbol>@ticker`
- Mark Price: `<symbol>@markPrice`
- User Data: Order updates, account updates

## Important Disclaimers

⚠️ **RISK WARNING**

1. **Past performance does not guarantee future results**
2. **Futures trading involves significant risk of loss**
3. **High leverage amplifies both gains AND losses**
4. **Never risk more than you can afford to lose**
5. **Backtests do not account for slippage, latency, or liquidity**
6. **Market conditions can change rapidly**
7. **ALWAYS start with TESTNET before using real capital**

## Pre-Live Trading Checklist

- [ ] Test on Binance Futures Testnet for at least 1 week
- [ ] Verify all API permissions (Futures trading enabled)
- [ ] Enable 2FA on exchange account
- [ ] Set up IP whitelist for API
- [ ] Configure withdrawal whitelist
- [ ] Test emergency stop functionality
- [ ] Set up monitoring/alerts (Telegram/Discord)
- [ ] Have backup internet connection ready
- [ ] Understand all risks involved

## File Structure

```
├── grid_trading_system.py    # Core grid trading engine
├── ultra_optimized_grid.py   # Ultra-optimized strategy
├── advanced_grid_features.py # ML signals and advanced features
├── production_config.py      # Production configurations
├── run_grid_analysis.py      # Main analysis runner
├── binance_connector.py      # Binance API connector
├── live_grid_bot.py          # Live trading bot
├── monitoring.py             # Monitoring & alerts
├── start_bot.py              # Easy launcher
├── requirements.txt          # Python dependencies
├── config.json               # Configuration file
├── BTCUSDT_1h_1year.csv      # Historical data
├── SOLUSDT_1h_1year.csv      # Historical data
├── XRPUSDT_1h_1year.csv      # Historical data
└── README.md                 # Documentation
```

## Dependencies

```bash
pip install pandas numpy aiohttp websockets
```

## License

For educational and research purposes only. Not financial advice.
