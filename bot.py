"""
Autonomous Trading Bot for Small Accounts
Connects to Binance USDT-M Futures and executes trades automatically

IMPORTANT: This bot trades with real money. Use at your own risk.
Always start with testnet and small amounts.
"""

import time
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import threading
from queue import Queue

from config import trading_config, strategy_config, algo_config
from binance_client import (
    BinanceFuturesClient, OrderSide, OrderType, Position, Order
)
from strategy_optimized import SimpleStrategy as SmallAccountStrategy, TradeSignal, SignalType


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Current state of the trading bot"""
    is_running: bool = False
    balance: float = 0.0
    equity: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    current_position: Optional[Dict] = None
    last_signal: Optional[Dict] = None
    last_error: Optional[str] = None
    start_time: datetime = None
    uptime_seconds: float = 0.0


class TradingBot:
    """
    Autonomous trading bot for Binance USDT-M Futures
    Optimized for small accounts starting with ~$4.5
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
        symbols: List[str] = None
    ):
        # API client
        self.client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )

        # Trading strategy
        self.strategy = SmallAccountStrategy()

        # Configuration
        self.testnet = testnet
        self.symbols = symbols or list(strategy_config.trading_pairs)
        self.primary_symbol = strategy_config.primary_pair

        # State
        self.state = BotState()
        self.running = False
        self.stop_event = threading.Event()

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.kline_limit = 500  # Number of candles to fetch

        # Risk management
        self.daily_loss_limit = strategy_config.max_daily_loss
        self.daily_start_balance = 0.0

        # Trade tracking
        self.active_orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []

        # Safety flags
        self.emergency_stop = False
        self.max_consecutive_losses = 5
        self.consecutive_losses = 0

        logger.info(f"Bot initialized - Testnet: {testnet}")
        logger.info(f"Trading symbols: {self.symbols}")

    def start(self):
        """Start the trading bot"""
        logger.info("="*50)
        logger.info("STARTING TRADING BOT")
        logger.info("="*50)

        if not self.client.api_key or not self.client.api_secret:
            logger.error("API credentials not configured!")
            logger.info("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
            return

        try:
            # Initialize
            self._initialize()

            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Main loop
            self.running = True
            self.state.is_running = True
            self.state.start_time = datetime.now()

            logger.info("Bot started successfully")
            self._main_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.state.last_error = str(e)
            raise
        finally:
            self._cleanup()

    def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping bot...")
        self.running = False
        self.stop_event.set()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _initialize(self):
        """Initialize bot state and configuration"""
        logger.info("Initializing bot...")

        # Get account balance
        self.state.balance = self.client.get_balance()
        self.daily_start_balance = self.state.balance
        logger.info(f"Account balance: ${self.state.balance:.4f}")

        if self.state.balance < 1.0:
            logger.warning("Very low balance! Ensure sufficient funds.")

        # Set up leverage and margin for each symbol
        for symbol in self.symbols:
            try:
                self.client.set_leverage(symbol, strategy_config.leverage)
                logger.info(f"{symbol}: Leverage set to {strategy_config.leverage}x")

                self.client.set_margin_type(symbol, strategy_config.margin_type)
                logger.info(f"{symbol}: Margin type set to {strategy_config.margin_type}")
            except Exception as e:
                logger.warning(f"Error setting up {symbol}: {e}")

        # Check for existing positions
        positions = self.client.get_positions()
        if positions:
            logger.info(f"Found {len(positions)} existing position(s)")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.side} {pos.size} @ {pos.entry_price}")
                if pos.symbol in self.symbols:
                    self.state.current_position = {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price
                    }

        # Fetch initial price data
        self._update_price_data()

    def _main_loop(self):
        """Main trading loop"""
        check_interval = 60  # Check every 60 seconds (1 minute)

        while self.running and not self.stop_event.is_set():
            try:
                loop_start = time.time()

                # Update state
                self._update_state()

                # Check safety conditions
                if not self._check_safety():
                    logger.warning("Safety check failed, skipping trade cycle")
                    time.sleep(check_interval)
                    continue

                # Update price data
                self._update_price_data()

                # Check open position
                if self.state.current_position:
                    self._manage_position()
                else:
                    # Look for new trading opportunities
                    self._scan_for_signals()

                # Log status
                self._log_status()

                # Calculate sleep time
                elapsed = time.time() - loop_start
                sleep_time = max(0, check_interval - elapsed)

                if sleep_time > 0:
                    self.stop_event.wait(timeout=sleep_time)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.state.last_error = str(e)
                time.sleep(10)  # Wait before retrying

    def _update_state(self):
        """Update bot state"""
        try:
            # Update balance
            self.state.balance = self.client.get_balance()

            # Calculate PnL
            self.state.daily_pnl = self.state.balance - self.daily_start_balance
            self.state.total_pnl = self.state.balance - strategy_config.initial_balance

            # Update uptime
            if self.state.start_time:
                self.state.uptime_seconds = (datetime.now() - self.state.start_time).total_seconds()

            # Check position
            positions = self.client.get_positions()
            if positions:
                pos = positions[0]  # We only trade one position at a time
                self.state.current_position = {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                self.state.equity = self.state.balance + pos.unrealized_pnl
            else:
                self.state.current_position = None
                self.state.equity = self.state.balance

        except Exception as e:
            logger.error(f"Error updating state: {e}")

    def _update_price_data(self):
        """Fetch latest price data for all symbols"""
        for symbol in self.symbols:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=strategy_config.primary_timeframe,
                    limit=self.kline_limit
                )

                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])

                self.price_data[symbol] = df

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

    def _check_safety(self) -> bool:
        """Check safety conditions before trading"""

        # Emergency stop
        if self.emergency_stop:
            logger.warning("Emergency stop active")
            return False

        # Daily loss limit
        daily_loss_pct = -self.state.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
        if daily_loss_pct >= self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {daily_loss_pct*100:.1f}%")
            return False

        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            return False

        # Minimum balance
        if self.state.balance < strategy_config.min_notional / strategy_config.leverage:
            logger.warning(f"Balance too low: ${self.state.balance:.4f}")
            return False

        return True

    def _scan_for_signals(self):
        """Scan all symbols for trading signals"""
        if self.state.current_position:
            return  # Already in a position

        best_signal = None
        best_symbol = None

        for symbol in self.symbols:
            if symbol not in self.price_data:
                continue

            df = self.price_data[symbol]
            if len(df) < 200:
                continue

            try:
                signal = self.strategy.analyze(df)
                if signal:
                    if best_signal is None or signal.confidence > best_signal.confidence:
                        best_signal = signal
                        best_symbol = symbol
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        if best_signal and best_signal.confidence >= 0.65:
            logger.info(f"Signal found for {best_symbol}: {best_signal.signal_type.name}")
            logger.info(f"Strategy: {best_signal.strategy.value}, Confidence: {best_signal.confidence:.2f}")
            logger.info(f"Reason: {best_signal.reason}")
            self._execute_signal(best_symbol, best_signal)

    def _execute_signal(self, symbol: str, signal: TradeSignal):
        """Execute a trading signal"""
        try:
            # Get current price
            current_price = self.client.get_price(symbol)

            # Calculate position size
            risk_amount = self.state.balance * strategy_config.max_risk_per_trade
            position_value = self.state.balance * strategy_config.leverage * signal.position_size_pct

            # Ensure minimum notional
            position_value = max(position_value, strategy_config.min_notional)

            # Don't exceed available margin
            max_position = self.state.balance * strategy_config.leverage * 0.9
            position_value = min(position_value, max_position)

            quantity = position_value / current_price

            # Determine order side
            side = OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SELL

            logger.info(f"Placing order: {side.value} {quantity:.4f} {symbol} @ market")

            # Place market order for entry
            order = self.client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )

            logger.info(f"Order placed: ID {order.order_id}, Status: {order.status}")

            if order.status in ['FILLED', 'NEW']:
                # Place stop loss
                sl_side = OrderSide.SELL if signal.signal_type == SignalType.LONG else OrderSide.BUY

                try:
                    sl_order = self.client.place_stop_loss(
                        symbol=symbol,
                        side=sl_side,
                        quantity=quantity,
                        stop_price=signal.stop_loss
                    )
                    logger.info(f"Stop loss placed at {signal.stop_loss:.4f}")
                    self.active_orders['sl'] = sl_order
                except Exception as e:
                    logger.error(f"Error placing stop loss: {e}")

                # Place take profit
                try:
                    tp_order = self.client.place_take_profit(
                        symbol=symbol,
                        side=sl_side,
                        quantity=quantity,
                        stop_price=signal.take_profit
                    )
                    logger.info(f"Take profit placed at {signal.take_profit:.4f}")
                    self.active_orders['tp'] = tp_order
                except Exception as e:
                    logger.error(f"Error placing take profit: {e}")

                # Update state
                self.state.current_position = {
                    'symbol': symbol,
                    'side': signal.signal_type.name,
                    'size': quantity,
                    'entry_price': current_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }
                self.state.last_signal = {
                    'type': signal.signal_type.name,
                    'strategy': signal.strategy.value,
                    'confidence': signal.confidence,
                    'reason': signal.reason
                }
                self.state.trades_today += 1

                # Record trade
                self.trade_history.append({
                    'time': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': signal.signal_type.name,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'signal': asdict(signal) if hasattr(signal, '__dict__') else str(signal)
                })

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            self.state.last_error = str(e)

    def _manage_position(self):
        """Manage open position (trailing stop, etc.)"""
        if not self.state.current_position:
            return

        pos = self.state.current_position
        symbol = pos['symbol']

        try:
            # Get current position from exchange
            position = self.client.get_position(symbol)

            if not position:
                # Position closed (by SL or TP)
                logger.info("Position closed")
                self._on_position_closed()
                return

            # Update unrealized PnL
            pos['unrealized_pnl'] = position.unrealized_pnl

            # Check if we should adjust stops (trailing)
            if strategy_config.use_trailing_stop:
                self._update_trailing_stop(symbol, position)

        except Exception as e:
            logger.error(f"Error managing position: {e}")

    def _update_trailing_stop(self, symbol: str, position: Position):
        """Update trailing stop if price moved favorably"""
        if symbol not in self.price_data:
            return

        current_price = self.price_data[symbol]['close'].iloc[-1]
        entry_price = position.entry_price

        if position.side == "LONG":
            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # Move stop if profit > 1%
            if profit_pct > 0.01:
                new_sl = current_price * (1 - strategy_config.trailing_stop_pct)

                # Only move up, never down
                current_sl = self.state.current_position.get('stop_loss', 0)
                if new_sl > current_sl:
                    try:
                        # Cancel old SL
                        if 'sl' in self.active_orders:
                            self.client.cancel_order(symbol, self.active_orders['sl'].order_id)

                        # Place new SL
                        quantity = position.size
                        new_sl_order = self.client.place_stop_loss(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=quantity,
                            stop_price=new_sl
                        )
                        self.active_orders['sl'] = new_sl_order
                        self.state.current_position['stop_loss'] = new_sl
                        logger.info(f"Trailing stop updated to {new_sl:.4f}")
                    except Exception as e:
                        logger.error(f"Error updating trailing stop: {e}")

        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price

            if profit_pct > 0.01:
                new_sl = current_price * (1 + strategy_config.trailing_stop_pct)
                current_sl = self.state.current_position.get('stop_loss', float('inf'))

                if new_sl < current_sl:
                    try:
                        if 'sl' in self.active_orders:
                            self.client.cancel_order(symbol, self.active_orders['sl'].order_id)

                        quantity = position.size
                        new_sl_order = self.client.place_stop_loss(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            stop_price=new_sl
                        )
                        self.active_orders['sl'] = new_sl_order
                        self.state.current_position['stop_loss'] = new_sl
                        logger.info(f"Trailing stop updated to {new_sl:.4f}")
                    except Exception as e:
                        logger.error(f"Error updating trailing stop: {e}")

    def _on_position_closed(self):
        """Handle position closure"""
        # Clear state
        old_pos = self.state.current_position
        self.state.current_position = None
        self.active_orders.clear()

        # Update consecutive losses
        if old_pos and old_pos.get('unrealized_pnl', 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update balance
        self.state.balance = self.client.get_balance()

        logger.info(f"Position closed. New balance: ${self.state.balance:.4f}")
        logger.info(f"Daily PnL: ${self.state.daily_pnl:.4f}")

    def _log_status(self):
        """Log current bot status"""
        logger.info("-" * 40)
        logger.info(f"Balance: ${self.state.balance:.4f} | Daily PnL: ${self.state.daily_pnl:+.4f}")
        if self.state.current_position:
            pos = self.state.current_position
            logger.info(f"Position: {pos['side']} {pos['symbol']} | Entry: {pos['entry_price']:.4f}")
        else:
            logger.info("Position: None")
        logger.info(f"Trades today: {self.state.trades_today}")

    def _cleanup(self):
        """Cleanup on shutdown"""
        logger.info("Cleaning up...")
        self.running = False
        self.state.is_running = False

        # Save trade history
        try:
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info("Trade history saved")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

        logger.info("Bot stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_running': self.state.is_running,
            'balance': self.state.balance,
            'equity': self.state.equity,
            'daily_pnl': self.state.daily_pnl,
            'total_pnl': self.state.total_pnl,
            'trades_today': self.state.trades_today,
            'position': self.state.current_position,
            'last_signal': self.state.last_signal,
            'uptime': self.state.uptime_seconds,
            'testnet': self.testnet
        }


def main():
    """Main entry point"""
    import os

    # Check for API credentials
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')

    # Determine mode
    testnet = os.getenv('TRADING_MODE', 'testnet').lower() != 'live'

    if not api_key or not api_secret:
        print("\n" + "="*60)
        print("BINANCE TRADING BOT - CONFIGURATION REQUIRED")
        print("="*60)
        print("\nTo run the bot, set the following environment variables:")
        print("  export BINANCE_API_KEY='your_api_key'")
        print("  export BINANCE_API_SECRET='your_api_secret'")
        print("\nFor testnet (recommended for testing):")
        print("  Get keys from: https://testnet.binancefuture.com/")
        print("\nFor live trading:")
        print("  export TRADING_MODE='live'")
        print("\nWARNING: This bot trades with real money in live mode!")
        print("="*60)
        return

    print("\n" + "="*60)
    print("BINANCE TRADING BOT")
    print("="*60)
    print(f"Mode: {'TESTNET' if testnet else 'LIVE'}")
    print(f"Initial Balance Target: ${strategy_config.initial_balance}")
    print(f"Leverage: {strategy_config.leverage}x")
    print(f"Trading Pairs: {', '.join(strategy_config.trading_pairs)}")
    print("="*60 + "\n")

    if not testnet:
        print("\n*** WARNING: LIVE TRADING MODE ***")
        print("You are about to trade with REAL MONEY!")
        response = input("Type 'YES' to confirm: ")
        if response != 'YES':
            print("Aborted.")
            return

    # Create and start bot
    bot = TradingBot(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )

    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
