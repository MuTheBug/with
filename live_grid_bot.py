#!/usr/bin/env python3
"""
AUTONOMOUS LIVE GRID TRADING BOT
=================================
Fully autonomous grid trading bot for Binance Futures.
Automatically manages orders, positions, and risk according to strategy.

Features:
- Real-time price monitoring via WebSocket
- Automatic grid order placement and management
- Dynamic leverage and position sizing
- Risk management with stop-loss and take-profit
- Position tracking and P&L calculation
- Automatic error recovery and reconnection
- Logging and monitoring
"""

import asyncio
import json
import time
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from binance_connector import (
    BinanceClient, BinanceConfig, BinanceFuturesREST,
    OrderSide, OrderType, PositionSide, TimeInForce,
    kline_stream, ticker_stream, mark_price_stream
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grid_bot.log')
    ]
)
logger = logging.getLogger('GridBot')


class GridOrderStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class GridLevel:
    """Represents a single grid level"""
    id: str
    price: float
    side: str  # 'BUY' or 'SELL'
    quantity: float
    status: GridOrderStatus = GridOrderStatus.PENDING
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None


@dataclass
class Position:
    """Active position tracking"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    leverage: int
    unrealized_pnl: float = 0.0
    liquidation_price: float = 0.0
    margin_type: str = "CROSSED"


@dataclass
class LiveGridConfig:
    """Configuration for live grid trading"""
    # Trading pair - Use XRPUSDT for small accounts ($4)
    # BTCUSDT requires $100 minimum notional
    symbol: str = "XRPUSDT"

    # Grid parameters
    num_grids: int = 8
    grid_spacing_pct: float = 0.4
    use_dynamic_spacing: bool = True

    # Position sizing
    total_investment: float = 4.0  # Total USDT to use
    position_size_pct: float = 3.0  # % per grid level
    max_positions: int = 6

    # Leverage
    leverage: int = 20
    max_leverage: int = 50

    # Risk management
    max_drawdown_pct: float = 10.0
    position_stop_loss_pct: float = 3.0
    take_profit_pct: float = 0.5
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.3

    # Grid management
    rebalance_threshold_pct: float = 5.0
    max_one_sided_fills: int = 4

    # Execution
    order_type: str = "LIMIT"  # LIMIT or MARKET
    time_in_force: str = "GTC"
    reduce_only_exits: bool = True

    # Safety
    emergency_stop_loss_pct: float = 15.0
    max_daily_loss_pct: float = 8.0
    pause_on_high_volatility: bool = True
    high_volatility_threshold: float = 5.0  # 5% in 1 hour

    # Monitoring
    heartbeat_interval: int = 60  # seconds
    position_check_interval: int = 10  # seconds


# Symbol-specific settings
SYMBOL_CONFIG = {
    'BTCUSDT': {'min_notional': 100, 'tick_size': 0.10, 'qty_precision': 3, 'price_precision': 2},
    'ETHUSDT': {'min_notional': 20, 'tick_size': 0.01, 'qty_precision': 3, 'price_precision': 2},
    'SOLUSDT': {'min_notional': 5, 'tick_size': 0.01, 'qty_precision': 0, 'price_precision': 2},
    'XRPUSDT': {'min_notional': 5, 'tick_size': 0.0001, 'qty_precision': 1, 'price_precision': 4},
    'DOGEUSDT': {'min_notional': 5, 'tick_size': 0.00001, 'qty_precision': 0, 'price_precision': 5},
}


def get_symbol_config(symbol: str) -> dict:
    """Get symbol-specific configuration"""
    return SYMBOL_CONFIG.get(symbol, {'min_notional': 5, 'tick_size': 0.0001, 'qty_precision': 1, 'price_precision': 4})


class RiskManager:
    """Live risk management system"""

    def __init__(self, config: LiveGridConfig):
        self.config = config
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.is_trading_allowed = True
        self.stop_reason = ""
        self.price_history = deque(maxlen=60)  # Last 60 prices for volatility

    def update_balance(self, balance: float):
        """Update balance tracking"""
        self.current_balance = balance

        if self.initial_balance == 0:
            self.initial_balance = balance
            self.daily_start_balance = balance

        if balance > self.peak_balance:
            self.peak_balance = balance

        # Calculate drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - balance) / self.peak_balance * 100
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Calculate daily P&L
        self.daily_pnl = (balance - self.daily_start_balance) / self.daily_start_balance * 100
        self.total_pnl = (balance - self.initial_balance) / self.initial_balance * 100

        # Check risk limits
        self._check_risk_limits()

    def update_price(self, price: float):
        """Update price history for volatility calculation"""
        self.price_history.append(price)

    def calculate_volatility(self) -> float:
        """Calculate recent volatility"""
        if len(self.price_history) < 10:
            return 0.0

        prices = list(self.price_history)
        returns = [(prices[i] - prices[i-1]) / prices[i-1]
                   for i in range(1, len(prices))]

        if not returns:
            return 0.0

        import statistics
        return statistics.stdev(returns) * 100  # As percentage

    def _check_risk_limits(self):
        """Check if any risk limits are breached"""
        # Max drawdown check
        if self.current_drawdown >= self.config.max_drawdown_pct:
            self.is_trading_allowed = False
            self.stop_reason = f"Max drawdown reached: {self.current_drawdown:.2f}%"
            logger.warning(self.stop_reason)
            return

        # Daily loss check
        if self.daily_pnl <= -self.config.max_daily_loss_pct:
            self.is_trading_allowed = False
            self.stop_reason = f"Max daily loss reached: {self.daily_pnl:.2f}%"
            logger.warning(self.stop_reason)
            return

        # Volatility check
        if self.config.pause_on_high_volatility:
            volatility = self.calculate_volatility()
            if volatility > self.config.high_volatility_threshold:
                self.is_trading_allowed = False
                self.stop_reason = f"High volatility: {volatility:.2f}%"
                logger.warning(self.stop_reason)
                return

        self.is_trading_allowed = True
        self.stop_reason = ""

    def can_open_position(self, side: str, current_positions: List[Position]) -> bool:
        """Check if a new position can be opened"""
        if not self.is_trading_allowed:
            return False

        # Count current positions
        if len(current_positions) >= self.config.max_positions:
            return False

        return True

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters"""
        if self.current_balance <= 0 or price <= 0:
            return 0.0

        base_size = (self.current_balance * self.config.position_size_pct / 100)
        leveraged_value = base_size * self.config.leverage
        quantity = leveraged_value / price

        # Reduce size if in drawdown
        if self.current_drawdown > self.config.max_drawdown_pct / 2:
            quantity *= 0.5

        # Binance minimum quantity checks (approximate)
        # BTC: 0.001, ETH: 0.001, most others: varies
        min_qty = 0.001
        if quantity < min_qty:
            quantity = min_qty

        return max(0.001, quantity)

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        self.daily_start_balance = self.current_balance
        self.daily_pnl = 0.0


class OrderManager:
    """Manages order placement and tracking"""

    def __init__(self, rest_client: BinanceFuturesREST, config: LiveGridConfig):
        self.rest = rest_client
        self.config = config
        self.active_orders: Dict[str, GridLevel] = {}
        self.order_history: List[Dict] = []

    async def place_grid_order(self, level: GridLevel) -> bool:
        """Place a single grid order"""
        try:
            # Validate inputs
            if level.price <= 0:
                logger.error(f"Invalid price: {level.price}")
                return False

            if level.quantity <= 0:
                logger.error(f"Invalid quantity: {level.quantity}")
                return False

            # Get symbol-specific configuration
            sym_config = get_symbol_config(self.config.symbol)
            min_notional = sym_config['min_notional']
            tick_size = sym_config['tick_size']
            qty_precision = sym_config['qty_precision']
            price_precision = sym_config['price_precision']

            # Calculate notional value (price * quantity)
            notional = level.price * level.quantity

            # Adjust quantity to meet minimum notional
            if notional < min_notional:
                level.quantity = (min_notional / level.price) * 1.1  # 10% buffer
                logger.info(f"Adjusted quantity to {level.quantity:.{qty_precision}f} to meet min notional ${min_notional}")

            client_order_id = f"GRID_{level.id}_{int(time.time()*1000)}"

            # Round to proper precision for this symbol
            quantity = round(level.quantity, qty_precision)

            # Round price to tick size
            price = round(level.price / tick_size) * tick_size
            price = round(price, price_precision)

            # Final validation
            if quantity <= 0:
                logger.error(f"Quantity rounded to zero or negative: {quantity}")
                return False

            order = await self.rest.place_order(
                symbol=self.config.symbol,
                side=OrderSide.BUY if level.side == 'BUY' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                time_in_force=TimeInForce.GTC,
                client_order_id=client_order_id
            )

            level.order_id = order['orderId']
            level.client_order_id = client_order_id
            level.status = GridOrderStatus.ACTIVE
            self.active_orders[client_order_id] = level

            logger.info(f"Placed {level.side} order: {quantity} @ ${price} (ID: {order['orderId']})")
            return True

        except Exception as e:
            logger.error(f"Failed to place order at {level.price}: {e}")
            return False

    async def place_all_grid_orders(self, levels: List[GridLevel]) -> int:
        """Place all grid orders"""
        success_count = 0

        # Place orders in batches of 5 (Binance limit)
        for i in range(0, len(levels), 5):
            batch = levels[i:i+5]
            tasks = [self.place_grid_order(level) for level in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count += sum(1 for r in results if r is True)
            await asyncio.sleep(0.1)  # Small delay between batches

        return success_count

    async def cancel_order(self, level: GridLevel) -> bool:
        """Cancel a specific order"""
        try:
            if level.order_id:
                await self.rest.cancel_order(
                    symbol=self.config.symbol,
                    order_id=level.order_id
                )
                level.status = GridOrderStatus.CANCELLED
                if level.client_order_id in self.active_orders:
                    del self.active_orders[level.client_order_id]
                logger.info(f"Cancelled order {level.order_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to cancel order {level.order_id}: {e}")
        return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            await self.rest.cancel_all_orders(self.config.symbol)
            self.active_orders.clear()
            logger.info("Cancelled all orders")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    async def place_stop_loss(self, position: Position) -> bool:
        """Place stop-loss order for a position"""
        try:
            # Get symbol-specific precision
            sym_config = get_symbol_config(self.config.symbol)
            tick_size = sym_config['tick_size']
            qty_precision = sym_config['qty_precision']
            price_precision = sym_config['price_precision']

            if position.side == 'LONG':
                stop_price = position.entry_price * (1 - self.config.position_stop_loss_pct / 100)
                side = OrderSide.SELL
            else:
                stop_price = position.entry_price * (1 + self.config.position_stop_loss_pct / 100)
                side = OrderSide.BUY

            # Round to tick size
            stop_price = round(stop_price / tick_size) * tick_size
            stop_price = round(stop_price, price_precision)
            quantity = round(abs(position.quantity), qty_precision)

            await self.rest.place_order(
                symbol=self.config.symbol,
                side=side,
                order_type=OrderType.STOP_MARKET,
                quantity=quantity,
                stop_price=stop_price,
                reduce_only=True
            )

            logger.info(f"Placed stop-loss at ${stop_price:.{price_precision}f}")
            return True

        except Exception as e:
            logger.error(f"Failed to place stop-loss: {e}")
            return False

    async def place_take_profit(self, position: Position) -> bool:
        """Place take-profit order for a position"""
        try:
            # Get symbol-specific precision
            sym_config = get_symbol_config(self.config.symbol)
            tick_size = sym_config['tick_size']
            qty_precision = sym_config['qty_precision']
            price_precision = sym_config['price_precision']

            if position.side == 'LONG':
                tp_price = position.entry_price * (1 + self.config.take_profit_pct / 100)
                side = OrderSide.SELL
            else:
                tp_price = position.entry_price * (1 - self.config.take_profit_pct / 100)
                side = OrderSide.BUY

            # Round to tick size
            tp_price = round(tp_price / tick_size) * tick_size
            tp_price = round(tp_price, price_precision)
            quantity = round(abs(position.quantity), qty_precision)

            await self.rest.place_order(
                symbol=self.config.symbol,
                side=side,
                order_type=OrderType.TAKE_PROFIT_MARKET,
                quantity=quantity,
                stop_price=tp_price,
                reduce_only=True
            )

            logger.info(f"Placed take-profit at ${tp_price:.{price_precision}f}")
            return True

        except Exception as e:
            logger.error(f"Failed to place take-profit: {e}")
            return False

    def handle_order_update(self, data: Dict):
        """Handle order update from WebSocket"""
        order_data = data.get('o', {})
        client_order_id = order_data.get('c', '')
        status = order_data.get('X', '')
        fill_price = float(order_data.get('ap', 0) or order_data.get('p', 0))

        if client_order_id in self.active_orders:
            level = self.active_orders[client_order_id]

            if status == 'FILLED':
                level.status = GridOrderStatus.FILLED
                level.fill_price = fill_price
                level.fill_time = datetime.now()
                logger.info(f"Order filled: {client_order_id} at {fill_price}")

                self.order_history.append({
                    'time': datetime.now().isoformat(),
                    'side': level.side,
                    'price': fill_price,
                    'quantity': level.quantity,
                    'order_id': level.order_id
                })

            elif status == 'CANCELED':
                level.status = GridOrderStatus.CANCELLED
                del self.active_orders[client_order_id]

            elif status == 'EXPIRED':
                level.status = GridOrderStatus.CANCELLED
                del self.active_orders[client_order_id]


class LiveGridBot:
    """Main autonomous grid trading bot"""

    def __init__(self, config: LiveGridConfig, binance_config: BinanceConfig):
        self.config = config
        self.binance_config = binance_config
        self.client: Optional[BinanceClient] = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager = RiskManager(config)

        # State
        self.is_running = False
        self.is_initialized = False
        self.current_price = 0.0
        self.grid_center = 0.0
        self.grid_levels: List[GridLevel] = []
        self.positions: List[Position] = []

        # Statistics
        self.start_time: Optional[datetime] = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    async def initialize(self):
        """Initialize the bot"""
        logger.info("Initializing Grid Bot...")

        # Create Binance client
        self.client = BinanceClient(self.binance_config)
        self.order_manager = OrderManager(self.client.rest, self.config)

        # Set leverage
        try:
            await self.client.rest.set_leverage(
                self.config.symbol,
                self.config.leverage
            )
            logger.info(f"Set leverage to {self.config.leverage}x")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")

        # Set margin type to CROSSED
        try:
            await self.client.rest.set_margin_type(self.config.symbol, "CROSSED")
        except Exception as e:
            logger.warning(f"Could not set margin type: {e}")

        # Get initial balance
        balance_info = await self.client.rest.get_balance()
        usdt_balance = next(
            (b for b in balance_info if b['asset'] == 'USDT'),
            None
        )
        if usdt_balance:
            self.risk_manager.update_balance(float(usdt_balance['balance']))
            logger.info(f"Initial balance: ${self.risk_manager.current_balance:.2f}")

        # Get current price
        ticker = await self.client.rest.get_ticker_price(self.config.symbol)
        self.current_price = float(ticker['price'])
        logger.info(f"Current {self.config.symbol} price: ${self.current_price:.2f}")

        self.is_initialized = True
        logger.info("Bot initialized successfully")

    async def setup_grid(self):
        """Setup initial grid orders"""
        logger.info("Setting up grid...")

        # Cancel any existing orders
        await self.order_manager.cancel_all_orders()

        # Validate current price
        if self.current_price <= 0:
            logger.error("Invalid current price, cannot setup grid")
            return False

        # Set grid center to current price
        self.grid_center = self.current_price

        # Calculate grid levels
        self.grid_levels = []
        grid_id = 0

        # Calculate spacing (ensure it's reasonable)
        if self.config.use_dynamic_spacing:
            volatility = self.risk_manager.calculate_volatility()
            spacing = max(self.config.grid_spacing_pct, volatility * 0.5)
        else:
            spacing = self.config.grid_spacing_pct

        # Cap spacing to prevent negative prices
        max_spacing = 90.0 / self.config.num_grids  # Max 90% total range
        spacing = min(spacing, max_spacing)

        logger.info(f"Grid spacing: {spacing:.3f}%")

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(self.current_price)

        if position_size <= 0:
            logger.error(f"Invalid position size: {position_size}, balance: {self.risk_manager.current_balance}")
            return False

        logger.info(f"Position size per grid: {position_size:.6f}")

        # Create buy levels (below current price)
        for i in range(1, self.config.num_grids + 1):
            price = self.grid_center * (1 - spacing * i / 100)
            if price > 0:  # Only add valid prices
                self.grid_levels.append(GridLevel(
                    id=f"B{grid_id}",
                    price=price,
                    side='BUY',
                    quantity=position_size
                ))
                grid_id += 1
            else:
                logger.warning(f"Skipping invalid buy price: {price}")

        # Create sell levels (above current price)
        for i in range(1, self.config.num_grids + 1):
            price = self.grid_center * (1 + spacing * i / 100)
            self.grid_levels.append(GridLevel(
                id=f"S{grid_id}",
                price=price,
                side='SELL',
                quantity=position_size
            ))
            grid_id += 1

        logger.info(f"Created {len(self.grid_levels)} grid levels")
        logger.info(f"Price range: ${min(l.price for l in self.grid_levels):.2f} - ${max(l.price for l in self.grid_levels):.2f}")

        # Place all orders
        success_count = await self.order_manager.place_all_grid_orders(self.grid_levels)
        logger.info(f"Placed {success_count}/{len(self.grid_levels)} grid orders")

        return success_count > 0

    async def handle_price_update(self, data: Dict):
        """Handle real-time price update from ticker stream"""
        new_price = 0.0

        # Binance ticker stream format:
        # 'c' = last price (current price) - USE THIS
        # 'p' = price change (NOT the current price!)
        # For ticker stream, always use 'c'
        if 'c' in data:
            new_price = float(data['c'])

        # Validate price before updating
        if new_price <= 0:
            return  # Invalid price, ignore

        # Sanity check: if we have a valid grid center, new price shouldn't be wildly different
        # (prevents parsing errors from corrupting our state)
        if self.grid_center > 0:
            price_diff_pct = abs(new_price - self.grid_center) / self.grid_center * 100
            if price_diff_pct > 50:  # More than 50% difference is likely a parsing error
                logger.warning(f"Suspicious price {new_price} (grid center: {self.grid_center}), ignoring")
                return

        # Only log occasionally to avoid spam (every ~50 updates)
        if random.random() < 0.02:
            logger.debug(f"Price update: ${new_price:.4f}")

        self.current_price = new_price
        self.risk_manager.update_price(self.current_price)

        # Check if grid needs rebalancing
        if self.grid_center > 0 and self.current_price > 0:
            price_change = abs(self.current_price - self.grid_center) / self.grid_center * 100
            if price_change >= self.config.rebalance_threshold_pct:
                logger.info(f"Price moved {price_change:.2f}% from grid center, rebalancing...")
                await self.rebalance_grid()

    async def handle_mark_price_update(self, data: Dict):
        """Handle mark price updates from WebSocket"""
        # Mark price stream format:
        # 'p' = mark price
        # 'i' = index price
        # 'r' = funding rate
        if 'p' in data:
            mark_price = float(data['p'])
            if mark_price > 0:
                # Use mark price as backup if we don't have a current price yet
                if self.current_price <= 0:
                    self.current_price = mark_price
                    logger.info(f"Set initial price from mark price: ${mark_price}")

    async def handle_order_update(self, data: Dict):
        """Handle order update from WebSocket"""
        self.order_manager.handle_order_update(data)

        order_data = data.get('o', {})
        status = order_data.get('X', '')
        side = order_data.get('S', '')
        fill_price = float(order_data.get('ap', 0) or 0)
        quantity = float(order_data.get('q', 0))

        if status == 'FILLED':
            self.total_trades += 1

            # Create counter order for grid profit
            if side == 'BUY':
                # Place sell order at higher price
                tp_price = fill_price * (1 + self.config.take_profit_pct / 100)
                await self._place_counter_order('SELL', tp_price, quantity)
            else:
                # Place buy order at lower price
                tp_price = fill_price * (1 - self.config.take_profit_pct / 100)
                await self._place_counter_order('BUY', tp_price, quantity)

    async def _place_counter_order(self, side: str, price: float, quantity: float):
        """Place counter order after a fill"""
        try:
            # Get symbol-specific precision
            sym_config = get_symbol_config(self.config.symbol)
            tick_size = sym_config['tick_size']
            qty_precision = sym_config['qty_precision']
            price_precision = sym_config['price_precision']

            # Round to proper precision
            quantity = round(quantity, qty_precision)
            price = round(price / tick_size) * tick_size
            price = round(price, price_precision)

            await self.client.rest.place_order(
                symbol=self.config.symbol,
                side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                time_in_force=TimeInForce.GTC,
                reduce_only=self.config.reduce_only_exits
            )
            logger.info(f"Placed counter {side} order at ${price:.{price_precision}f}")
        except Exception as e:
            logger.error(f"Failed to place counter order: {e}")

    async def handle_account_update(self, data: Dict):
        """Handle account update from WebSocket"""
        account_data = data.get('a', {})

        # Update balance
        balances = account_data.get('B', [])
        for balance in balances:
            if balance.get('a') == 'USDT':
                self.risk_manager.update_balance(float(balance.get('wb', 0)))

        # Update positions
        positions = account_data.get('P', [])
        self.positions = []
        for pos in positions:
            if pos.get('s') == self.config.symbol:
                quantity = float(pos.get('pa', 0))
                if quantity != 0:
                    self.positions.append(Position(
                        symbol=pos['s'],
                        side='LONG' if quantity > 0 else 'SHORT',
                        entry_price=float(pos.get('ep', 0)),
                        quantity=quantity,
                        leverage=self.config.leverage,
                        unrealized_pnl=float(pos.get('up', 0))
                    ))

    async def rebalance_grid(self):
        """Rebalance grid around new center price"""
        if not self.risk_manager.is_trading_allowed:
            logger.warning(f"Trading paused: {self.risk_manager.stop_reason}")
            return

        logger.info("Rebalancing grid...")
        await self.setup_grid()

    async def check_positions(self):
        """Periodically check and manage positions"""
        try:
            position_risk = await self.client.rest.get_position_risk(self.config.symbol)

            for pos in position_risk:
                quantity = float(pos.get('positionAmt', 0))
                if quantity == 0:
                    continue

                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                liquidation_price = float(pos.get('liquidationPrice', 0))

                # Calculate P&L percentage
                if entry_price > 0:
                    if quantity > 0:  # LONG
                        pnl_pct = (mark_price - entry_price) / entry_price * 100
                    else:  # SHORT
                        pnl_pct = (entry_price - mark_price) / entry_price * 100

                    # Emergency stop loss check
                    if pnl_pct <= -self.config.emergency_stop_loss_pct:
                        logger.warning(f"Emergency stop triggered: {pnl_pct:.2f}%")
                        await self.emergency_close_position(pos)

        except Exception as e:
            logger.error(f"Position check failed: {e}")

    async def emergency_close_position(self, position: Dict):
        """Emergency close a position"""
        try:
            quantity = abs(float(position.get('positionAmt', 0)))
            side = OrderSide.SELL if float(position.get('positionAmt', 0)) > 0 else OrderSide.BUY

            await self.client.rest.place_order(
                symbol=self.config.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                reduce_only=True
            )
            logger.info(f"Emergency closed position: {quantity}")

        except Exception as e:
            logger.error(f"Emergency close failed: {e}")

    async def run(self):
        """Main bot loop"""
        if not self.is_initialized:
            await self.initialize()

        self.is_running = True
        self.start_time = datetime.now()

        logger.info("="*50)
        logger.info("  GRID BOT STARTED")
        logger.info(f"  Symbol: {self.config.symbol}")
        logger.info(f"  Leverage: {self.config.leverage}x")
        logger.info(f"  Grids: {self.config.num_grids} each side")
        logger.info(f"  Spacing: {self.config.grid_spacing_pct}%")
        logger.info("="*50)

        # Setup initial grid
        await self.setup_grid()

        # Subscribe to WebSocket events
        self.client.ws.subscribe('ticker', self.handle_price_update)
        self.client.ws.subscribe('markPrice', self.handle_mark_price_update)
        self.client.ws.subscribe('order_update', self.handle_order_update)
        self.client.ws.subscribe('account_update', self.handle_account_update)

        # Start WebSocket streams
        streams = [
            ticker_stream(self.config.symbol),
            mark_price_stream(self.config.symbol)
        ]

        # Run tasks concurrently
        await asyncio.gather(
            self.client.ws.connect_market_stream(streams),
            self.client.ws.connect_user_stream(),
            self._position_monitor_loop(),
            self._heartbeat_loop(),
            self._status_loop()
        )

    async def _position_monitor_loop(self):
        """Monitor positions periodically"""
        while self.is_running:
            await asyncio.sleep(self.config.position_check_interval)
            await self.check_positions()

    async def _heartbeat_loop(self):
        """Heartbeat for monitoring"""
        while self.is_running:
            await asyncio.sleep(self.config.heartbeat_interval)
            logger.info(f"Heartbeat - Price: ${self.current_price:.2f}, "
                       f"Balance: ${self.risk_manager.current_balance:.2f}, "
                       f"DD: {self.risk_manager.current_drawdown:.2f}%")

    async def _status_loop(self):
        """Print status periodically"""
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            await self.print_status()

    async def print_status(self):
        """Print current bot status"""
        runtime = datetime.now() - self.start_time if self.start_time else None

        print("\n" + "="*50)
        print("  GRID BOT STATUS")
        print("="*50)
        print(f"  Runtime: {runtime}")
        print(f"  Current Price: ${self.current_price:.2f}")
        print(f"  Grid Center: ${self.grid_center:.2f}")
        print(f"  Balance: ${self.risk_manager.current_balance:.2f}")
        print(f"  Total P&L: {self.risk_manager.total_pnl:.2f}%")
        print(f"  Daily P&L: {self.risk_manager.daily_pnl:.2f}%")
        print(f"  Drawdown: {self.risk_manager.current_drawdown:.2f}%")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Active Orders: {len(self.order_manager.active_orders)}")
        print(f"  Trading Allowed: {self.risk_manager.is_trading_allowed}")
        print("="*50 + "\n")

    async def stop(self):
        """Stop the bot gracefully"""
        logger.info("Stopping Grid Bot...")
        self.is_running = False

        # Cancel all orders
        if self.order_manager:
            await self.order_manager.cancel_all_orders()

        # Close connections
        if self.client:
            await self.client.close()

        logger.info("Grid Bot stopped")


class BinanceGridAlgoBot:
    """
    Alternative bot using Binance's native Grid Trading Algo API
    NOTE: The Algo API requires special access/VIP level on Binance
    This falls back to custom implementation if algo API is unavailable
    """

    def __init__(self, config: LiveGridConfig, binance_config: BinanceConfig):
        self.config = config
        self.binance_config = binance_config
        self.client: Optional[BinanceClient] = None
        self.algo_id: Optional[int] = None
        self.fallback_bot: Optional[LiveGridBot] = None

    async def initialize(self):
        """Initialize the bot"""
        self.client = BinanceClient(self.binance_config)
        logger.info("Binance Grid Algo Bot initialized")

    async def start_grid_algo(self):
        """
        Start Binance's native grid trading algo
        NOTE: This API may require VIP access or may not be available
        """
        try:
            # Get current price
            ticker = await self.client.rest.get_ticker_price(self.config.symbol)
            current_price = float(ticker['price'])

            # Calculate grid bounds
            price_range = current_price * (self.config.num_grids * self.config.grid_spacing_pct / 100)
            price_upper = current_price + price_range
            price_lower = current_price - price_range

            # Calculate quantity
            balance_info = await self.client.rest.get_balance()
            usdt_balance = next(
                (float(b['balance']) for b in balance_info if b['asset'] == 'USDT'),
                0
            )
            quantity = (usdt_balance * 0.9) / current_price  # Use 90% of balance

            # Try the algo endpoint (may require VIP access)
            # Binance Algo API endpoints vary by region and account type
            result = await self.client.rest.place_grid_algo(
                symbol=self.config.symbol,
                side='NEUTRAL',
                quantity=quantity,
                grid_count=self.config.num_grids * 2,
                price_upper=price_upper,
                price_lower=price_lower
            )

            self.algo_id = result.get('algoId')
            logger.info(f"Started Grid Algo: {self.algo_id}")
            logger.info(f"  Price Range: ${price_lower:.2f} - ${price_upper:.2f}")
            logger.info(f"  Grids: {self.config.num_grids * 2}")

            return result

        except Exception as e:
            logger.warning(f"Binance Algo API not available: {e}")
            logger.info("Falling back to custom grid implementation...")
            return await self._fallback_to_custom()

    async def _fallback_to_custom(self):
        """Fall back to custom grid implementation"""
        logger.info("Starting custom grid bot as fallback...")
        self.fallback_bot = LiveGridBot(self.config, self.binance_config)
        await self.fallback_bot.initialize()
        return {'status': 'fallback', 'message': 'Using custom grid implementation'}

    async def stop_grid_algo(self):
        """Stop the grid algo"""
        if self.fallback_bot:
            await self.fallback_bot.stop()
            return

        if self.algo_id:
            try:
                await self.client.rest.cancel_grid_algo(self.algo_id)
                logger.info(f"Stopped Grid Algo: {self.algo_id}")
            except Exception as e:
                logger.error(f"Failed to stop grid algo: {e}")

    async def get_algo_status(self) -> Dict:
        """Get current algo status"""
        if self.fallback_bot:
            return {'status': 'running', 'mode': 'custom_fallback'}

        if self.algo_id:
            try:
                orders = await self.client.rest.get_grid_algo_orders(self.config.symbol)
                return next((o for o in orders if o.get('algoId') == self.algo_id), {})
            except:
                return {}
        return {}

    async def run(self):
        """Run the algo bot with monitoring"""
        await self.initialize()
        result = await self.start_grid_algo()

        # If we fell back to custom bot, run that instead
        if self.fallback_bot:
            await self.fallback_bot.run()
            return

        # Subscribe to grid updates
        self.client.ws.subscribe('grid_update', self._handle_grid_update)

        # Monitor loop
        try:
            while True:
                await asyncio.sleep(60)
                status = await self.get_algo_status()
                if status:
                    logger.info(f"Grid Algo Status: {status.get('status')}, "
                               f"PnL: {status.get('totalPnl', 0)}")
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop_grid_algo()
            if self.client:
                await self.client.close()

    async def _handle_grid_update(self, data: Dict):
        """Handle grid update events"""
        logger.info(f"Grid Update: {data}")


def load_config_from_env() -> Tuple[LiveGridConfig, BinanceConfig]:
    """Load configuration from environment variables"""
    binance_config = BinanceConfig(
        api_key=os.getenv('BINANCE_API_KEY', ''),
        api_secret=os.getenv('BINANCE_API_SECRET', ''),
        testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    )

    live_config = LiveGridConfig(
        symbol=os.getenv('TRADING_SYMBOL', 'BTCUSDT'),
        num_grids=int(os.getenv('NUM_GRIDS', '10')),
        grid_spacing_pct=float(os.getenv('GRID_SPACING', '0.3')),
        total_investment=float(os.getenv('TOTAL_INVESTMENT', '4.0')),
        leverage=int(os.getenv('LEVERAGE', '20')),
        max_drawdown_pct=float(os.getenv('MAX_DRAWDOWN', '10.0')),
    )

    return live_config, binance_config


async def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         AUTONOMOUS GRID TRADING BOT                        ║
    ║         For Binance Futures                                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Load config
    live_config, binance_config = load_config_from_env()

    # Validate API keys
    if not binance_config.api_key or not binance_config.api_secret:
        print("ERROR: Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("\nExample:")
        print("  export BINANCE_API_KEY='your_api_key'")
        print("  export BINANCE_API_SECRET='your_api_secret'")
        print("  export BINANCE_TESTNET='true'  # Use testnet first!")
        return

    # Create and run bot
    bot = LiveGridBot(live_config, binance_config)

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
