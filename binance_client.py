"""
Binance USDT-M Futures Client
Supports regular orders and new Algo Orders (TWAP/VP)
"""

import hashlib
import hmac
import time
import json
import urllib.parse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import requests
from config import trading_config, strategy_config


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class PositionSide(Enum):
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


class AlgoOrderType(Enum):
    VP = "VP"  # Volume Participation
    TWAP = "TWAP"  # Time-Weighted Average Price


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str


@dataclass
class Order:
    order_id: int
    symbol: str
    side: str
    type: str
    price: float
    quantity: float
    status: str
    filled_qty: float = 0.0


class BinanceFuturesClient:
    """
    Binance USDT-M Futures API Client
    Supports both regular orders and new Algo Orders
    """

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key or trading_config.api_key
        self.api_secret = api_secret or trading_config.api_secret
        self.testnet = testnet

        if testnet:
            self.base_url = trading_config.testnet_base_url
        else:
            self.base_url = trading_config.prod_base_url

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _sign(self, params: Dict[str, Any]) -> str:
        """Create HMAC SHA256 signature"""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling"""

        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        params = params or {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params["signature"] = self._sign(params)

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            elif method == "POST":
                response = self.session.post(url, params=params)
            elif method == "DELETE":
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.last_request_time = time.time()

            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                raise Exception(f"API Error {response.status_code}: {error_data}")

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    # ==================== Account & Position Methods ====================

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including balance"""
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_balance(self) -> float:
        """Get USDT balance"""
        account = self.get_account_info()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])
        return 0.0

    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        account = self.get_account_info()
        positions = []
        for pos in account.get("positions", []):
            size = float(pos["positionAmt"])
            if size != 0:
                positions.append(Position(
                    symbol=pos["symbol"],
                    side="LONG" if size > 0 else "SHORT",
                    size=abs(size),
                    entry_price=float(pos["entryPrice"]),
                    unrealized_pnl=float(pos["unrealizedProfit"]),
                    leverage=int(pos["leverage"]),
                    margin_type=pos["marginType"]
                ))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # ==================== Leverage & Margin ====================

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol"""
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return self._request("POST", "/fapi/v1/leverage", params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """Set margin type (ISOLATED or CROSSED)"""
        params = {
            "symbol": symbol,
            "marginType": margin_type
        }
        try:
            return self._request("POST", "/fapi/v1/marginType", params, signed=True)
        except Exception as e:
            # Ignore if margin type is already set
            if "No need to change margin type" in str(e):
                return {"msg": "Margin type already set"}
            raise

    # ==================== Market Data ====================

    def get_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        result = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        return float(result["price"])

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int = None,
        end_time: int = None
    ) -> List[List]:
        """Get kline/candlestick data"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._request("GET", "/fapi/v1/klines", params)

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book"""
        return self._request("GET", "/fapi/v1/depth", {
            "symbol": symbol,
            "limit": limit
        })

    def get_exchange_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get exchange trading rules"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/exchangeInfo", params)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get trading rules for specific symbol"""
        info = self.get_exchange_info(symbol)
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return {}

    # ==================== Regular Orders ====================

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = None,
        stop_price: float = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_position: bool = False,
        activation_price: float = None,
        callback_rate: float = None,
        position_side: PositionSide = PositionSide.BOTH
    ) -> Order:
        """Place a new order"""

        params = {
            "symbol": symbol,
            "side": side.value if isinstance(side, OrderSide) else side,
            "type": order_type.value if isinstance(order_type, OrderType) else order_type,
            "quantity": self._format_quantity(symbol, quantity),
            "positionSide": position_side.value if isinstance(position_side, PositionSide) else position_side
        }

        if price:
            params["price"] = self._format_price(symbol, price)

        if order_type in [OrderType.LIMIT, "LIMIT"]:
            params["timeInForce"] = time_in_force

        if stop_price:
            params["stopPrice"] = self._format_price(symbol, stop_price)

        if reduce_only:
            params["reduceOnly"] = "true"

        if close_position:
            params["closePosition"] = "true"
            del params["quantity"]

        # Trailing stop parameters
        if activation_price:
            params["activationPrice"] = self._format_price(symbol, activation_price)
        if callback_rate:
            params["callbackRate"] = callback_rate

        result = self._request("POST", "/fapi/v1/order", params, signed=True)

        return Order(
            order_id=result["orderId"],
            symbol=result["symbol"],
            side=result["side"],
            type=result["type"],
            price=float(result.get("price", 0)),
            quantity=float(result["origQty"]),
            status=result["status"],
            filled_qty=float(result.get("executedQty", 0))
        )

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False
    ) -> Order:
        """Place market order"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only
        )

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: str = "GTC"
    ) -> Order:
        """Place limit order"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )

    def place_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float
    ) -> Order:
        """Place stop loss order"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=True
        )

    def place_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float
    ) -> Order:
        """Place take profit order"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=True
        )

    def place_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        callback_rate: float,
        activation_price: float = None
    ) -> Order:
        """Place trailing stop order"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP_MARKET,
            quantity=quantity,
            callback_rate=callback_rate,
            activation_price=activation_price,
            reduce_only=True
        )

    # ==================== Algo Orders (TWAP/VP) ====================

    def place_algo_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        algo_type: AlgoOrderType,
        duration: int = None,  # For TWAP, in seconds
        urgency: str = "MEDIUM",  # LOW, MEDIUM, HIGH
    ) -> Dict[str, Any]:
        """
        Place new algo order (TWAP or VP)
        Reference: https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/New-Algo-Order
        """
        params = {
            "symbol": symbol,
            "side": side.value if isinstance(side, OrderSide) else side,
            "quantity": self._format_quantity(symbol, quantity),
            "algoType": algo_type.value if isinstance(algo_type, AlgoOrderType) else algo_type,
        }

        if algo_type == AlgoOrderType.TWAP and duration:
            params["duration"] = duration

        params["urgency"] = urgency

        return self._request("POST", "/fapi/v1/algo/futures/newOrderTwap", params, signed=True)

    def place_twap_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration: int = 300,  # 5 minutes default
        urgency: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Place TWAP order - splits order over time"""
        return self.place_algo_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            algo_type=AlgoOrderType.TWAP,
            duration=duration,
            urgency=urgency
        )

    def place_vp_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        urgency: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Place VP order - participates in volume"""
        return self.place_algo_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            algo_type=AlgoOrderType.VP,
            urgency=urgency
        )

    def get_algo_orders(
        self,
        symbol: str = None,
        algo_id: int = None,
        start_time: int = None,
        end_time: int = None
    ) -> List[Dict[str, Any]]:
        """Get algo order history"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        if algo_id:
            params["algoId"] = algo_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._request("GET", "/fapi/v1/algo/futures/openOrders", params, signed=True)

    def cancel_algo_order(self, algo_id: int) -> Dict[str, Any]:
        """Cancel an algo order"""
        return self._request("DELETE", "/fapi/v1/algo/futures/order", {
            "algoId": algo_id
        }, signed=True)

    # ==================== Order Management ====================

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order"""
        return self._request("DELETE", "/fapi/v1/order", {
            "symbol": symbol,
            "orderId": order_id
        }, signed=True)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol"""
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {
            "symbol": symbol
        }, signed=True)

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status"""
        return self._request("GET", "/fapi/v1/order", {
            "symbol": symbol,
            "orderId": order_id
        }, signed=True)

    # ==================== Utility Methods ====================

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol's precision"""
        # Default to 3 decimal places, could be enhanced with exchange info
        return f"{quantity:.3f}"

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol's precision"""
        # Adjust based on asset
        if "BTC" in symbol:
            return f"{price:.2f}"
        elif "SOL" in symbol:
            return f"{price:.4f}"
        elif "XRP" in symbol:
            return f"{price:.5f}"
        return f"{price:.4f}"

    def calculate_position_size(
        self,
        symbol: str,
        balance: float,
        risk_pct: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk management

        For small accounts, we need to ensure we meet minimum notional
        while not risking too much
        """
        # Calculate risk amount
        risk_amount = balance * risk_pct

        # Calculate price difference
        price_diff = abs(entry_price - stop_loss_price)
        price_diff_pct = price_diff / entry_price

        # Position size based on risk
        if price_diff_pct > 0:
            position_value = risk_amount / price_diff_pct
        else:
            position_value = balance * strategy_config.position_size_pct

        # Apply leverage
        leveraged_balance = balance * strategy_config.leverage

        # Ensure we don't exceed leveraged balance
        position_value = min(position_value, leveraged_balance * 0.9)

        # Ensure minimum notional
        position_value = max(position_value, strategy_config.min_notional)

        # Calculate quantity
        quantity = position_value / entry_price

        return quantity


# Create global client instance
client = BinanceFuturesClient(testnet=trading_config.use_testnet)
