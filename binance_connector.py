#!/usr/bin/env python3
"""
BINANCE FUTURES API CONNECTOR
=============================
Complete API connector for Binance Futures with:
- REST API for order management
- WebSocket for real-time data
- Algo API endpoints for grid trading
- Signature generation and authentication
"""

import hashlib
import hmac
import time
import json
import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing (Post Only)


@dataclass
class BinanceConfig:
    """Binance API configuration"""
    api_key: str
    api_secret: str
    testnet: bool = True
    recv_window: int = 5000

    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    @property
    def ws_url(self) -> str:
        if self.testnet:
            return "wss://stream.binancefuture.com"
        return "wss://fstream.binance.com"


class BinanceAuth:
    """Handle Binance API authentication"""

    def __init__(self, config: BinanceConfig):
        self.config = config

    def generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get_headers(self) -> Dict:
        """Get authenticated headers"""
        return {
            'X-MBX-APIKEY': self.config.api_key,
            'Content-Type': 'application/json'
        }

    def sign_params(self, params: Dict) -> Dict:
        """Add timestamp and signature to params"""
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = self.config.recv_window
        params['signature'] = self.generate_signature(params)
        return params


class BinanceFuturesREST:
    """Binance Futures REST API client"""

    def __init__(self, config: BinanceConfig):
        self.config = config
        self.auth = BinanceAuth(config)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, params: Dict = None,
                       signed: bool = False) -> Dict:
        """Make API request"""
        await self._ensure_session()

        url = f"{self.config.base_url}{endpoint}"
        params = params or {}

        if signed:
            params = self.auth.sign_params(params)

        headers = self.auth.get_headers() if signed else {}

        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == "POST":
                async with self.session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == "DELETE":
                async with self.session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == "PUT":
                async with self.session.put(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")

            if 'code' in data and data['code'] != 200:
                logger.error(f"API Error: {data}")
                raise Exception(f"Binance API Error: {data.get('msg', data)}")

            return data

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise

    # ==================== Account Endpoints ====================

    async def get_account_info(self) -> Dict:
        """Get futures account information"""
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self) -> List[Dict]:
        """Get futures account balance"""
        return await self._request("GET", "/fapi/v2/balance", signed=True)

    async def get_positions(self) -> List[Dict]:
        """Get all positions"""
        account = await self.get_account_info()
        return [p for p in account.get('positions', []) if float(p['positionAmt']) != 0]

    async def get_position_risk(self, symbol: str = None) -> List[Dict]:
        """Get position risk information"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/fapi/v2/positionRisk", params, signed=True)

    # ==================== Market Data Endpoints ====================

    async def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol info"""
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def get_ticker_price(self, symbol: str = None) -> Dict:
        """Get latest price for symbol(s)"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/fapi/v1/ticker/price", params)

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book"""
        return await self._request("GET", "/fapi/v1/depth", {
            'symbol': symbol,
            'limit': limit
        })

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List:
        """Get candlestick data"""
        return await self._request("GET", "/fapi/v1/klines", {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        })

    async def get_mark_price(self, symbol: str = None) -> Dict:
        """Get mark price and funding rate"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/fapi/v1/premiumIndex", params)

    # ==================== Order Management Endpoints ====================

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = None,
        stop_price: float = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
        client_order_id: str = None
    ) -> Dict:
        """Place a new order"""
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': quantity,
            'positionSide': position_side.value,
        }

        if price and order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT]:
            params['price'] = price
            params['timeInForce'] = time_in_force.value

        if stop_price and order_type in [OrderType.STOP, OrderType.STOP_MARKET,
                                          OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
            params['stopPrice'] = stop_price

        if reduce_only:
            params['reduceOnly'] = 'true'

        if client_order_id:
            params['newClientOrderId'] = client_order_id

        return await self._request("POST", "/fapi/v1/order", params, signed=True)

    async def place_batch_orders(self, orders: List[Dict]) -> List[Dict]:
        """Place multiple orders at once (max 5)"""
        params = {
            'batchOrders': json.dumps(orders)
        }
        return await self._request("POST", "/fapi/v1/batchOrders", params, signed=True)

    async def cancel_order(self, symbol: str, order_id: int = None,
                          client_order_id: str = None) -> Dict:
        """Cancel an order"""
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        if client_order_id:
            params['origClientOrderId'] = client_order_id
        return await self._request("DELETE", "/fapi/v1/order", params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol"""
        return await self._request("DELETE", "/fapi/v1/allOpenOrders",
                                   {'symbol': symbol}, signed=True)

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    async def get_order(self, symbol: str, order_id: int = None,
                       client_order_id: str = None) -> Dict:
        """Get order status"""
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        if client_order_id:
            params['origClientOrderId'] = client_order_id
        return await self._request("GET", "/fapi/v1/order", params, signed=True)

    # ==================== Leverage & Margin Endpoints ====================

    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol"""
        return await self._request("POST", "/fapi/v1/leverage", {
            'symbol': symbol,
            'leverage': leverage
        }, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """Set margin type (ISOLATED or CROSSED)"""
        return await self._request("POST", "/fapi/v1/marginType", {
            'symbol': symbol,
            'marginType': margin_type
        }, signed=True)

    async def set_position_mode(self, dual_side: bool) -> Dict:
        """Set position mode (Hedge Mode or One-way Mode)"""
        return await self._request("POST", "/fapi/v1/positionSide/dual", {
            'dualSidePosition': 'true' if dual_side else 'false'
        }, signed=True)

    # ==================== Algo Trading Endpoints ====================

    async def place_algo_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        algo_type: str,  # 'VP' (Volume Participation), 'TWAP', etc.
        duration: int,  # Duration in seconds
        **kwargs
    ) -> Dict:
        """Place algorithmic order using Algo API"""
        params = {
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'algoType': algo_type,
            'duration': duration,
            **kwargs
        }
        return await self._request("POST", "/fapi/v1/algo/order", params, signed=True)

    async def cancel_algo_order(self, algo_id: int) -> Dict:
        """Cancel an algo order"""
        return await self._request("DELETE", "/fapi/v1/algo/order", {
            'algoId': algo_id
        }, signed=True)

    async def get_algo_orders(self, symbol: str = None,
                              algo_id: int = None) -> List[Dict]:
        """Get algo order history"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        if algo_id:
            params['algoId'] = algo_id
        return await self._request("GET", "/fapi/v1/algo/openOrders", params, signed=True)

    # ==================== Grid Trading Algo Endpoints ====================

    async def place_grid_algo(
        self,
        symbol: str,
        side: str,  # 'NEUTRAL', 'LONG', 'SHORT'
        quantity: float,
        grid_count: int,
        price_upper: float,
        price_lower: float,
        **kwargs
    ) -> Dict:
        """
        Place grid trading algo order
        Uses Binance's native grid trading API
        """
        params = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'gridCount': grid_count,
            'priceUpper': price_upper,
            'priceLower': price_lower,
            **kwargs
        }
        return await self._request("POST", "/fapi/v1/algo/futures/grid", params, signed=True)

    async def modify_grid_algo(self, algo_id: int, **kwargs) -> Dict:
        """Modify grid trading algo parameters"""
        params = {'algoId': algo_id, **kwargs}
        return await self._request("PUT", "/fapi/v1/algo/futures/grid", params, signed=True)

    async def cancel_grid_algo(self, algo_id: int) -> Dict:
        """Cancel grid trading algo"""
        return await self._request("DELETE", "/fapi/v1/algo/futures/grid", {
            'algoId': algo_id
        }, signed=True)

    async def get_grid_algo_orders(self, symbol: str = None) -> List[Dict]:
        """Get grid algo orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/fapi/v1/algo/futures/grid/openOrders",
                                   params, signed=True)

    async def get_grid_algo_history(self, symbol: str = None,
                                    start_time: int = None,
                                    end_time: int = None) -> List[Dict]:
        """Get grid algo order history"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        return await self._request("GET", "/fapi/v1/algo/futures/grid/historyOrders",
                                   params, signed=True)


class BinanceWebSocket:
    """Binance Futures WebSocket client for real-time data"""

    def __init__(self, config: BinanceConfig):
        self.config = config
        self.ws = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.listen_key: Optional[str] = None
        self.rest_client: Optional[BinanceFuturesREST] = None

    def set_rest_client(self, client: BinanceFuturesREST):
        """Set REST client for listen key management"""
        self.rest_client = client

    async def get_listen_key(self) -> str:
        """Get listen key for user data stream"""
        if not self.rest_client:
            raise Exception("REST client not set")

        data = await self.rest_client._request(
            "POST", "/fapi/v1/listenKey", signed=True
        )
        return data['listenKey']

    async def keepalive_listen_key(self):
        """Keep listen key alive (call every 30 minutes)"""
        if not self.rest_client or not self.listen_key:
            return
        await self.rest_client._request(
            "PUT", "/fapi/v1/listenKey", signed=True
        )

    def subscribe(self, event: str, callback: Callable):
        """Subscribe to an event"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    async def _emit(self, event: str, data: Any):
        """Emit event to callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error for {event}: {e}")

    async def connect_market_stream(self, streams: List[str]):
        """Connect to market data streams"""
        stream_path = "/".join(streams)
        url = f"{self.config.ws_url}/stream?streams={stream_path}"

        self.running = True

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    logger.info(f"Connected to market stream: {streams}")
                    await self._emit('connected', {'streams': streams})

                    async for message in ws:
                        data = json.loads(message)
                        stream = data.get('stream', '')
                        payload = data.get('data', {})

                        # Parse stream type and emit appropriate event
                        if '@kline' in stream:
                            await self._emit('kline', payload)
                        elif '@trade' in stream:
                            await self._emit('trade', payload)
                        elif '@depth' in stream:
                            await self._emit('depth', payload)
                        elif '@ticker' in stream:
                            await self._emit('ticker', payload)
                        elif '@markPrice' in stream:
                            await self._emit('markPrice', payload)
                        elif '@aggTrade' in stream:
                            await self._emit('aggTrade', payload)

                        await self._emit('message', data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)

    async def connect_user_stream(self):
        """Connect to user data stream"""
        self.listen_key = await self.get_listen_key()
        url = f"{self.config.ws_url}/ws/{self.listen_key}"

        self.running = True

        # Start keepalive task
        asyncio.create_task(self._keepalive_loop())

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    logger.info("Connected to user data stream")
                    await self._emit('user_connected', {})

                    async for message in ws:
                        data = json.loads(message)
                        event_type = data.get('e', '')

                        if event_type == 'ORDER_TRADE_UPDATE':
                            await self._emit('order_update', data)
                        elif event_type == 'ACCOUNT_UPDATE':
                            await self._emit('account_update', data)
                        elif event_type == 'MARGIN_CALL':
                            await self._emit('margin_call', data)
                        elif event_type == 'ACCOUNT_CONFIG_UPDATE':
                            await self._emit('config_update', data)
                        elif event_type == 'STRATEGY_UPDATE':
                            await self._emit('strategy_update', data)
                        elif event_type == 'GRID_UPDATE':
                            await self._emit('grid_update', data)

                        await self._emit('user_message', data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("User stream closed, reconnecting...")
                self.listen_key = await self.get_listen_key()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"User stream error: {e}")
                await asyncio.sleep(5)

    async def _keepalive_loop(self):
        """Keep listen key alive every 30 minutes"""
        while self.running:
            await asyncio.sleep(30 * 60)  # 30 minutes
            try:
                await self.keepalive_listen_key()
                logger.info("Listen key keepalive sent")
            except Exception as e:
                logger.error(f"Keepalive failed: {e}")

    async def close(self):
        """Close WebSocket connection"""
        self.running = False
        if self.ws:
            await self.ws.close()


class BinanceClient:
    """Combined Binance client with REST and WebSocket"""

    def __init__(self, config: BinanceConfig):
        self.config = config
        self.rest = BinanceFuturesREST(config)
        self.ws = BinanceWebSocket(config)
        self.ws.set_rest_client(self.rest)

    async def close(self):
        """Close all connections"""
        await self.rest.close()
        await self.ws.close()


# Helper functions for creating common stream names
def kline_stream(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"

def trade_stream(symbol: str) -> str:
    return f"{symbol.lower()}@trade"

def ticker_stream(symbol: str) -> str:
    return f"{symbol.lower()}@ticker"

def depth_stream(symbol: str, level: str = "20") -> str:
    return f"{symbol.lower()}@depth{level}"

def mark_price_stream(symbol: str) -> str:
    return f"{symbol.lower()}@markPrice"

def agg_trade_stream(symbol: str) -> str:
    return f"{symbol.lower()}@aggTrade"


if __name__ == "__main__":
    # Example usage
    async def main():
        config = BinanceConfig(
            api_key="your_api_key",
            api_secret="your_api_secret",
            testnet=True
        )

        client = BinanceClient(config)

        # Get account info
        try:
            info = await client.rest.get_ticker_price("BTCUSDT")
            print(f"BTC Price: {info}")
        except Exception as e:
            print(f"Error: {e}")

        await client.close()

    asyncio.run(main())
