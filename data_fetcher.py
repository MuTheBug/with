"""
Binance Futures data fetcher and order executor.
Handles all exchange communication with retry logic.
"""

import time
import hmac
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from config import BotConfig, TimeframeConfig
from strategies import SignalDirection

logger = logging.getLogger(__name__)


class BinanceFuturesClient:
    """
    Client for Binance USDT-M Futures API.
    Supports both testnet and production.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.bc = self.config.binance
        self.base_url = self.bc.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.bc.api_key,
        })
        self._exchange_info_cache = None
        self._exchange_info_time = 0

    def _sign(self, params: dict) -> dict:
        """Add HMAC SHA256 signature to request params."""
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(
            self.bc.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self, method: str, endpoint: str, params: dict = None,
        signed: bool = False, retries: int = 3,
    ) -> dict:
        """Make API request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params = self._sign(params)

        for attempt in range(retries):
            try:
                if method == "GET":
                    resp = self.session.get(url, params=params, timeout=10)
                elif method == "POST":
                    resp = self.session.post(url, params=params, timeout=10)
                elif method == "DELETE":
                    resp = self.session.delete(url, params=params, timeout=10)
                else:
                    raise ValueError(f"Unknown method: {method}")

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limited
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    error = resp.json() if resp.text else {}
                    logger.error(f"API error {resp.status_code}: {error}")
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                    return {"error": error, "status": resp.status_code}

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    # ---- Market Data ----

    def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch kline/candlestick data."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = self._request("GET", "/fapi/v1/klines", params)

        if isinstance(data, dict) and "error" in data:
            logger.error(f"Failed to fetch klines: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                     "taker_buy_base", "taker_buy_quote"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        return df

    def get_multi_tf_data(
        self, symbol: str, tf_config: Optional[TimeframeConfig] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured timeframes."""
        if tf_config is None:
            tf_config = self.config.timeframes

        data = {}
        for tf in tf_config.timeframes:
            limit = tf_config.lookback_bars.get(tf, 200)
            df = self.get_klines(symbol, tf, limit)
            if not df.empty:
                data[tf] = df
            time.sleep(0.1)  # Rate limit courtesy
        return data

    def get_ticker_price(self, symbol: str) -> float:
        """Get current price."""
        data = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        if isinstance(data, dict) and "price" in data:
            return float(data["price"])
        return 0.0

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price and funding rate."""
        data = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        if isinstance(data, dict) and "markPrice" in data:
            return {
                "mark_price": float(data["markPrice"]),
                "funding_rate": float(data.get("lastFundingRate", 0)),
                "next_funding_time": data.get("nextFundingTime", 0),
            }
        return {"mark_price": 0, "funding_rate": 0, "next_funding_time": 0}

    def get_orderbook(self, symbol: str, limit: int = 5) -> dict:
        """Get order book."""
        data = self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})
        if isinstance(data, dict) and "bids" in data:
            return {
                "bids": [(float(p), float(q)) for p, q in data["bids"]],
                "asks": [(float(p), float(q)) for p, q in data["asks"]],
            }
        return {"bids": [], "asks": []}

    # ---- Account ----

    def get_balance(self) -> float:
        """Get USDT futures balance."""
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        if isinstance(data, list):
            for asset in data:
                if asset.get("asset") == "USDT":
                    return float(asset.get("balance", 0))
        return 0.0

    def get_account_info(self) -> dict:
        """Get account information."""
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_positions(self) -> List[dict]:
        """Get open positions."""
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if isinstance(data, list):
            return [
                p for p in data
                if float(p.get("positionAmt", 0)) != 0
            ]
        return []

    # ---- Trading ----

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol."""
        params = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        """Set margin type (ISOLATED or CROSSED)."""
        params = {"symbol": symbol, "marginType": margin_type}
        return self._request("POST", "/fapi/v1/marginType", params, signed=True)

    def place_market_order(
        self, symbol: str, side: str, quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        """Place a market order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(f"Placing market order: {side} {quantity} {symbol}")
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float,
        time_in_force: str = "GTC", reduce_only: bool = False,
    ) -> dict:
        """Place a limit order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force,
        }
        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(f"Placing limit order: {side} {quantity} {symbol} @ {price}")
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_market(
        self, symbol: str, side: str, quantity: float, stop_price: float,
    ) -> dict:
        """Place a stop-market order (for stop loss)."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": quantity,
            "stopPrice": stop_price,
            "reduceOnly": "true",
        }
        logger.info(f"Placing stop market: {side} {quantity} {symbol} @ {stop_price}")
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_take_profit_market(
        self, symbol: str, side: str, quantity: float, stop_price: float,
    ) -> dict:
        """Place a take-profit market order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": quantity,
            "stopPrice": stop_price,
            "reduceOnly": "true",
        }
        logger.info(f"Placing TP market: {side} {quantity} {symbol} @ {stop_price}")
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol}
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True)

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/fapi/v1/openOrders", params, signed=True)
        return data if isinstance(data, list) else []

    # ---- Composite Operations ----

    def open_position(
        self,
        symbol: str,
        direction: SignalDirection,
        quantity: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[dict, dict, dict]:
        """
        Open a full position with SL and TP orders.
        Returns (entry_result, sl_result, tp_result).
        """
        # Set leverage first
        self.set_leverage(symbol, leverage)

        # Set isolated margin
        self.set_margin_type(symbol, "ISOLATED")

        # Entry order
        side = "BUY" if direction == SignalDirection.LONG else "SELL"
        entry = self.place_market_order(symbol, side, quantity)

        if isinstance(entry, dict) and "error" in entry:
            return entry, {}, {}

        # Stop loss
        sl_side = "SELL" if direction == SignalDirection.LONG else "BUY"
        sl_result = self.place_stop_market(symbol, sl_side, quantity, stop_loss)

        # Take profit
        tp_result = self.place_take_profit_market(symbol, sl_side, quantity, take_profit)

        return entry, sl_result, tp_result

    def close_position(self, symbol: str, direction: SignalDirection, quantity: float) -> dict:
        """Close an open position."""
        # Cancel existing orders first
        self.cancel_all_orders(symbol)

        side = "SELL" if direction == SignalDirection.LONG else "BUY"
        return self.place_market_order(symbol, side, quantity, reduce_only=True)


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load historical data from CSV file (for backtesting)."""
    df = pd.read_csv(filepath)

    # Ensure proper column names
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "open_time" in col_lower or col_lower == "timestamp":
            rename_map[col] = "open_time"
        elif col_lower == "open":
            rename_map[col] = "open"
        elif col_lower == "high":
            rename_map[col] = "high"
        elif col_lower == "low":
            rename_map[col] = "low"
        elif col_lower == "close":
            rename_map[col] = "close"
        elif col_lower == "volume":
            rename_map[col] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open_time" in df.columns:
        if df["open_time"].dtype == np.int64 or df["open_time"].dtype == np.float64:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        else:
            df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time")

    return df


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    Input should have DatetimeIndex.
    """
    tf_map = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
        "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
    }
    rule = tf_map.get(target_tf, target_tf)

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Carry over additional columns if they exist
    extra_cols = ["quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    for col in extra_cols:
        if col in df.columns:
            resampled[col] = df[col].resample(rule).sum()

    return resampled
