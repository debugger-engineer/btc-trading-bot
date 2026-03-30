import logging
import os
import time

import requests.exceptions
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ServerError

load_dotenv()

logger = logging.getLogger(__name__)

def _hl_call(fn, *args, retries: int = 3, **kwargs):
    """Call a Hyperliquid API function, retrying on transient 5xx / network errors."""
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except ServerError as e:
            if attempt == retries or e.status_code < 500:
                raise
            wait = attempt * 5
            logger.warning(
                "Hyperliquid %d error (attempt %d/%d), retrying in %ds",
                e.status_code, attempt, retries, wait,
            )
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            if attempt == retries:
                raise
            wait = attempt * 5
            logger.warning(
                "Hyperliquid connection error (attempt %d/%d), retrying in %ds: %s",
                attempt, retries, wait, e,
            )
            time.sleep(wait)


class HyperliquidTrader:
    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol

        # Support builder-deployed perps (e.g. "xyz:GOLD" → dex="xyz", coin_name="GOLD")
        if ":" in symbol:
            self._dex = symbol.split(":")[0]
            self._coin_name = symbol.split(":")[1]
        else:
            self._dex = ""
            self._coin_name = symbol

        private_key = os.getenv("HL_PRIVATE_KEY")
        self.wallet_address = os.getenv("HL_WALLET_ADDRESS")
        use_testnet = False

        if not private_key or private_key == "your_private_key_here":
            raise ValueError("HL_PRIVATE_KEY not set in .env")
        if not self.wallet_address or self.wallet_address == "your_wallet_address_here":
            raise ValueError("HL_WALLET_ADDRESS not set in .env")

        api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        logger.info("Connecting to Hyperliquid %s", "testnet" if use_testnet else "mainnet")

        perp_dexs = [self._dex] if self._dex else None
        self.wallet = Account.from_key(private_key)
        self.info = Info(api_url, skip_ws=True, perp_dexs=perp_dexs)
        self.exchange = Exchange(self.wallet, api_url, account_address=self.wallet_address, perp_dexs=perp_dexs)

        # Fetch sz_decimals from Hyperliquid metadata
        self.sz_decimals = self._fetch_sz_decimals()

    def _fetch_sz_decimals(self) -> int:
        """Query Hyperliquid's asset metadata to get the correct size decimals for this symbol."""
        meta = _hl_call(self.info.meta, dex=self._dex)
        for asset in meta.get("universe", []):
            if asset["name"] == self.symbol:
                sz_dec = asset["szDecimals"]
                logger.info("%s sz_decimals=%d (from exchange metadata)", self.symbol, sz_dec)
                return sz_dec
        available = [a["name"] for a in meta.get("universe", [])]
        raise ValueError(f"Symbol '{self.symbol}' not found in Hyperliquid metadata (dex='{self._dex}'). Available: {available}")

    def get_perp_position(self) -> dict | None:
        """Return current perp position for this symbol, or None if flat."""
        state = _hl_call(self.info.user_state, self.wallet_address, dex=self._dex)
        for pos in state.get("assetPositions", []):
            if pos["position"]["coin"] == self.symbol:
                szi = float(pos["position"]["szi"])
                if szi != 0:
                    return {"side": "LONG" if szi > 0 else "SHORT", "size": abs(szi)}
        return None

    def get_perp_usdc_balance(self) -> float:
        """Return available margin (withdrawable) from perp account."""
        state = _hl_call(self.info.user_state, self.wallet_address, dex=self._dex)
        return float(state.get("withdrawable", 0))

    def get_account_equity(self) -> float:
        """Return total account equity (margin + unrealized PnL). Safe for multi-token capital allocation."""
        state = _hl_call(self.info.user_state, self.wallet_address)
        return float(state.get("marginSummary", {}).get("accountValue", 0))

    def get_last_fill(self) -> dict:
        """Return most recent perp fill for this symbol with actual price and fee."""
        fills = _hl_call(self.info.user_fills, self.wallet_address)
        for fill in fills:  # API returns newest-first
            if fill.get("coin") == self.symbol:
                return {"px": float(fill["px"]), "fee": float(fill["fee"])}
        raise RuntimeError(f"No {self.symbol} perp fill found in recent history")

    def open_long_limit(self, size_usd: float, limit_px: float, tif: str = "Alo") -> dict:
        """Place a limit buy order for a long perp position. Default TIF is ALO (maker-only)."""
        sz  = round(size_usd / limit_px, self.sz_decimals)
        px  = self._round_perp_price(limit_px)
        result = _hl_call(
            self.exchange.order,
            self.symbol, is_buy=True, sz=sz,
            limit_px=px, order_type={"limit": {"tif": tif}}, reduce_only=False,
        )
        logger.info("LIMIT LONG %s sz=%s @ $%.1f [%s] result: %s", self.symbol, sz, px, tif, result)
        return result

    def open_short_limit(self, size_usd: float, limit_px: float, tif: str = "Alo") -> dict:
        """Place a limit sell order for a short perp position. Default TIF is ALO (maker-only)."""
        sz  = round(size_usd / limit_px, self.sz_decimals)
        px  = self._round_perp_price(limit_px)
        result = _hl_call(
            self.exchange.order,
            self.symbol, is_buy=False, sz=sz,
            limit_px=px, order_type={"limit": {"tif": tif}}, reduce_only=False,
        )
        logger.info("LIMIT SHORT %s sz=%s @ $%.1f [%s] result: %s", self.symbol, sz, px, tif, result)
        return result

    def close_perp_position(self) -> dict:
        """Market-close the current perp position for this symbol."""
        result = _hl_call(self.exchange.market_close, self.symbol)
        logger.info("Close %s position result: %s", self.symbol, result)
        return result

    def update_leverage(self, leverage: int):
        """Set isolated-margin leverage for this symbol."""
        result = _hl_call(self.exchange.update_leverage, leverage, self.symbol, is_cross=False)
        logger.info("%s leverage set to %dx: %s", self.symbol, leverage, result)

    @staticmethod
    def _round_perp_price(px: float) -> float:
        """Round to Hyperliquid's required format: 5 significant figures, 1 decimal place (BTC szDecimals=5)."""
        return round(float(f"{px:.5g}"), 1)

    def place_exit_limit_perp(self, side: str, sz: float, limit_px: float) -> dict:
        """Place a reduce-only GTC limit order at the exit target (maker fee)."""
        is_buy     = side == "SHORT"   # buy to close SHORT, sell to close LONG
        px         = self._round_perp_price(limit_px)
        order_type = {"limit": {"tif": "Gtc"}}
        result = _hl_call(
            self.exchange.order,
            self.symbol, is_buy=is_buy, sz=round(sz, self.sz_decimals),
            limit_px=px, order_type=order_type, reduce_only=True,
        )
        logger.info("%s exit limit %s @ $%.1f sz=%s result: %s", self.symbol, side, px, sz, result)
        return result

    def place_stop_loss_perp(self, side: str, sz: float, stop_price: float) -> dict:
        """Place a reduce-only stop-market order to close a perp position."""
        is_buy     = side == "SHORT"   # buy to close SHORT, sell to close LONG
        trigger_px = self._round_perp_price(stop_price)
        limit_px   = self._round_perp_price(stop_price * (0.95 if not is_buy else 1.05))
        order_type = {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "sl"}}
        result = _hl_call(
            self.exchange.order,
            self.symbol, is_buy=is_buy, sz=round(sz, self.sz_decimals),
            limit_px=limit_px, order_type=order_type, reduce_only=True,
        )
        logger.info("%s SL order placed — %s SL=$%.2f sz=%s result: %s", self.symbol, side, stop_price, sz, result)
        return result

    def cancel_order(self, coin: str, oid: int) -> bool:
        """Cancel a single open order by oid. Returns True on success."""
        result = _hl_call(self.exchange.cancel, coin, oid)
        logger.info("Cancel order coin=%s oid=%d result: %s", coin, oid, result)
        return result.get("status") == "ok"

    def cancel_all_open_orders(self, coin: str):
        """Cancel all open orders for the given coin."""
        orders = _hl_call(self.info.open_orders, self.wallet_address, dex=self._dex)
        for order in orders:
            if order.get("coin") == self.symbol:
                _hl_call(self.exchange.cancel, coin, order["oid"])
                logger.info("Cancelled open order oid=%d for %s", order["oid"], coin)

    def get_open_orders(self) -> list:
        """Return open orders for this symbol's dex."""
        return _hl_call(self.info.open_orders, self.wallet_address, dex=self._dex)

    def get_mid_price(self) -> float:
        order_book = _hl_call(self.info.l2_snapshot, self.symbol)
        best_bid = float(order_book["levels"][0][0]["px"])
        best_ask = float(order_book["levels"][1][0]["px"])
        return (best_bid + best_ask) / 2
