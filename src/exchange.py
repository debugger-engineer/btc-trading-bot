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

# Hyperliquid perp asset name for BTC
BTC_PERP_ASSET = "BTC"


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
    def __init__(self):
        private_key = os.getenv("HL_PRIVATE_KEY")
        self.wallet_address = os.getenv("HL_WALLET_ADDRESS")
        use_testnet = False

        if not private_key or private_key == "your_private_key_here":
            raise ValueError("HL_PRIVATE_KEY not set in .env")
        if not self.wallet_address or self.wallet_address == "your_wallet_address_here":
            raise ValueError("HL_WALLET_ADDRESS not set in .env")

        api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        logger.info("Connecting to Hyperliquid %s", "testnet" if use_testnet else "mainnet")

        self.wallet = Account.from_key(private_key)
        self.info = Info(api_url, skip_ws=True)
        self.exchange = Exchange(self.wallet, api_url, account_address=self.wallet_address)

    def get_perp_position(self) -> dict | None:
        """Return current BTC perp position or None if flat."""
        state = _hl_call(self.info.user_state, self.wallet_address)
        for pos in state.get("assetPositions", []):
            if pos["position"]["coin"] == BTC_PERP_ASSET:
                szi = float(pos["position"]["szi"])
                if szi != 0:
                    return {"side": "LONG" if szi > 0 else "SHORT", "size": abs(szi)}
        return None

    def get_perp_usdc_balance(self) -> float:
        """Return available margin (withdrawable) from perp account."""
        state = _hl_call(self.info.user_state, self.wallet_address)
        return float(state.get("withdrawable", 0))

    def get_last_btc_perp_fill(self) -> dict:
        """Return most recent BTC perp fill with actual price and fee."""
        fills = _hl_call(self.info.user_fills, self.wallet_address)
        for fill in fills:  # API returns newest-first
            if fill.get("coin") == BTC_PERP_ASSET:
                return {"px": float(fill["px"]), "fee": float(fill["fee"])}
        raise RuntimeError("No BTC perp fill found in recent history")

    def open_long_limit(self, size_usd: float, limit_px: float, tif: str = "Alo") -> dict:
        """Place a limit buy order for a BTC long perp position. Default TIF is ALO (maker-only)."""
        sz  = round(size_usd / limit_px, 5)
        px  = self._round_perp_price(limit_px)
        result = _hl_call(
            self.exchange.order,
            BTC_PERP_ASSET, is_buy=True, sz=sz,
            limit_px=px, order_type={"limit": {"tif": tif}}, reduce_only=False,
        )
        logger.info("LIMIT LONG %.5f BTC @ $%.1f [%s] result: %s", sz, px, tif, result)
        return result

    def open_short_limit(self, size_usd: float, limit_px: float, tif: str = "Alo") -> dict:
        """Place a limit sell order for a BTC short perp position. Default TIF is ALO (maker-only)."""
        sz  = round(size_usd / limit_px, 5)
        px  = self._round_perp_price(limit_px)
        result = _hl_call(
            self.exchange.order,
            BTC_PERP_ASSET, is_buy=False, sz=sz,
            limit_px=px, order_type={"limit": {"tif": tif}}, reduce_only=False,
        )
        logger.info("LIMIT SHORT %.5f BTC @ $%.1f [%s] result: %s", sz, px, tif, result)
        return result

    def close_perp_position(self) -> dict:
        """Market-close the current BTC perp position."""
        result = _hl_call(self.exchange.market_close, BTC_PERP_ASSET)
        logger.info("Close position result: %s", result)
        return result

    def update_leverage(self, leverage: int):
        """Set isolated-margin leverage for the BTC perp asset."""
        result = _hl_call(self.exchange.update_leverage, leverage, BTC_PERP_ASSET, is_cross=False)
        logger.info("Leverage set to %dx: %s", leverage, result)

    @staticmethod
    def _round_perp_price(px: float) -> float:
        """Round to Hyperliquid's required format: 5 significant figures, 1 decimal place (BTC szDecimals=5)."""
        return round(float(f"{px:.5g}"), 1)

    def place_exit_limit_perp(self, side: str, btc_sz: float, limit_px: float) -> dict:
        """Place a reduce-only GTC limit order at the exit target (maker fee)."""
        is_buy     = side == "SHORT"   # buy to close SHORT, sell to close LONG
        px         = self._round_perp_price(limit_px)
        order_type = {"limit": {"tif": "Gtc"}}
        result = _hl_call(
            self.exchange.order,
            BTC_PERP_ASSET, is_buy=is_buy, sz=round(btc_sz, 5),
            limit_px=px, order_type=order_type, reduce_only=True,
        )
        logger.info("Exit limit %s @ $%.1f sz=%.5f result: %s", side, px, btc_sz, result)
        return result

    def place_stop_loss_perp(self, side: str, btc_sz: float, stop_price: float) -> dict:
        """Place a reduce-only stop-market order to close a perp position."""
        is_buy     = side == "SHORT"   # buy to close SHORT, sell to close LONG
        trigger_px = self._round_perp_price(stop_price)
        limit_px   = self._round_perp_price(stop_price * (0.95 if not is_buy else 1.05))
        order_type = {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "sl"}}
        result = _hl_call(
            self.exchange.order,
            BTC_PERP_ASSET, is_buy=is_buy, sz=round(btc_sz, 5),
            limit_px=limit_px, order_type=order_type, reduce_only=True,
        )
        logger.info("Perp SL order placed — %s SL=$%.2f sz=%.6f result: %s", side, stop_price, btc_sz, result)
        return result

    def cancel_order(self, coin: str, oid: int) -> bool:
        """Cancel a single open order by oid. Returns True on success."""
        result = _hl_call(self.exchange.cancel, coin, oid)
        logger.info("Cancel order coin=%s oid=%d result: %s", coin, oid, result)
        return result.get("status") == "ok"

    def cancel_all_open_orders(self, coin: str):
        """Cancel all open orders for the given coin."""
        orders = _hl_call(self.info.open_orders, self.wallet_address)
        for order in orders:
            if order.get("coin") == coin:
                _hl_call(self.exchange.cancel, coin, order["oid"])
                logger.info("Cancelled open order oid=%d for %s", order["oid"], coin)

    def _get_btc_perp_mid_price(self) -> float:
        order_book = _hl_call(self.info.l2_snapshot, BTC_PERP_ASSET)
        best_bid = float(order_book["levels"][0][0]["px"])
        best_ask = float(order_book["levels"][1][0]["px"])
        return (best_bid + best_ask) / 2
