import logging
import os

from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

load_dotenv()

logger = logging.getLogger(__name__)

# Hyperliquid spot asset index for BTC/USDC
BTC_SPOT_ASSET = "@1"
# Hyperliquid perp asset name for BTC
BTC_PERP_ASSET = "BTC"


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

    def get_spot_balance(self, token: str) -> float:
        """Return the available spot balance for a given token symbol (e.g. 'USDC', 'BTC')."""
        state = self.info.spot_user_state(self.wallet_address)
        for balance in state.get("balances", []):
            if balance["coin"] == token:
                return float(balance["total"])
        return 0.0

    def get_usdc_balance(self) -> float:
        return self.get_spot_balance("USDC")

    def get_btc_balance(self) -> float:
        return self.get_spot_balance("BTC")

    def buy_btc(self, usdc_amount: float, price: float | None = None) -> dict:
        """Place a market buy order using the given USDC amount.

        Hyperliquid spot market buys are sized in the base asset (BTC).
        Pass `price` (from Binance klines) to avoid fetching the spot order book.
        """
        mid_price = price if price is not None else self._get_btc_perp_mid_price()
        btc_qty = round(usdc_amount / mid_price, 5)
        logger.info("BUY %.5f BTC (~$%.2f USDC) at mid $%.2f", btc_qty, usdc_amount, mid_price)
        result = self.exchange.market_open(BTC_SPOT_ASSET, is_buy=True, sz=btc_qty)
        logger.info("Buy order result: %s", result)
        return result

    def sell_btc(self, btc_amount: float) -> dict:
        """Place a market sell order for the given BTC amount."""
        btc_amount = round(btc_amount, 5)
        logger.info("SELL %.5f BTC", btc_amount)
        result = self.exchange.market_open(BTC_SPOT_ASSET, is_buy=False, sz=btc_amount)
        logger.info("Sell order result: %s", result)
        return result

    def place_stop_loss_spot(self, btc_amount: float, stop_price: float) -> dict:
        """Place a stop-market sell order to protect a spot BTC long position."""
        btc_amount = round(btc_amount, 5)
        limit_px   = round(stop_price * 0.95, 2)   # 5% slippage buffer for market fill
        order_type = {"trigger": {"triggerPx": round(stop_price, 2), "isMarket": True, "tpsl": "sl"}}
        result = self.exchange.order(
            BTC_SPOT_ASSET, is_buy=False, sz=btc_amount,
            limit_px=limit_px, order_type=order_type, reduce_only=False,
        )
        logger.info("Spot SL order placed — SL=$%.2f btc=%.6f result: %s", stop_price, btc_amount, result)
        return result

    def _get_btc_mid_price(self) -> float:
        """Fetch current BTC mid price from Hyperliquid order book."""
        order_book = self.info.l2_snapshot(BTC_SPOT_ASSET)
        best_bid = float(order_book["levels"][0][0]["px"])
        best_ask = float(order_book["levels"][1][0]["px"])
        return (best_bid + best_ask) / 2

    # --- Perps methods ---

    def get_perp_position(self) -> dict | None:
        """Return current BTC perp position or None if flat."""
        state = self.info.user_state(self.wallet_address)
        for pos in state.get("assetPositions", []):
            if pos["position"]["coin"] == BTC_PERP_ASSET:
                szi = float(pos["position"]["szi"])
                if szi != 0:
                    return {"side": "LONG" if szi > 0 else "SHORT", "size": abs(szi)}
        return None

    def get_perp_usdc_balance(self) -> float:
        """Return available margin (withdrawable) from perp account."""
        state = self.info.user_state(self.wallet_address)
        return float(state.get("withdrawable", 0))

    def get_last_btc_perp_fill(self) -> dict:
        """Return most recent BTC perp fill with actual price and fee."""
        fills = self.info.user_fills(self.wallet_address)
        for fill in fills:  # API returns newest-first
            if fill.get("coin") == BTC_PERP_ASSET:
                return {"px": float(fill["px"]), "fee": float(fill["fee"])}
        raise RuntimeError("No BTC perp fill found in recent history")

    def open_long(self, size_usd: float) -> dict:
        """Open a BTC long perp position sized in USDC notional."""
        mid = self._get_btc_perp_mid_price()
        sz = round(size_usd / mid, 5)
        logger.info("OPEN LONG %.5f BTC (~$%.2f USDC)", sz, size_usd)
        result = self.exchange.market_open(BTC_PERP_ASSET, is_buy=True, sz=sz)
        logger.info("Open long result: %s", result)
        return result

    def open_short(self, size_usd: float) -> dict:
        """Open a BTC short perp position sized in USDC notional."""
        mid = self._get_btc_perp_mid_price()
        sz = round(size_usd / mid, 5)
        logger.info("OPEN SHORT %.5f BTC (~$%.2f USDC)", sz, size_usd)
        result = self.exchange.market_open(BTC_PERP_ASSET, is_buy=False, sz=sz)
        logger.info("Open short result: %s", result)
        return result

    def open_long_limit(self, size_usd: float, limit_px: float) -> dict:
        """Place a GTC limit buy order for a BTC long perp position (maker fee)."""
        sz  = round(size_usd / limit_px, 5)
        px  = self._round_perp_price(limit_px)
        result = self.exchange.order(
            BTC_PERP_ASSET, is_buy=True, sz=sz,
            limit_px=px, order_type={"limit": {"tif": "Gtc"}}, reduce_only=False,
        )
        logger.info("LIMIT LONG %.5f BTC @ $%.1f result: %s", sz, px, result)
        return result

    def open_short_limit(self, size_usd: float, limit_px: float) -> dict:
        """Place a GTC limit sell order for a BTC short perp position (maker fee)."""
        sz  = round(size_usd / limit_px, 5)
        px  = self._round_perp_price(limit_px)
        result = self.exchange.order(
            BTC_PERP_ASSET, is_buy=False, sz=sz,
            limit_px=px, order_type={"limit": {"tif": "Gtc"}}, reduce_only=False,
        )
        logger.info("LIMIT SHORT %.5f BTC @ $%.1f result: %s", sz, px, result)
        return result

    def close_perp_position(self) -> dict:
        """Market-close the current BTC perp position."""
        result = self.exchange.market_close(BTC_PERP_ASSET)
        logger.info("Close position result: %s", result)
        return result

    def update_leverage(self, leverage: int):
        """Set isolated-margin leverage for the BTC perp asset."""
        result = self.exchange.update_leverage(leverage, BTC_PERP_ASSET, is_cross=False)
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
        result = self.exchange.order(
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
        result = self.exchange.order(
            BTC_PERP_ASSET, is_buy=is_buy, sz=round(btc_sz, 5),
            limit_px=limit_px, order_type=order_type, reduce_only=True,
        )
        logger.info("Perp SL order placed — %s SL=$%.2f sz=%.6f result: %s", side, stop_price, btc_sz, result)
        return result

    def cancel_order(self, coin: str, oid: int) -> bool:
        """Cancel a single open order by oid. Returns True on success."""
        result = self.exchange.cancel(coin, oid)
        logger.info("Cancel order coin=%s oid=%d result: %s", coin, oid, result)
        return result.get("status") == "ok"

    def cancel_all_open_orders(self, coin: str):
        """Cancel all open orders for the given coin."""
        orders = self.info.open_orders(self.wallet_address)
        for order in orders:
            if order.get("coin") == coin:
                self.exchange.cancel(coin, order["oid"])
                logger.info("Cancelled open order oid=%d for %s", order["oid"], coin)

    def _get_btc_perp_mid_price(self) -> float:
        order_book = self.info.l2_snapshot(BTC_PERP_ASSET)
        best_bid = float(order_book["levels"][0][0]["px"])
        best_ask = float(order_book["levels"][1][0]["px"])
        return (best_bid + best_ask) / 2
