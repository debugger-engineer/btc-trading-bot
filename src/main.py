import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import requests
import websocket as ws_lib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)

from perps_trading import CandleMaintainer, PerpsBot  # noqa: E402

BOT_ACCOUNT = os.getenv("BOT_ACCOUNT")
BOT_SYMBOL = os.getenv("BOT_SYMBOL", "BTC")

_config_path = Path(__file__).parent.parent / "config" / "accounts.json"
with open(_config_path) as _f:
    _cfg = json.load(_f)

_account = next(a for a in _cfg["accounts"] if a["name"] == BOT_ACCOUNT)

if not _account.get("active", True):
    logger.info("Account %s is inactive (active=false in accounts.json). Exiting.", BOT_ACCOUNT)
    sys.exit(0)

# -- Resolve token config (new format with backward compat) ----------------
_defaults = _cfg.get("defaults", {})

if "tokens" in _account:
    _token = next(
        (t for t in _account["tokens"] if t["symbol"] == BOT_SYMBOL),
        None,
    )
    if _token is None:
        logger.error("No token config for symbol=%s in account=%s. Exiting.", BOT_SYMBOL, BOT_ACCOUNT)
        sys.exit(1)
    if not _token.get("active", True):
        logger.info("Token %s in account %s is inactive. Exiting.", BOT_SYMBOL, BOT_ACCOUNT)
        sys.exit(0)
else:
    # Backward compat: old flat config -> synthesize a BTC token
    _token = {
        "symbol": "BTC",
        "binance_symbol": "BTCUSDT",
        "leverage": _account["leverage"],
        "active": True,
    }
    _old_risk = _cfg.get("target_risk_pct", 8.0)
    _old_sl = _cfg.get("stop_loss_pct", 2.5)
    _token["capital_pct"] = _old_risk / (_account["leverage"] * _old_sl / 100)

# Merge defaults with token-level overrides
ACCOUNT_NAME     = BOT_ACCOUNT
PERPS_SYMBOL     = _token["symbol"]
BINANCE_SYMBOL   = _token.get("binance_symbol", "")
PERPS_LEVERAGE      = _token["leverage"]
PERPS_STOP_LOSS_PCT = _token.get("stop_loss_pct", _defaults.get("stop_loss_pct", 2.5))
TARGET_RISK_PCT     = _cfg.get("target_risk_pct", 8.0)
_risk_pct = TARGET_RISK_PCT / (PERPS_LEVERAGE * PERPS_STOP_LOSS_PCT / 100)
_active_count = sum(1 for t in _account.get("tokens", []) if t.get("active", True))
_auto_capital_pct = 100.0 / _active_count if _active_count > 0 else 100.0
PERPS_CAPITAL_PCT   = min(_token.get("capital_pct", _auto_capital_pct), _risk_pct)
PERPS_BB_PERIOD  = _token.get("bb_period", _defaults.get("bb_period", 20))
PERPS_BB_STD     = _token.get("bb_std", _defaults.get("bb_std", 2.0))
PERPS_EMA_PERIOD = _token.get("ema_period", _defaults.get("ema_period", 200))
MIN_BAND_WIDTH_PCT = _token.get("min_band_width_pct", _defaults.get("min_band_width_pct", 0.10))
MAX_SAME_DIRECTION = _cfg.get("max_same_direction", 4)
FORCE_GTC = _token.get("force_gtc", False)

assert PERPS_LEVERAGE <= 15, (
    f"Leverage {PERPS_LEVERAGE}x unsafe for {PERPS_STOP_LOSS_PCT}% SL strategy (max 15x)"
)

logger.info(
    "Account: %s | symbol=%s | leverage=%dx | risk=%.1f%% | capital_pct=%.1f%% (auto=%d tokens) | SL=%.1f%% | BB(%d,%.1f) EMA(%d)",
    ACCOUNT_NAME, PERPS_SYMBOL, PERPS_LEVERAGE, TARGET_RISK_PCT,
    PERPS_CAPITAL_PCT, _active_count,
    PERPS_STOP_LOSS_PCT, PERPS_BB_PERIOD, PERPS_BB_STD, PERPS_EMA_PERIOD,
)


# -- WebSocket price feeds -------------------------------------------------

class PriceFeed:
    """Thread-safe price feed via WebSocket with HTTP fallback.

    Also subscribes to orderUpdates for instant fill detection (HL only).
    """

    def __init__(self, symbol: str, binance_symbol: str = "", wallet_address: str = ""):
        self._price: float | None = None
        self._lock = threading.Lock()
        self._binance_symbol = binance_symbol
        self._symbol = symbol
        self._wallet_address = wallet_address
        self._ws: ws_lib.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._connected = threading.Event()
        # Order update tracking (filled/cancelled oids)
        self._order_updates_lock = threading.Lock()
        self._filled_oids: set[int] = set()
        self._cancelled_oids: set[int] = set()

    @property
    def price(self) -> float | None:
        with self._lock:
            return self._price

    def _set_price(self, px: float):
        with self._lock:
            self._price = px

    def is_order_filled(self, oid: int) -> bool:
        with self._order_updates_lock:
            return oid in self._filled_oids

    def is_order_gone(self, oid: int) -> bool:
        """Return True if the order was filled or cancelled (no longer resting)."""
        with self._order_updates_lock:
            return oid in self._filled_oids or oid in self._cancelled_oids

    def clear_order(self, oid: int):
        """Remove a tracked oid after it's been processed."""
        with self._order_updates_lock:
            self._filled_oids.discard(oid)
            self._cancelled_oids.discard(oid)

    def start(self):
        if self._binance_symbol:
            self._start_binance_ws()
        else:
            self._start_hl_ws()

    def _start_binance_ws(self):
        """Subscribe to Binance real-time trade stream."""
        stream = self._binance_symbol.lower()
        url = f"wss://stream.binance.com:9443/ws/{stream}@trade"

        def on_message(_ws, message):
            data = json.loads(message)
            self._set_price(float(data["p"]))

        def on_open(_ws):
            logger.info("Binance WS connected for %s", self._binance_symbol)
            self._connected.set()

        def on_error(_ws, error):
            logger.warning("Binance WS error: %s", error)

        def on_close(_ws, close_status, close_msg):
            logger.warning("Binance WS closed: %s %s — reconnecting in 5s", close_status, close_msg)
            self._connected.clear()
            time.sleep(5)
            self._start_binance_ws()

        self._ws = ws_lib.WebSocketApp(url, on_message=on_message, on_open=on_open,
                                       on_error=on_error, on_close=on_close)
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def _start_hl_ws(self):
        """Subscribe to Hyperliquid allMids + orderUpdates via WebSocket."""
        url = "wss://api.hyperliquid.xyz/ws"
        symbol = self._symbol
        dex = symbol.split(":")[0] if ":" in symbol else ""
        wallet = self._wallet_address

        def on_message(_ws, message):
            if message == "Websocket connection established.":
                return
            data = json.loads(message)
            channel = data.get("channel")
            if channel == "allMids":
                mids = data.get("data", {}).get("mids", {})
                px = mids.get(symbol)
                if px is not None:
                    self._set_price(float(px))
            elif channel == "orderUpdates":
                self._handle_order_updates(data.get("data", []))
            elif channel == "pong":
                pass

        def on_open(_ws):
            logger.info("Hyperliquid WS connected for %s", symbol)
            # Subscribe to price feed
            sub = {"method": "subscribe", "subscription": {"type": "allMids"}}
            if dex:
                sub["subscription"]["dex"] = dex
            _ws.send(json.dumps(sub))
            # Subscribe to order updates for fill detection
            if wallet:
                order_sub = {"method": "subscribe", "subscription": {"type": "orderUpdates", "user": wallet}}
                _ws.send(json.dumps(order_sub))
                logger.info("Subscribed to orderUpdates for %s", wallet)
            self._connected.set()

        def on_error(_ws, error):
            logger.warning("Hyperliquid WS error: %s", error)

        def on_close(_ws, close_status, close_msg):
            logger.warning("Hyperliquid WS closed: %s %s — reconnecting in 5s", close_status, close_msg)
            self._connected.clear()
            time.sleep(5)
            self._start_hl_ws()

        def ping_loop(ws_app):
            while ws_app.keep_running:
                try:
                    ws_app.send(json.dumps({"method": "ping"}))
                except Exception:
                    break
                time.sleep(50)

        self._ws = ws_lib.WebSocketApp(url, on_message=on_message, on_open=on_open,
                                       on_error=on_error, on_close=on_close)
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        # Ping thread to keep HL connection alive
        threading.Thread(target=ping_loop, args=(self._ws,), daemon=True).start()

    def wait_for_connection(self, timeout: float = 15.0) -> bool:
        return self._connected.wait(timeout)

    def _handle_order_updates(self, updates):
        """Process orderUpdates WS messages to track filled/cancelled orders."""
        for order in updates:
            try:
                oid = order.get("order", {}).get("oid") or order.get("oid")
                status = order.get("status", "")
                if oid is None:
                    continue
                oid = int(oid)
                if status == "filled":
                    with self._order_updates_lock:
                        self._filled_oids.add(oid)
                    logger.debug("WS orderUpdate: oid=%d filled", oid)
                elif status in ("canceled", "cancelled", "rejected"):
                    with self._order_updates_lock:
                        self._cancelled_oids.add(oid)
                    logger.debug("WS orderUpdate: oid=%d %s", oid, status)
            except (ValueError, TypeError, AttributeError):
                continue

    def fetch_price_http(self) -> float | None:
        """HTTP fallback if WS hasn't delivered a price yet."""
        try:
            if self._binance_symbol:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={self._binance_symbol}"
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                return float(resp.json()["price"])
            else:
                dex = self._symbol.split(":")[0] if ":" in self._symbol else ""
                resp = requests.post("https://api.hyperliquid.xyz/info",
                                     json={"type": "allMids", "dex": dex}, timeout=5)
                resp.raise_for_status()
                px = resp.json().get(self._symbol)
                return float(px) if px else None
        except Exception as exc:
            logger.warning("HTTP price fallback failed: %s", exc)
            return None


def main():
    perps_bot = PerpsBot(
        dry_run=False,
        leverage=PERPS_LEVERAGE,
        capital_percent=PERPS_CAPITAL_PCT,
        bb_period=PERPS_BB_PERIOD,
        bb_std=PERPS_BB_STD,
        ema_period=PERPS_EMA_PERIOD,
        stop_loss_pct=PERPS_STOP_LOSS_PCT,
        account_name=ACCOUNT_NAME,
        symbol=PERPS_SYMBOL,
        binance_symbol=BINANCE_SYMBOL,
        min_band_width_pct=MIN_BAND_WIDTH_PCT,
        max_same_direction=MAX_SAME_DIRECTION,
        force_gtc=FORCE_GTC,
    )

    INDICATOR_INTERVAL = 10  # seconds (keep BB bands fresh for tighter entry fills)

    # Start WebSocket price feed (+ orderUpdates for HL symbols)
    wallet_address = os.getenv("HL_WALLET_ADDRESS", "")
    feed = PriceFeed(PERPS_SYMBOL, BINANCE_SYMBOL, wallet_address=wallet_address)
    perps_bot.price_feed = feed
    feed.start()
    if feed.wait_for_connection(timeout=15):
        logger.info("WebSocket price feed connected")
    else:
        logger.warning("WebSocket connection timeout — will use HTTP fallback until connected")

    # Local candle maintenance — REST fetch only at startup + periodic 5-min resync
    num_candles = max(PERPS_EMA_PERIOD, PERPS_BB_PERIOD) * 3
    candle_maintainer = CandleMaintainer(num_candles)
    candle_maintainer.initialize(perps_bot._fetch_klines)
    perps_bot.candle_maintainer = candle_maintainer

    perps_bot.update_indicators()
    last_indicator_update = time.monotonic()

    while True:
        now = time.monotonic()

        price = feed.price
        if price is None:
            price = feed.fetch_price_http()

        if price is not None:
            candle_maintainer.update_price(price, time.time())
            if now - last_indicator_update >= INDICATOR_INTERVAL:
                perps_bot.update_indicators()
                last_indicator_update = now
            perps_bot.check_signals(price)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
