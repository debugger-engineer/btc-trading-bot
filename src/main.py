import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
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

from perps_trading import PerpsBot  # noqa: E402

BOT_ACCOUNT = os.getenv("BOT_ACCOUNT")
BOT_SYMBOL = os.getenv("BOT_SYMBOL", "BTC")

_config_path = Path(__file__).parent.parent / "config" / "accounts.json"
with open(_config_path) as _f:
    _cfg = json.load(_f)

_account = next(a for a in _cfg["accounts"] if a["name"] == BOT_ACCOUNT)

if not _account.get("active", True):
    logger.info("Account %s is inactive (active=false in accounts.json). Exiting.", BOT_ACCOUNT)
    sys.exit(0)

# ── Resolve token config (new format with backward compat) ──────────────
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
    # Backward compat: old flat config → synthesize a BTC token
    _token = {
        "symbol": "BTC",
        "binance_symbol": "BTCUSDT",
        "leverage": _account["leverage"],
        "active": True,
    }
    # Derive capital_pct from old risk-based formula
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
# Auto equal-split capital across active tokens, capped by risk formula
_active_count = sum(1 for t in _account.get("tokens", []) if t.get("active", True))
_auto_capital_pct = 100.0 / _active_count if _active_count > 0 else 100.0
PERPS_CAPITAL_PCT   = min(_token.get("capital_pct", _auto_capital_pct), _risk_pct)
PERPS_BB_PERIOD  = _token.get("bb_period", _defaults.get("bb_period", 20))
PERPS_BB_STD     = _token.get("bb_std", _defaults.get("bb_std", 2.0))
PERPS_EMA_PERIOD = _token.get("ema_period", _defaults.get("ema_period", 200))

assert PERPS_LEVERAGE <= 15, (
    f"Leverage {PERPS_LEVERAGE}x unsafe for {PERPS_STOP_LOSS_PCT}% SL strategy (max 15x)"
)

logger.info(
    "Account: %s | symbol=%s | leverage=%dx | risk=%.1f%% | capital_pct=%.1f%% (auto=%d tokens) | SL=%.1f%% | BB(%d,%.1f) EMA(%d)",
    ACCOUNT_NAME, PERPS_SYMBOL, PERPS_LEVERAGE, TARGET_RISK_PCT,
    PERPS_CAPITAL_PCT, _active_count,
    PERPS_STOP_LOSS_PCT, PERPS_BB_PERIOD, PERPS_BB_STD, PERPS_EMA_PERIOD,
)


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
    )

    INDICATOR_INTERVAL = 60  # seconds

    if BINANCE_SYMBOL:
        TICKER_URL = f"https://api.binance.com/api/v3/ticker/price?symbol={BINANCE_SYMBOL}"

        def fetch_price() -> float | None:
            try:
                resp = requests.get(TICKER_URL, timeout=5)
                resp.raise_for_status()
                return float(resp.json()["price"])
            except Exception as exc:
                logger.warning("Price fetch failed: %s", exc)
                return None
    else:
        # No Binance pair — fetch mid price from Hyperliquid
        HL_INFO_URL = "https://api.hyperliquid.xyz/info"
        _dex = PERPS_SYMBOL.split(":")[0] if ":" in PERPS_SYMBOL else ""

        def fetch_price() -> float | None:
            try:
                resp = requests.post(HL_INFO_URL, json={"type": "allMids", "dex": _dex}, timeout=5)
                resp.raise_for_status()
                mids = resp.json()
                px = mids.get(PERPS_SYMBOL)
                if px is None:
                    logger.warning("No mid price for %s in Hyperliquid allMids", PERPS_SYMBOL)
                    return None
                return float(px)
            except Exception as exc:
                logger.warning("Price fetch failed: %s", exc)
                return None

    perps_bot.update_indicators()
    last_indicator_update = time.monotonic()

    while True:
        now = time.monotonic()
        if now - last_indicator_update >= INDICATOR_INTERVAL:
            perps_bot.update_indicators()
            last_indicator_update = now

        price = fetch_price()
        if price is not None:
            perps_bot.check_signals(price)

        time.sleep(1)


if __name__ == "__main__":
    main()
