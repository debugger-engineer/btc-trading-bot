import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Ensure logs directory exists
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "app.log"),
    ],
)

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from perps_trading import PerpsBot  # noqa: E402

BOT_ACCOUNT = os.getenv("BOT_ACCOUNT")

_config_path = Path(__file__).parent.parent / "config" / "accounts.json"
with open(_config_path) as _f:
    _cfg = json.load(_f)

_account = next(a for a in _cfg["accounts"] if a["name"] == BOT_ACCOUNT)

if not _account.get("active", True):
    logger.info("Account %s is inactive (active=false in accounts.json). Exiting.", BOT_ACCOUNT)
    sys.exit(0)

ACCOUNT_NAME = BOT_ACCOUNT

# Perps
PERPS_LIVE          = _cfg["perps_live"]
PERPS_LEVERAGE      = _account["leverage"]
PERPS_STOP_LOSS_PCT = _cfg["stop_loss_pct"]
PERPS_BB_PERIOD     = _cfg["bb_period"]
PERPS_BB_STD        = _cfg["bb_std"]
PERPS_EMA_PERIOD    = _cfg["ema_period"]
PERPS_CAPITAL_PCT   = _cfg["target_risk_pct"] / (PERPS_LEVERAGE * PERPS_STOP_LOSS_PCT / 100)

assert PERPS_LEVERAGE <= 15, (
    f"Leverage {PERPS_LEVERAGE}x unsafe for {PERPS_STOP_LOSS_PCT}% SL strategy (max 15x)"
)

logger.info("Account: %s | leverage=%dx | capital_pct=%.1f%% | risk/trade=%.1f%%",
            ACCOUNT_NAME, PERPS_LEVERAGE, PERPS_CAPITAL_PCT, _cfg["target_risk_pct"])


def main():
    perps_bot = PerpsBot(
        dry_run=not PERPS_LIVE,
        leverage=PERPS_LEVERAGE,
        capital_percent=PERPS_CAPITAL_PCT,
        bb_period=PERPS_BB_PERIOD,
        bb_std=PERPS_BB_STD,
        ema_period=PERPS_EMA_PERIOD,
        stop_loss_pct=PERPS_STOP_LOSS_PCT,
        account_name=ACCOUNT_NAME,
    )

    INDICATOR_INTERVAL = 5 * 60  # seconds
    TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def fetch_price() -> float | None:
        try:
            resp = requests.get(TICKER_URL, timeout=5)
            resp.raise_for_status()
            return float(resp.json()["price"])
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
