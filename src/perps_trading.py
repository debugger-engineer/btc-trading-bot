import logging
import re
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import db

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"


class CandleMaintainer:
    """Maintains a local 5m candle DataFrame, updating the current candle
    from WS prices and re-syncing from REST periodically."""

    CANDLE_INTERVAL_S = 300  # 5 minutes
    RESYNC_INTERVAL_S = 300  # re-fetch from REST every 5 minutes

    def __init__(self, num_candles: int):
        self._num_candles = num_candles
        self._df: pd.DataFrame | None = None
        self._lock = threading.Lock()
        self._last_resync: float = 0.0
        self._current_candle_open_ms: int | None = None
        self._initialized = False

    def _candle_open_ms(self, timestamp_s: float) -> int:
        epoch_s = int(timestamp_s)
        candle_start_s = epoch_s - (epoch_s % self.CANDLE_INTERVAL_S)
        return candle_start_s * 1000

    def initialize(self, fetch_fn):
        """Fetch historical candles via REST once at startup."""
        df = fetch_fn()
        with self._lock:
            self._df = df.copy()
            # Determine current candle boundary from latest candle
            self._current_candle_open_ms = self._candle_open_ms(time.time())
            self._initialized = True
            self._last_resync = time.monotonic()
        logger.info("CandleMaintainer initialized with %d candles", len(df))

    def update_price(self, price: float, timestamp_s: float):
        """Update the current candle's close from the live WS price."""
        if not self._initialized or self._df is None:
            return
        current_ms = self._candle_open_ms(timestamp_s)
        with self._lock:
            if current_ms == self._current_candle_open_ms:
                # Same candle — update close
                self._df.at[self._df.index[-1], "close"] = price
            elif current_ms > self._current_candle_open_ms:
                # New candle boundary — append new row
                new_row = pd.DataFrame([{
                    "close": price,
                    "open_time": pd.Timestamp(current_ms, unit="ms", tz="UTC"),
                }])
                self._df = pd.concat([self._df, new_row], ignore_index=True)
                if len(self._df) > self._num_candles:
                    self._df = self._df.iloc[-self._num_candles:].reset_index(drop=True)
                self._current_candle_open_ms = current_ms

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return self._df.copy() if self._df is not None else pd.DataFrame()

    def maybe_resync(self, fetch_fn):
        """Periodically re-fetch from REST to correct any drift."""
        now = time.monotonic()
        if now - self._last_resync < self.RESYNC_INTERVAL_S:
            return
        try:
            fresh_df = fetch_fn()
            with self._lock:
                # Preserve local current candle close (more recent than REST)
                local_close = self._df.at[self._df.index[-1], "close"] if self._df is not None else None
                self._df = fresh_df.copy()
                if local_close is not None and self._candle_open_ms(time.time()) == self._current_candle_open_ms:
                    self._df.at[self._df.index[-1], "close"] = local_close
            self._last_resync = now
            logger.debug("CandleMaintainer re-synced with %d candles", len(fresh_df))
        except Exception as exc:
            logger.warning("CandleMaintainer re-sync failed: %s", exc)


def _extract_oid(result: dict) -> int | None:
    """Pull the resting order-id from a Hyperliquid order placement response."""
    try:
        return result["response"]["data"]["statuses"][0]["resting"]["oid"]
    except (KeyError, IndexError, TypeError):
        return None


def _is_filled(result: dict) -> tuple[bool, float]:
    """Return (True, avgPx) if the order was immediately filled."""
    try:
        filled = result["response"]["data"]["statuses"][0]["filled"]
        return True, float(filled.get("avgPx", 0))
    except (KeyError, IndexError, TypeError):
        return False, 0.0


def _is_alo_rejected(result: dict) -> bool:
    """Return True if an ALO order was rejected because it would have crossed the spread."""
    try:
        status = result["response"]["data"]["statuses"][0]
        # Hyperliquid returns {"error": "..."} or {"canceled": ...} for ALO kills
        return "error" in status or "canceled" in status
    except (KeyError, IndexError, TypeError):
        return False


def _is_margin_error(result: dict) -> bool:
    """Return True if the order was rejected due to insufficient margin."""
    try:
        error = result["response"]["data"]["statuses"][0].get("error", "")
        return "Insufficient margin" in error
    except (KeyError, IndexError, TypeError):
        return False


def _parse_bbo_from_rejection(result: dict) -> tuple[float, float] | None:
    """Return (bid, ask) parsed from an ALO rejection error string, or None if not parseable.

    Hyperliquid includes the BBO in the error: "bbo was {bid}@{ask}. asset=N"
    """
    try:
        error_msg = result["response"]["data"]["statuses"][0].get("error", "")
        m = re.search(r"bbo was (\d+(?:\.\d+)?)@(\d+(?:\.\d+)?)", error_msg)
        if m:
            return float(m.group(1)), float(m.group(2))
    except (KeyError, IndexError, TypeError):
        pass
    return None


class PerpsBot:
    """BB mean-reversion BTC perpetuals bot.

    Strategy:
    - Entry: price touches/crosses outside Bollinger Band → LONG (lower band) or SHORT (upper band)
    - Exit:  price reaches the opposite band
    - Stop-loss: 2% from entry price (cuts runaway moves before they compound)
    - Trend filter: EMA(200) — only longs when price > EMA, shorts when price < EMA
    - Timeframe: 5m candles (~2-3 trades/day)

    Validated backtest: BB(20,2) + opposite band exit + SL 2% + EMA(200)
    → +88.8% annually, 69% win rate, 872 trades (5m, 365 days)
    """

    def __init__(
        self,
        dry_run: bool,
        leverage: int = 3,
        capital_percent: float = 50.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_period: int = 200,
        stop_loss_pct: float = 2.0,
        account_name: str = "",
        symbol: str = "BTC",
        binance_symbol: str = "BTCUSDT",
        min_band_width_pct: float = 0.10,
        max_same_direction: int = 4,
        force_gtc: bool = False,
        _trader=None,
        _db=None,
    ):
        self.dry_run = dry_run
        self.leverage = leverage
        self.capital_percent = capital_percent
        self.account_name = account_name
        self.symbol = symbol
        self.binance_symbol = binance_symbol
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_period = ema_period
        self.stop_loss_pct = stop_loss_pct
        self.min_band_width_pct = min_band_width_pct
        self.max_same_direction = max_same_direction
        self.force_gtc = force_gtc

        self.position = None        # "LONG", "SHORT", or None
        self.entry_price = None     # price at entry (for stop-loss check)
        self.stop_order_id = None
        self.exit_order_id = None   # resting limit exit order (maker fee)
        self.current_trade_id = None
        self._position_sz: float | None = None  # size of current position in base asset
        self._entry_fee: float = 0.0
        self._last_hourly_log: int = -1  # tracks last hour the status was logged
        self._last_debug_log: float = 0.0

        self._bb_upper: float | None = None
        self._bb_lower: float | None = None
        self._bb_mid: float | None = None
        self._ema: float | None = None
        self._last_close: float | None = None
        self._db = _db or db  # injected for testing; falls back to real db module

        # Pending limit-entry order state
        self._pending_entry_oid: int | None = None
        self._pending_entry_side: str | None = None
        self._pending_entry_px: float | None = None
        self._pending_entry_time: float | None = None
        self._margin_cooldown_until: float = 0.0
        self._last_pos_check: float = 0.0  # throttle exchange position polling
        self._pending_entry_size_usd: float | None = None
        self._pending_bb_snapshot: dict | None = None

        self._log_tag = f"[PERPS:{self.symbol}]"

        if self.dry_run:
            logger.info(
                "%s [DRY RUN] BB(%d,%.1f) EMA(%d) SL=%.0f%% leverage=%dx capital=%.0f%%",
                self._log_tag, bb_period, bb_std, ema_period, stop_loss_pct, leverage, capital_percent,
            )
        else:
            if _trader is not None:
                self.trader = _trader
            else:
                from exchange import HyperliquidTrader
                self.trader = HyperliquidTrader(symbol=self.symbol)
            self.trader.update_leverage(self.leverage)
            self._db.init_bb_db()
            logger.info(
                "%s [LIVE] BB(%d,%.1f) EMA(%d) SL=%.0f%% leverage=%dx capital=%.0f%%",
                self._log_tag, bb_period, bb_std, ema_period, stop_loss_pct, leverage, capital_percent,
            )
            self._restore_state()

    # ── State management ───────────────────────────────────────────────────────

    def _restore_state(self):
        """On restart, recover position from DB and re-place the stop-loss order."""
        # Cancel any orphaned limit entry orders from a previous session
        self.trader.cancel_all_open_orders(self.symbol)
        open_trade = self._db.get_open_bb_trade(symbol=self.symbol, account_name=self.account_name)
        if not open_trade:
            return
        self.position         = open_trade["direction"]
        self.entry_price      = open_trade["entry_price"]
        self.current_trade_id = open_trade["id"]
        # Verify a matching position still exists on the exchange
        actual = self.trader.get_perp_position()
        if not actual or actual["side"] != self.position:
            logger.warning("%s DB has open trade but no matching exchange position — closing DB record", self._log_tag)
            self._db.close_bb_trade(self.current_trade_id, self.entry_price, stopped=False)
            self._reset_position()
            return
        stop_price = (
            self.entry_price * (1 - self.stop_loss_pct / 100) if self.position == "LONG"
            else self.entry_price * (1 + self.stop_loss_pct / 100)
        )
        self._position_sz = actual["size"]
        self.trader.cancel_all_open_orders(self.symbol)
        sl_result = self.trader.place_stop_loss_perp(self.position, actual["size"], stop_price)
        self.stop_order_id = _extract_oid(sl_result)
        # Exit limit will be placed by the first update_indicators() call in the main loop
        logger.info(
            "%s Restored open %s trade id=%d @ $%.2f | SL=$%.2f (stop_oid=%s) — exit limit pending next indicator update",
            self._log_tag, self.position, self.current_trade_id, self.entry_price, stop_price, self.stop_order_id,
        )

    def _reset_position(self):
        self.position            = None
        self.entry_price         = None
        self.stop_order_id       = None
        self.exit_order_id       = None
        self.current_trade_id    = None
        self._position_sz        = None

    def _reset_pending_entry(self):
        self._pending_entry_oid       = None
        self._pending_entry_side      = None
        self._pending_entry_px        = None
        self._pending_entry_time      = None
        self._pending_entry_size_usd  = None
        self._pending_bb_snapshot     = None

    # ── Market data ────────────────────────────────────────────────────────────

    def _fetch_klines(self) -> pd.DataFrame:
        if self.binance_symbol:
            return self._fetch_klines_binance()
        return self._fetch_klines_hl()

    def _fetch_klines_binance(self) -> pd.DataFrame:
        limit = max(self.ema_period, self.bb_period) * 3
        params = {"symbol": self.binance_symbol, "interval": "5m", "limit": limit}
        for attempt in range(1, 4):
            try:
                resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
                resp.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                logger.warning("%s Network/HTTP error (attempt %d/3), retrying in %ds: %s", self._log_tag, attempt, wait, e)
                time.sleep(wait)
        df = pd.DataFrame(resp.json(), columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["close"] = df["close"].astype(float)
        return df

    def _fetch_klines_hl(self) -> pd.DataFrame:
        """Fetch 5m candles from Hyperliquid candleSnapshot API (for tokens without a Binance pair)."""
        limit = max(self.ema_period, self.bb_period) * 3
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - limit * 5 * 60 * 1000  # 5m per candle
        for attempt in range(1, 4):
            try:
                resp = requests.post(HL_INFO_URL, json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": self.symbol,
                        "interval": "5m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                    }
                }, timeout=15)
                resp.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                logger.warning("%s Network/HTTP error (attempt %d/3), retrying in %ds: %s", self._log_tag, attempt, wait, e)
                time.sleep(wait)
        data = resp.json()
        if not data:
            raise RuntimeError(f"No candle data from Hyperliquid for {self.symbol}")
        df = pd.DataFrame(data)
        df["close"] = df["c"].astype(float)
        df["open_time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        return df

    def _compute_indicators(self, df: pd.DataFrame):
        closes = df["close"]
        bb_mid   = closes.rolling(self.bb_period).mean()
        bb_upper = bb_mid + self.bb_std * closes.rolling(self.bb_period).std()
        bb_lower = bb_mid - self.bb_std * closes.rolling(self.bb_period).std()
        ema      = closes.ewm(span=self.ema_period, adjust=False).mean()
        return (
            float(bb_upper.iloc[-1]),
            float(bb_lower.iloc[-1]),
            float(bb_mid.iloc[-1]),
            float(ema.iloc[-1]),
        )

    def update_indicators(self):
        try:
            maintainer = getattr(self, "candle_maintainer", None)
            if maintainer:
                maintainer.maybe_resync(self._fetch_klines)
                df = maintainer.get_dataframe()
            else:
                df = self._fetch_klines()
            self._last_close = float(df["close"].iloc[-1])
            self._bb_upper, self._bb_lower, self._bb_mid, self._ema = self._compute_indicators(df)
            self._margin_cooldown_until = 0.0
            logger.debug(
                "%s Indicators updated: BB upper=%.2f lower=%.2f mid=%.2f EMA=%.2f",
                self._log_tag, self._bb_upper, self._bb_lower, self._bb_mid, self._ema,
            )
            # Replace exit limit at updated target band whenever we have an open position
            if not self.dry_run and self.position and self._position_sz:
                self._replace_exit_limit()
        except Exception as exc:
            logger.error("%s Indicator update failed: %s", self._log_tag, exc, exc_info=True)

    def _is_inverted(self) -> bool:
        """Return True when the dynamic TP target has drifted past entry price (guaranteed loss if filled)."""
        if self.position == "SHORT":
            return self._bb_lower is not None and self._bb_lower > self.entry_price
        if self.position == "LONG":
            return self._bb_upper is not None and self._bb_upper < self.entry_price
        return False

    def _place_exit_limit_alo(self, target_px: float) -> int | None:
        """Place an ALO exit limit with retry logic (same pattern as entry ALO).

        Returns the resting order oid, or None if all attempts fail.
        """
        result = self.trader.place_exit_limit_perp(self.position, self._position_sz, target_px)
        oid = _extract_oid(result)
        if oid is not None:
            return oid
        # Check if it filled immediately (rare but possible)
        filled, filled_px = _is_filled(result)
        if filled:
            logger.info("%s Exit limit filled immediately @ $%.2f (maker)", self._log_tag, filled_px)
            return None  # caller will detect position closed via exchange polling
        # ALO rejected — retry up to 3 times using BBO
        limit_px = target_px
        for attempt in range(1, 4):
            if not _is_alo_rejected(result):
                break
            bbo = _parse_bbo_from_rejection(result)
            if bbo:
                bid, ask = bbo
                # Closing LONG = sell → place at ask to rest as maker
                # Closing SHORT = buy → place at bid to rest as maker
                retry_px = ask if self.position == "LONG" else bid
            else:
                nudge = 1.0 if self.position == "LONG" else -1.0
                retry_px = round(limit_px + nudge, 1)
            logger.info(
                "%s ALO exit rejected at $%.1f — retry %d/3 at $%.1f",
                self._log_tag, limit_px, attempt, retry_px,
            )
            result = self.trader.place_exit_limit_perp(self.position, self._position_sz, retry_px)
            oid = _extract_oid(result)
            if oid is not None:
                return oid
            limit_px = retry_px
        # All ALO retries exhausted — fall back to GTC
        logger.warning("%s ALO exit rejected after 3 retries — falling back to GTC at $%.1f", self._log_tag, limit_px)
        result = self.trader.place_exit_limit_perp(self.position, self._position_sz, limit_px, tif="Gtc")
        oid = _extract_oid(result)
        if oid is None:
            filled, filled_px = _is_filled(result)
            if filled:
                logger.info("%s GTC exit filled immediately @ $%.2f (taker)", self._log_tag, filled_px)
            else:
                logger.error("%s Exit limit failed after all retries: %s", self._log_tag, result)
        return oid

    def _replace_exit_limit(self):
        """Cancel the old exit limit (if any) and place a fresh one at the current target band.

        If the TP has inverted (band drifted past entry price), the exit is placed at
        entry price (break-even) instead of the band, capping the loss at fees only.
        Skips replacement if the target price hasn't changed (avoids unnecessary order churn).
        """
        inverted = self._is_inverted()
        if inverted:
            target_px = self.entry_price
        else:
            target_px = self._bb_upper if self.position == "LONG" else self._bb_lower
        # Skip if target price barely moved (< 0.01%) and exit order is already resting
        if self.exit_order_id and hasattr(self, "_last_exit_target_px"):
            pct_change = abs(target_px - self._last_exit_target_px) / self._last_exit_target_px
            if pct_change < 0.0001:  # 0.01%
                return
        if self.exit_order_id:
            self.trader.cancel_order(self.symbol, self.exit_order_id)
        self._last_exit_target_px = target_px
        self.exit_order_id = self._place_exit_limit_alo(target_px)
        logger.info(
            "%s Exit limit refreshed — %s @ $%.1f (oid=%s)%s",
            self._log_tag, self.position, target_px, self.exit_order_id,
            " [BREAK-EVEN: TP inverted]" if inverted else "",
        )

    # ── Core logic ─────────────────────────────────────────────────────────────

    def _is_stop_hit(self, price: float) -> bool:
        if self.entry_price is None:
            return False
        if self.position == "LONG":
            return price <= self.entry_price * (1 - self.stop_loss_pct / 100)
        if self.position == "SHORT":
            return price >= self.entry_price * (1 + self.stop_loss_pct / 100)
        return False

    def _min_band_width_ok(self, bb_upper: float, bb_lower: float, price: float) -> bool:
        """Return True only if the BB band width provides enough profit margin to overcome fees + slippage."""
        band_width_pct = (bb_upper - bb_lower) / price * 100
        if band_width_pct < self.min_band_width_pct:
            logger.debug(
                "%s Band width too narrow (%.4f%% < %.4f%%) — skipping entry",
                self._log_tag, band_width_pct, self.min_band_width_pct,
            )
            return False
        return True

    def check_signals(self, price: float):
        if self._bb_upper is None:
            return  # indicators not ready yet
        try:
            bb_upper = self._bb_upper
            bb_lower = self._bb_lower
            bb_mid   = self._bb_mid
            ema      = self._ema

            # Poll pending limit entry order for fill or cancellation
            if not self.dry_run:
                self._check_pending_entry(price)

            # Detect if position was closed exchange-side (SL or exit limit)
            # First check via WS for instant detection, fall back to 10s REST poll
            feed = getattr(self, "price_feed", None)
            if self.position and not self.dry_run:
                ws_exit_filled = feed and self.exit_order_id and feed.is_order_filled(self.exit_order_id)
                ws_sl_filled   = feed and self.stop_order_id and feed.is_order_filled(self.stop_order_id)
                # WS instant detection
                if ws_exit_filled or ws_sl_filled:
                    stopped = ws_sl_filled and not ws_exit_filled
                    if stopped:
                        # SL filled → cancel exit limit
                        if self.exit_order_id:
                            self.trader.cancel_order(self.symbol, self.exit_order_id)
                        logger.info("%s Position closed [SL] (WS) — cancelled exit limit", self._log_tag)
                    else:
                        # Exit filled → cancel SL
                        if self.stop_order_id:
                            self.trader.cancel_order(self.symbol, self.stop_order_id)
                        logger.info("%s Position closed [TARGET HIT] (WS) — exit limit filled (maker)", self._log_tag)
                    if feed:
                        if self.exit_order_id:
                            feed.clear_order(self.exit_order_id)
                        if self.stop_order_id:
                            feed.clear_order(self.stop_order_id)
                    if self.current_trade_id:
                        exit_fill = self.trader.get_last_fill()
                        self._db.close_bb_trade(
                            self.current_trade_id, exit_fill["px"], stopped=stopped,
                            exit_bb_upper=self._bb_upper, exit_bb_lower=self._bb_lower,
                            exit_bb_mid=self._bb_mid,
                            entry_fee=self._entry_fee, exit_fee=exit_fill["fee"],
                        )
                    self._reset_position()
                    return
                # REST fallback — throttled to every 10s
                if time.monotonic() - self._last_pos_check >= 10:
                    self._last_pos_check = time.monotonic()
                    if self.trader.get_perp_position() is None:
                        open_oids = {
                            o["oid"] for o in self.trader.get_open_orders()
                            if o.get("coin") == self.symbol
                        }
                        exit_still_open = bool(self.exit_order_id and self.exit_order_id in open_oids)
                        sl_still_open   = bool(self.stop_order_id and self.stop_order_id in open_oids)
                        if exit_still_open:
                            stopped = True
                            self.trader.cancel_order(self.symbol, self.exit_order_id)
                            logger.info("%s Position gone [SL] — cancelled exit limit oid=%d", self._log_tag, self.exit_order_id)
                        else:
                            stopped = False
                            if sl_still_open:
                                self.trader.cancel_order(self.symbol, self.stop_order_id)
                            logger.info("%s Position gone [TARGET HIT] — exit limit filled (maker)", self._log_tag)
                        if self.current_trade_id:
                            exit_fill = self.trader.get_last_fill()
                            self._db.close_bb_trade(
                                self.current_trade_id, exit_fill["px"], stopped=stopped,
                                exit_bb_upper=self._bb_upper, exit_bb_lower=self._bb_lower,
                                exit_bb_mid=self._bb_mid,
                                entry_fee=self._entry_fee, exit_fee=exit_fill["fee"],
                            )
                        self._reset_position()
                        return

            now = time.monotonic()
            if now - self._last_debug_log >= 60:
                self._last_debug_log = now
                logger.debug(
                    "%s $%.2f | BB upper=%.2f lower=%.2f mid=%.2f | EMA=%.2f | pos=%s",
                    self._log_tag, price, bb_upper, bb_lower, bb_mid, ema, self.position,
                )
            current_hour = datetime.now(timezone.utc).hour
            if current_hour != self._last_hourly_log:
                self._last_hourly_log = current_hour
                logger.info(
                    "%s $%.2f | BB upper=%.2f lower=%.2f mid=%.2f | EMA=%.2f | pos=%s",
                    self._log_tag, price, bb_upper, bb_lower, bb_mid, ema, self.position,
                )

            if self.position is None and self._pending_entry_oid is None:
                if time.monotonic() < self._margin_cooldown_until:
                    pass  # suppressed — already logged once
                elif not self._min_band_width_ok(bb_upper, bb_lower, price):
                    pass  # bands too narrow — profit can't overcome fees
                elif price <= bb_lower and price > ema:
                    self._dry_or_live(
                        f"[DRY RUN] ${price:.2f} ≤ BB lower ${bb_lower:.2f} & above EMA — LONG entry",
                        lambda: self._place_entry_limit("LONG", price, bb_upper, bb_lower, bb_mid),
                    )
                elif price >= bb_upper and price < ema:
                    self._dry_or_live(
                        f"[DRY RUN] ${price:.2f} ≥ BB upper ${bb_upper:.2f} & below EMA — SHORT entry",
                        lambda: self._place_entry_limit("SHORT", price, bb_upper, bb_lower, bb_mid),
                    )

            elif self.position == "LONG":
                if self._is_stop_hit(price):
                    sl_price = self.entry_price * (1 - self.stop_loss_pct / 100)
                    self._dry_or_live(
                        f"[DRY RUN] Stop-loss hit — close LONG @ ${price:.2f} (SL=${sl_price:.2f})",
                        lambda: self._close(price, stopped=True),
                    )
                elif bb_upper < self.entry_price and price <= self.entry_price:
                    self._dry_or_live(
                        f"[DRY RUN] BB inverted (upper=${bb_upper:.2f} < entry=${self.entry_price:.2f}) — close LONG at break-even @ ${price:.2f}",
                        lambda: self._close(price),
                    )
                elif price >= bb_upper:
                    self._dry_or_live(
                        f"[DRY RUN] ${price:.2f} ≥ BB upper ${bb_upper:.2f} — close LONG (target hit)",
                        lambda: self._close(price),
                    )

            elif self.position == "SHORT":
                if self._is_stop_hit(price):
                    sl_price = self.entry_price * (1 + self.stop_loss_pct / 100)
                    self._dry_or_live(
                        f"[DRY RUN] Stop-loss hit — close SHORT @ ${price:.2f} (SL=${sl_price:.2f})",
                        lambda: self._close(price, stopped=True),
                    )
                elif bb_lower > self.entry_price and price >= self.entry_price:
                    self._dry_or_live(
                        f"[DRY RUN] BB inverted (lower=${bb_lower:.2f} > entry=${self.entry_price:.2f}) — close SHORT at break-even @ ${price:.2f}",
                        lambda: self._close(price),
                    )
                elif price <= bb_lower:
                    self._dry_or_live(
                        f"[DRY RUN] ${price:.2f} ≤ BB lower ${bb_lower:.2f} — close SHORT (target hit)",
                        lambda: self._close(price),
                    )

        except Exception as exc:
            logger.error("%s ERROR: %s", self._log_tag, exc, exc_info=True)

    def run_cycle(self):
        self.update_indicators()
        if self._last_close is not None:
            self.check_signals(self._last_close)

    # ── Execution ──────────────────────────────────────────────────────────────

    def _dry_or_live(self, dry_msg: str, live_fn):
        if self.dry_run:
            logger.info(dry_msg)
        else:
            live_fn()

    def _entry_size_usd(self) -> float:
        equity = self.trader.get_account_equity()
        return equity * (self.capital_percent / 100) * self.leverage

    def _place_entry_limit(self, side: str, price: float, bb_upper: float, bb_lower: float, bb_mid: float):
        """Phase 1: place an ALO (post-only/maker) limit order at the trigger band level.

        On ALO rejection (order would cross the spread), retry up to 3 times using the
        actual BBO from the rejection message to ensure the retry price lands on the maker
        side of the book. If all ALO attempts fail, fall back to a GTC taker order.
        """
        # Portfolio correlation guard: limit same-direction exposure across all symbols
        if not self.dry_run and self.max_same_direction > 0:
            open_count = self._db.count_open_bb_trades_by_direction(side, self.account_name)
            if open_count >= self.max_same_direction:
                logger.info(
                    "%s Skipping %s — max %d same-direction positions reached (%d open)",
                    self._log_tag, side, self.max_same_direction, open_count,
                )
                return
        size_usd = self._entry_size_usd()
        if size_usd < 1.0:
            self._margin_cooldown_until = time.monotonic() + 300
            logger.warning("%s Skipping — insufficient margin (%.2f USDC). Retry in 5m.", self._log_tag, size_usd)
            return
        self.trader.update_leverage(self.leverage)
        # Use the current mid-price (not the stale band level) as the limit.
        # Signal detection already confirmed price is at/past the band, so
        # entering at 'price' gives an immediate maker fill instead of an
        # ALO rejection cascade when the band value is stale.
        limit_px = price

        # For low-liquidity tokens (force_gtc=True), skip ALO entirely
        # and place a GTC limit order directly. This avoids the ALO rejection
        # cascade that causes missed entries on wide-spread pairs like GOLD.
        if self.force_gtc:
            logger.info("%s GTC %s at $%.1f (force_gtc mode)", self._log_tag, side, limit_px)
            result = (
                self.trader.open_long_limit(size_usd, limit_px, tif="Gtc")
                if side == "LONG"
                else self.trader.open_short_limit(size_usd, limit_px, tif="Gtc")
            )
            oid = _extract_oid(result)
            if oid is None:
                filled, filled_px = _is_filled(result)
                if filled:
                    logger.info("%s GTC %s filled immediately @ $%.2f", self._log_tag, side, filled_px)
                    self._on_entry_filled(side, filled_px, size_usd, bb_upper, bb_lower, bb_mid)
                    return
                if _is_margin_error(result):
                    self._margin_cooldown_until = time.monotonic() + 300
                    logger.warning("%s Insufficient margin from exchange. Retry in 5m.", self._log_tag)
                    return
                logger.error("%s GTC %s order failed: %s", self._log_tag, side, result)
                return
        else:
            result = (
                self.trader.open_long_limit(size_usd, limit_px)
                if side == "LONG"
                else self.trader.open_short_limit(size_usd, limit_px)
            )
            oid = _extract_oid(result)
            if oid is None:
                filled, filled_px = _is_filled(result)
                if filled:
                    # ALO orders should not fill immediately, but handle defensively
                    logger.info("%s Limit %s filled immediately @ $%.2f", self._log_tag, side, filled_px)
                    self._on_entry_filled(side, filled_px, size_usd, bb_upper, bb_lower, bb_mid)
                    return
                # ALO rejected: retry up to 3 times using the BBO from the rejection message
                for attempt in range(1, 4):
                    if not _is_alo_rejected(result):
                        break
                    if _is_margin_error(result):
                        self._margin_cooldown_until = time.monotonic() + 300
                        logger.warning("%s Insufficient margin from exchange. Retry in 5m.", self._log_tag)
                        return
                    bbo = _parse_bbo_from_rejection(result)
                    if bbo:
                        bid, ask = bbo
                        # Place at current bid (LONG) or ask (SHORT) — guaranteed maker side
                        retry_px = bid if side == "LONG" else ask
                    else:
                        # BBO not parseable: nudge $1 further inside the book from last price
                        retry_px = round(limit_px - 1.0, 1) if side == "LONG" else round(limit_px + 1.0, 1)
                    logger.info(
                        "%s ALO %s rejected at $%.1f — retry %d/3 at $%.1f",
                        self._log_tag, side, limit_px, attempt, retry_px,
                    )
                    result = (
                        self.trader.open_long_limit(size_usd, retry_px)
                        if side == "LONG"
                        else self.trader.open_short_limit(size_usd, retry_px)
                    )
                    oid = _extract_oid(result)
                    limit_px = retry_px
                # All ALO attempts exhausted — fall back to GTC (taker)
                if oid is None:
                    logger.warning(
                        "%s ALO %s rejected after 3 retries — falling back to GTC taker at $%.1f",
                        self._log_tag, side, limit_px,
                    )
                    result = (
                        self.trader.open_long_limit(size_usd, limit_px, tif="Gtc")
                        if side == "LONG"
                        else self.trader.open_short_limit(size_usd, limit_px, tif="Gtc")
                    )
                    oid = _extract_oid(result)
                    filled, filled_px = _is_filled(result)
                    if filled:
                        logger.info("%s GTC fallback %s filled @ $%.2f (taker)", self._log_tag, side, filled_px)
                        self._on_entry_filled(side, filled_px, size_usd, bb_upper, bb_lower, bb_mid)
                        return
                if oid is None:
                    logger.error("%s Limit %s order failed after all retries: %s", self._log_tag, side, result)
                    return
        self._pending_entry_oid      = oid
        self._pending_entry_side     = side
        self._pending_entry_px       = limit_px
        self._pending_entry_time     = time.monotonic()
        self._pending_entry_size_usd = size_usd
        self._pending_bb_snapshot    = {"bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid}
        logger.info(
            "%s LIMIT %s order placed oid=%d @ $%.1f size=$%.2f (cancels if no fill in 60s)",
            self._log_tag, side, oid, limit_px, size_usd,
        )

    def _on_entry_filled(self, side: str, fill_px: float, size_usd: float,
                         bb_upper: float, bb_lower: float, bb_mid: float):
        """Phase 2: complete position setup after the limit entry is confirmed filled."""
        self.position    = side
        self.entry_price = fill_px
        stop_price = (
            fill_px * (1 - self.stop_loss_pct / 100) if side == "LONG"
            else fill_px * (1 + self.stop_loss_pct / 100)
        )
        target_price = bb_upper if side == "LONG" else bb_lower
        entry_fill = self.trader.get_last_fill()
        self._entry_fee = entry_fill["fee"]
        self.current_trade_id = self._db.open_bb_trade(
            side, self.leverage, fill_px, size_usd,
            bb_upper, bb_lower, bb_mid, stop_price,
            account_name=self.account_name, symbol=self.symbol,
        )
        sz = round(size_usd / fill_px, self.trader.sz_decimals)
        self._position_sz = sz
        sl_result = self.trader.place_stop_loss_perp(side, sz, stop_price)
        self.stop_order_id = _extract_oid(sl_result)
        self.exit_order_id = self._place_exit_limit_alo(target_price)
        self._reset_pending_entry()
        logger.info(
            "%s OPEN %s $%.2f notional @ $%.2f | target=$%.2f (exit_oid=%s) | SL=$%.2f (stop_oid=%s) (trade_id=%d)",
            self._log_tag, side, size_usd, fill_px, target_price, self.exit_order_id, stop_price, self.stop_order_id, self.current_trade_id,
        )

    def _check_pending_entry(self, price: float):
        """Detect fill via WebSocket orderUpdates, or cancel on timeout/price drift."""
        if self._pending_entry_oid is None:
            return
        side     = self._pending_entry_side
        limit_px = self._pending_entry_px
        snap     = self._pending_bb_snapshot
        oid      = self._pending_entry_oid
        elapsed  = time.monotonic() - self._pending_entry_time

        # Instant fill detection via WebSocket (no REST call)
        feed = getattr(self, "price_feed", None)
        if feed and feed.is_order_gone(oid):
            filled = feed.is_order_filled(oid)
            feed.clear_order(oid)
            if filled:
                # Confirm position exists on exchange (single REST call, only on fill)
                actual_pos = self.trader.get_perp_position()
                if actual_pos and actual_pos["side"] == side:
                    logger.info("%s Limit entry oid=%d filled (WS) — position confirmed", self._log_tag, oid)
                    self._on_entry_filled(
                        side, limit_px, self._pending_entry_size_usd,
                        snap["bb_upper"], snap["bb_lower"], snap["bb_mid"],
                    )
                    return
            # Order was cancelled/rejected by exchange
            logger.info("%s Limit entry oid=%d cancelled/rejected (WS)", self._log_tag, oid)
            self._reset_pending_entry()
            return

        timed_out        = elapsed > 60.0
        price_moved_away = (
            (side == "LONG"  and price > limit_px * 1.001) or
            (side == "SHORT" and price < limit_px * 0.999)
        )
        if timed_out or price_moved_away:
            # Check if the order already filled before trying to cancel
            actual_pos = self.trader.get_perp_position()
            if actual_pos and actual_pos["side"] == side:
                logger.info("%s Limit entry oid=%d filled — position confirmed", self._log_tag, oid)
                self._on_entry_filled(
                    side, limit_px, self._pending_entry_size_usd,
                    snap["bb_upper"], snap["bb_lower"], snap["bb_mid"],
                )
                return
            reason = "timeout" if timed_out else f"price moved away (${price:.2f})"
            logger.info("%s Cancelling pending %s limit oid=%d — %s", self._log_tag, side, oid, reason)
            self.trader.cancel_order(self.symbol, oid)
            if feed:
                feed.clear_order(oid)
            self._reset_pending_entry()

    def _close(self, price: float, stopped: bool = False):
        # Cancel both resting orders before closing to avoid double-close
        if self.exit_order_id:
            self.trader.cancel_order(self.symbol, self.exit_order_id)
            self.exit_order_id = None
        if self.stop_order_id:
            self.trader.cancel_order(self.symbol, self.stop_order_id)
            self.stop_order_id = None
        result = self.trader.close_perp_position()
        if result.get("status") == "ok":
            if self.current_trade_id:
                exit_fill = self.trader.get_last_fill()
                self._db.close_bb_trade(
                    self.current_trade_id, exit_fill["px"], stopped,
                    exit_bb_upper=self._bb_upper, exit_bb_lower=self._bb_lower, exit_bb_mid=self._bb_mid,
                    entry_fee=self._entry_fee, exit_fee=exit_fill["fee"],
                )
            logger.info(
                "%s CLOSE %s @ $%.2f%s (trade_id=%s)",
                self._log_tag, self.position, price,
                " [STOP LOSS]" if stopped else " [TARGET HIT]",
                self.current_trade_id,
            )
            self._reset_position()
        else:
            logger.error("%s Close failed: %s", self._log_tag, result)
