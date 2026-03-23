import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import db

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BTC_PERP_COIN = "BTC"  # Hyperliquid perp asset name


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
        _trader=None,
        _db=None,
    ):
        self.dry_run = dry_run
        self.leverage = leverage
        self.capital_percent = capital_percent
        self.account_name = account_name
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_period = ema_period
        self.stop_loss_pct = stop_loss_pct

        self.position = None        # "LONG", "SHORT", or None
        self.entry_price = None     # price at entry (for stop-loss check)
        self.stop_order_id = None
        self.exit_order_id = None   # resting limit exit order (maker fee)
        self.current_trade_id = None
        self._position_btc_sz: float | None = None  # BTC size of current position
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
        self._pending_entry_size_usd: float | None = None
        self._pending_bb_snapshot: dict | None = None

        if self.dry_run:
            logger.info(
                "PerpsBot [DRY RUN] BB(%d,%.1f) EMA(%d) SL=%.0f%% leverage=%dx capital=%.0f%%",
                bb_period, bb_std, ema_period, stop_loss_pct, leverage, capital_percent,
            )
        else:
            if _trader is not None:
                self.trader = _trader
            else:
                from exchange import HyperliquidTrader
                self.trader = HyperliquidTrader()
            self.trader.update_leverage(self.leverage)
            self._db.init_bb_db()
            logger.info(
                "PerpsBot [LIVE] BB(%d,%.1f) EMA(%d) SL=%.0f%% leverage=%dx capital=%.0f%%",
                bb_period, bb_std, ema_period, stop_loss_pct, leverage, capital_percent,
            )
            self._restore_state()

    # ── State management ───────────────────────────────────────────────────────

    def _restore_state(self):
        """On restart, recover position from DB and re-place the stop-loss order."""
        # Cancel any orphaned limit entry orders from a previous session
        self.trader.cancel_all_open_orders(BTC_PERP_COIN)
        open_trade = self._db.get_open_bb_trade()
        if not open_trade:
            return
        self.position         = open_trade["direction"]
        self.entry_price      = open_trade["entry_price"]
        self.current_trade_id = open_trade["id"]
        # Verify a matching position still exists on the exchange
        actual = self.trader.get_perp_position()
        if not actual or actual["side"] != self.position:
            logger.warning("[PERPS] DB has open trade but no matching exchange position — closing DB record")
            self._db.close_bb_trade(self.current_trade_id, self.entry_price, stopped=False)
            self._reset_position()
            return
        stop_price = (
            self.entry_price * (1 - self.stop_loss_pct / 100) if self.position == "LONG"
            else self.entry_price * (1 + self.stop_loss_pct / 100)
        )
        self._position_btc_sz = actual["size"]
        self.trader.cancel_all_open_orders(BTC_PERP_COIN)
        sl_result = self.trader.place_stop_loss_perp(self.position, actual["size"], stop_price)
        self.stop_order_id = _extract_oid(sl_result)
        # Exit limit will be placed by the first update_indicators() call in the main loop
        logger.info(
            "[PERPS] Restored open %s trade id=%d @ $%.2f | SL=$%.2f (stop_oid=%s) — exit limit pending next indicator update",
            self.position, self.current_trade_id, self.entry_price, stop_price, self.stop_order_id,
        )

    def _reset_position(self):
        self.position            = None
        self.entry_price         = None
        self.stop_order_id       = None
        self.exit_order_id       = None
        self.current_trade_id    = None
        self._position_btc_sz    = None

    def _reset_pending_entry(self):
        self._pending_entry_oid       = None
        self._pending_entry_side      = None
        self._pending_entry_px        = None
        self._pending_entry_time      = None
        self._pending_entry_size_usd  = None
        self._pending_bb_snapshot     = None

    # ── Market data ────────────────────────────────────────────────────────────

    def _fetch_klines(self) -> pd.DataFrame:
        limit = max(self.ema_period, self.bb_period) * 3
        params = {"symbol": "BTCUSDT", "interval": "5m", "limit": limit}
        for attempt in range(1, 4):
            try:
                resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
                resp.raise_for_status()
                break
            except requests.exceptions.ConnectionError as e:
                if attempt == 3:
                    raise
                wait = attempt * 5
                logger.warning("[PERPS] Network error (attempt %d/3), retrying in %ds: %s", attempt, wait, e)
                time.sleep(wait)
        df = pd.DataFrame(resp.json(), columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["close"] = df["close"].astype(float)
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
            df = self._fetch_klines()
            self._last_close = float(df["close"].iloc[-1])
            self._bb_upper, self._bb_lower, self._bb_mid, self._ema = self._compute_indicators(df)
            logger.debug(
                "[PERPS] Indicators updated: BB upper=%.2f lower=%.2f mid=%.2f EMA=%.2f",
                self._bb_upper, self._bb_lower, self._bb_mid, self._ema,
            )
            # Replace exit limit at updated target band whenever we have an open position
            if not self.dry_run and self.position and self._position_btc_sz:
                self._replace_exit_limit()
        except Exception as exc:
            logger.error("[PERPS] Indicator update failed: %s", exc, exc_info=True)

    def _is_inverted(self) -> bool:
        """Return True when the dynamic TP target has drifted past entry price (guaranteed loss if filled)."""
        if self.position == "SHORT":
            return self._bb_lower is not None and self._bb_lower > self.entry_price
        if self.position == "LONG":
            return self._bb_upper is not None and self._bb_upper < self.entry_price
        return False

    def _replace_exit_limit(self):
        """Cancel the old exit limit (if any) and place a fresh one at the current target band.

        If the TP has inverted (band drifted past entry price), the exit is placed at
        entry price (break-even) instead of the band, capping the loss at fees only.
        """
        if self.exit_order_id:
            self.trader.cancel_order(BTC_PERP_COIN, self.exit_order_id)
        inverted = self._is_inverted()
        if inverted:
            target_px = self.entry_price
        else:
            target_px = self._bb_upper if self.position == "LONG" else self._bb_lower
        result = self.trader.place_exit_limit_perp(self.position, self._position_btc_sz, target_px)
        self.exit_order_id = _extract_oid(result)
        logger.info(
            "[PERPS] Exit limit refreshed — %s @ $%.1f (oid=%s)%s",
            self.position, target_px, self.exit_order_id,
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
            if self.position and not self.dry_run:
                if self.trader.get_perp_position() is None:
                    open_oids = {
                        o["oid"] for o in self.trader.info.open_orders(self.trader.wallet_address)
                        if o.get("coin") == BTC_PERP_COIN
                    }
                    exit_still_open = bool(self.exit_order_id and self.exit_order_id in open_oids)
                    sl_still_open   = bool(self.stop_order_id and self.stop_order_id in open_oids)
                    if exit_still_open:
                        # Exit limit still resting → SL must have fired
                        stopped = True
                        self.trader.cancel_order(BTC_PERP_COIN, self.exit_order_id)
                        logger.info("[PERPS] Position gone [SL] — cancelled exit limit oid=%d", self.exit_order_id)
                    else:
                        # Exit limit gone → target hit (maker fill)
                        stopped = False
                        if sl_still_open:
                            self.trader.cancel_order(BTC_PERP_COIN, self.stop_order_id)
                        logger.info("[PERPS] Position gone [TARGET HIT] — exit limit filled (maker)")
                    if self.current_trade_id:
                        exit_fill = self.trader.get_last_btc_perp_fill()
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
                    "[PERPS] BTC=$%.2f | BB upper=%.2f lower=%.2f mid=%.2f | EMA=%.2f | pos=%s",
                    price, bb_upper, bb_lower, bb_mid, ema, self.position,
                )
            current_hour = datetime.now(timezone.utc).hour
            if current_hour != self._last_hourly_log:
                self._last_hourly_log = current_hour
                logger.info(
                    "[PERPS] BTC=$%.2f | BB upper=%.2f lower=%.2f mid=%.2f | EMA=%.2f | pos=%s",
                    price, bb_upper, bb_lower, bb_mid, ema, self.position,
                )

            if self.position is None and self._pending_entry_oid is None:
                if price <= bb_lower and price > ema:
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
                        f"[DRY RUN] Inverted TP — close LONG @ break-even ${self.entry_price:.2f}",
                        lambda: self._close(self.entry_price),
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
                        f"[DRY RUN] Inverted TP — close SHORT @ break-even ${self.entry_price:.2f}",
                        lambda: self._close(self.entry_price),
                    )
                elif price <= bb_lower:
                    self._dry_or_live(
                        f"[DRY RUN] ${price:.2f} ≤ BB lower ${bb_lower:.2f} — close SHORT (target hit)",
                        lambda: self._close(price),
                    )

        except Exception as exc:
            logger.error("[PERPS] ERROR: %s", exc, exc_info=True)

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
        balance = self.trader.get_perp_usdc_balance()
        return balance * (self.capital_percent / 100) * self.leverage

    def _place_entry_limit(self, side: str, price: float, bb_upper: float, bb_lower: float, bb_mid: float):
        """Phase 1: place an ALO (post-only/maker) limit order at the trigger band level.

        On ALO rejection (order would cross the spread), retry up to twice with the price
        nudged further inside the book (-0.1, -0.2 for LONG; +0.1, +0.2 for SHORT).
        If all ALO attempts fail, fall back to a GTC taker order at the last nudged price.
        """
        size_usd = self._entry_size_usd()
        if size_usd < 1.0:
            logger.warning("[PERPS] Skipping — insufficient margin (%.2f USDC)", size_usd)
            return
        self.trader.update_leverage(self.leverage)
        limit_px = bb_lower if side == "LONG" else bb_upper
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
                logger.info("[PERPS] Limit %s filled immediately @ $%.2f", side, filled_px)
                self._on_entry_filled(side, filled_px, size_usd, bb_upper, bb_lower, bb_mid)
                return
            # ALO rejected: retry twice, nudging one more tick inside the book each time
            for tick in (1, 2):
                if not _is_alo_rejected(result):
                    break
                retry_px = round(price - tick * 0.1, 1) if side == "LONG" else round(price + tick * 0.1, 1)
                logger.info(
                    "[PERPS] ALO %s rejected at $%.1f — retry %d/2 at $%.1f",
                    side, limit_px, tick, retry_px,
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
                    "[PERPS] ALO %s rejected after 2 retries — falling back to GTC taker at $%.1f",
                    side, limit_px,
                )
                result = (
                    self.trader.open_long_limit(size_usd, limit_px, tif="Gtc")
                    if side == "LONG"
                    else self.trader.open_short_limit(size_usd, limit_px, tif="Gtc")
                )
                oid = _extract_oid(result)
                filled, filled_px = _is_filled(result)
                if filled:
                    logger.info("[PERPS] GTC fallback %s filled @ $%.2f (taker)", side, filled_px)
                    self._on_entry_filled(side, filled_px, size_usd, bb_upper, bb_lower, bb_mid)
                    return
            if oid is None:
                logger.error("[PERPS] Limit %s order failed after all retries: %s", side, result)
                return
        self._pending_entry_oid      = oid
        self._pending_entry_side     = side
        self._pending_entry_px       = limit_px
        self._pending_entry_time     = time.monotonic()
        self._pending_entry_size_usd = size_usd
        self._pending_bb_snapshot    = {"bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid}
        logger.info(
            "[PERPS] LIMIT %s order placed oid=%d @ $%.1f size=$%.2f (cancels if no fill in 60s)",
            side, oid, limit_px, size_usd,
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
        entry_fill = self.trader.get_last_btc_perp_fill()
        self._entry_fee = entry_fill["fee"]
        self.current_trade_id = self._db.open_bb_trade(
            side, self.leverage, fill_px, size_usd,
            bb_upper, bb_lower, bb_mid, stop_price,
            account_name=self.account_name,
        )
        btc_sz = round(size_usd / fill_px, 5)
        self._position_btc_sz = btc_sz
        sl_result = self.trader.place_stop_loss_perp(side, btc_sz, stop_price)
        self.stop_order_id = _extract_oid(sl_result)
        exit_result = self.trader.place_exit_limit_perp(side, btc_sz, target_price)
        self.exit_order_id = _extract_oid(exit_result)
        self._reset_pending_entry()
        logger.info(
            "[PERPS] OPEN %s $%.2f notional @ $%.2f | target=$%.2f (exit_oid=%s) | SL=$%.2f (stop_oid=%s) (trade_id=%d)",
            side, size_usd, fill_px, target_price, self.exit_order_id, stop_price, self.stop_order_id, self.current_trade_id,
        )

    def _check_pending_entry(self, price: float):
        """Poll for fill or cancel the pending limit entry order."""
        if self._pending_entry_oid is None:
            return
        side     = self._pending_entry_side
        limit_px = self._pending_entry_px
        snap     = self._pending_bb_snapshot
        elapsed  = time.monotonic() - self._pending_entry_time

        timed_out        = elapsed > 60.0
        price_moved_away = (
            (side == "LONG"  and price > limit_px * 1.001) or
            (side == "SHORT" and price < limit_px * 0.999)
        )
        if timed_out or price_moved_away:
            reason = "timeout" if timed_out else f"price moved away (${price:.2f})"
            logger.info("[PERPS] Cancelling pending %s limit oid=%d — %s", side, self._pending_entry_oid, reason)
            self.trader.cancel_order(BTC_PERP_COIN, self._pending_entry_oid)
            self._reset_pending_entry()
            return

        # Fill detection: oid absent from open orders means filled (or externally cancelled)
        open_oids = {
            o["oid"] for o in self.trader.info.open_orders(self.trader.wallet_address)
            if o.get("coin") == BTC_PERP_COIN
        }
        if self._pending_entry_oid not in open_oids:
            actual_pos = self.trader.get_perp_position()
            if actual_pos and actual_pos["side"] == side:
                logger.info("[PERPS] Limit entry oid=%d filled — position confirmed", self._pending_entry_oid)
                self._on_entry_filled(
                    side, limit_px, self._pending_entry_size_usd,
                    snap["bb_upper"], snap["bb_lower"], snap["bb_mid"],
                )
            else:
                logger.warning(
                    "[PERPS] Limit entry oid=%d gone but no position — treating as cancelled",
                    self._pending_entry_oid,
                )
                self._reset_pending_entry()

    def _close(self, price: float, stopped: bool = False):
        # Cancel both resting orders before closing to avoid double-close
        if self.exit_order_id:
            self.trader.cancel_order(BTC_PERP_COIN, self.exit_order_id)
            self.exit_order_id = None
        if self.stop_order_id:
            self.trader.cancel_order(BTC_PERP_COIN, self.stop_order_id)
            self.stop_order_id = None
        result = self.trader.close_perp_position()
        if result.get("status") == "ok":
            if self.current_trade_id:
                exit_fill = self.trader.get_last_btc_perp_fill()
                self._db.close_bb_trade(
                    self.current_trade_id, exit_fill["px"], stopped,
                    exit_bb_upper=self._bb_upper, exit_bb_lower=self._bb_lower, exit_bb_mid=self._bb_mid,
                    entry_fee=self._entry_fee, exit_fee=exit_fill["fee"],
                )
            logger.info(
                "[PERPS] CLOSE %s @ $%.2f%s (trade_id=%s)",
                self.position, price,
                " [STOP LOSS]" if stopped else " [TARGET HIT]",
                self.current_trade_id,
            )
            self._reset_position()
        else:
            logger.error("[PERPS] Close failed: %s", result)
