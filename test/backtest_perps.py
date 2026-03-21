"""
Backtest script — BB mean-reversion strategy comparison.

Edit the config block at the top, then run:
    python test/backtest.py
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL         = "BTCUSDT"   # "BTCUSDT" | "ETHUSDT" | "SOLUSDT"
INTERVAL       = "5m"
LEVERAGE       = 5
TRADE_SIZE_USD = 95.0
EMA_PERIOD     = 200
DAYS_LIST      = [30, 60, 90, 180, 365, 730]

# Configs to compare — add or remove freely
CONFIGS = [
    {"label": "BTC CURRENT STRATEGY BB(20,2.0)  SL=2.5%",  "bp": 20, "bs": 2.0, "sl": 2.5}, # CURRENT BTC STRATEGY ON 5M, NEVER REMOVE!!!
    {"label": "BTC BB(20,2.0)  SL=3.0%",  "bp": 20, "bs": 2.0, "sl": 3.0},
    {"label": "BTC BB(20,2.0)  SL=3.5%",  "bp": 20, "bs": 2.0, "sl": 3.5},
    {"label": "BTC BB(20,2.0)  SL=4.0%",  "bp": 20, "bs": 2.0, "sl": 4.0},
    {"label": "BTC BB(20,2.0)  SL=4.5%",  "bp": 20, "bs": 2.0, "sl": 4.5},
    {"label": "BTC BB(20,2.0)  SL=5.0%",  "bp": 20, "bs": 2.0, "sl": 5.0},

]

# ── Fee model (Hyperliquid) ────────────────────────────────────────────────────
MAKER_FEE = 0.0001   # 0.010% — limit entry + limit exit (target hit)
TAKER_FEE = 0.0005   # 0.050% — stop-market exit (stop-loss)

# ── Internals ──────────────────────────────────────────────────────────────────
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MS_PER_CANDLE      = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    direction: str
    entry_timestamp: datetime = None
    entry_price: float = 0.0
    size_usd: float = 0.0
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    stopped_out: bool = False

    @property
    def gross_pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) / self.entry_price * self.size_usd
        return (self.entry_price - self.exit_price) / self.entry_price * self.size_usd

    @property
    def fees(self) -> float:
        exit_fee = TAKER_FEE if self.stopped_out else MAKER_FEE
        return self.size_usd * (MAKER_FEE + exit_fee)

    @property
    def realized_pnl(self) -> Optional[float]:
        if self.gross_pnl is None:
            return None
        return self.gross_pnl - self.fees

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.realized_pnl is None or self.size_usd == 0:
            return None
        return self.realized_pnl / (self.size_usd / LEVERAGE) * 100


# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch_klines(days: int) -> pd.DataFrame:
    ms_per_candle = MS_PER_CANDLE[INTERVAL]
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_rows = []
    cursor   = start_ms
    print(f"Fetching {SYMBOL} {INTERVAL} klines for last {days} days…")

    while cursor < end_ms:
        resp = requests.get(BINANCE_KLINES_URL, params={
            "symbol": SYMBOL, "interval": INTERVAL,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + ms_per_candle

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["close"]    = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    print(f"  {len(df)} candles  ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})\n")
    return df


# ── Simulation ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, stop_loss_pct: float, bb_period: int, bb_std: float) -> List[Trade]:
    ema_series = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    bb_mid_s   = df["close"].rolling(bb_period).mean()
    bb_upper_s = bb_mid_s + bb_std * df["close"].rolling(bb_period).std()
    bb_lower_s = bb_mid_s - bb_std * df["close"].rolling(bb_period).std()
    warmup     = max(EMA_PERIOD, bb_period) * 3

    position, entry_candle_i, current = None, None, None
    trades = []

    def entry_size() -> float:
        return TRADE_SIZE_USD * LEVERAGE

    def close_trade(ts, price, stopped=False):
        nonlocal position, current, entry_candle_i
        current.exit_timestamp = ts
        current.exit_price     = price
        current.stopped_out    = stopped
        trades.append(current)
        position, current, entry_candle_i = None, None, None

    def is_stop_hit(price: float) -> bool:
        if current is None:
            return False
        if position == "LONG":
            return price <= current.entry_price * (1 - stop_loss_pct / 100)
        if position == "SHORT":
            return price >= current.entry_price * (1 + stop_loss_pct / 100)
        return False

    for i in range(warmup, len(df)):
        price    = df["close"].iloc[i]
        ts       = df["open_time"].iloc[i]
        ema      = float(ema_series.iloc[i])
        bb_upper = float(bb_upper_s.iloc[i])
        bb_lower = float(bb_lower_s.iloc[i])

        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(ema):
            continue

        half_band = (bb_upper - bb_lower) / 2

        if position is None:
            if price <= bb_lower and price > ema:
                current = Trade(direction="LONG", entry_timestamp=ts,
                                entry_price=price, size_usd=entry_size())
                position, entry_candle_i = "LONG", i
            elif price >= bb_upper and price < ema:
                current = Trade(direction="SHORT", entry_timestamp=ts,
                                entry_price=price, size_usd=entry_size())
                position, entry_candle_i = "SHORT", i

        elif position == "LONG":
            if is_stop_hit(price):
                close_trade(ts, price, stopped=True)
            elif bb_upper < current.entry_price and price <= current.entry_price:
                # Inverted TP: upper BB drifted below entry → exit at break-even
                close_trade(ts, current.entry_price)
            elif price >= bb_lower + 2 * half_band:
                close_trade(ts, price)

        elif position == "SHORT":
            if is_stop_hit(price):
                close_trade(ts, price, stopped=True)
            elif bb_lower > current.entry_price and price >= current.entry_price:
                # Inverted TP: lower BB drifted above entry → exit at break-even
                close_trade(ts, current.entry_price)
            elif price <= bb_upper - 2 * half_band:
                close_trade(ts, price)

    if current is not None:
        current.exit_timestamp = df["open_time"].iloc[-1]
        current.exit_price     = df["close"].iloc[-1]
        trades.append(current)
        print("  ⚠  Open position at end of data — closed at last price (mark-to-market)\n")

    return trades


# ── Comparison ─────────────────────────────────────────────────────────────────
def run_comparison():
    klines_cache = {}

    for days in DAYS_LIST:
        if days not in klines_cache:
            klines_cache[days] = fetch_klines(days)
        df = klines_cache[days]

        rows = []
        for cfg in CONFIGS:
            trades      = run_backtest(df, stop_loss_pct=cfg["sl"],
                                       bb_period=cfg["bp"], bb_std=cfg["bs"])
            n           = len([t for t in trades if t.realized_pnl is not None])
            wins        = sum(1 for t in trades if (t.realized_pnl or 0) > 0)
            stops       = sum(1 for t in trades if t.stopped_out)
            total_gross = sum(t.gross_pnl or 0 for t in trades)
            total_fees  = sum(t.fees for t in trades)
            total_pnl   = sum(t.realized_pnl or 0 for t in trades)
            win_rate    = (wins / n * 100) if n else 0
            avg_pnl     = total_pnl / n if n else 0
            rows.append({
                "label": cfg["label"], "trades": n, "win_rate": win_rate,
                "sl_hits": stops, "total_gross": total_gross,
                "total_fees": total_fees, "total_pnl": total_pnl, "avg_pnl": avg_pnl,
            })

        sep = "─" * 105
        print("\n" + sep)
        print(
            f"  {days} days  │  PERPS {LEVERAGE}x  │  {SYMBOL} {INTERVAL}  │  "
            f"${TRADE_SIZE_USD:,.0f}/trade  │  fees: maker {MAKER_FEE*100:.3f}% / taker {TAKER_FEE*100:.3f}%"
        )
        print(sep)
        print(f"  {'Config':<36}  {'Trades':>6}  {'Win%':>5}  {'SLs':>4}  {'Gross $':>10}  {'Fees $':>8}  {'Net PnL $':>10}  {'Avg/trade':>10}")
        print(sep)
        for r in rows:
            nsign = "+" if r["total_pnl"]   >= 0 else ""
            gsign = "+" if r["total_gross"] >= 0 else ""
            asign = "+" if r["avg_pnl"]     >= 0 else ""
            print(
                f"  {r['label']:<36}  {r['trades']:>6}  {r['win_rate']:>4.0f}%  {r['sl_hits']:>4}  "
                f"  {gsign}${r['total_gross']:>7,.2f}  -${r['total_fees']:>5,.2f}  "
                f"  {nsign}${r['total_pnl']:>7,.2f}  {asign}${r['avg_pnl']:>7,.2f}"
            )
        print(sep + "\n")


# ── Tee stdout to file ─────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker   = SYMBOL.replace("USDT", "").lower()
    out_path = os.path.join(os.path.dirname(__file__), f"backtest_perps_results_{ticker}.txt")
    out_file = open(out_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out_file)
    try:
        run_comparison()
    finally:
        sys.stdout = sys.__stdout__
        out_file.close()
        print(f"Results saved to {out_path}")
