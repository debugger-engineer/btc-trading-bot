"""
Backtest script — simulates the DCA perps strategy against historical Binance data.

Usage:
    python src/backtest.py                  # last 30 days, default config
    python src/backtest.py --days 60
    python src/backtest.py --days 7 --leverage 2

No exchange credentials or DB required.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

# ── Strategy config (mirrors main.py defaults) ────────────────────────────────
INITIAL_BALANCE    = 10_000.0   # USDC
LEVERAGE           = 3
CAPITAL_PERCENT    = 50         # % of balance allocated to perps
ENTRY_ALLOC        = [75, 15, 10]  # % of capital per DCA entry
RSI_PERIOD         = 14
LONG_ENTRIES       = [30, 25, 20]
SHORT_ENTRIES      = [70, 75, 80]
CLOSE_LONG_AT      = 70
CLOSE_SHORT_AT     = 30
INTERVAL           = "15m"
SYMBOL             = "BTCUSDT"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class Entry:
    level: int
    timestamp: datetime
    price: float
    size_usd: float
    rsi: float


@dataclass
class Trade:
    direction: str          # LONG or SHORT
    entries: List[Entry] = field(default_factory=list)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_rsi: Optional[float] = None
    stopped_out: bool = False   # True if closed by stop-loss

    @property
    def avg_entry_price(self) -> float:
        total_w = sum(e.price * e.size_usd for e in self.entries)
        total_s = sum(e.size_usd for e in self.entries)
        return total_w / total_s if total_s else 0

    @property
    def total_size_usd(self) -> float:
        return sum(e.size_usd for e in self.entries)

    @property
    def realized_pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.direction == "LONG":
            return (self.exit_price - self.avg_entry_price) / self.avg_entry_price * self.total_size_usd
        else:
            return (self.avg_entry_price - self.exit_price) / self.avg_entry_price * self.total_size_usd

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.realized_pnl is None or self.total_size_usd == 0:
            return None
        return self.realized_pnl / (self.total_size_usd / LEVERAGE) * 100


# ── Binance data fetching ──────────────────────────────────────────────────────
def fetch_klines(days: int) -> pd.DataFrame:
    """Fetch historical 15m klines for the last N days, paginating as needed."""
    ms_per_candle = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}[INTERVAL]
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    all_rows = []
    cursor = start_ms
    print(f"Fetching {SYMBOL} {INTERVAL} klines for last {days} days…")

    while cursor < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + ms_per_candle  # advance past last candle

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    print(f"  {len(df)} candles fetched  ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})\n")
    return df


def compute_rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()


# ── Simulation ─────────────────────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    leverage: int,
    initial_balance: float = INITIAL_BALANCE,
    stop_loss_pct: Optional[float] = None,
    trend_filter: bool = False,
    ema_period: int = 200,
) -> List[Trade]:
    rsi_series = compute_rsi(df["close"], RSI_PERIOD)
    ema_series = compute_ema(df["close"], ema_period) if trend_filter else None
    warmup = max(RSI_PERIOD * 3, ema_period)  # wait for both indicators to stabilize

    balance   = initial_balance
    position  = None      # "LONG", "SHORT", or None
    entry_lvl = 0
    current   = None      # Trade in progress
    trades    = []

    def entry_size(level: int) -> float:
        budget = balance * (CAPITAL_PERCENT / 100)
        return budget * (ENTRY_ALLOC[level - 1] / 100) * leverage

    def close_trade(ts, price, rsi, stopped=False):
        nonlocal position, entry_lvl, current, balance
        current.exit_timestamp = ts
        current.exit_price     = price
        current.exit_rsi       = rsi
        current.stopped_out    = stopped
        balance += current.realized_pnl
        trades.append(current)
        position, entry_lvl, current = None, 0, None

    def is_stop_hit(price: float) -> bool:
        if stop_loss_pct is None or current is None:
            return False
        avg = current.avg_entry_price
        if position == "LONG":
            return price <= avg * (1 - stop_loss_pct / 100)
        if position == "SHORT":
            return price >= avg * (1 + stop_loss_pct / 100)
        return False

    for i in range(warmup, len(df)):
        rsi   = rsi_series.iloc[i]
        price = df["close"].iloc[i]
        ts    = df["open_time"].iloc[i]

        if pd.isna(rsi):
            continue

        ema = float(ema_series.iloc[i]) if ema_series is not None else None

        if position is None:
            in_uptrend   = (ema is None or price > ema)
            in_downtrend = (ema is None or price < ema)

            if rsi <= LONG_ENTRIES[0] and in_uptrend:
                current = Trade(direction="LONG")
                current.entries.append(Entry(1, ts, price, entry_size(1), rsi))
                position, entry_lvl = "LONG", 1

            elif rsi >= SHORT_ENTRIES[0] and in_downtrend:
                current = Trade(direction="SHORT")
                current.entries.append(Entry(1, ts, price, entry_size(1), rsi))
                position, entry_lvl = "SHORT", 1

        elif position == "LONG":
            if is_stop_hit(price):
                close_trade(ts, price, rsi, stopped=True)
            elif rsi >= CLOSE_LONG_AT:
                close_trade(ts, price, rsi)
            elif entry_lvl < len(LONG_ENTRIES) and rsi <= LONG_ENTRIES[entry_lvl]:
                entry_lvl += 1
                current.entries.append(Entry(entry_lvl, ts, price, entry_size(entry_lvl), rsi))

        elif position == "SHORT":
            if is_stop_hit(price):
                close_trade(ts, price, rsi, stopped=True)
            elif rsi <= CLOSE_SHORT_AT:
                close_trade(ts, price, rsi)
            elif entry_lvl < len(SHORT_ENTRIES) and rsi >= SHORT_ENTRIES[entry_lvl]:
                entry_lvl += 1
                current.entries.append(Entry(entry_lvl, ts, price, entry_size(entry_lvl), rsi))

    # Close any open trade at last price (mark-to-market)
    if current is not None:
        last_price = df["close"].iloc[-1]
        last_ts    = df["open_time"].iloc[-1]
        current.exit_timestamp = last_ts
        current.exit_price     = last_price
        current.exit_rsi       = rsi_series.iloc[-1]
        trades.append(current)
        print("  ⚠  Open position at end of data — closed at last price (mark-to-market)\n")

    return trades


# ── Reporting ──────────────────────────────────────────────────────────────────
def print_report(
    trades: List[Trade],
    days: int,
    leverage: int,
    initial_balance: float = INITIAL_BALANCE,
    stop_loss_pct: Optional[float] = None,
    trend_filter: bool = False,
    ema_period: int = 200,
):
    sl_label     = f"  │  SL {stop_loss_pct}%" if stop_loss_pct else ""
    trend_label  = f"  │  trend filter EMA({ema_period})" if trend_filter else ""
    sep = "─" * 120
    print(sep)
    print(f"  BACKTEST RESULTS  │  {SYMBOL} {INTERVAL}  │  last {days} days  │  "
          f"{leverage}x leverage  │  capital {CAPITAL_PERCENT}%  │  alloc {ENTRY_ALLOC}{sl_label}{trend_label}")
    print(sep)

    if not trades:
        print("  No trades triggered in this period.")
        print(sep)
        return

    # Header
    print(f"  {'#':>3}  {'Dir':5}  {'#Ent':4}  {'AvgEntry':>10}  {'Exit':>10}  "
          f"{'Notional':>10}  {'PnL $':>9}  {'PnL%':>7}  "
          f"{'Open':>19}  {'Close':>19}")
    print(sep)

    total_pnl = 0.0
    wins = 0

    for i, t in enumerate(trades, 1):
        pnl   = t.realized_pnl or 0.0
        pct   = t.pnl_pct or 0.0
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        sign     = "+" if pnl >= 0 else ""
        sl_tag   = " SL" if t.stopped_out else "   "
        open_dt  = t.entries[0].timestamp.strftime("%Y-%m-%d %H:%M")
        close_dt = t.exit_timestamp.strftime("%Y-%m-%d %H:%M") if t.exit_timestamp else "open"
        entry_details = "  ".join(
            f"L{e.level}@{e.price:,.0f}(RSI{e.rsi:.0f})" for e in t.entries
        )
        print(
            f"  {i:>3}  {t.direction:5}  {len(t.entries):>4}  "
            f"${t.avg_entry_price:>9,.2f}  ${t.exit_price or 0:>9,.2f}  "
            f"${t.total_size_usd:>9,.0f}  "
            f"{sign}${pnl:>7.2f}  {sign}{pct:>5.1f}%{sl_tag}  "
            f"{open_dt}  {close_dt}"
        )
        print(f"       Entries: {entry_details}")

    n_closed = len([t for t in trades if t.realized_pnl is not None])

    print(sep)
    win_rate = (wins / n_closed * 100) if n_closed else 0
    final_balance = initial_balance + total_pnl
    total_return  = (total_pnl / initial_balance) * 100

    stops = sum(1 for t in trades if t.stopped_out)
    print(f"  Trades closed : {n_closed}   Wins: {wins}   Win rate: {win_rate:.0f}%   Stop-losses: {stops}")
    print(f"  Starting bal  : ${initial_balance:,.2f}")
    print(f"  Ending bal    : ${final_balance:,.2f}")
    sign = "+" if total_pnl >= 0 else ""
    print(f"  Total PnL     : {sign}${total_pnl:,.2f}  ({sign}{total_return:.2f}%)")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backtest the DCA perps strategy")
    parser.add_argument("--days",         type=int,   default=30,       help="Days of history (default 30)")
    parser.add_argument("--leverage",     type=int,   default=LEVERAGE, help=f"Leverage (default {LEVERAGE})")
    parser.add_argument("--capital",      type=float, default=INITIAL_BALANCE, help="Starting balance USDC")
    parser.add_argument("--stop-loss",    type=float, default=None,
                        help="Close if price moves X%% against avg entry (e.g. 8)")
    parser.add_argument("--trend-filter", action="store_true",
                        help="Only open longs above EMA(200), shorts below EMA(200)")
    parser.add_argument("--ema-period",   type=int,   default=200,
                        help="EMA period for trend filter (default 200)")
    args = parser.parse_args()

    initial_balance = args.capital
    stop_loss_pct   = args.stop_loss
    trend_filter    = args.trend_filter
    ema_period      = args.ema_period

    df     = fetch_klines(args.days)
    trades = run_backtest(df, args.leverage, initial_balance, stop_loss_pct, trend_filter, ema_period)
    print_report(trades, args.days, args.leverage, initial_balance, stop_loss_pct, trend_filter, ema_period)


if __name__ == "__main__":
    main()
