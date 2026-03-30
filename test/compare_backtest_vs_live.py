"""
Compare backtest predictions vs actual live trades from the database.

Usage:
    python test/compare_backtest_vs_live.py
    python test/compare_backtest_vs_live.py --account bot-5x --days 30
    python test/compare_backtest_vs_live.py --tolerance-minutes 10
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Strategy constants (must match live config & backtest) ────────────────────
SYMBOL         = "BTCUSDT"
INTERVAL       = "5m"
LEVERAGE       = 5
TRADE_SIZE_USD = 95.0        # backtest uses fixed size
EMA_PERIOD     = 200
BB_PERIOD      = 20
BB_STD         = 2.0
STOP_LOSS_PCT  = 2.5

MAKER_FEE = 0.0001   # 0.010%
TAKER_FEE = 0.0005   # 0.050%

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MS_PER_CANDLE      = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


# ── Trade dataclass (copied from backtest_perps.py) ──────────────────────────
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


# ── Database ──────────────────────────────────────────────────────────────────
def fetch_live_trades(account_name: str, start_date: datetime) -> list:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, opened_at, closed_at, direction, leverage,
                       entry_price, entry_size_usd,
                       bb_upper, bb_lower, bb_mid, stop_price,
                       exit_price, exit_bb_upper, exit_bb_lower, exit_bb_mid,
                       gross_pnl, fees, net_pnl, closed_by_sl, account_name
                FROM perps_trades
                WHERE status = 'CLOSED'
                  AND symbol = 'BTC'
                  AND account_name = %s
                  AND opened_at >= %s
                ORDER BY opened_at ASC
            """, (account_name, start_date))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    trades = []
    for row in rows:
        t = dict(zip(cols, row))
        # Ensure datetimes are timezone-aware
        for k in ("opened_at", "closed_at"):
            if t[k] and t[k].tzinfo is None:
                t[k] = t[k].replace(tzinfo=timezone.utc)
        # Cast numerics to float
        for k in ("entry_price", "entry_size_usd", "bb_upper", "bb_lower", "bb_mid",
                   "stop_price", "exit_price", "exit_bb_upper", "exit_bb_lower",
                   "exit_bb_mid", "gross_pnl", "fees", "net_pnl"):
            if t[k] is not None:
                t[k] = float(t[k])
        trades.append(t)
    return trades


# ── Kline fetching (copied from backtest_perps.py) ───────────────────────────
def fetch_klines(days: int) -> pd.DataFrame:
    ms_per_candle = MS_PER_CANDLE[INTERVAL]
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_rows = []
    cursor   = start_ms
    print(f"Fetching {SYMBOL} {INTERVAL} klines for last {days} days...")

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
    df["close"]     = df["close"].astype(float)
    df["high"]      = df["high"].astype(float)
    df["low"]       = df["low"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    print(f"  {len(df)} candles  ({df['open_time'].iloc[0].date()} -> {df['open_time'].iloc[-1].date()})\n")
    return df


# ── Backtest engine (copied from backtest_perps.py) ──────────────────────────
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
                close_trade(ts, current.entry_price)
            elif price >= bb_lower + 2 * half_band:
                close_trade(ts, price)

        elif position == "SHORT":
            if is_stop_hit(price):
                close_trade(ts, price, stopped=True)
            elif bb_lower > current.entry_price and price >= current.entry_price:
                close_trade(ts, current.entry_price)
            elif price <= bb_upper - 2 * half_band:
                close_trade(ts, price)

    if current is not None:
        current.exit_timestamp = df["open_time"].iloc[-1]
        current.exit_price     = df["close"].iloc[-1]
        trades.append(current)

    return trades


# ── Trade matching ────────────────────────────────────────────────────────────
def match_trades(live_trades: list, bt_trades: List[Trade],
                 tolerance_min: int) -> tuple:
    tolerance = timedelta(minutes=tolerance_min)
    bt_used = set()
    matched = []
    live_only = []

    for lt in live_trades:
        best_idx = None
        best_delta = None
        for j, bt in enumerate(bt_trades):
            if j in bt_used:
                continue
            if bt.direction != lt["direction"]:
                continue
            delta = abs((lt["opened_at"] - bt.entry_timestamp).total_seconds())
            if delta <= tolerance.total_seconds():
                if best_delta is None or delta < best_delta:
                    best_idx = j
                    best_delta = delta
        if best_idx is not None:
            bt_used.add(best_idx)
            matched.append((lt, bt_trades[best_idx]))
        else:
            live_only.append(lt)

    bt_only = [bt for j, bt in enumerate(bt_trades) if j not in bt_used]
    return matched, live_only, bt_only


# ── Heuristic cause for missed/extra trades ──────────────────────────────────
def infer_live_only_cause(lt: dict, bt_trades: List[Trade]) -> str:
    """Why did the bot trade but the backtest didn't?"""
    # Check if a BT trade exists nearby but in opposite direction
    for bt in bt_trades:
        delta = abs((lt["opened_at"] - bt.entry_timestamp).total_seconds())
        if delta < 600 and bt.direction != lt["direction"]:
            return "Direction mismatch (EMA filter edge)"
    return "Intra-candle wick (price dipped below/above band mid-candle)"


def infer_bt_only_cause(bt: Trade, live_trades: list) -> str:
    """Why did the backtest signal but the bot didn't trade?"""
    for lt in live_trades:
        opened = lt["opened_at"]
        closed = lt["closed_at"]
        if closed and opened <= bt.entry_timestamp <= closed:
            return "Bot was in existing position"
    return "Likely ALO rejection / fill timeout"


# ── Report formatting ────────────────────────────────────────────────────────
def fmt_ts(dt: datetime) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%m-%d %H:%M")


def fmt_price(p) -> str:
    if p is None:
        return "N/A"
    return f"${p:,.2f}"


def fmt_pnl(p) -> str:
    if p is None:
        return "N/A"
    sign = "+" if p >= 0 else ""
    return f"{sign}${p:,.2f}"


def fmt_pct(p) -> str:
    if p is None:
        return "N/A"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"


def print_report(matched, live_only, bt_only, live_trades, bt_trades, days):
    sep = "=" * 120

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  BACKTEST vs LIVE COMPARISON  |  Last {days} days  |  Account trades: {len(live_trades)}  |  Backtest trades: {len(bt_trades)}")
    print(sep)

    n_matched = len(matched)
    n_live = len(live_trades)
    match_rate = (n_matched / n_live * 100) if n_live else 0

    print(f"\n  Matched trades:      {n_matched} / {n_live} live  ({match_rate:.1f}%)")
    print(f"  Live-only trades:    {len(live_only)}  (bot traded, backtest did not)")
    print(f"  Backtest-only:       {len(bt_only)}  (backtest signaled, bot did not trade)")

    # ── Aggregate PnL ─────────────────────────────────────────────────────────
    live_total_pnl = sum(lt["net_pnl"] or 0 for lt in live_trades)
    live_wins = sum(1 for lt in live_trades if (lt["net_pnl"] or 0) > 0)
    live_sls = sum(1 for lt in live_trades if lt["closed_by_sl"])
    live_wr = (live_wins / n_live * 100) if n_live else 0

    bt_closed = [bt for bt in bt_trades if bt.realized_pnl is not None]
    bt_total_pnl = sum(bt.realized_pnl for bt in bt_closed)
    bt_wins = sum(1 for bt in bt_closed if bt.realized_pnl > 0)
    bt_sls = sum(1 for bt in bt_closed if bt.stopped_out)
    bt_wr = (bt_wins / len(bt_closed) * 100) if bt_closed else 0

    print(f"\n  {'Metric':<30} {'Live':>15} {'Backtest':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Total trades':<30} {n_live:>15} {len(bt_closed):>15}")
    print(f"  {'Win rate':<30} {f'{live_wr:.1f}%':>15} {f'{bt_wr:.1f}%':>15}")
    print(f"  {'Stop-losses':<30} {live_sls:>15} {bt_sls:>15}")
    print(f"  {'Total net PnL':<30} {fmt_pnl(live_total_pnl):>15} {fmt_pnl(bt_total_pnl):>15}")

    # ── Matched PnL comparison (% based for fairness) ─────────────────────────
    if matched:
        print(f"\n  Among {n_matched} matched trades:")
        entry_deltas = []
        exit_price_deltas = []
        same_exit_type = 0
        pnl_sign_match = 0
        for lt, bt in matched:
            delta_sec = (lt["opened_at"] - bt.entry_timestamp).total_seconds()
            entry_deltas.append(delta_sec)
            if lt["exit_price"] and bt.exit_price:
                exit_price_deltas.append(
                    (lt["exit_price"] - bt.exit_price) / bt.exit_price * 100
                )
            if lt["closed_by_sl"] == bt.stopped_out:
                same_exit_type += 1
            lt_win = (lt["net_pnl"] or 0) > 0
            bt_win = (bt.realized_pnl or 0) > 0
            if lt_win == bt_win:
                pnl_sign_match += 1

        avg_entry_delta = sum(entry_deltas) / len(entry_deltas) if entry_deltas else 0
        avg_exit_price_d = sum(exit_price_deltas) / len(exit_price_deltas) if exit_price_deltas else 0

        print(f"    Avg entry time delta:     {avg_entry_delta:+.1f} sec ({avg_entry_delta/60:+.1f} min)")
        print(f"    Avg exit price delta:     {avg_exit_price_d:+.4f}%")
        print(f"    Same exit type (SL/TP):   {same_exit_type} / {n_matched} ({same_exit_type/n_matched*100:.0f}%)")
        print(f"    Same PnL sign (W/L):      {pnl_sign_match} / {n_matched} ({pnl_sign_match/n_matched*100:.0f}%)")

    # ── Matched trades table ──────────────────────────────────────────────────
    if matched:
        print(f"\n{'─'*120}")
        print(f"  MATCHED TRADES (side-by-side)")
        print(f"{'─'*120}")
        header = (
            f"  {'#':>3}  {'Dir':<5}  {'BT Entry Time':<14}  {'Live Entry':<14}  "
            f"{'dTime':>6}  {'BT Entry$':>12}  {'Live Entry$':>12}  "
            f"{'BT Exit$':>12}  {'Live Exit$':>12}  "
            f"{'BT PnL%':>8}  {'Live PnL%':>9}  {'Exit':>6}"
        )
        print(header)
        print(f"  {'─'*116}")

        for idx, (lt, bt) in enumerate(matched, 1):
            delta_sec = (lt["opened_at"] - bt.entry_timestamp).total_seconds()
            delta_str = f"{delta_sec/60:+.0f}m"

            bt_pnl_pct = bt.pnl_pct
            # Live PnL % of margin
            live_margin = (lt["entry_size_usd"] or 1) / (lt["leverage"] or LEVERAGE)
            live_pnl_pct = ((lt["net_pnl"] or 0) / live_margin * 100) if live_margin else 0

            exit_type = ""
            if lt["closed_by_sl"] and bt.stopped_out:
                exit_type = "SL/SL"
            elif not lt["closed_by_sl"] and not bt.stopped_out:
                exit_type = "TP/TP"
            else:
                exit_type = f"{'SL' if lt['closed_by_sl'] else 'TP'}/{'SL' if bt.stopped_out else 'TP'}"

            flag = ""
            if lt["closed_by_sl"] != bt.stopped_out:
                flag = " !!"
            elif bt_pnl_pct is not None and (
                (live_pnl_pct > 0) != (bt_pnl_pct > 0)
            ):
                flag = " !"

            print(
                f"  {idx:>3}  {bt.direction:<5}  {fmt_ts(bt.entry_timestamp):<14}  "
                f"{fmt_ts(lt['opened_at']):<14}  {delta_str:>6}  "
                f"{fmt_price(bt.entry_price):>12}  {fmt_price(lt['entry_price']):>12}  "
                f"{fmt_price(bt.exit_price):>12}  {fmt_price(lt['exit_price']):>12}  "
                f"{fmt_pct(bt_pnl_pct):>8}  {fmt_pct(live_pnl_pct):>9}  "
                f"{exit_type:>6}{flag}"
            )

    # ── Live-only trades ──────────────────────────────────────────────────────
    if live_only:
        print(f"\n{'─'*120}")
        print(f"  LIVE-ONLY TRADES (bot traded, backtest did NOT predict)")
        print(f"{'─'*120}")
        print(
            f"  {'#':>3}  {'Dir':<5}  {'Opened':<14}  {'Entry$':>12}  "
            f"{'Exit$':>12}  {'Net PnL':>10}  {'SL?':>4}  {'Possible Cause'}"
        )
        print(f"  {'─'*116}")
        for idx, lt in enumerate(live_only, 1):
            cause = infer_live_only_cause(lt, bt_trades)
            print(
                f"  {idx:>3}  {lt['direction']:<5}  {fmt_ts(lt['opened_at']):<14}  "
                f"{fmt_price(lt['entry_price']):>12}  {fmt_price(lt['exit_price']):>12}  "
                f"{fmt_pnl(lt['net_pnl']):>10}  {'Yes' if lt['closed_by_sl'] else 'No':>4}  "
                f"{cause}"
            )

    # ── Backtest-only trades ──────────────────────────────────────────────────
    if bt_only:
        print(f"\n{'─'*120}")
        print(f"  BACKTEST-ONLY TRADES (backtest signaled, bot did NOT trade)")
        print(f"{'─'*120}")
        print(
            f"  {'#':>3}  {'Dir':<5}  {'BT Entry Time':<14}  {'BT Entry$':>12}  "
            f"{'BT Exit$':>12}  {'BT PnL':>10}  {'SL?':>4}  {'Possible Cause'}"
        )
        print(f"  {'─'*116}")
        for idx, bt in enumerate(bt_only, 1):
            cause = infer_bt_only_cause(bt, live_trades)
            print(
                f"  {idx:>3}  {bt.direction:<5}  {fmt_ts(bt.entry_timestamp):<14}  "
                f"{fmt_price(bt.entry_price):>12}  {fmt_price(bt.exit_price):>12}  "
                f"{fmt_pnl(bt.realized_pnl):>10}  {'Yes' if bt.stopped_out else 'No':>4}  "
                f"{cause}"
            )

    # ── Divergence summary ────────────────────────────────────────────────────
    print(f"\n{'─'*120}")
    print(f"  DIVERGENCE SUMMARY")
    print(f"{'─'*120}")

    exit_type_mismatch = sum(1 for lt, bt in matched if lt["closed_by_sl"] != bt.stopped_out)
    pnl_sign_mismatch = sum(
        1 for lt, bt in matched
        if ((lt["net_pnl"] or 0) > 0) != ((bt.realized_pnl or 0) > 0)
    )
    in_position = sum(1 for bt in bt_only if infer_bt_only_cause(bt, live_trades) == "Bot was in existing position")
    alo_timeout = len(bt_only) - in_position
    wick_entries = len(live_only)

    print(f"  Exit type mismatches (SL vs TP):    {exit_type_mismatch}")
    print(f"  PnL sign mismatches (win vs loss):  {pnl_sign_mismatch}")
    print(f"  Missed signals (bot in position):   {in_position}")
    print(f"  Missed signals (ALO/timeout):       {alo_timeout}")
    print(f"  Extra live entries (wick/realtime):  {wick_entries}")
    print(f"{'─'*120}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare backtest vs live trades")
    parser.add_argument("--account", default="bot-5x", help="Account name (default: bot-5x)")
    parser.add_argument("--days", type=int, default=30, help="Days of trades to compare (default: 30)")
    parser.add_argument("--tolerance-minutes", type=int, default=15,
                        help="Max minutes between BT and live entry to consider a match (default: 15)")
    args = parser.parse_args()

    start_date = datetime.now(timezone.utc) - timedelta(days=args.days)

    # 1. Fetch live trades
    print(f"Fetching live trades for account '{args.account}' since {start_date.date()}...")
    live_trades = fetch_live_trades(args.account, start_date)
    if not live_trades:
        print("No closed BTC trades found in the database for this period.")
        sys.exit(0)
    print(f"  Found {len(live_trades)} closed trades\n")

    # 2. Determine kline range (add warmup buffer)
    earliest = min(t["opened_at"] for t in live_trades)
    buffer_days = 4  # extra days for EMA(200) warmup on 5m candles
    kline_days = (datetime.now(timezone.utc) - earliest).days + buffer_days
    kline_days = max(kline_days, args.days + buffer_days)

    # 3. Fetch klines and run backtest
    df = fetch_klines(kline_days)

    # Filter backtest to only produce trades within our comparison window
    bt_all = run_backtest(df, STOP_LOSS_PCT, BB_PERIOD, BB_STD)
    bt_trades = [
        t for t in bt_all
        if t.entry_timestamp >= start_date
    ]
    print(f"  Backtest produced {len(bt_trades)} trades in the comparison window "
          f"({len(bt_all)} total including warmup)\n")

    # 4. Match and report
    matched, live_only, bt_only = match_trades(live_trades, bt_trades, args.tolerance_minutes)
    print_report(matched, live_only, bt_only, live_trades, bt_trades, args.days)


if __name__ == "__main__":
    main()
