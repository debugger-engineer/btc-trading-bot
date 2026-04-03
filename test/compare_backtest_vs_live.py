"""
Compare backtest predictions vs actual live trades from the database.

Multi-asset version: reads per-asset config from accounts.json and compares
all active symbols, not just BTC.

Usage:
    python test/compare_backtest_vs_live.py
    python test/compare_backtest_vs_live.py --account bot-5x --days 2
    python test/compare_backtest_vs_live.py --tolerance-minutes 10
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import psycopg2
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
INTERVAL       = "5m"
TRADE_SIZE_USD = 95.0
MAKER_FEE      = 0.0001   # 0.010%
TAKER_FEE      = 0.0005   # 0.050%

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MS_PER_CANDLE      = 5 * 60 * 1000


# ── Trade dataclass ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    direction: str
    entry_timestamp: Optional[datetime] = None
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
        return self.realized_pnl / self.size_usd * 100


# ── Config loading ────────────────────────────────────────────────────────────
def load_account_config(account_name: str) -> list[dict]:
    """Load active token configs for the given account from accounts.json."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "accounts.json")
    with open(config_path) as f:
        data = json.load(f)

    for acct in data["accounts"]:
        if acct["name"] == account_name:
            return [t for t in acct["tokens"] if t.get("active", False)]

    raise ValueError(f"Account '{account_name}' not found in accounts.json")


# ── Database ──────────────────────────────────────────────────────────────────
def fetch_live_trades(account_name: str, start_date: datetime) -> list[dict]:
    """Fetch all closed trades for the account from the database."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, opened_at, closed_at, symbol, direction, leverage,
                       entry_price, entry_size_usd,
                       bb_upper, bb_lower, bb_mid, stop_price,
                       exit_price, exit_bb_upper, exit_bb_lower, exit_bb_mid,
                       gross_pnl, fees, net_pnl, closed_by_sl, account_name
                FROM perps_trades
                WHERE status = 'CLOSED'
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
        for k in ("opened_at", "closed_at"):
            if t[k] and t[k].tzinfo is None:
                t[k] = t[k].replace(tzinfo=timezone.utc)
        for k in ("entry_price", "entry_size_usd", "bb_upper", "bb_lower", "bb_mid",
                   "stop_price", "exit_price", "exit_bb_upper", "exit_bb_lower",
                   "exit_bb_mid", "gross_pnl", "fees", "net_pnl"):
            if t[k] is not None:
                t[k] = float(t[k])
        trades.append(t)
    return trades


# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_klines_yahoo(symbol: str, days: int) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval=INTERVAL, progress=False)
    if df.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    df["close"] = df["close"].astype(float)
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "open_time", "Date": "open_time"})
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    return df


def fetch_klines_binance(symbol: str, days: int) -> pd.DataFrame:
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_rows, cursor = [], start_ms
    while cursor < end_ms:
        resp = requests.get(BINANCE_KLINES_URL, params={
            "symbol": symbol, "interval": INTERVAL,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + MS_PER_CANDLE
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["close"]     = df["close"].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


def fetch_klines_for_token(token: dict, days: int) -> pd.DataFrame:
    """Fetch candles using the appropriate data source for the token."""
    binance_sym = token.get("binance_symbol", "")
    yahoo_sym = token.get("yahoo_symbol", "")
    if binance_sym:
        return fetch_klines_binance(binance_sym, days)
    if yahoo_sym:
        return fetch_klines_yahoo(yahoo_sym, days)
    raise RuntimeError(f"No data source for {token['symbol']}")


# ── Backtest engine (with timestamps) ────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame, stop_loss_pct: float, bb_period: int, bb_std: float,
    ema_period: int, leverage: int,
) -> List[Trade]:
    ema_series = df["close"].ewm(span=ema_period, adjust=False).mean()
    bb_mid_s   = df["close"].rolling(bb_period).mean()
    bb_upper_s = bb_mid_s + bb_std * df["close"].rolling(bb_period).std()
    bb_lower_s = bb_mid_s - bb_std * df["close"].rolling(bb_period).std()
    warmup     = max(ema_period, bb_period) * 3

    position, current = None, None
    trades = []
    entry_size = TRADE_SIZE_USD * leverage

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
                                entry_price=price, size_usd=entry_size)
                position = "LONG"
            elif price >= bb_upper and price < ema:
                current = Trade(direction="SHORT", entry_timestamp=ts,
                                entry_price=price, size_usd=entry_size)
                position = "SHORT"

        elif position == "LONG":
            sl_hit = price <= current.entry_price * (1 - stop_loss_pct / 100)
            if sl_hit:
                current.exit_price = price; current.exit_timestamp = ts; current.stopped_out = True
                trades.append(current); position, current = None, None
            elif bb_upper < current.entry_price and price <= current.entry_price:
                current.exit_price = current.entry_price; current.exit_timestamp = ts
                trades.append(current); position, current = None, None
            elif price >= bb_lower + 2 * half_band:
                current.exit_price = price; current.exit_timestamp = ts
                trades.append(current); position, current = None, None

        elif position == "SHORT":
            sl_hit = price >= current.entry_price * (1 + stop_loss_pct / 100)
            if sl_hit:
                current.exit_price = price; current.exit_timestamp = ts; current.stopped_out = True
                trades.append(current); position, current = None, None
            elif bb_lower > current.entry_price and price >= current.entry_price:
                current.exit_price = current.entry_price; current.exit_timestamp = ts
                trades.append(current); position, current = None, None
            elif price <= bb_upper - 2 * half_band:
                current.exit_price = price; current.exit_timestamp = ts
                trades.append(current); position, current = None, None

    if current is not None:
        current.exit_price = df["close"].iloc[-1]
        current.exit_timestamp = df["open_time"].iloc[-1]
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
        best_idx, best_delta = None, None
        for j, bt in enumerate(bt_trades):
            if j in bt_used or bt.direction != lt["direction"]:
                continue
            delta = abs((lt["opened_at"] - bt.entry_timestamp).total_seconds())
            if delta <= tolerance.total_seconds():
                if best_delta is None or delta < best_delta:
                    best_idx, best_delta = j, delta
        if best_idx is not None:
            bt_used.add(best_idx)
            matched.append((lt, bt_trades[best_idx]))
        else:
            live_only.append(lt)

    bt_only = [bt for j, bt in enumerate(bt_trades) if j not in bt_used]
    return matched, live_only, bt_only


# ── Divergence cause inference ────────────────────────────────────────────────
def infer_live_only_cause(lt: dict, bt_trades: List[Trade]) -> str:
    for bt in bt_trades:
        if bt.entry_timestamp is None:
            continue
        delta = abs((lt["opened_at"] - bt.entry_timestamp).total_seconds())
        if delta < 600 and bt.direction != lt["direction"]:
            return "Direction mismatch (EMA filter edge)"
    return "Intra-candle wick (price touched band mid-candle)"


def infer_bt_only_cause(bt: Trade, live_trades: list) -> str:
    for lt in live_trades:
        opened = lt["opened_at"]
        closed = lt["closed_at"]
        if closed and opened <= bt.entry_timestamp <= closed:
            return "Bot was in existing position"
    return "ALO rejection / fill timeout"


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_ts(dt) -> str:
    if dt is None:
        return "N/A"
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return dt.strftime("%m-%d %H:%M")

def fmt_price(p) -> str:
    return "N/A" if p is None else f"${p:,.2f}"

def fmt_pnl(p) -> str:
    if p is None:
        return "N/A"
    return f"{'+' if p >= 0 else ''}${p:,.2f}"

def fmt_pct(p) -> str:
    if p is None:
        return "N/A"
    return f"{'+' if p >= 0 else ''}{p:.2f}%"


# ── Per-symbol report ─────────────────────────────────────────────────────────
def print_symbol_report(
    symbol: str, token: dict, matched: list, live_only: list, bt_only: list,
    live_trades_sym: list, bt_trades_sym: List[Trade],
):
    n_live = len(live_trades_sym)
    n_bt_closed = len([bt for bt in bt_trades_sym if bt.realized_pnl is not None])
    n_matched = len(matched)
    match_rate = (n_matched / n_live * 100) if n_live else 0

    data_src = token.get("binance_symbol") or token.get("yahoo_symbol", "?")
    cfg_str = f"BB({token['bb_period']},{token['bb_std']:.1f}) SL={token['stop_loss_pct']}% EMA={token['ema_period']}"

    print(f"\n{'='*110}")
    print(f"  {symbol}  |  {cfg_str}  |  data: {data_src}")
    print(f"{'='*110}")

    # Summary
    print(f"  Matched: {n_matched}/{n_live} live trades ({match_rate:.0f}%)  |  "
          f"Live-only: {len(live_only)}  |  BT-only: {len(bt_only)}  |  BT total: {n_bt_closed}")

    # PnL comparison
    live_pnl = sum(lt["net_pnl"] or 0 for lt in live_trades_sym)
    live_wins = sum(1 for lt in live_trades_sym if (lt["net_pnl"] or 0) > 0)
    live_sls = sum(1 for lt in live_trades_sym if lt["closed_by_sl"])
    live_wr = (live_wins / n_live * 100) if n_live else 0

    bt_closed = [bt for bt in bt_trades_sym if bt.realized_pnl is not None]
    bt_pnl = sum(bt.realized_pnl for bt in bt_closed)
    bt_wins = sum(1 for bt in bt_closed if bt.realized_pnl > 0)
    bt_sls = sum(1 for bt in bt_closed if bt.stopped_out)
    bt_wr = (bt_wins / len(bt_closed) * 100) if bt_closed else 0

    print(f"\n  {'Metric':<25} {'Live':>12} {'Backtest':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Trades':<25} {n_live:>12} {n_bt_closed:>12}")
    print(f"  {'Win rate':<25} {f'{live_wr:.0f}%':>12} {f'{bt_wr:.0f}%':>12}")
    print(f"  {'Stop-losses':<25} {live_sls:>12} {bt_sls:>12}")
    print(f"  {'Net PnL':<25} {fmt_pnl(live_pnl):>12} {fmt_pnl(bt_pnl):>12}")

    # Matched trades table
    if matched:
        print(f"\n  {'#':>3}  {'Dir':<5}  {'BT Entry':<12}  {'Live Entry':<12}  "
              f"{'dTime':>6}  {'BT Entry$':>11}  {'Live Entry$':>11}  "
              f"{'BT Exit$':>11}  {'Live Exit$':>11}  "
              f"{'BT PnL':>9}  {'Live PnL':>9}  {'Exit':>6}")
        print(f"  {'-'*106}")
        for idx, (lt, bt) in enumerate(matched, 1):
            delta_sec = (lt["opened_at"] - bt.entry_timestamp).total_seconds()
            delta_str = f"{delta_sec/60:+.0f}m"
            exit_type = ""
            if lt["closed_by_sl"] and bt.stopped_out:
                exit_type = "SL/SL"
            elif not lt["closed_by_sl"] and not bt.stopped_out:
                exit_type = "TP/TP"
            else:
                exit_type = f"{'SL' if lt['closed_by_sl'] else 'TP'}/{'SL' if bt.stopped_out else 'TP'}"
            flag = " !!" if lt["closed_by_sl"] != bt.stopped_out else ""
            print(
                f"  {idx:>3}  {bt.direction:<5}  {fmt_ts(bt.entry_timestamp):<12}  "
                f"{fmt_ts(lt['opened_at']):<12}  {delta_str:>6}  "
                f"{fmt_price(bt.entry_price):>11}  {fmt_price(lt['entry_price']):>11}  "
                f"{fmt_price(bt.exit_price):>11}  {fmt_price(lt['exit_price']):>11}  "
                f"{fmt_pnl(bt.realized_pnl):>9}  {fmt_pnl(lt['net_pnl']):>9}  "
                f"{exit_type:>6}{flag}"
            )

    # Live-only trades
    if live_only:
        print(f"\n  LIVE-ONLY (bot traded, backtest did not):")
        for idx, lt in enumerate(live_only, 1):
            cause = infer_live_only_cause(lt, bt_trades_sym)
            print(f"    {idx}. {lt['direction']:<5} {fmt_ts(lt['opened_at'])} @ {fmt_price(lt['entry_price'])} "
                  f"-> {fmt_price(lt['exit_price'])}  PnL={fmt_pnl(lt['net_pnl'])}  | {cause}")

    # BT-only trades
    if bt_only:
        print(f"\n  BT-ONLY (backtest signaled, bot did not trade):")
        for idx, bt in enumerate(bt_only, 1):
            cause = infer_bt_only_cause(bt, live_trades_sym)
            print(f"    {idx}. {bt.direction:<5} {fmt_ts(bt.entry_timestamp)} @ {fmt_price(bt.entry_price)} "
                  f"-> {fmt_price(bt.exit_price)}  PnL={fmt_pnl(bt.realized_pnl)}  | {cause}")


# ── Overall report ────────────────────────────────────────────────────────────
def print_overall_report(all_results: list, start_date: datetime):
    total_live = sum(r["n_live"] for r in all_results)
    total_matched = sum(r["n_matched"] for r in all_results)
    total_live_only = sum(r["n_live_only"] for r in all_results)
    total_bt_only = sum(r["n_bt_only"] for r in all_results)
    total_live_pnl = sum(r["live_pnl"] for r in all_results)
    total_bt_pnl = sum(r["bt_pnl"] for r in all_results)
    match_rate = (total_matched / total_live * 100) if total_live else 0

    hours = (datetime.now(timezone.utc) - start_date).total_seconds() / 3600
    window_str = f"Since {start_date.strftime('%m-%d %H:%M UTC')} ({hours:.0f}h)"

    print(f"\n{'='*110}")
    print(f"  OVERALL SUMMARY  |  {window_str}  |  {len(all_results)} symbols")
    print(f"{'='*110}")

    print(f"\n  Total live trades:    {total_live}")
    print(f"  Matched:              {total_matched} ({match_rate:.0f}%)")
    print(f"  Live-only:            {total_live_only}")
    print(f"  BT-only:              {total_bt_only}")
    print(f"\n  Live net PnL:         {fmt_pnl(total_live_pnl)}")
    print(f"  Backtest net PnL:     {fmt_pnl(total_bt_pnl)}")
    print(f"  Delta:                {fmt_pnl(total_live_pnl - total_bt_pnl)}")

    # Per-symbol summary table
    print(f"\n  {'Symbol':<18} {'Live':>6} {'Match':>6} {'Rate':>6} {'Live PnL':>10} {'BT PnL':>10} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
        rate = (r["n_matched"] / r["n_live"] * 100) if r["n_live"] else 0
        delta = r["live_pnl"] - r["bt_pnl"]
        print(f"  {r['symbol']:<18} {r['n_live']:>6} {r['n_matched']:>6} {f'{rate:.0f}%':>6} "
              f"{fmt_pnl(r['live_pnl']):>10} {fmt_pnl(r['bt_pnl']):>10} {fmt_pnl(delta):>10}")

    # Divergence summary
    total_exit_mismatch = sum(r["exit_mismatches"] for r in all_results)
    total_pnl_sign_mismatch = sum(r["pnl_sign_mismatches"] for r in all_results)
    print(f"\n  DIVERGENCES:")
    print(f"    Exit type mismatches (SL vs TP):   {total_exit_mismatch}")
    print(f"    PnL sign mismatches (win vs loss): {total_pnl_sign_mismatch}")
    print(f"    Live-only entries (wick/realtime):  {total_live_only}")
    print(f"    BT-only signals (missed by bot):   {total_bt_only}")
    print(f"{'='*110}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare backtest vs live trades (multi-asset)")
    parser.add_argument("--account", default="bot-5x", help="Account name (default: bot-5x)")
    parser.add_argument("--days", type=int, default=None, help="Days of trades to compare (default: auto-detect from first trade)")
    parser.add_argument("--since", default=None,
                        help="Start time in ISO format, e.g. '2026-03-30T19:56' (default: auto-detect from first trade)")
    parser.add_argument("--tolerance-minutes", type=int, default=15,
                        help="Max minutes between BT and live entry to consider a match (default: 15)")
    args = parser.parse_args()

    if args.since:
        start_date = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    elif args.days:
        start_date = datetime.now(timezone.utc) - timedelta(days=args.days)
    else:
        start_date = None  # will auto-detect from first trade

    # 1. Load config
    tokens = load_account_config(args.account)
    print(f"Account '{args.account}': {len(tokens)} active tokens")
    for t in tokens:
        src = t.get("binance_symbol") or t.get("yahoo_symbol", "?")
        print(f"  {t['symbol']:<18} BB({t['bb_period']},{t['bb_std']:.1f}) SL={t['stop_loss_pct']}% src={src}")

    # 2. Fetch live trades (auto-detect start if not specified)
    if start_date is None:
        # Fetch all trades to find the earliest, then use that as start
        all_live = fetch_live_trades(args.account, datetime(2000, 1, 1, tzinfo=timezone.utc))
        if not all_live:
            print("No closed trades found in the database.")
            sys.exit(0)
        start_date = min(t["opened_at"] for t in all_live)
        print(f"\nAuto-detected bot start: {start_date.strftime('%Y-%m-%d %H:%M UTC')}")
        live_trades = all_live
    else:
        print(f"\nFetching live trades since {start_date.strftime('%Y-%m-%d %H:%M UTC')}...")
        live_trades = fetch_live_trades(args.account, start_date)

    if not live_trades:
        print("No closed trades found in the database for this period.")
        sys.exit(0)
    print(f"  Found {len(live_trades)} closed trades")

    # Group by symbol
    live_by_symbol = {}
    for lt in live_trades:
        live_by_symbol.setdefault(lt["symbol"], []).append(lt)

    # 3. Per-symbol: fetch candles, run backtest, match, report
    all_results = []

    for token in tokens:
        symbol = token["symbol"]
        live_sym = live_by_symbol.get(symbol, [])

        # Determine kline fetch period
        days_since_start = (datetime.now(timezone.utc) - start_date).days + 1
        if live_sym:
            earliest = min(t["opened_at"] for t in live_sym)
            kline_days = max(
                (datetime.now(timezone.utc) - earliest).days + 8,
                days_since_start + 8,
            )
        else:
            kline_days = days_since_start + 8

        # Cap Yahoo at 59 days
        if not token.get("binance_symbol"):
            kline_days = min(kline_days, 59)

        # Fetch candles
        src = token.get("binance_symbol") or token.get("yahoo_symbol", "?")
        print(f"\nFetching {symbol} ({src}) {kline_days}d candles...")
        try:
            df = fetch_klines_for_token(token, kline_days)
        except Exception as e:
            print(f"  SKIP {symbol}: {e}")
            continue
        print(f"  {len(df)} candles ({df['open_time'].iloc[0].date()} -> {df['open_time'].iloc[-1].date()})")

        # Run backtest
        bt_all = run_backtest(
            df,
            stop_loss_pct=token["stop_loss_pct"],
            bb_period=token["bb_period"],
            bb_std=token["bb_std"],
            ema_period=token["ema_period"],
            leverage=token["leverage"],
        )
        bt_trades = [t for t in bt_all if t.entry_timestamp and t.entry_timestamp >= start_date]
        print(f"  Backtest: {len(bt_trades)} trades in window ({len(bt_all)} total incl. warmup)")

        # Match
        matched, live_only, bt_only = match_trades(live_sym, bt_trades, args.tolerance_minutes)

        # Compute stats
        exit_mismatches = sum(1 for lt, bt in matched if lt["closed_by_sl"] != bt.stopped_out)
        pnl_sign_mismatches = sum(
            1 for lt, bt in matched
            if ((lt["net_pnl"] or 0) > 0) != ((bt.realized_pnl or 0) > 0)
        )

        all_results.append({
            "symbol": symbol,
            "n_live": len(live_sym),
            "n_matched": len(matched),
            "n_live_only": len(live_only),
            "n_bt_only": len(bt_only),
            "live_pnl": sum(lt["net_pnl"] or 0 for lt in live_sym),
            "bt_pnl": sum(bt.realized_pnl or 0 for bt in bt_trades if bt.realized_pnl is not None),
            "exit_mismatches": exit_mismatches,
            "pnl_sign_mismatches": pnl_sign_mismatches,
        })

        # Print per-symbol report
        if live_sym or bt_trades:
            print_symbol_report(symbol, token, matched, live_only, bt_only, live_sym, bt_trades)

    # 4. Overall report
    if all_results:
        print_overall_report(all_results, start_date)


if __name__ == "__main__":
    main()
