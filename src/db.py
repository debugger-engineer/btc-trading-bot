import logging
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)



# BB mean-reversion perp trades — one row per trade (single entry, opposite-band exit).
CREATE_BB_TABLE = """
CREATE TABLE IF NOT EXISTS perps_trades (
    id              SERIAL PRIMARY KEY,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    status          VARCHAR(6)  NOT NULL DEFAULT 'OPEN',  -- OPEN, CLOSED
    symbol          TEXT        NOT NULL DEFAULT 'BTC',
    direction       VARCHAR(5)  NOT NULL,                 -- LONG, SHORT
    leverage        INTEGER     NOT NULL,
    entry_price     NUMERIC     NOT NULL,
    entry_size_usd  NUMERIC     NOT NULL,
    bb_upper        NUMERIC,
    bb_lower        NUMERIC,
    bb_mid          NUMERIC,
    stop_price      NUMERIC,
    exit_price      NUMERIC,
    exit_bb_upper   NUMERIC,
    exit_bb_lower   NUMERIC,
    exit_bb_mid     NUMERIC,
    gross_pnl       NUMERIC,   -- price-move PnL before fees
    fees            NUMERIC,   -- total round-trip fees paid
    net_pnl         NUMERIC,   -- gross_pnl - fees (true realized PnL)
    closed_by_sl    BOOLEAN     NOT NULL DEFAULT FALSE,
    account_name    TEXT
);
"""


def _connect():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def compute_pnl(
    direction: str, entry_price: float, exit_price: float,
    size_usd: float, entry_fee: float, exit_fee: float,
) -> dict:
    """Pure Python PnL calculation — mirrors the SQL in close_bb_trade."""
    gross = (
        (exit_price - entry_price) if direction == "LONG"
        else (entry_price - exit_price)
    ) / entry_price * size_usd
    fees = entry_fee + exit_fee
    return {"gross_pnl": gross, "fees": fees, "net_pnl": gross - fees}


def _migrate_add_symbol_column():
    """Add symbol column to existing perps_trades table if missing."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'perps_trades' AND column_name = 'symbol'
        """)
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE perps_trades ADD COLUMN symbol TEXT NOT NULL DEFAULT 'BTC'")
            logger.info("Migration: added 'symbol' column to perps_trades (default='BTC')")


def init_bb_db():
    try:
        with _connect() as conn, conn.cursor() as cur:
            try:
                cur.execute(CREATE_BB_TABLE)
            except psycopg2.errors.UniqueViolation:
                conn.rollback()  # table created by another bot concurrently
        _migrate_add_symbol_column()
        logger.info("DB connected — perps_trades table ready")
    except Exception as exc:
        logger.error("Perps DB init failed: %s", exc)
        raise


# --- BB trade logging ---

def open_bb_trade(
    direction: str, leverage: int, entry_price: float, entry_size_usd: float,
    bb_upper: float, bb_lower: float, bb_mid: float, stop_price: float,
    account_name: str = "", symbol: str = "BTC",
) -> int:
    """Insert a new BB mean-reversion trade. Returns the row id."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO perps_trades
               (symbol, direction, leverage, entry_price, entry_size_usd,
                bb_upper, bb_lower, bb_mid, stop_price, account_name)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (symbol, direction, leverage, entry_price, entry_size_usd,
             bb_upper, bb_lower, bb_mid, stop_price, account_name),
        )
        trade_id = cur.fetchone()[0]
    logger.info("BB trade opened — id=%d %s %s @ $%.2f size=$%.2f SL=$%.2f",
                trade_id, symbol, direction, entry_price, entry_size_usd, stop_price)
    return trade_id


def get_open_bb_trade(symbol: str = "BTC", account_name: str = "") -> dict | None:
    """Return the most recent open BB perp trade row for the given symbol/account, or None."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, direction, entry_price, entry_size_usd, stop_price
               FROM perps_trades
               WHERE status = 'OPEN' AND symbol = %s AND account_name = %s
               ORDER BY opened_at DESC LIMIT 1""",
            (symbol, account_name),
        )
        row = cur.fetchone()
    if row:
        return {
            "id": row[0], "direction": row[1],
            "entry_price": float(row[2]), "entry_size_usd": float(row[3]),
            "stop_price": float(row[4]),
        }
    return None


def close_bb_trade(
    trade_id: int, exit_price: float, stopped: bool = False,
    exit_bb_upper: float | None = None, exit_bb_lower: float | None = None, exit_bb_mid: float | None = None,
    entry_fee: float = 0.0, exit_fee: float = 0.0,
):
    """Close a BB trade, computing gross PnL, fees, and net PnL from actual fill data."""
    total_fees = entry_fee + exit_fee
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE perps_trades
               SET closed_at     = NOW(),
                   status        = 'CLOSED',
                   exit_price    = %s,
                   exit_bb_upper = %s,
                   exit_bb_lower = %s,
                   exit_bb_mid   = %s,
                   closed_by_sl  = %s,
                   gross_pnl     = CASE direction
                       WHEN 'LONG'  THEN (%s - entry_price) / entry_price * entry_size_usd
                       WHEN 'SHORT' THEN (entry_price - %s) / entry_price * entry_size_usd
                   END,
                   fees          = %s,
                   net_pnl       = CASE direction
                       WHEN 'LONG'  THEN (%s - entry_price) / entry_price * entry_size_usd
                       WHEN 'SHORT' THEN (entry_price - %s) / entry_price * entry_size_usd
                   END - %s
               WHERE id = %s""",
            (
                exit_price, exit_bb_upper, exit_bb_lower, exit_bb_mid, stopped,
                exit_price, exit_price,        # gross_pnl
                total_fees,                    # fees
                exit_price, exit_price, total_fees,  # net_pnl
                trade_id,
            ),
        )
    logger.info(
        "BB trade closed — id=%d exit=$%.2f stopped=%s fees=%.4f (entry=%.4f exit=%.4f)",
        trade_id, exit_price, stopped, total_fees, entry_fee, exit_fee,
    )
