"""Integration tests for the exit flow:
_close(): cancel orders → market close → DB update.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_live_bot, make_mock_trader, make_mock_db


def _open_long_position(bot):
    """Put the bot into a LONG position state."""
    bot.position = "LONG"
    bot.entry_price = 94_000.0
    bot.exit_order_id = 1003
    bot.stop_order_id = 1002
    bot.current_trade_id = 42
    bot._position_btc_sz = 0.09574
    bot._bb_upper = 96_000.0
    bot._bb_lower = 94_000.0
    bot._bb_mid = 95_000.0


# ── _close ────────────────────────────────────────────────────────────────────

def test_close_cancels_both_resting_orders():
    """Both exit limit and SL orders must be cancelled before market-close."""
    bot = make_live_bot()
    _open_long_position(bot)
    bot._close(96_000.0, stopped=False)

    cancelled = [call[0][1] for call in bot.trader.cancel_order.call_args_list]
    assert 1003 in cancelled  # exit limit
    assert 1002 in cancelled  # stop loss


def test_close_calls_market_close():
    """close_perp_position must be called."""
    bot = make_live_bot()
    _open_long_position(bot)
    bot._close(96_000.0)

    bot.trader.close_perp_position.assert_called_once()


def test_close_resets_position_to_none():
    """After a successful close, position must be None."""
    bot = make_live_bot()
    _open_long_position(bot)
    bot._close(96_000.0)

    assert bot.position is None
    assert bot.entry_price is None
    assert bot.current_trade_id is None


def test_close_writes_to_db_with_stopped_false():
    """close_bb_trade must be called with stopped=False for a TP close."""
    bot = make_live_bot()
    _open_long_position(bot)
    bot._close(96_000.0, stopped=False)

    bot._db.close_bb_trade.assert_called_once()
    # close_bb_trade(trade_id, exit_price, stopped, ...)
    args = bot._db.close_bb_trade.call_args[0]
    assert args[0] == 42          # trade_id
    assert args[2] is False       # stopped


def test_close_writes_to_db_with_stopped_true():
    """close_bb_trade must be called with stopped=True when SL fires."""
    bot = make_live_bot()
    _open_long_position(bot)
    bot._close(93_100.0, stopped=True)

    args = bot._db.close_bb_trade.call_args[0]
    assert args[2] is True        # stopped
