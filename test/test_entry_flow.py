"""Integration tests for the entry order flow:
_place_entry_limit → _on_entry_filled → DB write + SL + exit orders.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_live_bot, make_mock_trader, make_mock_db


@pytest.fixture
def bot():
    return make_live_bot()


# ── _place_entry_limit ────────────────────────────────────────────────────────

def test_place_entry_limit_long_sets_pending_state(bot):
    """After placing a LONG entry, pending-order state must be fully populated."""
    bot._place_entry_limit("LONG", price=94_000, bb_upper=96_000, bb_lower=94_000, bb_mid=95_000)

    assert bot._pending_entry_oid == 1001
    assert bot._pending_entry_side == "LONG"
    assert bot._pending_entry_px == 94_000   # LONG limit at bb_lower
    assert bot._pending_entry_size_usd > 0


def test_place_entry_limit_short_sets_pending_state(bot):
    """SHORT entry limit is placed at bb_upper."""
    bot._place_entry_limit("SHORT", price=96_000, bb_upper=96_000, bb_lower=94_000, bb_mid=95_000)

    assert bot._pending_entry_oid == 1001
    assert bot._pending_entry_side == "SHORT"
    assert bot._pending_entry_px == 96_000   # SHORT limit at bb_upper


def test_place_entry_limit_skips_when_no_margin():
    """Zero balance → order never placed, pending state stays None."""
    trader = make_mock_trader(balance=0.0)
    bot = make_live_bot(trader=trader)
    bot._place_entry_limit("LONG", price=94_000, bb_upper=96_000, bb_lower=94_000, bb_mid=95_000)

    trader.open_long_limit.assert_not_called()
    assert bot._pending_entry_oid is None


# ── _on_entry_filled ──────────────────────────────────────────────────────────

def _call_on_entry_filled(bot, side="LONG", fill_px=94_000.0, size_usd=9_000.0):
    bot._on_entry_filled(side, fill_px, size_usd, bb_upper=96_000, bb_lower=94_000, bb_mid=95_000)


def test_on_entry_filled_sets_position_state(bot):
    """After fill confirmation, bot must have position and entry_price set."""
    _call_on_entry_filled(bot, side="LONG", fill_px=94_000)

    assert bot.position == "LONG"
    assert bot.entry_price == 94_000.0
    assert bot.current_trade_id == 42  # from make_mock_db default


def test_on_entry_filled_places_stop_loss(bot):
    """SL must be placed at entry*(1-sl_pct) for LONG."""
    _call_on_entry_filled(bot, side="LONG", fill_px=100_000.0, size_usd=9_000.0)

    expected_sl = 100_000.0 * (1 - bot.stop_loss_pct / 100)  # 98000 for sl=2%
    bot.trader.place_stop_loss_perp.assert_called_once()
    call_args = bot.trader.place_stop_loss_perp.call_args[0]
    assert call_args[0] == "LONG"
    assert abs(call_args[2] - expected_sl) < 0.01


def test_on_entry_filled_places_exit_limit_at_opposite_band(bot):
    """Exit limit must target bb_upper for LONG, bb_lower for SHORT."""
    _call_on_entry_filled(bot, side="LONG", fill_px=94_000.0)

    bot.trader.place_exit_limit_perp.assert_called_once()
    call_args = bot.trader.place_exit_limit_perp.call_args[0]
    assert call_args[0] == "LONG"
    assert call_args[2] == 96_000.0  # bb_upper


def test_on_entry_filled_writes_to_db(bot):
    """DB open_bb_trade must be called with correct direction and entry price."""
    _call_on_entry_filled(bot, side="SHORT", fill_px=96_000.0, size_usd=9_600.0)

    bot._db.open_bb_trade.assert_called_once()
    args = bot._db.open_bb_trade.call_args[0]
    # open_bb_trade(direction, leverage, entry_price, entry_size_usd, ...)
    assert args[0] == "SHORT"    # direction
    assert args[2] == 96_000.0   # entry_price (fill_px)
