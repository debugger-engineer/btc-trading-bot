"""Unit tests for _is_stop_hit() and _is_inverted() pure logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_dry_bot


# ── _is_stop_hit ──────────────────────────────────────────────────────────────

def test_stop_hit_long_at_exact_boundary():
    """LONG: price exactly at stop level → True."""
    bot = make_dry_bot(stop_loss_pct=2.0)
    bot.position = "LONG"
    bot.entry_price = 100_000.0
    # stop = 100000 * 0.98 = 98000
    assert bot._is_stop_hit(98_000.0) is True


def test_stop_not_hit_long_one_cent_above():
    """LONG: one cent above stop → False."""
    bot = make_dry_bot(stop_loss_pct=2.0)
    bot.position = "LONG"
    bot.entry_price = 100_000.0
    assert bot._is_stop_hit(98_000.01) is False


def test_stop_hit_short_at_exact_boundary():
    """SHORT: price exactly at stop level → True."""
    bot = make_dry_bot(stop_loss_pct=2.0)
    bot.position = "SHORT"
    bot.entry_price = 100_000.0
    # stop = 100000 * 1.02 = 102000
    assert bot._is_stop_hit(102_000.0) is True


def test_stop_not_hit_when_no_position():
    """No position → never triggers."""
    bot = make_dry_bot(stop_loss_pct=2.0)
    bot.position = None
    bot.entry_price = None
    assert bot._is_stop_hit(0.0) is False


# ── _is_inverted ──────────────────────────────────────────────────────────────

def test_inverted_long_when_upper_below_entry():
    """LONG: bb_upper drifted below entry_price → inverted."""
    bot = make_dry_bot()
    bot.position = "LONG"
    bot.entry_price = 96_000.0
    bot._bb_upper = 95_000.0  # band drifted below entry
    assert bot._is_inverted() is True


def test_not_inverted_long_when_upper_above_entry():
    """LONG: bb_upper still above entry → not inverted."""
    bot = make_dry_bot()
    bot.position = "LONG"
    bot.entry_price = 94_000.0
    bot._bb_upper = 96_000.0
    assert bot._is_inverted() is False


def test_inverted_short_when_lower_above_entry():
    """SHORT: bb_lower drifted above entry_price → inverted."""
    bot = make_dry_bot()
    bot.position = "SHORT"
    bot.entry_price = 94_000.0
    bot._bb_lower = 95_000.0  # band drifted above entry
    assert bot._is_inverted() is True


def test_not_inverted_when_no_position():
    """No position → always False."""
    bot = make_dry_bot()
    bot.position = None
    assert bot._is_inverted() is False
