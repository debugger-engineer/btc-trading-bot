"""Unit tests for check_signals() entry/exit conditions using dry_run mode."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_dry_bot


def _bot_with_indicators(bb_lower=94_000.0, bb_upper=96_000.0, bb_mid=95_000.0, ema=95_000.0, **kwargs):
    bot = make_dry_bot(**kwargs)
    bot._bb_lower = bb_lower
    bot._bb_upper = bb_upper
    bot._bb_mid = bb_mid
    bot._ema = ema
    return bot


# ── Entry signal tests ─────────────────────────────────────────────────────────

def test_long_entry_triggered():
    """Price at/below bb_lower and above EMA → LONG entry."""
    bot = _bot_with_indicators(bb_lower=94_000, ema=93_000)
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(93_999)
    mock_fn.assert_called_once()
    assert "LONG entry" in mock_fn.call_args[0][0]


def test_short_entry_triggered():
    """Price at/above bb_upper and below EMA → SHORT entry."""
    bot = _bot_with_indicators(bb_upper=96_000, ema=97_000)
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(96_001)
    mock_fn.assert_called_once()
    assert "SHORT entry" in mock_fn.call_args[0][0]


def test_no_entry_price_inside_bands():
    """Price between the bands → no signal."""
    bot = _bot_with_indicators(bb_lower=94_000, bb_upper=96_000, ema=95_000)
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(95_000)
    mock_fn.assert_not_called()


def test_no_long_entry_when_price_below_ema():
    """Price ≤ bb_lower but also below EMA (downtrend) → no LONG."""
    bot = _bot_with_indicators(bb_lower=94_000, ema=95_000)
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(93_500)  # below bb_lower AND below ema
    mock_fn.assert_not_called()


def test_no_entry_when_position_already_open():
    """Existing position blocks new entries."""
    bot = _bot_with_indicators(bb_upper=96_000, ema=97_000)
    bot.position = "LONG"
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(96_001)  # SHORT condition met
    mock_fn.assert_not_called()


def test_no_entry_when_indicators_not_ready():
    """No exception and no signal if indicators haven't been computed yet."""
    bot = make_dry_bot()  # _bb_upper is None by default
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(95_000)
    mock_fn.assert_not_called()


# ── Exit signal tests ──────────────────────────────────────────────────────────

def test_long_stop_loss_triggered():
    """LONG position: price at stop level → stop-loss signal."""
    bot = _bot_with_indicators()
    bot.position = "LONG"
    bot.entry_price = 95_000.0
    # stop at 95000 * (1 - 0.02) = 93100
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(93_100)
    mock_fn.assert_called_once()
    assert "Stop-loss hit" in mock_fn.call_args[0][0]
    assert "LONG" in mock_fn.call_args[0][0]


def test_long_target_hit():
    """LONG position: price reaches/exceeds bb_upper → target hit signal."""
    bot = _bot_with_indicators(bb_upper=96_000)
    bot.position = "LONG"
    bot.entry_price = 94_000.0
    with patch.object(bot, "_dry_or_live") as mock_fn:
        bot.check_signals(96_001)
    mock_fn.assert_called_once()
    assert "target hit" in mock_fn.call_args[0][0]
    assert "LONG" in mock_fn.call_args[0][0]
