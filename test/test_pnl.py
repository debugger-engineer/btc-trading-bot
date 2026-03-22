"""Unit tests for compute_pnl() and _extract_oid()."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db import compute_pnl
from perps_trading import _extract_oid


# ── compute_pnl ───────────────────────────────────────────────────────────────

def test_long_profitable_trade():
    result = compute_pnl("LONG", entry_price=90_000, exit_price=95_000,
                          size_usd=9_000, entry_fee=0.90, exit_fee=0.45)
    expected_gross = (95_000 - 90_000) / 90_000 * 9_000  # = 500.0
    assert abs(result["gross_pnl"] - expected_gross) < 0.01
    assert abs(result["fees"] - 1.35) < 0.001
    assert abs(result["net_pnl"] - (expected_gross - 1.35)) < 0.01


def test_long_losing_trade():
    """LONG stopped out at -2%."""
    result = compute_pnl("LONG", entry_price=95_000, exit_price=93_100,
                          size_usd=9_500, entry_fee=0.95, exit_fee=4.75)
    assert result["gross_pnl"] < 0
    assert result["net_pnl"] < result["gross_pnl"]  # fees make it worse


def test_short_profitable_trade():
    result = compute_pnl("SHORT", entry_price=95_000, exit_price=90_000,
                          size_usd=9_500, entry_fee=0.95, exit_fee=0.475)
    expected_gross = (95_000 - 90_000) / 95_000 * 9_500
    assert abs(result["gross_pnl"] - expected_gross) < 0.01
    assert result["net_pnl"] < result["gross_pnl"]  # fees reduce profit


def test_break_even_trade():
    """exit == entry → gross = 0, net = -fees."""
    result = compute_pnl("LONG", entry_price=95_000, exit_price=95_000,
                          size_usd=9_500, entry_fee=0.95, exit_fee=0.95)
    assert result["gross_pnl"] == 0.0
    assert abs(result["net_pnl"] - (-1.90)) < 0.001


# ── _extract_oid ──────────────────────────────────────────────────────────────

def test_extract_oid_resting_response():
    response = {"response": {"data": {"statuses": [{"resting": {"oid": 42}}]}}}
    assert _extract_oid(response) == 42


def test_extract_oid_returns_none_on_filled_response():
    response = {"response": {"data": {"statuses": [{"filled": {"avgPx": "95000"}}]}}}
    assert _extract_oid(response) is None


def test_extract_oid_returns_none_on_empty():
    assert _extract_oid({}) is None
