"""Shared helper factories for the test suite.

These are plain functions (not pytest fixtures) so they can be imported
directly by test files at any nesting level.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_mock_trader(
    balance: float = 10_000.0,
    position: dict | None = None,
    resting_oid: int = 1001,
) -> MagicMock:
    """Return a MagicMock for HyperliquidTrader with sensible defaults."""
    trader = MagicMock()
    trader.wallet_address = "0xTESTWALLET"
    trader.sz_decimals = 5
    trader._coin_name = "BTC"

    trader.get_perp_usdc_balance.return_value = balance
    trader.get_account_equity.return_value = balance
    trader.get_perp_position.return_value = position
    trader.get_open_orders.return_value = []
    trader.get_last_fill.return_value = {"px": 95_000.0, "fee": 0.95}

    def _resting(oid):
        return {"response": {"data": {"statuses": [{"resting": {"oid": oid}}]}}}

    trader.open_long_limit.return_value = _resting(resting_oid)
    trader.open_short_limit.return_value = _resting(resting_oid)
    trader.place_stop_loss_perp.return_value = _resting(resting_oid + 1)
    trader.place_exit_limit_perp.return_value = _resting(resting_oid + 2)

    trader.cancel_order.return_value = True
    trader.cancel_all_open_orders.return_value = None
    trader.close_perp_position.return_value = {"status": "ok"}
    trader.update_leverage.return_value = None

    return trader


def make_mock_db(trade_id: int = 42) -> MagicMock:
    """Return a MagicMock for the db module."""
    mock = MagicMock()
    mock.get_open_bb_trade.return_value = None
    mock.open_bb_trade.return_value = trade_id
    mock.close_bb_trade.return_value = None
    mock.init_bb_db.return_value = None
    mock.count_open_bb_trades_by_direction.return_value = 0
    return mock


def make_dry_bot(**kwargs):
    """Instantiate PerpsBot in dry_run=True mode — no exchange or DB access."""
    from perps_trading import PerpsBot
    defaults = dict(
        dry_run=True,
        leverage=5,
        capital_percent=50.0,
        bb_period=20,
        bb_std=2.0,
        ema_period=200,
        stop_loss_pct=2.0,
    )
    defaults.update(kwargs)
    return PerpsBot(**defaults)


def make_live_bot(trader=None, mock_db=None, **kwargs):
    """Instantiate PerpsBot in dry_run=False mode with injected mocks."""
    from perps_trading import PerpsBot
    defaults = dict(
        dry_run=False,
        leverage=5,
        capital_percent=50.0,
        bb_period=20,
        bb_std=2.0,
        ema_period=200,
        stop_loss_pct=2.0,
        _trader=trader or make_mock_trader(),
        _db=mock_db or make_mock_db(),
    )
    defaults.update(kwargs)
    return PerpsBot(**defaults)
