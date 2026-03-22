"""Pytest fixtures — auto-loaded by pytest for all tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # make helpers importable

from helpers import make_dry_bot, make_live_bot, make_mock_trader, make_mock_db


@pytest.fixture
def dry_bot():
    return make_dry_bot()


@pytest.fixture
def mock_trader():
    return make_mock_trader()


@pytest.fixture
def mock_db():
    return make_mock_db()


@pytest.fixture
def live_bot(mock_trader, mock_db):
    return make_live_bot(trader=mock_trader, mock_db=mock_db)
