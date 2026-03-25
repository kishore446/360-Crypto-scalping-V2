"""Tests for session-adaptive threshold management (src/session_thresholds.py)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.session_thresholds import (
    TradingSession,
    _SESSION_ADJUSTMENTS,
    get_current_session,
    get_session_adjusted_thresholds,
)


class TestGetCurrentSession:
    """Test UTC-hour → trading session mapping."""

    @pytest.mark.parametrize("hour,expected", [
        (0, TradingSession.ASIAN),
        (3, TradingSession.ASIAN),
        (7, TradingSession.ASIAN),
        (8, TradingSession.EUROPEAN),
        (11, TradingSession.EUROPEAN),
        (13, TradingSession.EUROPEAN),
        (14, TradingSession.US),
        (17, TradingSession.US),
        (20, TradingSession.US),
        (21, TradingSession.OVERNIGHT),
        (23, TradingSession.OVERNIGHT),
    ])
    def test_session_by_hour(self, hour: int, expected: str):
        dt = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
        assert get_current_session(dt) == expected

    def test_auto_detect_uses_utc_now(self):
        """When no argument is given, auto-detection should return a valid session name."""
        session = get_current_session()
        assert session in {
            TradingSession.ASIAN,
            TradingSession.EUROPEAN,
            TradingSession.US,
            TradingSession.OVERNIGHT,
        }


class TestGetSessionAdjustedThresholds:
    """Test threshold adjustment logic per session."""

    def test_asian_session_spreads_wider(self):
        """Asian session should allow 50% wider spreads (mult=1.5)."""
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session=TradingSession.ASIAN,
        )
        assert abs(adj_spread - 0.03) < 1e-9   # 0.02 * 1.5
        assert abs(adj_volume - 3_000_000) < 1  # 5_000_000 * 0.6
        assert offset == -2.0

    def test_us_session_uses_standard_thresholds(self):
        """US session should return unchanged (mult=1.0) values."""
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session=TradingSession.US,
        )
        assert adj_spread == 0.02
        assert adj_volume == 5_000_000.0
        assert offset == 1.0

    def test_european_session_multipliers(self):
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session=TradingSession.EUROPEAN,
        )
        assert abs(adj_spread - round(0.02 * 1.2, 6)) < 1e-9
        assert abs(adj_volume - round(5_000_000 * 0.8, 2)) < 1
        assert offset == 0.0

    def test_overnight_session_multipliers(self):
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session=TradingSession.OVERNIGHT,
        )
        assert abs(adj_spread - round(0.02 * 1.3, 6)) < 1e-9
        assert abs(adj_volume - round(5_000_000 * 0.7, 2)) < 1
        assert offset == -1.0

    def test_unknown_session_falls_back_to_us(self):
        """An unrecognised session name should silently fall back to US thresholds."""
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session="UNKNOWN_SESSION",
        )
        # Expect US defaults
        assert adj_spread == 0.02
        assert adj_volume == 5_000_000.0
        assert offset == 1.0

    def test_none_session_auto_detects(self):
        """Passing session=None should auto-detect and return valid floats."""
        adj_spread, adj_volume, offset = get_session_adjusted_thresholds(
            channel_spread_max=0.02,
            channel_min_volume=5_000_000,
            session=None,
        )
        assert adj_spread > 0.0
        assert adj_volume > 0.0
        assert isinstance(offset, float)


class TestPassBasicFiltersSessionAware:
    """Verify that BaseChannel._pass_basic_filters() uses session-adjusted thresholds."""

    def test_asian_session_relaxes_volume_filter(self):
        """During Asian session, a volume below the base threshold should still pass."""
        from src.channels.base import BaseChannel
        from unittest.mock import patch

        # Build a minimal config-like object
        cfg = MagicMock()
        cfg.spread_max = 0.02
        cfg.min_volume = 5_000_000

        chan = BaseChannel.__new__(BaseChannel)
        chan.config = cfg

        # Asian session volume_mult = 0.6 → adjusted min_volume = 3_000_000
        # A volume of 3_500_000 should pass
        with patch("src.session_thresholds.get_current_session", return_value=TradingSession.ASIAN):
            result = chan._pass_basic_filters(spread_pct=0.01, volume_24h_usd=3_500_000)
        assert result is True

    def test_us_session_blocks_low_volume(self):
        """During US session (standard thresholds), low volume should be blocked."""
        from src.channels.base import BaseChannel

        cfg = MagicMock()
        cfg.spread_max = 0.02
        cfg.min_volume = 5_000_000

        chan = BaseChannel.__new__(BaseChannel)
        chan.config = cfg

        with patch("src.session_thresholds.get_current_session", return_value=TradingSession.US):
            result = chan._pass_basic_filters(spread_pct=0.01, volume_24h_usd=3_500_000)
        assert result is False
