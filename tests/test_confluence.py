"""Tests for cross-channel confluence scoring (src/confluence.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.confluence import (
    _CONFLUENCE_BONUS,
    _CONFLUENCE_CHANNELS,
    _MAX_CONFLUENCE_BONUS,
    compute_confluence_bonus,
)
from src.smc import Direction


def _make_chan(name: str, direction: Direction | None = None) -> MagicMock:
    """Create a mock channel that returns a signal with the given direction."""
    chan = MagicMock()
    chan.config.name = name
    if direction is None:
        chan.evaluate.return_value = None
    else:
        sig = MagicMock()
        sig.direction.value = direction.value
        chan.evaluate.return_value = sig
    return chan


_KWARGS = dict(
    symbol="BTCUSDT",
    candles={},
    indicators={},
    smc_data={},
    spread_pct=0.01,
    volume_24h_usd=5_000_000,
)


class TestComputeConfluenceBonus:
    def test_zero_confirming_channels_gives_no_bonus(self):
        """When no other channel confirms, bonus should be 0."""
        channels = [
            _make_chan("360_SCALP_FVG", Direction.LONG),   # primary — excluded
            _make_chan("360_SCALP_CVD", None),             # no signal
            _make_chan("360_SCALP_VWAP", None),            # no signal
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert confirming == []

    def test_one_confirming_channel_gives_3_bonus(self):
        """1 confirming channel → bonus of 3.0."""
        channels = [
            _make_chan("360_SCALP_FVG"),                   # primary
            _make_chan("360_SCALP_CVD", Direction.LONG),   # confirms
            _make_chan("360_SCALP_VWAP", None),            # no signal
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 3.0
        assert confirming == ["360_SCALP_CVD"]

    def test_two_confirming_channels_gives_5_bonus(self):
        """2 confirming channels → bonus of 5.0."""
        channels = [
            _make_chan("360_SCALP_FVG"),                    # primary
            _make_chan("360_SCALP_CVD", Direction.LONG),    # confirms
            _make_chan("360_SCALP_VWAP", Direction.LONG),   # confirms
            _make_chan("360_SCALP_OBI", None),              # no signal
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 5.0
        assert set(confirming) == {"360_SCALP_CVD", "360_SCALP_VWAP"}

    def test_bonus_capped_at_max(self):
        """Bonus should never exceed _MAX_CONFLUENCE_BONUS."""
        channels = [
            _make_chan("360_SCALP"),
            _make_chan("360_SCALP_FVG", Direction.SHORT),
            _make_chan("360_SCALP_CVD", Direction.SHORT),
            _make_chan("360_SCALP_VWAP", Direction.SHORT),
            _make_chan("360_SCALP_OBI", Direction.SHORT),
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP",
            primary_direction=Direction.SHORT.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus <= _MAX_CONFLUENCE_BONUS
        assert len(confirming) == 4

    def test_primary_channel_excluded_from_own_check(self):
        """The primary channel must not count itself as a confirming channel."""
        channels = [
            _make_chan("360_SCALP_FVG", Direction.LONG),  # primary — should be skipped
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert "360_SCALP_FVG" not in confirming

    def test_non_scalp_channel_does_not_participate(self):
        """Non-scalp channels (e.g. 360_SPOT) must be ignored."""
        channels = [
            _make_chan("360_SCALP_FVG"),                   # primary
            _make_chan("360_SPOT", Direction.LONG),         # NOT a scalp channel
            _make_chan("360_SWING", Direction.LONG),        # NOT a scalp channel
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert confirming == []

    def test_non_scalp_primary_returns_zero(self):
        """If the primary channel is not a scalp channel, return 0 immediately."""
        channels = [
            _make_chan("360_SPOT"),
            _make_chan("360_SCALP_FVG", Direction.LONG),
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SPOT",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert confirming == []

    def test_opposite_direction_does_not_confirm(self):
        """A channel that signals in the opposite direction should not confirm."""
        channels = [
            _make_chan("360_SCALP_FVG"),                    # primary LONG
            _make_chan("360_SCALP_CVD", Direction.SHORT),   # opposite direction
        ]
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert confirming == []

    def test_exception_in_evaluate_is_silenced(self):
        """An exception raised by a channel's evaluate() should not propagate."""
        chan_error = MagicMock()
        chan_error.config.name = "360_SCALP_CVD"
        chan_error.evaluate.side_effect = RuntimeError("boom")

        channels = [
            _make_chan("360_SCALP_FVG"),   # primary
            chan_error,
        ]
        # Should not raise
        bonus, confirming = compute_confluence_bonus(
            primary_channel="360_SCALP_FVG",
            primary_direction=Direction.LONG.value,
            channels=channels,
            **_KWARGS,
        )
        assert bonus == 0.0
        assert confirming == []
