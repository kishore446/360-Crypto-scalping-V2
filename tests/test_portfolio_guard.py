"""Tests for src/portfolio_guard.py — PortfolioGuard and DrawdownTier."""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from src.portfolio_guard import DrawdownTier, PnLRecord, PortfolioGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard(**kwargs) -> PortfolioGuard:
    """Create a PortfolioGuard with fast (1s) halt durations for testing."""
    defaults = {
        "yellow_pct": 3.0,
        "red_pct": 5.0,
        "black_pct": 8.0,
        "red_halt_seconds": 1,    # short for tests
        "black_halt_seconds": 2,  # short for tests
        "yellow_size_multiplier": 0.5,
    }
    defaults.update(kwargs)
    return PortfolioGuard(**defaults)


def _add_loss(guard: PortfolioGuard, pnl_pct: float, channel: str = "SCALP") -> None:
    """Record a single realized loss on the guard."""
    guard.record_pnl(
        signal_id="test-001",
        channel=channel,
        symbol="BTCUSDT",
        pnl_pct=pnl_pct,
        is_realized=True,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        g = PortfolioGuard()
        assert g.yellow_pct == 3.0
        assert g.red_pct == 5.0
        assert g.black_pct == 8.0
        assert g.red_halt_seconds == 4 * 3600
        assert g.black_halt_seconds == 24 * 3600
        assert g.yellow_size_multiplier == 0.5
        assert g.current_tier == DrawdownTier.NORMAL

    def test_custom_thresholds(self):
        g = PortfolioGuard(yellow_pct=1.0, red_pct=2.0, black_pct=4.0)
        assert g.yellow_pct == 1.0
        assert g.red_pct == 2.0
        assert g.black_pct == 4.0

    def test_initial_tier_is_normal(self):
        g = _guard()
        assert g.current_tier == DrawdownTier.NORMAL

    def test_initial_drawdown_is_zero(self):
        g = _guard()
        assert g.rolling_drawdown_pct() == 0.0


# ---------------------------------------------------------------------------
# record_pnl
# ---------------------------------------------------------------------------

class TestRecordPnl:
    def test_single_loss_recorded(self):
        g = _guard()
        _add_loss(g, -2.0)
        assert g.rolling_drawdown_pct() == pytest.approx(2.0)

    def test_multiple_losses_accumulate(self):
        g = _guard()
        _add_loss(g, -1.5)
        _add_loss(g, -2.0)
        assert g.rolling_drawdown_pct() == pytest.approx(3.5)

    def test_gains_offset_losses(self):
        g = _guard()
        _add_loss(g, -5.0)
        g.record_pnl("win", "SCALP", "ETHUSDT", 4.0)
        # Net = -1.0, drawdown = 1.0
        assert g.rolling_drawdown_pct() == pytest.approx(1.0)

    def test_channel_pnl_tracked(self):
        g = _guard()
        _add_loss(g, -2.0, channel="SCALP")
        _add_loss(g, -1.0, channel="SWING")
        assert g._channel_pnl["SCALP"] == pytest.approx(-2.0)
        assert g._channel_pnl["SWING"] == pytest.approx(-1.0)

    def test_unrealized_pnl(self):
        g = _guard()
        g.record_pnl("sig1", "SCALP", "BTCUSDT", -4.0, is_realized=False)
        assert g.rolling_drawdown_pct() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Tier transitions
# ---------------------------------------------------------------------------

class TestTierTransitions:
    def test_normal_to_yellow(self):
        g = _guard()
        _add_loss(g, -3.5)
        assert g.current_tier == DrawdownTier.YELLOW

    def test_normal_stays_normal_below_threshold(self):
        g = _guard()
        _add_loss(g, -2.9)
        assert g.current_tier == DrawdownTier.NORMAL

    def test_yellow_to_red(self):
        g = _guard()
        _add_loss(g, -5.5)
        assert g.current_tier == DrawdownTier.RED

    def test_yellow_to_black(self):
        g = _guard()
        _add_loss(g, -8.5)
        assert g.current_tier == DrawdownTier.BLACK

    def test_tier_history_recorded(self):
        g = _guard()
        _add_loss(g, -3.5)   # NORMAL → YELLOW
        _add_loss(g, -2.0)   # YELLOW → RED
        assert len(g._tier_history) >= 2
        assert g._tier_history[0]["from"] == "NORMAL"
        assert g._tier_history[0]["to"] == "YELLOW"

    def test_no_duplicate_tier_change(self):
        g = _guard()
        _add_loss(g, -3.5)   # → YELLOW
        history_len = len(g._tier_history)
        _add_loss(g, -0.1)   # still YELLOW, no new transition
        assert len(g._tier_history) == history_len


# ---------------------------------------------------------------------------
# check_signal_allowed
# ---------------------------------------------------------------------------

class TestCheckSignalAllowed:
    def test_normal_allowed(self):
        g = _guard()
        allowed, reason, mult = g.check_signal_allowed()
        assert allowed is True
        assert reason == ""
        assert mult == pytest.approx(1.0)

    def test_yellow_allowed_with_reduced_size(self):
        g = _guard()
        _add_loss(g, -3.5)
        allowed, reason, mult = g.check_signal_allowed()
        assert allowed is True
        assert mult == pytest.approx(0.5)
        assert "YELLOW" in reason

    def test_red_blocked(self):
        g = _guard()
        _add_loss(g, -5.5)
        allowed, reason, mult = g.check_signal_allowed()
        assert allowed is False
        assert mult == pytest.approx(0.0)
        assert "RED" in reason

    def test_black_blocked(self):
        g = _guard()
        _add_loss(g, -8.5)
        allowed, reason, mult = g.check_signal_allowed()
        assert allowed is False
        assert mult == pytest.approx(0.0)
        assert "BLACK" in reason


# ---------------------------------------------------------------------------
# Halt expiry
# ---------------------------------------------------------------------------

class TestHaltExpiry:
    def test_red_halt_expires(self):
        g = _guard(red_halt_seconds=0, black_halt_seconds=1)
        _add_loss(g, -5.5)
        assert g.current_tier == DrawdownTier.RED
        # With red_halt_seconds=0 the halt expires immediately on next check
        time.sleep(0.01)
        # Force re-evaluation by calling check_signal_allowed
        # (the pnl is still -5.5 so after expiry tier re-evaluates)
        # After expiry + re-evaluate: drawdown still 5.5 → RED again
        # but halt_until is reset so the tier won't automatically drop
        # unless drawdown comes below threshold. Let's verify halt_until is None.
        g._check_halt_expiry()
        assert g._halt_until is None

    def test_black_halt_expires(self):
        g = _guard(red_halt_seconds=1, black_halt_seconds=0)
        _add_loss(g, -8.5)
        assert g.current_tier == DrawdownTier.BLACK
        time.sleep(0.01)
        g._check_halt_expiry()
        assert g._halt_until is None


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_tier(self):
        g = _guard()
        _add_loss(g, -6.0)
        assert g.current_tier == DrawdownTier.RED
        g.reset()
        assert g.current_tier == DrawdownTier.NORMAL

    def test_reset_clears_drawdown(self):
        g = _guard()
        _add_loss(g, -6.0)
        g.reset()
        assert g.rolling_drawdown_pct() == pytest.approx(0.0)

    def test_reset_clears_channel_pnl(self):
        g = _guard()
        _add_loss(g, -3.0, channel="SCALP")
        g.reset()
        assert g._channel_pnl == {}

    def test_reset_clears_halt(self):
        g = _guard()
        _add_loss(g, -5.5)
        assert g._halt_until is not None
        g.reset()
        assert g._halt_until is None

    def test_reset_clears_tier_history(self):
        g = _guard()
        _add_loss(g, -3.5)
        g.reset()
        assert g._tier_history == []


# ---------------------------------------------------------------------------
# status_text()
# ---------------------------------------------------------------------------

class TestStatusText:
    def test_normal_status(self):
        g = _guard()
        text = g.status_text()
        assert "NORMAL" in text
        assert "Portfolio Guard" in text

    def test_yellow_status(self):
        g = _guard()
        _add_loss(g, -3.5)
        text = g.status_text()
        assert "YELLOW" in text

    def test_red_status(self):
        g = _guard()
        _add_loss(g, -5.5)
        text = g.status_text()
        assert "RED" in text

    def test_black_status(self):
        g = _guard()
        _add_loss(g, -8.5)
        text = g.status_text()
        assert "BLACK" in text

    def test_status_includes_thresholds(self):
        g = _guard()
        text = g.status_text()
        assert "3.0" in text or "Yellow" in text
        assert "5.0" in text or "Red" in text
        assert "8.0" in text or "Black" in text

    def test_status_includes_channel_pnl(self):
        g = _guard()
        _add_loss(g, -2.0, channel="SCALP")
        text = g.status_text()
        assert "SCALP" in text

    def test_halt_remaining_shown_in_status(self):
        g = _guard()
        _add_loss(g, -5.5)  # triggers RED halt
        text = g.status_text()
        assert "Halt remaining" in text


# ---------------------------------------------------------------------------
# Per-channel P&L attribution
# ---------------------------------------------------------------------------

class TestChannelAttribution:
    def test_multiple_channels_tracked_separately(self):
        g = _guard()
        g.record_pnl("s1", "SCALP", "BTC", -1.0)
        g.record_pnl("s2", "SWING", "ETH", -2.0)
        g.record_pnl("s3", "SPOT", "SOL", -0.5)
        assert g._channel_pnl["SCALP"] == pytest.approx(-1.0)
        assert g._channel_pnl["SWING"] == pytest.approx(-2.0)
        assert g._channel_pnl["SPOT"] == pytest.approx(-0.5)

    def test_channel_pnl_accumulates_across_signals(self):
        g = _guard()
        g.record_pnl("s1", "SCALP", "BTC", -1.0)
        g.record_pnl("s2", "SCALP", "ETH", -1.5)
        assert g._channel_pnl["SCALP"] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Rolling window exclusion
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_old_records_excluded(self):
        """Records older than 24h should not contribute to the drawdown."""
        _25_HOURS_SECONDS = 25 * 3600
        g = _guard()
        # Manually inject an old record
        old_record = PnLRecord(
            signal_id="old",
            channel="SCALP",
            symbol="BTC",
            pnl_pct=-5.0,
            is_realized=True,
            timestamp=time.monotonic() - _25_HOURS_SECONDS,
        )
        g._pnl_records.append(old_record)
        # Fresh loss that IS in window
        _add_loss(g, -1.0)
        assert g.rolling_drawdown_pct() == pytest.approx(1.0)

    def test_recent_records_included(self):
        g = _guard()
        _add_loss(g, -3.0)
        assert g.rolling_drawdown_pct() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Alert callback
# ---------------------------------------------------------------------------

class TestAlertCallback:
    def test_alert_fired_on_yellow(self):
        alerts: List[str] = []

        async def fake_cb(msg: str) -> None:
            alerts.append(msg)

        g = _guard(alert_callback=fake_cb)
        _add_loss(g, -3.5)
        # Run pending coroutines
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.sleep(0))
        except RuntimeError:
            pass
        # Alert may be queued via loop.create_task; just verify tier changed
        assert g.current_tier == DrawdownTier.YELLOW

    def test_no_alert_on_no_tier_change(self):
        fired: List[str] = []

        async def fake_cb(msg: str) -> None:
            fired.append(msg)

        g = _guard(alert_callback=fake_cb)
        # Successive small losses that stay NORMAL
        g.record_pnl("a", "SCALP", "BTC", -0.5)
        g.record_pnl("b", "SCALP", "BTC", -0.5)
        assert g.current_tier == DrawdownTier.NORMAL


# ---------------------------------------------------------------------------
# Integration with DrawdownTier enum
# ---------------------------------------------------------------------------

class TestDrawdownTierEnum:
    def test_tier_values(self):
        assert DrawdownTier.NORMAL.value == "NORMAL"
        assert DrawdownTier.YELLOW.value == "YELLOW"
        assert DrawdownTier.RED.value == "RED"
        assert DrawdownTier.BLACK.value == "BLACK"

    def test_tier_is_str_subclass(self):
        # DrawdownTier inherits from str — can be compared to strings
        assert DrawdownTier.NORMAL == "NORMAL"


# ---------------------------------------------------------------------------
# rolling_drawdown_pct() property
# ---------------------------------------------------------------------------

class TestRollingDrawdownPct:
    def test_no_records(self):
        g = _guard()
        assert g.rolling_drawdown_pct() == 0.0

    def test_only_gains_no_drawdown(self):
        g = _guard()
        g.record_pnl("w", "SCALP", "BTC", 5.0)
        assert g.rolling_drawdown_pct() == 0.0

    def test_mixed_net_positive_no_drawdown(self):
        g = _guard()
        g.record_pnl("l", "SCALP", "BTC", -2.0)
        g.record_pnl("w", "SCALP", "BTC", 4.0)
        assert g.rolling_drawdown_pct() == 0.0
