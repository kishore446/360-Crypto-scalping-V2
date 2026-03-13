"""Tests for src.select_mode – SelectModeFilter ultra-selective signal gating."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.select_mode import SelectModeConfig, SelectModeFilter
from src.smc import Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    symbol: str = "BTCUSDT",
    direction: Direction = Direction.LONG,
    channel: str = "360_SCALP",
) -> MagicMock:
    sig = MagicMock()
    sig.symbol = symbol
    sig.direction = direction
    sig.channel = channel
    return sig


def _default_indicators(direction: str = "LONG") -> dict:
    """Return a minimal indicators dict that passes all EMA/momentum filters."""
    ema9 = 100.0 if direction == "LONG" else 90.0
    ema21 = 90.0 if direction == "LONG" else 100.0
    return {
        "5m": {
            "adx_last": 30.0,
            "rsi_last": 50.0,
            "ema9_last": ema9,
            "ema21_last": ema21,
            "momentum_last": 1.0 if direction == "LONG" else -1.0,
        },
        "1m": {
            "adx_last": 30.0,
            "rsi_last": 50.0,
            "ema9_last": ema9,
            "ema21_last": ema21,
            "momentum_last": 1.0 if direction == "LONG" else -1.0,
        },
    }


def _default_smc(has_sweep: bool = True, has_fvg: bool = False, has_mss: bool = False) -> dict:
    return {
        "sweeps": [object()] if has_sweep else [],
        "fvg": [object()] if has_fvg else [],
        "mss": object() if has_mss else None,
    }


def _make_filter(enabled: bool = True, **kwargs) -> SelectModeFilter:
    config = SelectModeConfig(enabled=enabled, **kwargs)
    return SelectModeFilter(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectModeOffAlwaysPasses:
    """When select mode is OFF, should_publish must always return True."""

    def test_off_by_default(self):
        f = SelectModeFilter()
        assert not f.enabled

    def test_off_passes_any_signal(self):
        f = SelectModeFilter()
        sig = _make_signal()
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=10.0,
            indicators={},
            smc_data={},
            ai_sentiment={"label": "bearish"},
            cross_exchange_verified=False,
            volume_24h=0.0,
            spread_pct=99.0,
        )
        assert allowed is True
        assert reason == ""

    def test_enable_disable_toggle(self):
        f = SelectModeFilter()
        f.enable()
        assert f.enabled
        f.disable()
        assert not f.enabled


class TestSelectModeOnFilters:
    """Individual filter checks when select mode is ON."""

    def test_low_confidence_rejected(self):
        f = _make_filter(min_confidence=80.0)
        sig = _make_signal()
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=70.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "bullish"},
            cross_exchange_verified=True,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "Confidence" in reason

    def test_sufficient_confidence_passes(self):
        f = _make_filter(
            min_confidence=80.0,
            require_smc_event=False,
            require_ai_sentiment_match=False,
            require_cross_exchange=False,
        )
        sig = _make_signal()
        allowed, _ = f.should_publish(
            signal=sig,
            confidence=85.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert allowed

    def test_no_smc_event_rejected(self):
        f = _make_filter(require_smc_event=True)
        sig = _make_signal()
        smc_no_events = {"sweeps": [], "fvg": [], "mss": None}
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators(),
            smc_data=smc_no_events,
            ai_sentiment={"label": "bullish"},
            cross_exchange_verified=True,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "SMC" in reason

    def test_adx_below_threshold_rejected(self):
        f = _make_filter(min_adx=25.0, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal()
        inds = _default_indicators()
        inds["5m"]["adx_last"] = 10.0  # below threshold
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=inds,
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "ADX" in reason

    def test_spread_too_high_rejected(self):
        f = _make_filter(max_spread_pct=0.015, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal()
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.02,  # too high
        )
        assert not allowed
        assert "Spread" in reason

    def test_low_volume_rejected(self):
        f = _make_filter(min_volume_24h=10_000_000.0, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal()
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=500_000.0,  # too low
            spread_pct=0.005,
        )
        assert not allowed
        assert "Volume" in reason

    def test_rsi_extreme_high_rejected(self):
        f = _make_filter(rsi_max=70.0, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal()
        inds = _default_indicators()
        inds["5m"]["rsi_last"] = 80.0  # above max
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=inds,
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "RSI" in reason

    def test_rsi_none_does_not_block(self):
        """Missing RSI data must not block the signal."""
        f = _make_filter(require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal()
        inds = _default_indicators()
        inds["5m"]["rsi_last"] = None
        allowed, _ = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=inds,
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert allowed

    def test_ai_sentiment_mismatch_rejected(self):
        f = _make_filter(require_ai_sentiment_match=True, require_smc_event=False, require_cross_exchange=False)
        sig = _make_signal(direction=Direction.LONG)
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators("LONG"),
            smc_data=_default_smc(),
            ai_sentiment={"label": "bearish"},  # mismatch
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "sentiment" in reason.lower()

    def test_ai_sentiment_neutral_passes(self):
        f = _make_filter(require_ai_sentiment_match=True, require_smc_event=False, require_cross_exchange=False)
        sig = _make_signal(direction=Direction.LONG)
        allowed, _ = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators("LONG"),
            smc_data=_default_smc(),
            ai_sentiment={"label": "Neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert allowed

    def test_cross_exchange_failed_rejected(self):
        f = _make_filter(require_cross_exchange=True, require_smc_event=False, require_ai_sentiment_match=False)
        sig = _make_signal()
        allowed, reason = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=False,  # failed
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert not allowed
        assert "Cross-exchange" in reason

    def test_cross_exchange_none_does_not_block(self):
        """None means no second exchange is configured – should not block."""
        f = _make_filter(require_cross_exchange=True, require_smc_event=False, require_ai_sentiment_match=False)
        sig = _make_signal()
        allowed, _ = f.should_publish(
            signal=sig,
            confidence=90.0,
            indicators=_default_indicators(),
            smc_data=_default_smc(),
            ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None,
            volume_24h=20_000_000.0,
            spread_pct=0.005,
        )
        assert allowed


class TestDailyCapEnforcement:
    """Daily signal cap should prevent signals above the per-channel limit."""

    def test_daily_cap_blocks_after_limit(self):
        f = _make_filter(max_daily_signals=2, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal(channel="360_SCALP")

        # Pass once
        allowed1, _ = f.should_publish(
            signal=sig, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        # Pass twice
        allowed2, _ = f.should_publish(
            signal=sig, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        # Third should be blocked
        allowed3, reason3 = f.should_publish(
            signal=sig, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        assert allowed1
        assert allowed2
        assert not allowed3
        assert "Daily cap" in reason3

    def test_daily_cap_per_channel_independent(self):
        """Different channels have independent daily counters."""
        f = _make_filter(max_daily_signals=1, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig_scalp = _make_signal(channel="360_SCALP")
        sig_swing = _make_signal(channel="360_SWING")

        # Both pass first time
        a1, _ = f.should_publish(
            signal=sig_scalp, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        a2, _ = f.should_publish(
            signal=sig_swing, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        assert a1
        assert a2


class TestMultiTFConfluence:
    """Multi-timeframe EMA confluence filter."""

    def test_insufficient_confluence_rejected(self):
        f = _make_filter(min_confluence_timeframes=2, require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal(direction=Direction.LONG)
        # Only 1m agrees, 5m disagrees
        inds = {
            "1m": {"ema9_last": 110.0, "ema21_last": 90.0, "adx_last": 30.0, "rsi_last": 50.0, "momentum_last": 1.0},
            "5m": {"ema9_last": 80.0, "ema21_last": 100.0, "adx_last": 30.0, "rsi_last": 50.0, "momentum_last": 1.0},
        }
        allowed, reason = f.should_publish(
            signal=sig, confidence=90.0, indicators=inds,
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        assert not allowed
        assert "EMA confluence" in reason


class TestUpdateConfig:
    """update_config should update config attributes correctly."""

    def test_update_confidence(self):
        f = _make_filter()
        ok, msg = f.update_config("confidence", "85.0")
        assert ok
        assert f.config.min_confidence == 85.0

    def test_update_daily_cap(self):
        f = _make_filter()
        ok, msg = f.update_config("daily_cap", "3")
        assert ok
        assert f.config.max_daily_signals == 3

    def test_update_adx(self):
        f = _make_filter()
        ok, msg = f.update_config("adx", "30.0")
        assert ok
        assert f.config.min_adx == 30.0

    def test_unknown_key_returns_error(self):
        f = _make_filter()
        ok, msg = f.update_config("unknown_key", "10")
        assert not ok
        assert "Unknown config key" in msg

    def test_invalid_value_returns_error(self):
        f = _make_filter()
        ok, msg = f.update_config("confidence", "not_a_number")
        assert not ok
        assert "Invalid value" in msg


class TestStatusText:
    """status_text should return the correct mode string."""

    def test_status_off(self):
        f = SelectModeFilter()
        text = f.status_text()
        assert "OFF" in text

    def test_status_on(self):
        f = _make_filter()
        text = f.status_text()
        assert "ON" in text
        assert "ultra-selective" in text.lower()

    def test_status_shows_today_totals(self):
        """After passing a signal, today's count should appear in status."""
        f = _make_filter(require_smc_event=False, require_ai_sentiment_match=False, require_cross_exchange=False)
        sig = _make_signal(channel="360_SCALP")
        f.should_publish(
            signal=sig, confidence=90.0, indicators=_default_indicators(),
            smc_data=_default_smc(), ai_sentiment={"label": "neutral"},
            cross_exchange_verified=None, volume_24h=20_000_000.0, spread_pct=0.005,
        )
        text = f.status_text()
        assert "360_SCALP" in text
