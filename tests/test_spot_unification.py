"""Tests for SpotChannel unification with build_channel_signal (Phase 2B)."""

from __future__ import annotations

import numpy as np
import pytest

from src.channels.spot import SpotChannel
from src.smc import Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_candles(n: int = 60, base: float = 100.0):
    """Candles where the last bar is a confirmed LONG breakout above recent highs."""
    closes = np.cumsum(np.ones(n) * 0.1) + base
    # Last close clearly above the previous 9 bars' high
    closes[-1] = max(float(h) for h in closes[-10:-1]) + 2.0
    highs = closes + 0.5
    lows = closes - 0.5
    # Volume: last bar 2× the 9-bar average (above 1.8× expansion threshold)
    volumes = np.ones(n) * 1000.0
    volumes[-1] = float(np.mean(volumes[:-1])) * 2.0
    return {
        "open": closes - 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def _make_short_candles(n: int = 60, base: float = 100.0):
    """Candles where the last bar is a confirmed SHORT breakdown below recent lows."""
    closes = np.cumsum(np.ones(n) * -0.1) + base
    # Last close clearly below the previous 9 bars' low
    closes[-1] = min(float(c) for c in closes[-10:-1]) - 2.0
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.ones(n) * 1000.0
    volumes[-1] = float(np.mean(volumes[:-1])) * 2.0
    return {
        "open": closes + 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def _make_indicators(adx_val: float = 25, ema200: float = 90.0):
    return {
        "adx_last": adx_val,
        "atr_last": 0.5,
        "ema200_last": ema200,
        "rsi_last": 50.0,
        "bb_width_pct": 2.0,
    }


def _get_long_signal():
    """Build a valid Spot LONG signal via evaluate()."""
    ch = SpotChannel()
    candles_data = _make_long_candles()
    # ema200 well below close so LONG path fires
    indicators = {"4h": _make_indicators(adx_val=25, ema200=90.0)}
    candles = {"4h": candles_data}
    return ch.evaluate("BTCUSDT", candles, indicators, {}, 0.01, 5_000_000,
                       regime="TRENDING_UP")


def _get_short_signal():
    """Build a valid Spot SHORT signal via evaluate()."""
    ch = SpotChannel()
    candles_data = _make_short_candles(base=110.0)
    close_last = float(candles_data["close"][-1])
    # ema200 above close AND ema50_daily above close so SHORT path fires
    ema200 = close_last + 5.0
    indicators_4h = {
        "adx_last": 25.0,
        "atr_last": 0.5,
        "ema200_last": ema200,
        "rsi_last": 50.0,
        "bb_width_pct": 2.0,
    }
    ema50_daily = close_last + 3.0
    indicators = {
        "4h": indicators_4h,
        "1d": {"ema50_last": ema50_daily},
    }
    candles = {"4h": candles_data}
    return ch.evaluate("BTCUSDT", candles, indicators, {}, 0.01, 5_000_000,
                       regime="TRENDING_DOWN")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpotNoBuildSignalMethod:
    def test_spot_channel_no_build_signal_method(self):
        """SpotChannel should no longer have a _build_signal method."""
        assert not hasattr(SpotChannel, "_build_signal")


class TestSpotLongSignal:
    def test_spot_long_signal_has_entry_zone(self):
        """Spot LONG signals should have entry_zone_low/high set to DCA zone."""
        sig = _get_long_signal()
        assert sig is not None, "Expected a LONG signal to be generated"
        assert sig.entry_zone_low is not None
        assert sig.entry_zone_high is not None
        assert sig.entry_zone_low == pytest.approx(sig.dca_zone_lower)
        assert sig.entry_zone_high == pytest.approx(sig.dca_zone_upper)

    def test_spot_long_signal_has_original_values(self):
        """Spot LONG signals must always set original_entry and original_tp*."""
        sig = _get_long_signal()
        assert sig is not None
        assert sig.original_entry == pytest.approx(sig.entry)
        assert sig.original_tp1 == pytest.approx(sig.tp1)
        assert sig.original_tp2 == pytest.approx(sig.tp2)

    def test_spot_long_signal_has_setup_class(self):
        """Spot LONG signals should have setup_class set."""
        sig = _get_long_signal()
        assert sig is not None
        assert sig.setup_class in ("BREAKOUT_RETEST", "BREAKOUT_INITIAL")

    def test_spot_long_signal_has_correct_direction(self):
        sig = _get_long_signal()
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_spot_long_signal_id_prefix(self):
        """Spot LONG signal IDs should start with 'SPOT-' (not 'SPOT-SHORT-')."""
        sig = _get_long_signal()
        assert sig is not None
        assert sig.signal_id.startswith("SPOT-")
        assert not sig.signal_id.startswith("SPOT-SHORT-")


class TestSpotShortSignal:
    def test_spot_short_signal_no_dca_zone(self):
        """Spot SHORT signals should NOT have DCA zone (Spot-specific behaviour preserved)."""
        sig = _get_short_signal()
        assert sig is not None, "Expected a SHORT signal to be generated"
        assert sig.dca_zone_lower == pytest.approx(0.0)
        assert sig.dca_zone_upper == pytest.approx(0.0)

    def test_spot_short_signal_has_entry_zone(self):
        """Spot SHORT signals should have entry_zone_low/high set via build_channel_signal."""
        sig = _get_short_signal()
        assert sig is not None
        assert sig.entry_zone_low is not None
        assert sig.entry_zone_high is not None

    def test_spot_short_signal_has_confidence_boost(self):
        """Spot SHORT signals must include the _SHORT_CONFIDENCE_BOOST offset."""
        from src.channels.spot import _SHORT_CONFIDENCE_BOOST
        sig = _get_short_signal()
        assert sig is not None
        assert sig.confidence >= _SHORT_CONFIDENCE_BOOST

    def test_spot_short_signal_has_setup_class(self):
        """Spot SHORT signals should have setup_class set."""
        sig = _get_short_signal()
        assert sig is not None
        assert sig.setup_class in ("BREAKOUT_RETEST", "BREAKOUT_INITIAL")

    def test_spot_short_signal_id_prefix(self):
        """Spot SHORT signal IDs should start with 'SPOT-SHORT-'."""
        sig = _get_short_signal()
        assert sig is not None
        assert sig.signal_id.startswith("SPOT-SHORT-")

    def test_spot_short_signal_has_correct_direction(self):
        sig = _get_short_signal()
        assert sig is not None
        assert sig.direction == Direction.SHORT
