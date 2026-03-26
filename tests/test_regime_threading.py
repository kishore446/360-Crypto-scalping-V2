"""Tests that regime is threaded from scanner through channels to build_channel_signal."""

from __future__ import annotations

import pytest
import numpy as np

from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.spot import SpotChannel
from src.channels.scalp_cvd import ScalpCVDChannel
from src.channels.scalp_fvg import ScalpFVGChannel
from src.channels.scalp_obi import ScalpOBIChannel
from src.channels.scalp_vwap import ScalpVWAPChannel
from src.smc import Direction, LiquiditySweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n=60, base=100.0, trend=0.1):
    """Create synthetic OHLCV candle data."""
    close = np.cumsum(np.ones(n) * trend) + base
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000
    return {"open": close - 0.1, "high": high, "low": low, "close": close, "volume": volume}


def _make_indicators(adx_val=30, atr_val=0.5, ema9=101, ema21=100, ema200=95,
                     rsi_val=50, bb_upper=103, bb_mid=100, bb_lower=97, mom=0.5):
    return {
        "adx_last": adx_val,
        "atr_last": atr_val,
        "ema9_last": ema9,
        "ema21_last": ema21,
        "ema200_last": ema200,
        "rsi_last": rsi_val,
        "bb_upper_last": bb_upper,
        "bb_mid_last": bb_mid,
        "bb_lower_last": bb_lower,
        "momentum_last": mom,
    }


# ---------------------------------------------------------------------------
# Tests: each channel evaluate() accepts regime parameter
# ---------------------------------------------------------------------------

class TestScalpChannelRegimeParam:
    def test_evaluate_accepts_regime_parameter(self):
        """ScalpChannel.evaluate() should accept regime kwarg without TypeError."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {"5m": _make_indicators(adx_val=30, mom=0.5, ema9=101, ema21=100)}
        smc_data = {"sweeps": [sweep]}
        # Should not raise TypeError for unknown keyword argument
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000,
                          regime="TRENDING_UP")
        # Signal may or may not be generated — just verifying regime is accepted
        assert sig is None or sig.channel == "360_SCALP"

    def test_evaluate_regime_default_empty_string(self):
        """ScalpChannel.evaluate() works without regime (backward compatible)."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {"5m": _make_indicators(adx_val=30, mom=0.5, ema9=101, ema21=100)}
        smc_data = {"sweeps": [sweep]}
        # Calling without regime should still work
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None or sig.channel == "360_SCALP"


class TestSwingChannelRegimeParam:
    def test_evaluate_accepts_regime_parameter(self):
        """SwingChannel.evaluate() should accept regime kwarg without TypeError."""
        ch = SwingChannel()
        candles = {
            "4h": _make_candles(60, base=100.0),
            "1h": _make_candles(60, base=100.0),
        }
        indicators = {
            "4h": _make_indicators(adx_val=30),
            "1h": _make_indicators(adx_val=30, ema200=95),
        }
        smc_data = {"sweeps": [], "mss": None}
        # Should not raise TypeError
        try:
            ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000,
                        regime="TRENDING_UP")
        except TypeError:
            pytest.fail("SwingChannel.evaluate() does not accept regime parameter")
        except Exception:
            pass  # Other errors are fine (missing data etc)

    def test_evaluate_regime_default_empty_string(self):
        """SwingChannel.evaluate() works without regime (backward compatible)."""
        ch = SwingChannel()
        candles = {
            "4h": _make_candles(60, base=100.0),
            "1h": _make_candles(60, base=100.0),
        }
        indicators = {
            "4h": _make_indicators(adx_val=30),
            "1h": _make_indicators(adx_val=30, ema200=95),
        }
        smc_data = {"sweeps": [], "mss": None}
        try:
            ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        except TypeError:
            pytest.fail("SwingChannel.evaluate() is not backward compatible without regime")
        except Exception:
            pass


class TestSpotChannelRegimeParam:
    def test_evaluate_accepts_regime_parameter(self):
        """SpotChannel.evaluate() should accept regime kwarg without TypeError."""
        ch = SpotChannel()
        candles = {"4h": _make_candles(60, base=100.0)}
        indicators = {"4h": _make_indicators(adx_val=30, ema200=95)}
        smc_data = {}
        try:
            ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000,
                        regime="RANGING")
        except TypeError:
            pytest.fail("SpotChannel.evaluate() does not accept regime parameter")
        except Exception:
            pass

    def test_evaluate_stores_regime_as_current_regime(self):
        """SpotChannel.evaluate() stores regime in self._current_regime."""
        ch = SpotChannel()
        candles = {"4h": _make_candles(60, base=100.0)}
        indicators = {"4h": _make_indicators(adx_val=30, ema200=95)}
        smc_data = {}
        try:
            ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000,
                        regime="VOLATILE")
        except Exception:
            pass  # May raise due to missing data, but _current_regime should be set
        assert ch._current_regime == "VOLATILE"

    def test_evaluate_regime_default_empty_string(self):
        """SpotChannel.evaluate() works without regime (backward compatible)."""
        ch = SpotChannel()
        candles = {"4h": _make_candles(60, base=100.0)}
        indicators = {"4h": _make_indicators(adx_val=30, ema200=95)}
        smc_data = {}
        try:
            ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        except TypeError:
            pytest.fail("SpotChannel.evaluate() is not backward compatible without regime")
        except Exception:
            pass


class TestAllSubChannelsRegimeParam:
    """All scalp sub-channels should accept a regime parameter."""

    @pytest.mark.parametrize("ChannelClass", [
        ScalpCVDChannel,
        ScalpFVGChannel,
        ScalpOBIChannel,
        ScalpVWAPChannel,
    ])
    def test_evaluate_accepts_regime_parameter(self, ChannelClass):
        """Channel.evaluate() should accept regime kwarg without TypeError."""
        ch = ChannelClass()
        try:
            ch.evaluate("BTCUSDT", {}, {}, {}, 0.01, 10_000_000, regime="RANGING")
        except TypeError as exc:
            pytest.fail(
                f"{ChannelClass.__name__}.evaluate() does not accept regime parameter: {exc}"
            )
        except Exception:
            pass  # Other errors are fine (missing data etc) — we only check the signature

    @pytest.mark.parametrize("ChannelClass", [
        ScalpCVDChannel,
        ScalpFVGChannel,
        ScalpOBIChannel,
        ScalpVWAPChannel,
    ])
    def test_evaluate_backward_compatible_without_regime(self, ChannelClass):
        """Calling evaluate() without regime should still work (backward compatible)."""
        ch = ChannelClass()
        try:
            ch.evaluate("BTCUSDT", {}, {}, {}, 0.01, 10_000_000)
        except TypeError as exc:
            pytest.fail(
                f"{ChannelClass.__name__}.evaluate() broke backward compatibility: {exc}"
            )
        except Exception:
            pass  # Other errors are fine
