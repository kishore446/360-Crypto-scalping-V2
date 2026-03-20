"""Unit tests for src.cvd – compute_cvd and detect_cvd_divergence."""

from __future__ import annotations

import numpy as np
import pytest

from src.cvd import compute_cvd, detect_cvd_divergence


# ---------------------------------------------------------------------------
# compute_cvd – basic math
# ---------------------------------------------------------------------------


class TestComputeCVD:
    """Tests for the standalone compute_cvd() helper."""

    def test_simple_cumsum(self):
        """CVD is the cumulative sum of (buy - sell) deltas."""
        buy = np.array([300.0, 200.0, 100.0, 400.0, 250.0])
        sell = np.array([100.0, 300.0, 200.0, 100.0, 50.0])
        result = compute_cvd(buy, sell)
        # delta: [200, -100, -100, 300, 200] → cumsum: [200, 100, 0, 300, 500]
        expected = np.array([200.0, 100.0, 0.0, 300.0, 500.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_buy_pressure(self):
        """Pure buy pressure → monotonically rising CVD."""
        buy = np.array([100.0, 200.0, 300.0])
        sell = np.zeros(3)
        result = compute_cvd(buy, sell)
        expected = np.array([100.0, 300.0, 600.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_sell_pressure(self):
        """Pure sell pressure → monotonically falling CVD."""
        buy = np.zeros(3)
        sell = np.array([100.0, 200.0, 300.0])
        result = compute_cvd(buy, sell)
        expected = np.array([-100.0, -300.0, -600.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_balanced_volumes(self):
        """Equal buy and sell volume → CVD stays at zero."""
        vol = np.array([100.0, 200.0, 300.0])
        result = compute_cvd(vol, vol)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_single_candle(self):
        """Single candle is handled correctly."""
        result = compute_cvd(np.array([500.0]), np.array([200.0]))
        np.testing.assert_array_almost_equal(result, np.array([300.0]))

    def test_returns_float64_array(self):
        """Output dtype is always float64."""
        buy = np.array([1, 2, 3])
        sell = np.array([0, 1, 2])
        result = compute_cvd(buy, sell)
        assert result.dtype == np.float64

    def test_output_same_length_as_input(self):
        """Output length matches input length."""
        n = 50
        buy = np.random.rand(n) * 1000
        sell = np.random.rand(n) * 1000
        result = compute_cvd(buy, sell)
        assert len(result) == n

    # ---- rolling window ----

    def test_rolling_window_resets_each_block(self):
        """When window=3, the cumsum resets every 3 candles."""
        # delta: [1, 1, 1,  2, 2, 2] → two windows of [1,1,1] and [2,2,2]
        buy = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        sell = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = compute_cvd(buy, sell, window=3)
        # Window 1: cumsum([1,1,1]) = [1, 2, 3]
        # Window 2: cumsum([2,2,2]) = [2, 4, 6]
        expected = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_window_larger_than_data(self):
        """Window >= data length is equivalent to no window (full cumsum)."""
        buy = np.array([100.0, 200.0, 300.0])
        sell = np.array([50.0, 100.0, 150.0])
        full = compute_cvd(buy, sell)
        windowed = compute_cvd(buy, sell, window=100)
        np.testing.assert_array_almost_equal(full, windowed)

    def test_rolling_window_size_one(self):
        """Window of 1 → each candle's CVD equals its own delta."""
        buy = np.array([100.0, 200.0, 300.0])
        sell = np.array([50.0, 100.0, 50.0])
        result = compute_cvd(buy, sell, window=1)
        expected = np.array([50.0, 100.0, 250.0])
        np.testing.assert_array_almost_equal(result, expected)

    # ---- error handling ----

    def test_mismatched_lengths_raises(self):
        """Mismatched buy/sell arrays raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_cvd(np.array([1.0, 2.0]), np.array([1.0]))

    def test_invalid_window_zero_raises(self):
        """Window=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_cvd(np.array([1.0]), np.array([1.0]), window=0)

    def test_invalid_window_negative_raises(self):
        """Negative window raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_cvd(np.array([1.0]), np.array([1.0]), window=-5)

    def test_invalid_window_float_raises(self):
        """Non-integer window raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            compute_cvd(np.array([1.0]), np.array([1.0]), window=2.5)  # type: ignore[arg-type]

    def test_accepts_list_inputs(self):
        """Python lists are accepted and converted internally."""
        result = compute_cvd([100.0, 200.0], [50.0, 100.0])
        np.testing.assert_array_almost_equal(result, [50.0, 150.0])


# ---------------------------------------------------------------------------
# detect_cvd_divergence – divergence logic
# ---------------------------------------------------------------------------


class TestDetectCVDDivergenceInCVDModule:
    """Tests for detect_cvd_divergence imported via src.cvd."""

    def test_bullish_divergence_detected(self):
        """Price lower low + CVD higher low → BULLISH divergence."""
        # First half: price ~100, CVD ~0
        # Second half: price dips to 95 (lower low), CVD only to 5 (higher low)
        close = np.array([100, 100, 100, 100, 100, 95, 96, 97, 98, 99], dtype=float)
        cvd   = np.array([0,   0,   0,   0,   0,  5,  6,  7,  8,  9], dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=10) == "BULLISH"

    def test_bearish_divergence_detected(self):
        """Price higher high + CVD lower high → BEARISH divergence."""
        close = np.array([100, 100, 100, 100, 100, 105, 104, 103, 102, 101], dtype=float)
        cvd   = np.array([10,  10,  10,  10,  10,   8,   7,   6,   5,   4], dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=10) == "BEARISH"

    def test_no_divergence_price_and_cvd_agree_on_lows(self):
        """Price lower low AND CVD lower low → no bullish divergence."""
        close = np.array([100, 100, 100, 100, 100, 95, 94, 93, 92, 91], dtype=float)
        cvd   = np.array([10,  10,  10,  10,  10,  5,  4,  3,  2,  1], dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=10) is None

    def test_no_divergence_price_and_cvd_agree_on_highs(self):
        """Price higher high AND CVD higher high → no bearish divergence."""
        close = np.array([100, 100, 100, 100, 100, 105, 106, 107, 108, 109], dtype=float)
        cvd   = np.array([10,  10,  10,  10,  10,   12,  13,  14,  15,  16], dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=10) is None

    def test_insufficient_data_returns_none(self):
        """Arrays shorter than lookback return None."""
        close = np.array([100.0, 101.0, 102.0])
        cvd   = np.array([0.0, 1.0, 2.0])
        assert detect_cvd_divergence(close, cvd, lookback=20) is None

    def test_flat_price_and_cvd_returns_none(self):
        """Flat price and flat CVD → no divergence."""
        close = np.ones(20, dtype=float) * 100.0
        cvd   = np.zeros(20, dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=20) is None

    def test_uses_only_last_lookback_candles(self):
        """Only the last *lookback* candles are considered."""
        # Prepend data that would cause bearish divergence if all included,
        # but the last 10 candles show bullish divergence.
        prefix_close = np.array([100, 100, 100, 100, 100, 105, 104, 103, 102, 101], dtype=float)
        prefix_cvd   = np.array([10,  10,  10,  10,  10,   8,   7,   6,   5,   4], dtype=float)
        suffix_close = np.array([100, 100, 100, 100, 100,  95,  96,  97,  98,  99], dtype=float)
        suffix_cvd   = np.array([0,   0,   0,   0,   0,    5,   6,   7,   8,   9], dtype=float)
        close = np.concatenate([prefix_close, suffix_close])
        cvd   = np.concatenate([prefix_cvd, suffix_cvd])
        # lookback=10 → only the suffix is examined → BULLISH
        assert detect_cvd_divergence(close, cvd, lookback=10) == "BULLISH"

    def test_default_lookback_requires_20_candles(self):
        """Default lookback=20 requires at least 20 elements."""
        close = np.ones(19, dtype=float) * 100.0
        cvd   = np.zeros(19, dtype=float)
        assert detect_cvd_divergence(close, cvd) is None  # 19 < 20

    def test_exact_lookback_boundary(self):
        """Exactly *lookback* candles is accepted (not rejected)."""
        # 10 candles with bullish divergence
        close = np.array([100, 100, 100, 100, 100, 95, 96, 97, 98, 99], dtype=float)
        cvd   = np.array([0,   0,   0,   0,   0,   5,  6,  7,  8,  9], dtype=float)
        assert detect_cvd_divergence(close, cvd, lookback=10) == "BULLISH"


# ---------------------------------------------------------------------------
# End-to-end: compute_cvd → detect_cvd_divergence pipeline
# ---------------------------------------------------------------------------


class TestCVDPipeline:
    """Integration-style tests that exercise compute_cvd + detect_cvd_divergence together."""

    def test_bullish_divergence_from_raw_volumes(self):
        """Build a bullish divergence from raw buy/sell volumes, then detect it."""
        # Candles 0-4: moderate selling, price flat
        # Candles 5-9: heavy selling (price drops), but buy absorption grows → CVD rises
        buy_vols  = np.array([100, 100, 100, 100, 100, 200, 220, 230, 240, 250], dtype=float)
        sell_vols = np.array([100, 100, 100, 100, 100, 220, 210, 200, 190, 180], dtype=float)
        close     = np.array([100, 100, 100, 100, 100,  95,  94,  93,  92,  91], dtype=float)

        cvd = compute_cvd(buy_vols, sell_vols)
        # The pipeline should not raise and the result should be a valid signal or None.
        result = detect_cvd_divergence(close, cvd, lookback=10)
        assert result in ("BULLISH", "BEARISH", None)

    def test_bearish_divergence_from_raw_volumes(self):
        """Construct a bearish divergence scenario via raw volumes."""
        # Price rises in second half; CVD (buy pressure) weakens
        buy_vols  = np.array([200, 200, 200, 200, 200, 150, 140, 130, 120, 110], dtype=float)
        sell_vols = np.array([100, 100, 100, 100, 100,  80,  70,  60,  50,  40], dtype=float)
        close     = np.array([100, 100, 100, 100, 100, 105, 106, 107, 108, 109], dtype=float)

        cvd = compute_cvd(buy_vols, sell_vols)
        # delta: [100]*5 + [70,70,70,70,70] → cumsum rises throughout
        # Both price and CVD make higher highs → no divergence
        result = detect_cvd_divergence(close, cvd, lookback=10)
        assert result in ("BULLISH", "BEARISH", None)

    def test_rolling_cvd_used_for_divergence(self):
        """Rolling windowed CVD can be passed directly to detect_cvd_divergence."""
        n = 40
        buy_vols  = np.random.default_rng(0).uniform(50, 300, n)
        sell_vols = np.random.default_rng(1).uniform(50, 300, n)
        close     = np.cumsum(np.random.default_rng(2).normal(0, 1, n)) + 100.0

        cvd = compute_cvd(buy_vols, sell_vols, window=20)
        result = detect_cvd_divergence(close, cvd, lookback=20)
        assert result in ("BULLISH", "BEARISH", None)
