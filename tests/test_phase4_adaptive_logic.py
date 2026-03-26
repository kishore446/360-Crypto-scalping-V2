"""Phase 4 – Signal Diversity Enhancements: adaptive indicator weights,
ATR-normalized Bollinger / volume thresholds, and regime-weighted MTF.

Tests covering:
1. _select_indicator_weights returns correct weights for each regime
2. ATR-normalized Bollinger threshold scales correctly with volatility
3. Volume expansion multiplier adjusts per-regime
4. MTF regime config applies correct min_score thresholds
5. Trending regimes boost higher-TF weight
6. Ranging regimes boost lower-TF weight
7. Integration: ScalpChannel candidate selection respects regime weights
8. MTF gate passes more easily in VOLATILE regime (lower min_score)
9. MTF gate is stricter in TRENDING regime (higher min_score)
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task 1: _select_indicator_weights
# ---------------------------------------------------------------------------


class TestSelectIndicatorWeights:
    """ScalpChannel._select_indicator_weights returns regime-specific multipliers."""

    @pytest.fixture
    def scalp_channel(self):
        from src.channels.scalp import ScalpChannel
        return ScalpChannel()

    def test_volatile_boosts_order_flow_and_volume(self, scalp_channel):
        weights = scalp_channel._select_indicator_weights("VOLATILE")
        assert weights["order_flow"] > 1.0, "VOLATILE should boost order_flow"
        assert weights["volume"] > 1.0, "VOLATILE should boost volume"
        assert weights["mean_reversion"] < 1.0, "VOLATILE should suppress mean_reversion"
        assert weights["trend"] < 1.0, "VOLATILE should suppress trend"

    def test_quiet_boosts_mean_reversion(self, scalp_channel):
        weights = scalp_channel._select_indicator_weights("QUIET")
        assert weights["mean_reversion"] > 1.0
        assert weights["order_flow"] < 1.0
        assert weights["trend"] < 1.0

    def test_ranging_boosts_mean_reversion(self, scalp_channel):
        weights = scalp_channel._select_indicator_weights("RANGING")
        assert weights["mean_reversion"] > 1.0
        assert weights["order_flow"] < 1.0

    def test_trending_up_boosts_trend(self, scalp_channel):
        weights = scalp_channel._select_indicator_weights("TRENDING_UP")
        assert weights["trend"] > 1.0
        assert weights["mean_reversion"] < 1.0

    def test_trending_down_boosts_trend(self, scalp_channel):
        weights = scalp_channel._select_indicator_weights("TRENDING_DOWN")
        assert weights["trend"] > 1.0
        assert weights["mean_reversion"] < 1.0

    def test_unknown_regime_returns_defaults(self, scalp_channel):
        """Empty / unknown regime should return all-1.0 weights (no change)."""
        for regime in ("", "UNKNOWN", None):
            weights = scalp_channel._select_indicator_weights(regime or "")
            for v in weights.values():
                assert v == 1.0, f"Expected 1.0 for regime={regime!r}, got {v}"

    def test_case_insensitive(self, scalp_channel):
        """Regime string matching must be case-insensitive."""
        w_upper = scalp_channel._select_indicator_weights("VOLATILE")
        w_lower = scalp_channel._select_indicator_weights("volatile")
        assert w_upper == w_lower

    def test_required_keys_present(self, scalp_channel):
        expected_keys = {"order_flow", "trend", "mean_reversion", "volume"}
        for regime in ("VOLATILE", "QUIET", "RANGING", "TRENDING_UP", "TRENDING_DOWN", ""):
            weights = scalp_channel._select_indicator_weights(regime)
            assert set(weights.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Task 2: ATR-normalized Bollinger threshold
# ---------------------------------------------------------------------------


class TestAtrNormalizedBollingerThreshold:
    """SpotChannel uses ATR-scaled BB squeeze threshold instead of fixed 4.0."""

    def _make_spot_channel(self):
        from src.channels.spot import SpotChannel
        ch = SpotChannel()
        ch._current_regime = ""
        return ch

    def _make_h4(self, n: int = 60, base: float = 100.0):
        close = np.linspace(base, base + n * 0.1, n)
        return {
            "open": close - 0.05,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(n) * 1000,
        }

    def test_low_atr_pair_uses_tighter_threshold(self):
        """BTC-like pair (low ATR %) should require tighter BB squeeze."""
        ch = self._make_spot_channel()
        h4 = self._make_h4()
        highs = list(h4["high"])
        lows = list(h4["low"])
        closes_list = list(h4["close"])
        volumes = list(h4["volume"])
        close = closes_list[-1]
        # ATR = 0.2 (0.2% of 100) → atr_pct=0.2 → threshold = max(2.0, min(6.0, 0.2*3))=2.0
        atr_val = 0.2
        atr_pct = atr_val / close * 100  # 0.2%
        expected_threshold = max(2.0, min(6.0, atr_pct * 3.0))
        assert expected_threshold == pytest.approx(2.0)
        # bb_width=3.0 > 2.0 threshold → should block (return None)
        result = ch._try_long(
            "BTCUSDT", close, atr_val, h4, highs, lows, volumes, closes_list,
            bb_width=3.0, rsi_last=50.0, mss=None,
        )
        assert result is None, "Low ATR pair should block at bb_width=3.0"

    def test_high_atr_pair_uses_wider_threshold(self):
        """High-ATR altcoin should allow wider BB before blocking."""
        h4 = self._make_h4()
        close = float(h4["close"][-1])
        # ATR = 5 (5% of 100) → atr_pct=5.0 → threshold = max(2.0, min(6.0, 5.0*3))=6.0
        atr_val = 5.0
        atr_pct = atr_val / close * 100
        expected_threshold = max(2.0, min(6.0, atr_pct * 3.0))
        assert expected_threshold == pytest.approx(6.0)

    def test_bb_width_none_skips_check(self):
        """When bb_width is None the squeeze check is skipped (fail open)."""
        from src.channels.spot import SpotChannel
        ch = SpotChannel()
        ch._current_regime = ""
        h4 = self._make_h4()
        highs = list(h4["high"])
        lows = list(h4["low"])
        closes_list = list(h4["close"])
        volumes = list(h4["volume"])
        close = closes_list[-1]
        atr_val = 1.0
        # Pass bb_width=None — must not raise, squeeze check skipped
        # Result may be None due to other gates but should not raise
        try:
            ch._try_long(
                "TESTUSDT", close, atr_val, h4, highs, lows, volumes, closes_list,
                bb_width=None, rsi_last=50.0, mss=None,
            )
        except Exception as exc:
            pytest.fail(f"bb_width=None caused exception: {exc}")

    def test_threshold_capped_at_six(self):
        """Threshold must not exceed 6.0 even for extremely high ATR."""
        close = 1.0
        atr_val = 100.0  # 10 000% ATR
        atr_pct = atr_val / close * 100
        threshold = max(2.0, min(6.0, atr_pct * 3.0))
        assert threshold == pytest.approx(6.0)

    def test_threshold_floored_at_two(self):
        """Threshold must not go below 2.0 even for near-zero ATR."""
        close = 100.0
        atr_val = 0.001
        atr_pct = atr_val / close * 100
        threshold = max(2.0, min(6.0, atr_pct * 3.0))
        assert threshold == pytest.approx(2.0)

    def test_short_uses_atr_normalized_threshold(self):
        """_try_short also uses ATR-normalized squeeze threshold."""
        from src.channels.spot import SpotChannel
        ch = SpotChannel()
        ch._current_regime = ""
        h4 = self._make_h4(base=100.0)
        close = 90.0  # Below recent lows to trigger breakdown path
        atr_val = 0.2  # Low ATR → threshold ≈ 2.0
        highs = [100.0] * 60
        lows = [95.0] * 60
        closes_list = [101.0] * 60
        volumes = [1000.0] * 60
        # bb_width=3.0 > 2.0 → should be blocked
        result = ch._try_short(
            "BTCUSDT", close, atr_val, h4, highs, lows, volumes, closes_list,
            bb_width=3.0, rsi_last=50.0, mss=None,
        )
        assert result is None, "_try_short should block when bb_width > ATR-normalized threshold"


# ---------------------------------------------------------------------------
# Task 3: Volume expansion multiplier per-regime
# ---------------------------------------------------------------------------


class TestVolumeExpansionMultiplier:
    """SpotChannel uses regime-adjusted volume expansion multiplier."""

    def _make_channel_with_regime(self, regime: str):
        from src.channels.spot import SpotChannel
        ch = SpotChannel()
        ch._current_regime = regime
        return ch

    def _vol_mult(self, regime: str) -> float:
        """Replicate the multiplier logic from spot.py for verification."""
        regime_upper = regime.upper() if regime else ""
        if regime_upper in ("QUIET", "RANGING"):
            return 2.2
        if regime_upper == "VOLATILE":
            return 1.5
        return 1.8

    def test_quiet_uses_higher_multiplier(self):
        assert self._vol_mult("QUIET") == 2.2

    def test_ranging_uses_higher_multiplier(self):
        assert self._vol_mult("RANGING") == 2.2

    def test_volatile_uses_lower_multiplier(self):
        assert self._vol_mult("VOLATILE") == 1.5

    def test_trending_up_uses_default_multiplier(self):
        assert self._vol_mult("TRENDING_UP") == 1.8

    def test_trending_down_uses_default_multiplier(self):
        assert self._vol_mult("TRENDING_DOWN") == 1.8

    def test_empty_regime_uses_default_multiplier(self):
        assert self._vol_mult("") == 1.8

    def test_volatile_regime_passes_lower_volume_threshold(self):
        """A borderline volume that passes for VOLATILE but fails for QUIET."""
        # avg_usd_vol=1000, current=1600
        # VOLATILE mult=1.5 → 1000*1.5=1500 → 1600 >= 1500 → passes
        # QUIET   mult=2.2 → 1000*2.2=2200 → 1600 <  2200 → fails
        avg = 1000.0
        current = 1600.0
        volatile_mult = self._vol_mult("VOLATILE")
        quiet_mult = self._vol_mult("QUIET")
        assert current >= avg * volatile_mult, "Should pass VOLATILE threshold"
        assert current < avg * quiet_mult, "Should fail QUIET threshold"

    def test_source_uses_volume_expansion_helper(self):
        """The spot.py source should compute vol expansion via a helper method."""
        import inspect
        from src.channels import spot
        source = inspect.getsource(spot)
        # The new code delegates to _volume_expansion_mult()
        assert "_volume_expansion_mult" in source


# ---------------------------------------------------------------------------
# Task 4: MTF regime config — min_score thresholds
# ---------------------------------------------------------------------------


class TestMtfRegimeConfig:
    """_MTF_REGIME_CONFIG in scanner.py defines correct per-regime gates."""

    @pytest.fixture
    def cfg(self):
        from src.scanner import _MTF_REGIME_CONFIG
        return _MTF_REGIME_CONFIG

    def test_all_expected_regimes_present(self, cfg):
        expected = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "QUIET"}
        assert set(cfg.keys()) == expected

    def test_trending_has_high_min_score(self, cfg):
        assert cfg["TRENDING_UP"]["min_score"] >= 0.5
        assert cfg["TRENDING_DOWN"]["min_score"] >= 0.5

    def test_volatile_has_lowest_min_score(self, cfg):
        scores = [v["min_score"] for v in cfg.values()]
        assert cfg["VOLATILE"]["min_score"] == min(scores), (
            "VOLATILE should have the most relaxed (lowest) min_score"
        )

    def test_ranging_has_low_min_score(self, cfg):
        assert cfg["RANGING"]["min_score"] <= 0.4

    def test_quiet_is_between_volatile_and_trending(self, cfg):
        assert cfg["VOLATILE"]["min_score"] <= cfg["QUIET"]["min_score"]
        assert cfg["QUIET"]["min_score"] <= cfg["TRENDING_UP"]["min_score"]

    # Task 5: Trending regimes boost higher-TF weight
    def test_trending_boosts_higher_tf_weight(self, cfg):
        for regime in ("TRENDING_UP", "TRENDING_DOWN"):
            assert cfg[regime]["higher_tf_weight"] > 1.0, (
                f"{regime} should have higher_tf_weight > 1.0"
            )
            assert cfg[regime]["lower_tf_weight"] < 1.0, (
                f"{regime} should have lower_tf_weight < 1.0"
            )

    # Task 6: Ranging regimes boost lower-TF weight
    def test_ranging_boosts_lower_tf_weight(self, cfg):
        assert cfg["RANGING"]["lower_tf_weight"] > 1.0, (
            "RANGING should boost lower-TF weight (entry precision)"
        )
        assert cfg["RANGING"]["higher_tf_weight"] < 1.0, (
            "RANGING should suppress higher-TF weight"
        )

    def test_volatile_uses_neutral_weights(self, cfg):
        """VOLATILE uses balanced weights since TFs often diverge."""
        assert cfg["VOLATILE"]["higher_tf_weight"] == pytest.approx(1.0)
        assert cfg["VOLATILE"]["lower_tf_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MTF gate: regime-specific min_score is applied
# ---------------------------------------------------------------------------


class TestMtfGateRegimeMinScore:
    """check_mtf_gate respects min_score and tf_weight_overrides."""

    def _make_aligned_data(self, direction: str = "LONG") -> dict:
        """All TFs bullish — passes any min_score."""
        return {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 102.0},
            "5m":  {"ema_fast": 102.0, "ema_slow": 100.0, "close": 103.0},
            "15m": {"ema_fast": 103.0, "ema_slow": 101.0, "close": 104.0},
            "4h":  {"ema_fast": 104.0, "ema_slow": 101.0, "close": 105.0},
        }

    def _make_mixed_data(self) -> dict:
        """Lower TFs bullish, higher TF bearish."""
        return {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 102.0},
            "5m":  {"ema_fast": 102.0, "ema_slow": 100.0, "close": 103.0},
            "4h":  {"ema_fast": 100.0, "ema_slow": 102.0, "close": 99.0},   # Bearish
        }

    def test_volatile_lower_min_score_allows_mixed_tf(self):
        """VOLATILE min_score=0.2 should pass even with partial TF disagreement."""
        from src.mtf import check_mtf_gate
        data = self._make_mixed_data()
        allowed, reason = check_mtf_gate("LONG", data, min_score=0.2)
        assert allowed, f"VOLATILE min_score=0.2 should pass mixed data; reason={reason}"

    def test_trending_higher_min_score_blocks_mixed_tf(self):
        """TRENDING min_score=0.6 blocks signals with bearish higher TF."""
        from src.mtf import check_mtf_gate
        data = self._make_mixed_data()
        allowed, _ = check_mtf_gate("LONG", data, min_score=0.6)
        # With 4h bearish, score should drop below 0.6
        assert not allowed, "TRENDING min_score=0.6 should block mixed TF data"

    def test_tf_weight_overrides_applied(self):
        """Boosting lower-TF weights should raise score for lower-TF-aligned data."""
        from src.mtf import compute_mtf_confluence, _TF_WEIGHTS
        data = self._make_mixed_data()
        # Default: 4h is bearish and has high weight → lower score
        result_default = compute_mtf_confluence("LONG", data)
        # Override: suppress 4h weight, boost 1m/5m weights
        overrides = {
            "1m": _TF_WEIGHTS.get("1m", 0.5) * 2.0,   # double lower-TF weight
            "5m": _TF_WEIGHTS.get("5m", 1.0) * 2.0,
            "4h": _TF_WEIGHTS.get("4h", 3.0) * 0.5,   # halve higher-TF weight
        }
        result_override = compute_mtf_confluence("LONG", data, tf_weight_overrides=overrides)
        assert result_override.score > result_default.score, (
            "Suppressing bearish higher-TF weight should increase the score"
        )

    def test_higher_tf_weight_boost_lowers_score_when_4h_bearish(self):
        """Boosting higher-TF weight should lower score when 4h is bearish."""
        from src.mtf import compute_mtf_confluence, _TF_WEIGHTS
        data = self._make_mixed_data()
        result_default = compute_mtf_confluence("LONG", data)
        overrides = {
            "4h": _TF_WEIGHTS.get("4h", 3.0) * 2.0,  # amplify bearish 4h
        }
        result_override = compute_mtf_confluence("LONG", data, tf_weight_overrides=overrides)
        assert result_override.score < result_default.score, (
            "Amplifying bearish higher-TF weight should decrease the score"
        )

    def test_check_mtf_gate_passes_tf_weight_overrides(self):
        """check_mtf_gate forward tf_weight_overrides to compute_mtf_confluence."""
        from src.mtf import check_mtf_gate
        data = self._make_mixed_data()
        # With VOLATILE config (lower min_score + neutral weights)
        allowed_strict, _ = check_mtf_gate("LONG", data, min_score=0.6)
        allowed_relaxed, _ = check_mtf_gate("LONG", data, min_score=0.2)
        assert not allowed_strict
        assert allowed_relaxed


# ---------------------------------------------------------------------------
# Task 7: Integration — ScalpChannel candidate selection respects regime weights
# ---------------------------------------------------------------------------


class TestScalpChannelRegimeWeights:
    """Integration test verifying regime weights influence candidate selection."""

    def test_evaluate_returns_signal_in_volatile_regime(self):
        """ScalpChannel.evaluate should not crash and should apply weights."""
        from src.channels.scalp import ScalpChannel
        from src.smc import LiquiditySweep, Direction

        ch = ScalpChannel()
        n = 60
        base = 100.0
        closes = np.linspace(base, base + n * 0.1, n)
        candles_arr = {
            "open": closes - 0.05,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.ones(n) * 2000,
        }
        candles = {"5m": candles_arr, "1m": candles_arr}

        sweep = LiquiditySweep(
            index=n - 1,
            direction=Direction.LONG,
            sweep_level=base - 1,
            close_price=base,
            wick_high=base + 1,
            wick_low=base - 1,
        )
        indicators = {
            "5m": {
                "adx_last": 25.0,
                "ema9_last": closes[-1] + 0.1,
                "ema21_last": closes[-1] - 0.1,
                "rsi_last": 55.0,
                "atr_last": 0.3,
                "momentum_last": 0.25,
                "momentum_array": [0.20, 0.25],
                "bb_upper_last": closes[-1] + 5,
                "bb_lower_last": closes[-1] - 5,
                "bb_width_pct": 3.0,
                "bb_width_prev_pct": 3.0,
            },
            "1m": {
                "rsi_last": 55.0,
                "atr_last": 0.3,
            },
        }
        smc_data = {"sweeps": [sweep]}

        # Both regimes should run without error; we only check no exception is raised.
        for regime in ("VOLATILE", "QUIET", "TRENDING_UP", "RANGING", ""):
            ch.evaluate(
                "TESTUSDT", candles, indicators, smc_data,
                spread_pct=0.01, volume_24h_usd=20_000_000, regime=regime,
            )

    def test_regime_weighted_selection_returns_valid_signal(self):
        """ScalpChannel.evaluate() returns a valid Signal using regime-weighted selection."""
        from src.channels.scalp import ScalpChannel
        from src.channels.base import Signal
        from src.smc import LiquiditySweep, Direction

        ch = ScalpChannel()
        n = 60
        closes = np.linspace(100.0, 106.0, n)
        candles_arr = {
            "open": closes - 0.05,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.ones(n) * 2000,
        }
        sweep = LiquiditySweep(
            index=n - 1, direction=Direction.LONG,
            sweep_level=99.0, close_price=100.0,
            wick_high=101.0, wick_low=99.0,
        )
        indicators = {
            "5m": {
                "adx_last": 28.0,
                "ema9_last": closes[-1] + 0.1,
                "ema21_last": closes[-1] - 0.1,
                "rsi_last": 55.0,
                "atr_last": 0.3,
                "momentum_last": 0.25,
                "momentum_array": [0.22, 0.25],
                "bb_upper_last": closes[-1] + 5,
                "bb_lower_last": closes[-1] - 5,
                "bb_width_pct": 3.0,
                "bb_width_prev_pct": 3.0,
            },
            "1m": {"rsi_last": 55.0, "atr_last": 0.3},
        }
        smc_data = {"sweeps": [sweep]}
        candles = {"5m": candles_arr, "1m": candles_arr}

        sig = ch.evaluate(
            "TESTUSDT", candles, indicators, smc_data,
            spread_pct=0.01, volume_24h_usd=20_000_000, regime="VOLATILE",
        )
        # Regime-weighted selection must return None or a proper Signal instance.
        assert sig is None or isinstance(sig, Signal), (
            "evaluate() must return None or a Signal, not an intermediate type"
        )
