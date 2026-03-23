"""Tests for src/mtf.py – MTF confluence with NEUTRAL partial credit."""

from __future__ import annotations

import pytest

from src.mtf import compute_mtf_confluence


class TestNeutralPartialCredit:
    """Verify that NEUTRAL timeframes contribute 0.5 partial credit."""

    def test_neutral_contributes_half_credit(self):
        """3 TFs: 1m+5m NEUTRAL, 15m BULLISH → weighted score > 0.5 → is_aligned."""
        timeframes = {
            "1m":  {"ema_fast": 100.5, "ema_slow": 100.0, "close": 100.2},  # NEUTRAL
            "5m":  {"ema_fast": 100.5, "ema_slow": 100.0, "close": 100.2},  # NEUTRAL
            "15m": {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},  # BULLISH
        }
        result = compute_mtf_confluence("LONG", timeframes)
        # With TF weights: 1m=0.5, 5m=1.0, 15m=1.5; total=3.0
        # aligned = 0.5*0.5 + 1.0*0.5 + 1.5*1.0 = 0.25 + 0.5 + 1.5 = 2.25
        # score = 2.25/3.0 = 0.75
        assert result.score == pytest.approx(0.75, abs=0.01)
        assert result.is_aligned is True

    def test_all_neutral_passes_threshold(self):
        """All NEUTRAL TFs → score = 0.5 == min_score (0.5) → is_aligned."""
        timeframes = {
            "1m":  {"ema_fast": 100.5, "ema_slow": 100.0, "close": 100.2},  # NEUTRAL
            "5m":  {"ema_fast": 100.5, "ema_slow": 100.0, "close": 100.2},  # NEUTRAL
        }
        result = compute_mtf_confluence("LONG", timeframes)
        # With weights: 1m=0.5, 5m=1.0; total=1.5
        # aligned = 0.25 + 0.5 = 0.75; score = 0.75/1.5 = 0.5
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.is_aligned is True

    def test_bearish_tf_receives_no_credit_for_long(self):
        """BEARISH TF gives 0 credit for a LONG signal (unchanged behavior)."""
        timeframes = {
            "1m": {"ema_fast": 99.0, "ema_slow": 101.0, "close": 98.0},   # BEARISH
            "5m": {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},  # BULLISH
        }
        result = compute_mtf_confluence("LONG", timeframes)
        # With weights: 1m=0.5 (BEARISH→0), 5m=1.0 (BULLISH→1.0); total=1.5
        # score = 1.0/1.5 ≈ 0.667
        assert result.score == pytest.approx(1.0 / 1.5, abs=0.01)
        assert result.is_aligned is True

    def test_mixed_neutral_and_bearish_may_block(self):
        """1 NEUTRAL + 1 BEARISH → weighted score < 0.5 → blocked."""
        timeframes = {
            "1m": {"ema_fast": 99.0, "ema_slow": 101.0, "close": 98.0},   # BEARISH
            "5m": {"ema_fast": 100.5, "ema_slow": 100.0, "close": 100.2},  # NEUTRAL
        }
        result = compute_mtf_confluence("LONG", timeframes)
        # With weights: 1m=0.5 (BEARISH→0), 5m=1.0 (NEUTRAL→0.5); total=1.5
        # aligned = 0 + 0.5 = 0.5; score = 0.5/1.5 ≈ 0.333
        assert result.score == pytest.approx(0.5 / 1.5, abs=0.01)
        assert result.is_aligned is False

    def test_fully_bullish_gives_perfect_score(self):
        """All BULLISH TFs → score = 1.0."""
        timeframes = {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
            "5m":  {"ema_fast": 102.0, "ema_slow": 101.0, "close": 102.5},
            "15m": {"ema_fast": 103.0, "ema_slow": 101.5, "close": 103.5},
        }
        result = compute_mtf_confluence("LONG", timeframes)
        assert result.score == pytest.approx(1.0)
        assert result.is_aligned is True
        assert result.is_strong is True
