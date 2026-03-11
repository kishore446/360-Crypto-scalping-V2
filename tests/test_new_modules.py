"""Tests for the new src modules: detector, regime, filters, risk, binance, exchange, logger."""

from __future__ import annotations

import numpy as np
import pytest

from src.channels.base import Signal
from src.detector import SMCDetector, SMCResult
from src.filters import (
    check_adx,
    check_ema_alignment,
    check_rsi,
    check_spread,
    check_volume,
)
from src.regime import MarketRegime, MarketRegimeDetector
from src.risk import RiskAssessment, RiskManager
from src.smc import Direction
from src.utils import utcnow


# ---------------------------------------------------------------------------
# SMCDetector
# ---------------------------------------------------------------------------


class TestSMCDetector:
    def _make_candles(self, n: int = 60):
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        return {"high": high, "low": low, "close": close, "volume": np.ones(n)}

    def test_returns_smc_result(self):
        det = SMCDetector()
        candles = {"5m": self._make_candles()}
        result = det.detect("BTCUSDT", candles, [])
        assert isinstance(result, SMCResult)

    def test_no_ticks_gives_no_whale(self):
        det = SMCDetector()
        result = det.detect("BTCUSDT", {}, [])
        assert result.whale_alert is None
        assert result.volume_delta_spike is False

    def test_whale_detected_in_ticks(self):
        det = SMCDetector()
        ticks = [{"price": 50000.0, "qty": 30.0, "isBuyerMaker": False, "time": 1}]
        result = det.detect("BTCUSDT", {}, ticks)
        assert result.whale_alert is not None  # 50000 * 30 = 1.5M > threshold

    def test_as_dict_keys(self):
        det = SMCDetector()
        result = det.detect("BTCUSDT", {}, [])
        d = result.as_dict()
        for key in ("sweeps", "mss", "fvg", "whale_alert", "volume_delta_spike", "recent_ticks"):
            assert key in d

    def test_sweep_detected_in_candles(self):
        det = SMCDetector()
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        # Last candle: wick below recent low, close just inside
        high[-1] = 105.0
        low[-1] = 93.0
        close[-1] = 95.04
        candles = {"5m": {"high": high, "low": low, "close": close, "volume": np.ones(n)}}
        result = det.detect("BTCUSDT", candles, [])
        assert len(result.sweeps) >= 1


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------


class TestMarketRegimeDetector:
    def test_trending_up_high_adx_positive_ema(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 102.0, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.TRENDING_UP

    def test_trending_down_high_adx_negative_ema(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 98.0, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.TRENDING_DOWN

    def test_ranging_low_adx(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 15.0, "ema9_last": 100.1, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.RANGING

    def test_volatile_wide_bb(self):
        det = MarketRegimeDetector()
        ind = {
            "adx_last": 22.0,
            "bb_upper_last": 112.0,
            "bb_lower_last": 88.0,
            "bb_mid_last": 100.0,
        }
        result = det.classify(ind)
        assert result.regime == MarketRegime.VOLATILE

    def test_quiet_narrow_bb(self):
        det = MarketRegimeDetector()
        ind = {
            "adx_last": 22.0,
            "bb_upper_last": 100.5,
            "bb_lower_last": 99.5,
            "bb_mid_last": 100.0,
        }
        result = det.classify(ind)
        assert result.regime == MarketRegime.QUIET

    def test_empty_indicators_defaults_to_ranging(self):
        det = MarketRegimeDetector()
        result = det.classify({})
        assert result.regime == MarketRegime.RANGING

    def test_result_has_regime_attribute(self):
        det = MarketRegimeDetector()
        result = det.classify({"adx_last": 28.0, "ema9_last": 105.0, "ema21_last": 100.0})
        assert hasattr(result, "regime")
        assert isinstance(result.regime, MarketRegime)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_check_spread_pass(self):
        assert check_spread(0.01, 0.02) is True

    def test_check_spread_fail(self):
        assert check_spread(0.03, 0.02) is False

    def test_check_spread_equal(self):
        assert check_spread(0.02, 0.02) is True

    def test_check_adx_in_range(self):
        assert check_adx(30.0, 25.0, 60.0) is True

    def test_check_adx_below_min(self):
        assert check_adx(20.0, 25.0) is False

    def test_check_adx_none(self):
        assert check_adx(None, 25.0) is False

    def test_check_ema_alignment_long(self):
        assert check_ema_alignment(102.0, 100.0, "LONG") is True

    def test_check_ema_alignment_long_fail(self):
        assert check_ema_alignment(98.0, 100.0, "LONG") is False

    def test_check_ema_alignment_short(self):
        assert check_ema_alignment(98.0, 100.0, "SHORT") is True

    def test_check_ema_alignment_none(self):
        assert check_ema_alignment(None, 100.0, "LONG") is False

    def test_check_volume_pass(self):
        assert check_volume(10_000_000, 5_000_000) is True

    def test_check_volume_fail(self):
        assert check_volume(1_000_000, 5_000_000) is False

    def test_check_rsi_long_not_overbought(self):
        assert check_rsi(60.0, 70.0, 30.0, "LONG") is True

    def test_check_rsi_long_overbought(self):
        assert check_rsi(75.0, 70.0, 30.0, "LONG") is False

    def test_check_rsi_short_not_oversold(self):
        assert check_rsi(40.0, 70.0, 30.0, "SHORT") is True

    def test_check_rsi_short_oversold(self):
        assert check_rsi(25.0, 70.0, 30.0, "SHORT") is False

    def test_check_rsi_none_passes(self):
        assert check_rsi(None, 70.0, 30.0, "LONG") is True


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


def _make_signal_obj(
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    entry: float = 50000.0,
    sl: float = 49500.0,
    tp1: float = 50500.0,
    confidence: float = 75.0,
):
    class _FakeDir:
        value = direction

    class _FakeSig:
        pass

    s = _FakeSig()
    s.symbol = symbol
    s.direction = _FakeDir()
    s.entry = entry
    s.stop_loss = sl
    s.tp1 = tp1
    s.confidence = confidence
    s.spread_pct = 0.01
    return s


class TestRiskManager:
    def test_basic_risk_assessment(self):
        rm = RiskManager()
        sig = _make_signal_obj()
        result = rm.calculate_risk(sig, {"atr_last": 200.0}, volume_24h_usd=50_000_000)
        assert isinstance(result, RiskAssessment)
        assert result.allowed is True

    def test_risk_reward_ratio(self):
        rm = RiskManager()
        sig = _make_signal_obj(entry=100.0, sl=99.0, tp1=101.0)
        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000)
        assert abs(result.risk_reward - 1.0) < 0.01

    def test_low_risk_label(self):
        rm = RiskManager()
        sig = _make_signal_obj(confidence=80.0)
        result = rm.calculate_risk(sig, {"atr_last": 10.0}, volume_24h_usd=100_000_000)
        assert result.risk_label in ("Low", "Medium")

    def test_high_risk_label_low_volume(self):
        rm = RiskManager()
        sig = _make_signal_obj(confidence=45.0)
        result = rm.calculate_risk(sig, {"atr_last": 1000.0}, volume_24h_usd=500_000)
        assert result.risk_label in ("High", "Very High")

    def test_concurrent_signal_blocked(self):
        rm = RiskManager(max_concurrent_same_direction=1)
        sig = _make_signal_obj(symbol="BTCUSDT", direction="LONG")

        existing = _make_signal_obj(symbol="BTCUSDT", direction="LONG")
        active = {"BTC-001": existing}

        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000, active_signals=active)
        assert result.allowed is False

    def test_concurrent_different_direction_allowed(self):
        rm = RiskManager()
        sig = _make_signal_obj(symbol="BTCUSDT", direction="LONG")
        active = {"BTC-001": _make_signal_obj(symbol="BTCUSDT", direction="SHORT")}
        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000, active_signals=active)
        assert result.allowed is True
