"""Tests for the Backtester – backtest framework with synthetic candle data."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtester import Backtester, BacktestResult, _compute_indicators, _simulate_trade
from src.channels.scalp import ScalpChannel
from src.channels.range_channel import RangeChannel
from src.smc import Direction


def _make_candles(
    n: int = 100,
    base: float = 100.0,
    trend: float = 0.1,
    noise: float = 0.2,
) -> dict:
    """Generate synthetic OHLCV data."""
    close = np.cumsum(np.ones(n) * trend) + base
    close += np.random.default_rng(42).normal(0, noise, n)
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000.0
    return {"open": close - 0.05, "high": high, "low": low, "close": close, "volume": volume}


class TestBacktestResult:
    def test_summary_contains_channel(self):
        r = BacktestResult(channel="360_SCALP", total_signals=10, wins=7, losses=3)
        assert "360_SCALP" in r.summary()

    def test_summary_contains_win_rate(self):
        r = BacktestResult(channel="360_SCALP", win_rate=70.0)
        assert "70.0" in r.summary()

    def test_default_values(self):
        r = BacktestResult(channel="TEST")
        assert r.total_signals == 0
        assert r.wins == 0
        assert r.losses == 0
        assert r.signal_details == []


class TestComputeIndicators:
    def test_returns_dict_with_ema(self):
        n = 250
        c = np.cumsum(np.ones(n) * 0.1) + 100.0
        h = c + 0.5
        lo = c - 0.5
        candles = {"high": h, "low": lo, "close": c, "volume": np.ones(n)}
        ind = _compute_indicators(candles)
        assert "ema9_last" in ind
        assert "ema21_last" in ind

    def test_insufficient_data_skips_indicators(self):
        c = np.ones(5) * 100.0
        ind = _compute_indicators({"high": c + 1, "low": c - 1, "close": c})
        assert "ema9_last" not in ind  # need >= 21 candles


class TestSimulateTrade:
    def _fake_signal(self, direction="LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0):
        class _FakeDir:
            value = direction

        class _FakeSig:
            pass

        s = _FakeSig()
        s.direction = _FakeDir()
        s.entry = entry
        s.stop_loss = sl
        s.tp1 = tp1
        s.tp2 = tp2
        s.tp3 = None
        return s

    def test_long_tp1_hit(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5, 101.5]),
            "low": np.array([99.5, 100.0]),
            "close": np.array([100.5, 101.5]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is True
        assert tp_level == 1
        assert pnl > 0

    def test_long_sl_hit(self):
        sig = self._fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5, 100.0]),
            "low": np.array([99.5, 98.5]),  # low < SL
            "close": np.array([100.5, 99.0]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is False
        assert tp_level == 0

    def test_short_tp1_hit(self):
        sig = self._fake_signal("SHORT", entry=100.0, sl=101.0, tp1=99.0, tp2=98.0)
        future = {
            "high": np.array([100.5, 100.0]),
            "low": np.array([99.5, 98.8]),  # low <= TP1
            "close": np.array([99.5, 99.0]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is True
        assert tp_level == 1

    def test_no_future_candles(self):
        sig = self._fake_signal()
        future = {"high": np.array([]), "low": np.array([]), "close": np.array([])}
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is False
        assert pnl == 0.0


class TestBacktester:
    def test_run_returns_results_for_each_channel(self):
        bt = Backtester(min_window=30, lookahead_candles=5)
        candles = _make_candles(n=200)
        candles_by_tf = {
            "5m": candles,
            "1m": candles,
            "15m": candles,
            "1h": candles,
            "4h": candles,
        }
        results = bt.run(candles_by_tf, symbol="BTCUSDT")
        assert isinstance(results, list)
        assert len(results) == 4  # one per channel

    def test_run_single_channel(self):
        bt = Backtester(min_window=30, lookahead_candles=5)
        candles = _make_candles(n=200)
        candles_by_tf = {"5m": candles, "1m": candles}
        results = bt.run(candles_by_tf, channel_name="360_SCALP")
        assert len(results) == 1
        assert results[0].channel == "360_SCALP"

    def test_result_has_win_rate_in_range(self):
        bt = Backtester(min_window=30, lookahead_candles=10)
        candles = _make_candles(n=300, trend=0.5)
        candles_by_tf = {"5m": candles, "1m": candles}
        results = bt.run(candles_by_tf, channel_name="360_SCALP")
        if results[0].total_signals > 0:
            assert 0.0 <= results[0].win_rate <= 100.0

    def test_missing_timeframe_returns_empty_result(self):
        bt = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=5)
        # No 5m timeframe provided
        candles_by_tf = {"1h": _make_candles(200)}
        results = bt.run(candles_by_tf)
        assert results[0].total_signals == 0

    def test_custom_channel_list(self):
        bt = Backtester(
            channels=[RangeChannel()], min_window=30, lookahead_candles=5
        )
        candles = _make_candles(n=200)
        candles_by_tf = {"15m": candles, "5m": candles}
        results = bt.run(candles_by_tf)
        assert len(results) == 1
        assert results[0].channel == "360_RANGE"
