"""Tests for CircuitBreaker – tripping, reset, and edge cases."""

from __future__ import annotations

import time

import pytest

from src.circuit_breaker import CircuitBreaker


class TestCircuitBreakerInitialState:
    def test_not_tripped_initially(self):
        cb = CircuitBreaker()
        assert cb.is_tripped() is False

    def test_consecutive_sl_counter_starts_at_zero(self):
        cb = CircuitBreaker()
        assert cb._consecutive_sl == 0

    def test_status_text_ok(self):
        cb = CircuitBreaker()
        assert "OK" in cb.status_text()


class TestCircuitBreakerTripOnConsecutiveSL:
    def test_trips_after_max_consecutive_sl(self):
        cb = CircuitBreaker(max_consecutive_sl=3)
        cb.record_outcome("sig1", hit_sl=True, pnl_pct=-1.0)
        cb.record_outcome("sig2", hit_sl=True, pnl_pct=-1.0)
        assert cb.is_tripped() is False  # not yet
        cb.record_outcome("sig3", hit_sl=True, pnl_pct=-1.0)
        assert cb.is_tripped() is True

    def test_win_resets_consecutive_counter(self):
        cb = CircuitBreaker(max_consecutive_sl=3)
        cb.record_outcome("sig1", hit_sl=True, pnl_pct=-1.0)
        cb.record_outcome("sig2", hit_sl=True, pnl_pct=-1.0)
        cb.record_outcome("sig3", hit_sl=False, pnl_pct=1.5)  # WIN resets counter
        cb.record_outcome("sig4", hit_sl=True, pnl_pct=-1.0)
        assert cb.is_tripped() is False  # counter reset, only 1 consecutive SL

    def test_exactly_max_consecutive_trips(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-0.5)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-0.5)
        assert cb.is_tripped() is True

    def test_status_text_shows_tripped(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-0.5)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-0.5)
        assert "TRIPPED" in cb.status_text()
        assert cb._trip_reason != ""


class TestCircuitBreakerTripOnHourlySL:
    def test_trips_after_max_hourly_sl(self):
        cb = CircuitBreaker(max_consecutive_sl=100, max_hourly_sl=3)
        # Record 3 SL hits within the hour
        for i in range(3):
            cb.record_outcome(f"sig{i}", hit_sl=True, pnl_pct=-0.5)
        assert cb.is_tripped() is True

    def test_old_outcomes_not_counted(self):
        cb = CircuitBreaker(max_consecutive_sl=100, max_hourly_sl=2)
        # Inject two old SL hits (outside the 1-hour window)
        old_time = time.monotonic() - 3601
        cb._outcomes.append(
            __import__("src.circuit_breaker", fromlist=["OutcomeRecord"]).OutcomeRecord(
                signal_id="old1", hit_sl=True, pnl_pct=-1.0, timestamp=old_time
            )
        )
        cb._outcomes.append(
            __import__("src.circuit_breaker", fromlist=["OutcomeRecord"]).OutcomeRecord(
                signal_id="old2", hit_sl=True, pnl_pct=-1.0, timestamp=old_time
            )
        )
        # Should NOT trip yet since old ones are outside window
        cb.record_outcome("new1", hit_sl=True, pnl_pct=-0.5)
        assert cb.is_tripped() is False


class TestCircuitBreakerReset:
    def test_reset_clears_tripped_state(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-0.5)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-0.5)
        assert cb.is_tripped() is True
        cb.reset()
        assert cb.is_tripped() is False

    def test_reset_clears_consecutive_counter(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-0.5)
        cb.reset()
        assert cb._consecutive_sl == 0

    def test_can_record_after_reset(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-0.5)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-0.5)
        cb.reset()
        cb.record_outcome("c", hit_sl=True, pnl_pct=-0.5)
        assert cb.is_tripped() is False  # only 1 SL after reset


class TestCircuitBreakerDailyDrawdown:
    def test_trips_on_daily_drawdown(self):
        cb = CircuitBreaker(
            max_consecutive_sl=100,
            max_hourly_sl=100,
            max_daily_drawdown_pct=5.0,
        )
        cb.record_outcome("a", hit_sl=True, pnl_pct=-3.0)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-3.0)
        assert cb.is_tripped() is True

    def test_wins_do_not_prevent_drawdown_trip(self):
        cb = CircuitBreaker(
            max_consecutive_sl=100,
            max_hourly_sl=100,
            max_daily_drawdown_pct=5.0,
        )
        cb.record_outcome("a", hit_sl=True, pnl_pct=-4.0)
        cb.record_outcome("b", hit_sl=False, pnl_pct=1.0)  # a win
        cb.record_outcome("c", hit_sl=True, pnl_pct=-2.0)   # total losses = 6%
        assert cb.is_tripped() is True


class TestCircuitBreakerEdgeCases:
    def test_no_outcomes_not_tripped(self):
        cb = CircuitBreaker()
        assert cb.is_tripped() is False

    def test_all_wins_not_tripped(self):
        cb = CircuitBreaker(max_consecutive_sl=3)
        for i in range(10):
            cb.record_outcome(f"w{i}", hit_sl=False, pnl_pct=2.0)
        assert cb.is_tripped() is False

    def test_already_tripped_does_not_re_evaluate(self):
        cb = CircuitBreaker(max_consecutive_sl=2)
        cb.record_outcome("a", hit_sl=True, pnl_pct=-1.0)
        cb.record_outcome("b", hit_sl=True, pnl_pct=-1.0)
        # already tripped – manually verify trip_reason unchanged
        old_reason = cb._trip_reason
        cb.record_outcome("c", hit_sl=False, pnl_pct=5.0)
        assert cb._trip_reason == old_reason
