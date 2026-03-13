"""Tests for src.trade_monitor – minimum lifespan and SL/TP evaluation."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict
from unittest.mock import MagicMock

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.trade_monitor import TradeMonitor
from src.utils import utcnow


def _make_signal(
    channel: str = "360_SCALP",
    symbol: str = "BTCUSDT",
    direction: Direction = Direction.LONG,
    entry: float = 30000.0,
    stop_loss: float = 29850.0,
    tp1: float = 30150.0,
    tp2: float = 30300.0,
    signal_id: str = "TEST-SIG-001",
    age_seconds: float = 0.0,
) -> Signal:
    sig = Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        confidence=85.0,
        signal_id=signal_id,
    )
    # Backdate the timestamp to simulate a signal of `age_seconds` old
    if age_seconds > 0:
        sig.timestamp = utcnow() - timedelta(seconds=age_seconds)
    return sig


class TestMinimumLifespan:
    """The monitor must NOT trigger SL/TP checks for very new signals."""

    def _build_monitor(self, active: Dict[str, Signal]):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
        )
        return monitor, removed, sent

    @pytest.mark.asyncio
    async def test_sl_not_triggered_within_min_lifespan(self):
        """Brand-new SCALP signal (age=0) below SL should NOT be removed."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=0.0,  # just created
        )
        # Set current price below stop loss to simulate SL condition
        sig.current_price = 29800.0

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        # Signal must NOT be removed because the min lifespan hasn't passed
        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_sl_triggered_after_min_lifespan(self):
        """A SCALP signal older than 30s whose price is below SL SHOULD be removed."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,  # past the 30s SCALP minimum
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "SL_HIT"

    @pytest.mark.asyncio
    async def test_swing_min_lifespan_is_longer(self):
        """A SWING signal at age=15s (< 60s min) should NOT trigger SL."""
        sig = _make_signal(
            channel="360_SWING",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=15.0,  # below the 60s SWING minimum
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_tp_not_triggered_within_min_lifespan(self):
        """TP1 should NOT fire on a brand-new signal even if price reaches TP."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=0.0,
        )
        sig.current_price = 30200.0  # above TP1

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "ACTIVE"


class TestOutcomeRecording:
    """TradeMonitor must call performance_tracker and circuit_breaker when signals close."""

    def _build_monitor_with_trackers(self, active):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        perf_tracker = MagicMock()
        cb = MagicMock()

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
            performance_tracker=perf_tracker,
            circuit_breaker=cb,
        )
        return monitor, removed, sent, perf_tracker, cb

    @pytest.mark.asyncio
    async def test_sl_hit_records_outcome(self):
        """SL hit must call record_outcome with hit_sl=True on both trackers."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.current_price = 29800.0  # below SL

        monitor, removed, sent, perf, cb = self._build_monitor_with_trackers(
            {sig.signal_id: sig}
        )
        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        perf.record_outcome.assert_called_once()
        call_kwargs = perf.record_outcome.call_args[1]
        assert call_kwargs["hit_sl"] is True
        assert call_kwargs["hit_tp"] == 0

        cb.record_outcome.assert_called_once()
        cb_kwargs = cb.record_outcome.call_args[1]
        assert cb_kwargs["hit_sl"] is True

    @pytest.mark.asyncio
    async def test_tp3_hit_records_outcome(self):
        """TP3 hit must call record_outcome with hit_tp=3 and hit_sl=False."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=35.0,
        )
        sig.tp3 = 30450.0
        sig.current_price = 30500.0  # above TP3

        monitor, removed, sent, perf, cb = self._build_monitor_with_trackers(
            {sig.signal_id: sig}
        )
        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        perf.record_outcome.assert_called_once()
        call_kwargs = perf.record_outcome.call_args[1]
        assert call_kwargs["hit_tp"] == 3
        assert call_kwargs["hit_sl"] is False

        cb.record_outcome.assert_called_once()
        cb_kwargs = cb.record_outcome.call_args[1]
        assert cb_kwargs["hit_sl"] is False

    @pytest.mark.asyncio
    async def test_tp1_tp2_do_not_record_outcome(self):
        """Intermediate TP1/TP2 hits must NOT call record_outcome (signal still active)."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=35.0,
        )
        sig.current_price = 30200.0  # above TP1, below TP2

        monitor, removed, sent, perf, cb = self._build_monitor_with_trackers(
            {sig.signal_id: sig}
        )
        await monitor._evaluate_signal(sig)

        # Signal should NOT be removed — it's still running
        assert sig.signal_id not in removed
        perf.record_outcome.assert_not_called()
        cb.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_records_outcome(self):
        """Signals with invalid SL are CANCELLED and outcome is recorded with hit_sl=False."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=30100.0,  # invalid: SL > entry for LONG
            age_seconds=35.0,
        )
        sig.current_price = 29900.0

        monitor, removed, sent, perf, cb = self._build_monitor_with_trackers(
            {sig.signal_id: sig}
        )
        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        perf.record_outcome.assert_called_once()
        call_kwargs = perf.record_outcome.call_args[1]
        assert call_kwargs["hit_sl"] is False
        assert call_kwargs["hit_tp"] == 0

    @pytest.mark.asyncio
    async def test_no_trackers_does_not_raise(self):
        """TradeMonitor without trackers must still work as before."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.current_price = 29800.0

        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        # No performance_tracker or circuit_breaker passed
        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: {sig.signal_id: sig},
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
        )
        await monitor._evaluate_signal(sig)
        assert sig.signal_id in removed
