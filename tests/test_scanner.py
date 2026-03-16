"""Tests for Scanner – cooldown logic and regime-aware gating."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.regime import MarketRegime
from src.scanner import Scanner, _RANGING_ADX_SUPPRESS_THRESHOLD
from src.signal_quality import (
    ExecutionAssessment,
    MarketState,
    PairQualityAssessment,
    QualityTier,
    RiskAssessment,
    SetupAssessment,
    SetupClass,
)
from src.smc import Direction
from src.utils import utcnow


def _make_scanner(**kwargs) -> Scanner:
    """Create a minimal Scanner instance with mocked dependencies."""
    signal_queue = MagicMock()
    signal_queue.put = AsyncMock(return_value=True)
    defaults = dict(
        pair_mgr=MagicMock(),
        data_store=MagicMock(),
        channels=[],
        smc_detector=MagicMock(),
        regime_detector=MagicMock(),
        predictive=MagicMock(),
        exchange_mgr=MagicMock(),
        spot_client=None,
        telemetry=MagicMock(),
        signal_queue=signal_queue,
        router=MagicMock(active_signals={}),
    )
    defaults.update(kwargs)
    return Scanner(**defaults)


def _candles(length: int = 40) -> dict:
    base = [float(i + 1) for i in range(length)]
    return {
        "high": base,
        "low": [max(v - 0.5, 0.1) for v in base],
        "close": base,
        "volume": [100.0 for _ in base],
    }


def _make_signal(
    *,
    channel: str = "360_SCALP",
    signal_id: str = "SIG-001",
    confidence: float = 10.0,
) -> Signal:
    return Signal(
        channel=channel,
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        confidence=confidence,
        signal_id=signal_id,
        timestamp=utcnow(),
    )


def _make_scan_ready_scanner(
    *,
    channel: MagicMock,
    signal_queue: MagicMock,
    predictive: MagicMock | None = None,
    openai_evaluator: MagicMock | None = None,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
) -> Scanner:
    smc_result = SimpleNamespace(
        sweeps=[SimpleNamespace(direction=Direction.LONG, sweep_level=95.0)],
        fvg=[],
        mss=SimpleNamespace(direction=Direction.LONG, midpoint=98.0),
        as_dict=lambda: {
            "sweeps": [SimpleNamespace(direction=Direction.LONG, sweep_level=95.0)],
            "fvg": [],
            "mss": SimpleNamespace(direction=Direction.LONG, midpoint=98.0),
        },
    )
    if predictive is None:
        predictive = MagicMock(
            predict=AsyncMock(
                return_value=SimpleNamespace(
                    confidence_adjustment=0.0,
                    predicted_direction="NEUTRAL",
                    suggested_tp_adjustment=1.0,
                    suggested_sl_adjustment=1.0,
                )
            ),
            adjust_tp_sl=MagicMock(),
            update_confidence=MagicMock(),
        )

    return _make_scanner(
        pair_mgr=MagicMock(has_enough_history=MagicMock(return_value=True)),
        data_store=MagicMock(
            get_candles=MagicMock(side_effect=lambda _symbol, _interval: _candles()),
            ticks={"BTCUSDT": []},
        ),
        channels=[channel],
        smc_detector=MagicMock(detect=MagicMock(return_value=smc_result)),
        regime_detector=MagicMock(
            classify=MagicMock(return_value=SimpleNamespace(regime=regime))
        ),
        predictive=predictive,
        exchange_mgr=MagicMock(
            verify_signal_cross_exchange=AsyncMock(return_value=True)
        ),
        spot_client=MagicMock(
            fetch_order_book=AsyncMock(
                return_value={"bids": [["100.0", "1"]], "asks": [["100.01", "1"]]}
            )
        ),
        signal_queue=signal_queue,
        router=MagicMock(active_signals={}),
        openai_evaluator=openai_evaluator,
        onchain_client=MagicMock(get_exchange_flow=AsyncMock(return_value=None)),
    )


def _setup_pass() -> SetupAssessment:
    return SetupAssessment(
        setup_class=SetupClass.BREAKOUT_RETEST,
        thesis="Breakout Retest",
        channel_compatible=True,
        regime_compatible=True,
    )


def _execution_pass() -> ExecutionAssessment:
    return ExecutionAssessment(
        passed=True,
        trigger_confirmed=True,
        extension_ratio=0.6,
        anchor_price=99.0,
        entry_zone="99.0000 – 100.0000",
        execution_note="Retest hold confirmed.",
    )


def _risk_pass() -> RiskAssessment:
    return RiskAssessment(
        passed=True,
        stop_loss=95.0,
        tp1=106.5,
        tp2=111.5,
        tp3=117.0,
        r_multiple=1.3,
        invalidation_summary="Below 96.0000 structure + volatility buffer",
    )


class TestScannerCooldown:
    def test_no_cooldown_initially(self):
        scanner = _make_scanner()
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is False

    def test_cooldown_active_after_set(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is True

    def test_cooldown_expires(self):
        scanner = _make_scanner()
        # Manually set an already-expired cooldown
        scanner._cooldown_until[("BTCUSDT", "360_SCALP")] = (
            time.monotonic() - 1  # 1 second in the past
        )
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is False

    def test_cooldown_expires_cleans_up(self):
        scanner = _make_scanner()
        scanner._cooldown_until[("BTCUSDT", "360_SCALP")] = (
            time.monotonic() - 1
        )
        scanner._is_in_cooldown("BTCUSDT", "360_SCALP")
        assert ("BTCUSDT", "360_SCALP") not in scanner._cooldown_until

    def test_cooldown_separate_per_channel(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is True
        assert scanner._is_in_cooldown("BTCUSDT", "360_SWING") is False

    def test_cooldown_separate_per_symbol(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("ETHUSDT", "360_SCALP") is False

    def test_cooldown_duration_from_config(self):
        from config import SIGNAL_SCAN_COOLDOWN_SECONDS
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        expiry = scanner._cooldown_until[("BTCUSDT", "360_SCALP")]
        expected_duration = SIGNAL_SCAN_COOLDOWN_SECONDS.get("360_SCALP", 300)
        actual_duration = expiry - time.monotonic()
        assert abs(actual_duration - expected_duration) < 2  # within 2 seconds


class TestScannerCircuitBreaker:
    def test_circuit_breaker_not_set_by_default(self):
        scanner = _make_scanner()
        assert scanner.circuit_breaker is None

    @pytest.mark.asyncio
    async def test_scan_loop_skips_when_tripped(self):
        """Scan loop should skip evaluation when circuit breaker is tripped."""
        scanner = _make_scanner()
        cb = MagicMock()
        cb.is_tripped.return_value = True
        scanner.circuit_breaker = cb

        # Patch asyncio.sleep to avoid infinite loop
        sleep_count = 0

        async def mock_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        with patch("src.scanner.asyncio.sleep", side_effect=mock_sleep):
            try:
                await scanner.scan_loop()
            except asyncio.CancelledError:
                pass

        # pair_mgr should NOT have been accessed (scan was skipped)
        scanner.pair_mgr.pairs.items.assert_not_called()


class TestScannerRegimeGating:
    def test_ranging_adx_threshold_constant(self):
        assert _RANGING_ADX_SUPPRESS_THRESHOLD == 15.0

    def test_scanner_has_paused_channels_attribute(self):
        scanner = _make_scanner()
        assert isinstance(scanner.paused_channels, set)

    def test_scanner_has_confidence_overrides_attribute(self):
        scanner = _make_scanner()
        assert isinstance(scanner.confidence_overrides, dict)

    def test_scanner_paused_channels_shared_with_external_set(self):
        shared = set()
        scanner = _make_scanner()
        scanner.paused_channels = shared
        shared.add("360_SCALP")
        assert "360_SCALP" in scanner.paused_channels


class TestScannerAttributes:
    def test_force_scan_starts_false(self):
        scanner = _make_scanner()
        assert scanner.force_scan is False

    def test_force_scan_can_be_set(self):
        scanner = _make_scanner()
        scanner.force_scan = True
        assert scanner.force_scan is True

    def test_ws_spot_starts_none(self):
        scanner = _make_scanner()
        assert scanner.ws_spot is None

    def test_ws_futures_starts_none(self):
        scanner = _make_scanner()
        assert scanner.ws_futures is None


class TestScannerConfidencePipeline:
    @pytest.mark.asyncio
    async def test_adjustments_persist_and_final_clamp_applies_last(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_RANGE", min_confidence=60.0)
        channel.evaluate.return_value = _make_signal(channel="360_RANGE", signal_id="SIG-001")

        predictive = MagicMock()
        predictive.predict = AsyncMock(
            return_value=SimpleNamespace(
                confidence_adjustment=7.0,
                predicted_direction="UP",
                suggested_tp_adjustment=1.0,
                suggested_sl_adjustment=1.0,
            )
        )
        predictive.adjust_tp_sl = MagicMock()
        predictive.update_confidence = MagicMock()

        def _update_confidence(signal, _prediction):
            # Base confidence (55) plus the RANGE ranging boost (+5) must be in
            # place before predictive adjustments run.
            assert signal.confidence == 60.0
            signal.confidence += 7.0

        predictive.update_confidence.side_effect = _update_confidence

        openai_evaluator = MagicMock()
        openai_evaluator.enabled = True
        openai_evaluator.evaluate = AsyncMock(
            return_value=SimpleNamespace(
                adjustment=50.0,
                recommended=True,
                reasoning="aligned",
            )
        )
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)

        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            predictive=predictive,
            openai_evaluator=openai_evaluator,
            regime=MarketRegime.RANGING,
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=55.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        queued_signal = signal_queue.put.await_args.args[0]
        assert queued_signal.confidence == 100.0
        openai_evaluator.evaluate.assert_awaited_once()
        assert predictive.adjust_tp_sl.called
        assert predictive.update_confidence.called
        assert openai_evaluator.evaluate.await_args.kwargs["confidence_before"] == queued_signal.pre_ai_confidence
        assert queued_signal.post_ai_confidence == 100.0
        assert queued_signal.setup_class == SetupClass.BREAKOUT_RETEST.value

    @pytest.mark.asyncio
    async def test_signals_below_final_min_confidence_are_rejected_after_all_adjustments(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=80.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-LOW")

        predictive = MagicMock(
            predict=AsyncMock(
                return_value=SimpleNamespace(
                    confidence_adjustment=-5.0,
                    predicted_direction="DOWN",
                    suggested_tp_adjustment=1.0,
                    suggested_sl_adjustment=1.0,
                )
            ),
            adjust_tp_sl=MagicMock(),
            update_confidence=MagicMock(
                side_effect=lambda signal, _prediction: setattr(
                    signal, "confidence", signal.confidence - 5.0
                )
            ),
        )
        openai_evaluator = MagicMock(
            enabled=True,
            evaluate=AsyncMock(
                return_value=SimpleNamespace(
                    adjustment=-10.0,
                    recommended=True,
                    reasoning="weak setup",
                )
            ),
        )
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            predictive=predictive,
            openai_evaluator=openai_evaluator,
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=50.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert openai_evaluator.evaluate.await_args.kwargs["confidence_before"] > 70.0
        signal_queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_skip_prevents_enqueue(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-002")
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            openai_evaluator=MagicMock(
                enabled=True,
                evaluate=AsyncMock(
                    return_value=SimpleNamespace(
                        adjustment=0.0,
                        recommended=False,
                        reasoning="reject",
                    )
                ),
            ),
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=55.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        scanner.signal_queue.put.assert_not_awaited()


class TestScannerEnqueueSemantics:
    @pytest.mark.asyncio
    async def test_cooldown_not_started_when_enqueue_fails(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-DROP")
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=False)
        scanner = _make_scan_ready_scanner(channel=channel, signal_queue=signal_queue)

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=80.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert ("BTCUSDT", "360_SCALP") not in scanner._cooldown_until

    @pytest.mark.asyncio
    async def test_failed_enqueue_does_not_suppress_later_signal(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.side_effect = [
            _make_signal(channel="360_SCALP", signal_id="SIG-FIRST"),
            _make_signal(channel="360_SCALP", signal_id="SIG-SECOND"),
        ]
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(side_effect=[False, True])
        scanner = _make_scan_ready_scanner(channel=channel, signal_queue=signal_queue)

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=80.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert signal_queue.put.await_count == 2
        assert scanner._cooldown_until.get(("BTCUSDT", "360_SCALP")) is not None
