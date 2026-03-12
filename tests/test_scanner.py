"""Tests for Scanner – cooldown logic and regime-aware gating."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.scanner import Scanner, _RANGING_ADX_SUPPRESS_THRESHOLD


def _make_scanner(**kwargs) -> Scanner:
    """Create a minimal Scanner instance with mocked dependencies."""
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
        signal_queue=MagicMock(),
        router=MagicMock(),
    )
    defaults.update(kwargs)
    return Scanner(**defaults)


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
