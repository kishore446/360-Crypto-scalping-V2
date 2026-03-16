"""Focused lifecycle tests for CryptoSignalEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def _make_engine():
    with (
        patch("src.main.TelegramBot"),
        patch("src.main.TelemetryCollector"),
        patch("src.main.RedisClient"),
        patch("src.main.SignalQueue"),
        patch("src.main.StateCache"),
        patch("src.main.SignalRouter"),
        patch("src.main.TradeMonitor"),
        patch("src.main.PairManager"),
        patch("src.main.HistoricalDataStore"),
        patch("src.main.PredictiveEngine"),
        patch("src.main.ExchangeManager"),
        patch("src.main.SMCDetector"),
        patch("src.main.MarketRegimeDetector"),
    ):
        from src.main import CryptoSignalEngine

        return CryptoSignalEngine()


class _FakeWebSocket:
    def __init__(self) -> None:
        self.stop = AsyncMock()


@pytest.mark.asyncio
async def test_shutdown_is_idempotent():
    engine = _make_engine()
    engine._bootstrap.shutdown = AsyncMock()

    await engine.shutdown()
    await engine.shutdown()

    engine._bootstrap.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_websockets_if_pair_universe_changed_restarts():
    engine = _make_engine()
    engine.pair_mgr.spot_symbols = ["BTCUSDT"]
    engine.pair_mgr.futures_symbols = ["ETHUSDT"]
    old_spot_ws = _FakeWebSocket()
    old_futures_ws = _FakeWebSocket()
    new_spot_ws = object()
    new_futures_ws = object()
    engine._ws_spot = old_spot_ws
    engine._ws_futures = old_futures_ws

    async def fake_start_websockets() -> None:
        engine._ws_spot = new_spot_ws
        engine._ws_futures = new_futures_ws

    engine._bootstrap.start_websockets = AsyncMock(side_effect=fake_start_websockets)

    await engine._restart_websockets_if_pair_universe_changed(set(), set())

    old_spot_ws.stop.assert_awaited_once()
    old_futures_ws.stop.assert_awaited_once()
    engine._bootstrap.start_websockets.assert_awaited_once()
    assert engine._command_handler.ws_spot is new_spot_ws
    assert engine._command_handler.ws_futures is new_futures_ws


@pytest.mark.asyncio
async def test_restart_websockets_if_pair_universe_unchanged_does_nothing():
    engine = _make_engine()
    engine.pair_mgr.spot_symbols = ["BTCUSDT"]
    engine.pair_mgr.futures_symbols = ["ETHUSDT"]
    current_spot_ws = _FakeWebSocket()
    current_futures_ws = _FakeWebSocket()
    engine._ws_spot = current_spot_ws
    engine._ws_futures = current_futures_ws
    engine._bootstrap.start_websockets = AsyncMock()
    old_spot, old_futures = engine._current_ws_symbol_sets()

    await engine._restart_websockets_if_pair_universe_changed(old_spot, old_futures)

    current_spot_ws.stop.assert_not_awaited()
    current_futures_ws.stop.assert_not_awaited()
    engine._bootstrap.start_websockets.assert_not_awaited()
