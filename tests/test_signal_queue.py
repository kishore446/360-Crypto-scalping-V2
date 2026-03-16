"""Focused tests for SignalQueue Redis acceptance semantics."""

from __future__ import annotations

import json

import pytest

from src.channels.base import Signal
from src.signal_queue import QUEUE_KEY, SignalQueue
from src.smc import Direction
from src.utils import utcnow


def _make_signal(signal_id: str = "TEST-SIGNAL-001") -> Signal:
    return Signal(
        channel="360_SCALP",
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry=32000,
        stop_loss=31900,
        tp1=32100,
        tp2=32200,
        confidence=85,
        signal_id=signal_id,
        timestamp=utcnow(),
    )


class FakeRedisBackend:
    def __init__(self, size: int) -> None:
        self.available = True
        self.mode = "redis"
        self.client = self
        self._size = size
        self.llen_calls = 0
        self.rpush_calls = 0
        self.rpush_payloads: list[tuple[str, str]] = []
        self.marked_unavailable: list[tuple[str, Exception | None]] = []

    async def llen(self, key: str) -> int:
        assert key == QUEUE_KEY
        self.llen_calls += 1
        return self._size

    async def rpush(self, key: str, payload: str) -> int:
        assert key == QUEUE_KEY
        self.rpush_calls += 1
        self.rpush_payloads.append((key, payload))
        self._size += 1
        return self._size

    async def blpop(self, keys: list[str], timeout: int):
        return None

    def mark_unavailable(self, where: str, exc: Exception | None = None) -> None:
        self.available = False
        self.mode = "memory"
        self.marked_unavailable.append((where, exc))


@pytest.mark.asyncio
async def test_redis_full_queue_returns_false():
    redis_backend = FakeRedisBackend(size=500)
    queue = SignalQueue(redis_backend)

    ok = await queue.put(_make_signal("REDIS-FULL-1"))

    assert ok is False
    stats = queue.stats()
    assert stats["dropped_signals"] == 1
    assert stats["overflow_events"] == 1
    assert redis_backend.rpush_calls == 0


@pytest.mark.asyncio
async def test_redis_full_queue_records_drop_count():
    redis_backend = FakeRedisBackend(size=500)
    queue = SignalQueue(redis_backend)

    await queue.put(_make_signal("REDIS-FULL-COUNT"))

    stats = queue.stats()
    assert stats["dropped_signals"] == 1
    assert stats["overflow_events"] == 1


@pytest.mark.asyncio
async def test_redis_successful_enqueue_returns_true():
    redis_backend = FakeRedisBackend(size=0)
    queue = SignalQueue(redis_backend)

    ok = await queue.put(_make_signal("REDIS-OK-1"))

    assert ok is True
    stats = queue.stats()
    assert stats["dropped_signals"] == 0
    assert redis_backend.rpush_calls == 1
    _, payload = redis_backend.rpush_payloads[0]
    assert json.loads(payload)["signal_id"] == "REDIS-OK-1"


@pytest.mark.asyncio
async def test_redis_overflow_records_last_dropped_signal_id():
    redis_backend = FakeRedisBackend(size=500)
    queue = SignalQueue(redis_backend)

    await queue.put(_make_signal("REDIS-OVERFLOW-ID"))

    assert queue.stats()["last_dropped_signal_id"] == "REDIS-OVERFLOW-ID"
    assert redis_backend.rpush_calls == 0
