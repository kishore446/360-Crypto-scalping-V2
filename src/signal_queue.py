"""Signal queue with Redis persistence and asyncio.Queue fallback."""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional, Union

from src.redis_client import RedisClient
from src.channels.base import Signal
from src.utils import get_logger

log = get_logger("signal_queue")

QUEUE_KEY = "360crypto:signal_queue"
QUEUE_MAXSIZE = 500


class SignalQueue:
    """Hybrid signal queue with consistent full-queue semantics.

    Uses a Redis LIST when available and an ``asyncio.Queue`` fallback otherwise.
    :meth:`put` returns ``True`` only when the new signal is accepted and returns
    ``False`` when the active backend queue is full; it does not silently evict
    older queued items.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        alert_callback: Optional[Callable[[str], Coroutine[Any, Any, Any]]] = None,
    ) -> None:
        self._redis = redis_client
        self._fallback: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._alert_callback = alert_callback
        self._dropped_signals: int = 0
        self._overflow_events: int = 0
        self._last_dropped_signal_id: str = ""

    def _serialize(self, signal: Signal) -> str:
        d = asdict(signal)
        # datetime → ISO string for JSON
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return json.dumps(d)

    def _deserialize(self, raw: str) -> dict:
        return json.loads(raw)

    def _record_drop(self, signal: Signal, reason: str) -> None:
        self._dropped_signals += 1
        self._overflow_events += 1
        self._last_dropped_signal_id = signal.signal_id
        log.warning(
            "Signal queue drop ({}): {} [drops={}]",
            reason,
            signal.signal_id,
            self._dropped_signals,
        )
        if self._alert_callback and (self._dropped_signals % 10) == 0:
            asyncio.create_task(
                self._alert_callback(
                    "⚠️ Signal queue is dropping items "
                    f"({self._dropped_signals} total drops, latest={signal.signal_id})."
                )
            )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self._redis.mode,
            "dropped_signals": self._dropped_signals,
            "overflow_events": self._overflow_events,
            "last_dropped_signal_id": self._last_dropped_signal_id,
            "fallback_qsize": self._fallback.qsize(),
        }

    async def put(self, signal: Signal) -> bool:
        if self._redis.available:
            try:
                size = await self._redis.client.llen(QUEUE_KEY)  # type: ignore[union-attr,misc]
                if size >= QUEUE_MAXSIZE:
                    self._record_drop(signal, "redis_queue_full")
                    return False
                await self._redis.client.rpush(QUEUE_KEY, self._serialize(signal))  # type: ignore[union-attr,misc]
                return True
            except Exception as exc:
                self._redis.mark_unavailable("signal_queue.put", exc)
        # Fallback
        try:
            self._fallback.put_nowait(signal)
            return True
        except asyncio.QueueFull:
            self._record_drop(signal, "memory_queue_full")
            return False

    async def get(self, timeout: float = 1.0) -> Optional[Union[Signal, dict]]:
        if self._redis.available:
            try:
                result = await self._redis.client.blpop([QUEUE_KEY], timeout=int(timeout))  # type: ignore[union-attr,misc]
                if result:
                    _, raw = result
                    data = self._deserialize(raw)
                    # Return raw dict — caller reconstructs Signal
                    return data
            except Exception as exc:
                self._redis.mark_unavailable("signal_queue.get", exc)
        # Fallback
        try:
            return await asyncio.wait_for(self._fallback.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def qsize(self) -> int:
        if self._redis.available:
            try:
                return await self._redis.client.llen(QUEUE_KEY)  # type: ignore[union-attr,misc]
            except Exception as exc:
                self._redis.mark_unavailable("signal_queue.qsize", exc)
        return self._fallback.qsize()

    def put_nowait(self, signal: Signal) -> bool:
        """Best-effort synchronous put.

        In Redis mode, enqueue success cannot be confirmed synchronously, so this
        method returns ``False`` instead of reporting a misleading success.
        Callers that need confirmed acceptance must use :meth:`put`.
        """
        if self._redis.available:
            log.debug(
                "SignalQueue.put_nowait cannot confirm Redis enqueue for {}; use await put()",
                signal.signal_id,
            )
            return False
        try:
            self._fallback.put_nowait(signal)
            return True
        except asyncio.QueueFull:
            self._record_drop(signal, "memory_queue_full")
            return False

    async def empty(self) -> bool:
        return (await self.qsize()) == 0
