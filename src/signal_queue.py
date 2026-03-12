"""Signal queue with Redis persistence and asyncio.Queue fallback."""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union

from src.redis_client import RedisClient
from src.channels.base import Signal
from src.utils import get_logger

log = get_logger("signal_queue")

QUEUE_KEY = "360crypto:signal_queue"
QUEUE_MAXSIZE = 500


class SignalQueue:
    """Hybrid signal queue: uses Redis LIST when available, asyncio.Queue otherwise."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._fallback: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    def _serialize(self, signal: Signal) -> str:
        d = asdict(signal)
        # datetime → ISO string for JSON
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return json.dumps(d)

    def _deserialize(self, raw: str) -> dict:
        return json.loads(raw)

    async def put(self, signal: Signal) -> None:
        if self._redis.available:
            try:
                await self._redis.client.rpush(QUEUE_KEY, self._serialize(signal))  # type: ignore[union-attr,misc]
                # Cap queue length
                await self._redis.client.ltrim(QUEUE_KEY, -QUEUE_MAXSIZE, -1)  # type: ignore[union-attr,misc]
                return
            except Exception as exc:
                log.warning("Redis put failed (%s), falling back to memory queue.", exc)
        # Fallback
        try:
            self._fallback.put_nowait(signal)
        except asyncio.QueueFull:
            log.warning("In-memory signal queue full, dropping signal %s", signal.signal_id)

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
                log.warning("Redis get failed (%s), falling back to memory queue.", exc)
        # Fallback
        try:
            return await asyncio.wait_for(self._fallback.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def qsize(self) -> int:
        if self._redis.available:
            try:
                return await self._redis.client.llen(QUEUE_KEY)  # type: ignore[union-attr,misc]
            except Exception:
                pass
        return self._fallback.qsize()

    def put_nowait(self, signal: Signal) -> None:
        """Sync put — enqueues to fallback queue. For Redis, use async put()."""
        if self._redis.available:
            # Schedule async push without blocking
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.put(signal))
                return
            except RuntimeError:
                pass
        try:
            self._fallback.put_nowait(signal)
        except asyncio.QueueFull:
            log.warning("In-memory signal queue full, dropping signal %s", signal.signal_id)

    async def empty(self) -> bool:
        return (await self.qsize()) == 0
