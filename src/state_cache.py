"""Redis-backed state cache with TTL support."""

import json
from typing import Any, Optional

from src.redis_client import RedisClient
from src.utils import get_logger

log = get_logger("state_cache")

CACHE_PREFIX = "360crypto:cache:"


class StateCache:
    """Key-value cache backed by Redis, with optional TTL."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._local: dict = {}  # in-memory fallback

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if self._redis.available:
            try:
                if ttl > 0:
                    await self._redis.client.setex(f"{CACHE_PREFIX}{key}", ttl, serialized)  # type: ignore[union-attr]
                else:
                    await self._redis.client.set(f"{CACHE_PREFIX}{key}", serialized)  # type: ignore[union-attr]
                return
            except Exception as exc:
                log.warning("Redis set failed for %s: %s", key, exc)
        self._local[key] = serialized

    async def get(self, key: str) -> Optional[str]:
        if self._redis.available:
            try:
                return await self._redis.client.get(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("Redis get failed for %s: %s", key, exc)
        return self._local.get(key)

    async def delete(self, key: str) -> None:
        if self._redis.available:
            try:
                await self._redis.client.delete(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception:
                pass
        self._local.pop(key, None)

    async def incr(self, key: str) -> int:
        if self._redis.available:
            try:
                return await self._redis.client.incr(f"{CACHE_PREFIX}{key}")  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            val = int(self._local.get(key, 0)) + 1
        except (ValueError, TypeError):
            val = 1
        self._local[key] = str(val)
        return val
