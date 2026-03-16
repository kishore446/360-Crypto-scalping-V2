"""Historical data seeding – fetch OHLCV and recent trades on boot.

Uses public Binance REST endpoints with rate-limit-compliant delays.
Supports disk-based caching so restarts only fetch the data that is
missing since the last snapshot (gap-fill), cutting boot times from
3-5 minutes down to ~15 seconds after a brief outage.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    BATCH_REQUEST_DELAY,
    SEED_TICK_LIMIT,
    SEED_TIMEFRAMES,
)
from src.binance import BinanceClient
from src.pair_manager import PairManager
from src.utils import get_logger

log = get_logger("historical")

# ---------------------------------------------------------------------------
# Disk-cache paths
# ---------------------------------------------------------------------------
CACHE_DIR = Path("data/cache")
_TICKS_DIR = CACHE_DIR / "ticks"
_META_FILE = CACHE_DIR / "metadata.json"

# Maximum candles to retain per symbol-timeframe bucket
_MAX_CANDLES_PER_BUCKET: int = 1000
# Seconds per candle interval — used to estimate how many candles are missing
_INTERVAL_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "4h": 14_400,
}

# If the cache is older than this many seconds, do a full re-seed for that
# symbol/timeframe rather than trying to gap-fill (avoids huge limit requests)
_MAX_GAP_FILL_SECONDS = 24 * 3_600  # 24 hours

# Sentinel returned by _estimate_gap_candles to signal "cache too stale, do full fetch"
_FULL_FETCH_SENTINEL = 9_999

# Extra candles fetched on top of the estimated gap to cover partial candles and clock skew
_GAP_BUFFER_CANDLES = 5


class HistoricalDataStore:
    """In-memory store for OHLCV and tick data, keyed by symbol + timeframe."""

    def __init__(self) -> None:
        # candles[symbol][timeframe] = {"open": [], "high": [], "low": [], "close": [], "volume": []}
        self.candles: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        # ticks[symbol] = [{"price": float, "qty": float, "isBuyerMaker": bool, "time": int}, …]
        self.ticks: Dict[str, List[Dict[str, Any]]] = {}
        self._client = BinanceClient("spot")

    # ------------------------------------------------------------------
    # OHLCV fetch
    # ------------------------------------------------------------------

    async def fetch_candles(
        self, symbol: str, interval: str, limit: int, market: str = "spot",
    ) -> Dict[str, np.ndarray]:
        """Fetch OHLCV candles for one symbol/interval."""
        if market == "spot":
            client = self._client
            close_after = False
        else:
            client = BinanceClient(market)
            close_after = True
        try:
            raw = await client.fetch_klines(symbol, interval, limit)
        except Exception as exc:
            log.error("Candle fetch error %s %s: %s", symbol, interval, exc)
            return {}
        finally:
            if close_after:
                await client.close()

        if not raw:
            return {}

        opens = np.array([float(c[1]) for c in raw])
        highs = np.array([float(c[2]) for c in raw])
        lows = np.array([float(c[3]) for c in raw])
        closes = np.array([float(c[4]) for c in raw])
        volumes = np.array([float(c[5]) for c in raw])

        return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}

    # ------------------------------------------------------------------
    # Recent trades fetch
    # ------------------------------------------------------------------

    async def fetch_recent_trades(
        self, symbol: str, limit: int = SEED_TICK_LIMIT, market: str = "spot",
    ) -> List[Dict[str, Any]]:
        if market == "spot":
            client = self._client
            close_after = False
        else:
            client = BinanceClient(market)
            close_after = True
        capped_limit = min(limit, 1000)
        try:
            if market == "futures":
                raw = await client._get(
                    "/fapi/v1/trades",
                    params={"symbol": symbol, "limit": capped_limit},
                    weight=1,
                )
            else:
                raw = await client._get(
                    "/api/v3/trades",
                    params={"symbol": symbol, "limit": capped_limit},
                    weight=1,
                )
        except Exception as exc:
            log.error("Trade fetch error %s: %s", symbol, exc)
            return []
        finally:
            if close_after:
                await client.close()

        if not raw:
            return []

        return [
            {
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "isBuyerMaker": t.get("isBuyerMaker", False),
                "time": t.get("time", 0),
            }
            for t in raw
        ]

    # ------------------------------------------------------------------
    # Full seed for one symbol
    # ------------------------------------------------------------------

    async def seed_symbol(self, symbol: str, market: str = "spot") -> None:
        """Seed all timeframes + ticks for a single symbol."""
        self.candles.setdefault(symbol, {})

        for tf in SEED_TIMEFRAMES:
            data = await self.fetch_candles(symbol, tf.interval, tf.limit, market)
            if data:
                self.candles[symbol][tf.interval] = data
                log.debug("Seeded %s %s: %d candles", symbol, tf.interval, len(data["close"]))
            await asyncio.sleep(BATCH_REQUEST_DELAY)

        ticks = await self.fetch_recent_trades(symbol, SEED_TICK_LIMIT, market)
        if ticks:
            self.ticks[symbol] = ticks
            log.debug("Seeded %s ticks: %d", symbol, len(ticks))
        await asyncio.sleep(BATCH_REQUEST_DELAY)

    # ------------------------------------------------------------------
    # Full boot seed
    # ------------------------------------------------------------------

    async def seed_all(self, pair_mgr: PairManager) -> None:
        """Seed historical data for every active pair."""
        log.info("Starting historical data seed for %d pairs …", len(pair_mgr.pairs))
        for sym, info in pair_mgr.pairs.items():
            await self.seed_symbol(sym, info.market)
            for tf_name, data in self.candles.get(sym, {}).items():
                pair_mgr.record_candles(sym, tf_name, len(data.get("close", [])))
        log.info("Historical data seed complete.")

    # ------------------------------------------------------------------
    # Disk-cache: save
    # ------------------------------------------------------------------

    async def save_snapshot(self) -> None:
        """Persist current candle and tick data to disk for fast restarts."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _TICKS_DIR.mkdir(parents=True, exist_ok=True)

            saved_at = datetime.now(timezone.utc).isoformat()
            meta: Dict[str, Any] = {}
            saved_count = 0

            for symbol, timeframes in self.candles.items():
                for interval, arrays in timeframes.items():
                    if not arrays or "close" not in arrays or len(arrays["close"]) == 0:
                        continue
                    try:
                        path = CACHE_DIR / f"{symbol}_{interval}.npz"
                        np.savez_compressed(
                            path,
                            open=arrays["open"],
                            high=arrays["high"],
                            low=arrays["low"],
                            close=arrays["close"],
                            volume=arrays["volume"],
                        )
                        key = f"{symbol}:{interval}"
                        meta[key] = {
                            "count": int(len(arrays["close"])),
                            "saved_at": saved_at,
                        }
                        saved_count += 1
                    except Exception as exc:  # pragma: no cover
                        log.warning("Snapshot save failed for %s %s: %s", symbol, interval, exc)

            for symbol, ticks in self.ticks.items():
                if not ticks:
                    continue
                try:
                    tick_path = _TICKS_DIR / f"{symbol}.json"
                    with tick_path.open("w", encoding="utf-8") as fh:
                        json.dump(ticks, fh)
                except Exception as exc:  # pragma: no cover
                    log.warning("Tick snapshot save failed for %s: %s", symbol, exc)

            with _META_FILE.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh)

            log.info("Snapshot saved: %d symbol-timeframe combos (saved_at=%s)", saved_count, saved_at)
        except Exception as exc:  # pragma: no cover
            log.error("save_snapshot error: %s", exc)

    # ------------------------------------------------------------------
    # Disk-cache: load
    # ------------------------------------------------------------------

    def load_snapshot(self) -> bool:
        """Load cached candle and tick data from disk.

        Returns True if cache was found and loaded, False otherwise.
        """
        try:
            if not _META_FILE.exists():
                return False

            with _META_FILE.open("r", encoding="utf-8") as fh:
                meta: Dict[str, Any] = json.load(fh)

            if not meta:
                return False

            loaded_count = 0
            for key, info in meta.items():
                try:
                    symbol, interval = key.split(":", 1)
                    path = CACHE_DIR / f"{symbol}_{interval}.npz"
                    if not path.exists():
                        log.warning("Cache file missing: %s — skipping", path)
                        continue
                    with np.load(path, allow_pickle=False) as data:
                        self.candles.setdefault(symbol, {})[interval] = {
                            k: np.asarray(data[k], dtype=np.float64).ravel()
                            for k in ("open", "high", "low", "close", "volume")
                        }
                    loaded_count += 1
                except Exception as exc:
                    log.warning("Failed to load cache for %s: %s — skipping", key, exc)

            for tick_file in _TICKS_DIR.glob("*.json"):
                symbol = tick_file.stem
                try:
                    with tick_file.open("r", encoding="utf-8") as fh:
                        self.ticks[symbol] = json.load(fh)
                except Exception as exc:
                    log.warning("Failed to load ticks for %s: %s — skipping", symbol, exc)

            log.info("Snapshot loaded: %d symbol-timeframe combos from disk", loaded_count)
            return loaded_count > 0
        except Exception as exc:  # pragma: no cover
            log.error("load_snapshot error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Disk-cache: gap-fill
    # ------------------------------------------------------------------

    async def gap_fill(self, pair_mgr: PairManager) -> None:
        """Fetch only the candles missing since the last snapshot.

        For each symbol+timeframe already in cache, calculates how many
        candles are needed based on the elapsed time since ``saved_at``,
        then fetches and merges them.  For symbol+timeframe combos not in
        cache, a full seed is performed.
        """
        try:
            if not _META_FILE.exists():
                log.info("No metadata found — falling back to full seed")
                await self.seed_all(pair_mgr)
                return

            with _META_FILE.open("r", encoding="utf-8") as fh:
                meta: Dict[str, Any] = json.load(fh)
        except Exception as exc:
            log.error("gap_fill: cannot read metadata (%s) — falling back to full seed", exc)
            await self.seed_all(pair_mgr)
            return

        log.info("Gap-filling %d pairs …", len(pair_mgr.pairs))
        for sym, info in pair_mgr.pairs.items():
            self.candles.setdefault(sym, {})

            for tf in SEED_TIMEFRAMES:
                key = f"{sym}:{tf.interval}"
                if key in meta and sym in self.candles and tf.interval in self.candles[sym]:
                    saved_at_iso = meta[key].get("saved_at", "")
                    gap = self._estimate_gap_candles(saved_at_iso, tf.interval)
                    if gap >= tf.limit:
                        # Cache is stale — do a full fetch for this timeframe
                        log.debug(
                            "Cache stale for %s %s (gap=%d >= limit=%d) — full fetch",
                            sym, tf.interval, gap, tf.limit,
                        )
                        data = await self.fetch_candles(sym, tf.interval, tf.limit, info.market)
                        if data:
                            self.candles[sym][tf.interval] = data
                    elif gap > 0:
                        new_data = await self.fetch_candles(sym, tf.interval, gap, info.market)
                        if new_data:
                            self.candles[sym][tf.interval] = self._merge_candles(
                                self.candles[sym][tf.interval], new_data, tf.limit
                            )
                            log.debug(
                                "Gap-filled %s %s: fetched %d, total %d",
                                sym, tf.interval, len(new_data["close"]),
                                len(self.candles[sym][tf.interval]["close"]),
                            )
                    else:
                        log.debug("No gap needed for %s %s", sym, tf.interval)
                else:
                    # Not in cache — full fetch
                    data = await self.fetch_candles(sym, tf.interval, tf.limit, info.market)
                    if data:
                        self.candles[sym][tf.interval] = data
                        log.debug("Full-seeded (no cache) %s %s: %d candles", sym, tf.interval, len(data["close"]))
                await asyncio.sleep(BATCH_REQUEST_DELAY)

            # Always refresh ticks — they are cheap
            ticks = await self.fetch_recent_trades(sym, SEED_TICK_LIMIT, info.market)
            if ticks:
                self.ticks[sym] = ticks
            await asyncio.sleep(BATCH_REQUEST_DELAY)

            for tf_name, data in self.candles.get(sym, {}).items():
                pair_mgr.record_candles(sym, tf_name, len(data.get("close", [])))

        log.info("Gap-fill complete.")

    # ------------------------------------------------------------------
    # Disk-cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_gap_candles(saved_at_iso: str, interval: str) -> int:
        """Return the number of candles needed to cover the gap since *saved_at_iso*.

        A small buffer (+5) is added to account for partial candles and
        clock skew.  If the timestamp cannot be parsed, returns a large
        number so the caller falls back to a full fetch.
        """
        if not saved_at_iso:
            return _FULL_FETCH_SENTINEL
        try:
            saved_dt = datetime.fromisoformat(saved_at_iso)
            now_dt = datetime.now(timezone.utc)
            if saved_dt.tzinfo is None:
                saved_dt = saved_dt.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (now_dt - saved_dt).total_seconds())
            if elapsed > _MAX_GAP_FILL_SECONDS:
                return _FULL_FETCH_SENTINEL
            interval_secs = _INTERVAL_SECONDS.get(interval, 60)
            return int(elapsed / interval_secs) + _GAP_BUFFER_CANDLES
        except Exception:
            return _FULL_FETCH_SENTINEL

    @staticmethod
    def _merge_candles(
        existing: Dict[str, np.ndarray],
        new_data: Dict[str, np.ndarray],
        limit: int,
    ) -> Dict[str, np.ndarray]:
        """Append *new_data* arrays to *existing* and trim to *limit* candles."""
        result: Dict[str, np.ndarray] = {}
        for key in ("open", "high", "low", "close", "volume"):
            combined = np.concatenate([existing.get(key, np.array([])), new_data.get(key, np.array([]))])
            result[key] = combined[-limit:] if len(combined) > limit else combined
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str, interval: str) -> Optional[Dict[str, np.ndarray]]:
        return self.candles.get(symbol, {}).get(interval)

    def has_data(self) -> bool:
        """Return True if the store has any seeded candle data."""
        return bool(self.candles)

    def update_candle(self, symbol: str, interval: str, candle: Dict[str, float]) -> None:
        """Append a single candle (from WebSocket) to the store."""
        bucket = self.candles.setdefault(symbol, {}).setdefault(
            interval,
            {k: np.empty(0, dtype=np.float64) for k in ("open", "high", "low", "close", "volume")},
        )
        for key in ("open", "high", "low", "close", "volume"):
            arr = bucket[key]
            arr = np.append(arr, candle.get(key, 0.0))
            if len(arr) > _MAX_CANDLES_PER_BUCKET:
                arr = arr[-_MAX_CANDLES_PER_BUCKET:]
            bucket[key] = arr

    def append_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        self.ticks.setdefault(symbol, []).append(tick)
        # Keep only the last SEED_TICK_LIMIT ticks
        if len(self.ticks[symbol]) > SEED_TICK_LIMIT:
            self.ticks[symbol] = self.ticks[symbol][-SEED_TICK_LIMIT:]

    async def close(self) -> None:
        await self._client.close()
