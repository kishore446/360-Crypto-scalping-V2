"""Signal router – queue-based decoupled architecture.

Scanner → queue → Router → Telegram

The router:
  1. Consumes signals from an asyncio.Queue
  2. Enriches them with AI/predictive, confidence, risk
  3. Applies channel-specific min-confidence filter
  4. Posts to the appropriate Telegram channel
  5. Selects top 1–2 for the free channel
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from config import ALL_CHANNELS, CHANNEL_TELEGRAM_MAP, TELEGRAM_FREE_CHANNEL_ID
from src.channels.base import Signal
from src.risk import RiskManager
from src.smc import Direction
from src.utils import get_logger

log = get_logger("signal_router")


def _signal_from_dict(data: dict) -> Optional[Signal]:
    """Reconstruct a Signal from a Redis-deserialized dict."""
    try:
        d = data.copy()
        if isinstance(d.get("direction"), str):
            d["direction"] = Direction(d["direction"])
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return Signal(**d)
    except Exception as exc:
        log.warning("Failed to reconstruct Signal from dict: %s", exc)
        return None


class SignalRouter:
    """Consumes signals from a queue, scores, filters, and dispatches."""

    def __init__(
        self,
        queue: Any,
        send_telegram: Callable[[str, str], Coroutine],
        format_signal: Callable[[Signal], str],
    ) -> None:
        self._queue = queue
        self._send_telegram = send_telegram
        self._format_signal = format_signal
        self._active_signals: Dict[str, Signal] = {}
        self._daily_best: List[Signal] = []  # for free channel
        self._position_lock: Dict[str, Direction] = {}  # symbol → direction
        self._running = False
        self._free_limit: int = 2  # max daily free signals
        self._risk_mgr = RiskManager()
        # Detect whether queue.get() supports a timeout keyword argument
        self._queue_has_timeout = "timeout" in inspect.signature(queue.get).parameters

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        log.info("Signal router started")
        while self._running:
            try:
                if self._queue_has_timeout:
                    signal = await self._queue.get(timeout=1.0)
                    if signal is None:
                        continue
                else:
                    signal = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Router error: %s", exc)
                continue

            # Reconstruct Signal from dict (Redis deserialization path)
            if isinstance(signal, dict):
                signal = _signal_from_dict(signal)
                if signal is None:
                    continue

            await self._process(signal)

    async def stop(self) -> None:
        self._running = False
        log.info("Signal router stopped")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _process(self, signal: Signal) -> None:
        # Correlation lock
        existing_dir = self._position_lock.get(signal.symbol)
        if existing_dir and existing_dir != signal.direction:
            log.info(
                "Blocked %s %s – opposing %s position open",
                signal.symbol, signal.direction.value, existing_dir.value,
            )
            return

        # Channel min-confidence filter
        chan_cfg = next(
            (c for c in ALL_CHANNELS if c.name == signal.channel), None
        )
        if chan_cfg and signal.confidence < chan_cfg.min_confidence:
            log.debug(
                "Signal %s %s confidence %.1f < min %.1f – skipped",
                signal.channel, signal.symbol,
                signal.confidence, chan_cfg.min_confidence,
            )
            return

        # Risk assessment (performed before registration so the current signal
        # is not counted against its own concurrent-signal limits)
        # indicators and volume_24h_usd are not available at router level;
        # the router uses defaults ({} and 0) – actual risk values are set
        # by the scanner before the signal is enqueued.
        risk = self._risk_mgr.calculate_risk(
            signal, {}, volume_24h_usd=0, active_signals=self.active_signals
        )
        if not risk.allowed:
            log.warning(
                "Signal %s %s blocked by risk manager: %s",
                signal.symbol, signal.direction.value, risk.reason,
            )
            return
        signal.risk_label = risk.risk_label

        # Register
        self._active_signals[signal.signal_id] = signal
        self._position_lock[signal.symbol] = signal.direction

        # Format and send to premium channel
        text = self._format_signal(signal)
        channel_id = CHANNEL_TELEGRAM_MAP.get(signal.channel, "")
        if channel_id:
            await self._send_telegram(channel_id, text)
            log.info("Signal posted → %s | %s %s", signal.channel, signal.symbol, signal.direction.value)

        # Track for daily free-channel picks
        self._daily_best.append(signal)
        self._daily_best.sort(key=lambda s: s.confidence, reverse=True)
        self._trim_daily_best()

    # ------------------------------------------------------------------
    # Free-channel publication (call once/day or on demand)
    # ------------------------------------------------------------------

    def _trim_daily_best(self) -> None:
        """Trim ``_daily_best`` to the current free-signal limit."""
        self._daily_best = self._daily_best[:max(self._free_limit, 1)]

    def set_free_limit(self, limit: int) -> None:
        """Update the maximum number of daily free signals."""
        self._free_limit = max(0, limit)
        self._trim_daily_best()

    async def publish_free_signals(self) -> None:
        """Post the top free signals of the day to the free channel."""
        if not self._daily_best or not TELEGRAM_FREE_CHANNEL_ID:
            return
        for sig in self._daily_best:
            text = self._format_signal(sig)
            header = "🆓 *FREE SIGNAL OF THE DAY* 🆓\n\n"
            footer = (
                "\n\n📚 _Tip: Scalping requires discipline. "
                "Always use a stop-loss and manage risk._"
            )
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, header + text + footer)
        self._daily_best.clear()

    # ------------------------------------------------------------------
    # Active-signal helpers
    # ------------------------------------------------------------------

    @property
    def active_signals(self) -> Dict[str, Signal]:
        return dict(self._active_signals)

    def remove_signal(self, signal_id: str) -> None:
        sig = self._active_signals.pop(signal_id, None)
        if sig:
            self._position_lock.pop(sig.symbol, None)

    def update_signal(self, signal_id: str, **kwargs) -> None:
        sig = self._active_signals.get(signal_id)
        if sig:
            for k, v in kwargs.items():
                if hasattr(sig, k):
                    setattr(sig, k, v)
