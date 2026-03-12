"""Circuit Breaker for rapid-loss protection.

Tracks recent signal outcomes and trips when loss thresholds are exceeded,
pausing all signal generation until manually reset via Telegram command.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

from src.utils import get_logger

log = get_logger("circuit_breaker")

# Default thresholds (overridable via constructor)
_DEFAULT_MAX_CONSECUTIVE_SL: int = 3
_DEFAULT_MAX_HOURLY_SL: int = 5
_DEFAULT_MAX_DAILY_DRAWDOWN_PCT: float = 10.0


@dataclass
class OutcomeRecord:
    """A single recorded signal outcome."""

    signal_id: str
    hit_sl: bool
    pnl_pct: float
    timestamp: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """Tracks signal outcomes and trips when loss thresholds are exceeded.

    Parameters
    ----------
    max_consecutive_sl:
        Maximum consecutive SL hits before tripping.
    max_hourly_sl:
        Maximum SL hits within a rolling 1-hour window before tripping.
    max_daily_drawdown_pct:
        Maximum cumulative PnL loss (%) within a rolling 24-hour window
        before tripping.
    alert_callback:
        Optional async callable that receives a message string.  Used to
        send Telegram alerts when the circuit trips.
    """

    def __init__(
        self,
        max_consecutive_sl: int = _DEFAULT_MAX_CONSECUTIVE_SL,
        max_hourly_sl: int = _DEFAULT_MAX_HOURLY_SL,
        max_daily_drawdown_pct: float = _DEFAULT_MAX_DAILY_DRAWDOWN_PCT,
        alert_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.max_consecutive_sl = max_consecutive_sl
        self.max_hourly_sl = max_hourly_sl
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self._alert_callback = alert_callback

        # Rolling outcome history (keep last 1000 entries)
        self._outcomes: Deque[OutcomeRecord] = deque(maxlen=1000)

        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_time: Optional[float] = None
        self._consecutive_sl: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        signal_id: str,
        hit_sl: bool,
        pnl_pct: float,
    ) -> None:
        """Record the outcome of a completed signal.

        Parameters
        ----------
        signal_id:
            Unique identifier of the signal.
        hit_sl:
            ``True`` if the stop-loss was triggered (a loss).
        pnl_pct:
            Realised PnL as a percentage (negative for losses).
        """
        record = OutcomeRecord(signal_id=signal_id, hit_sl=hit_sl, pnl_pct=pnl_pct)
        self._outcomes.append(record)

        if hit_sl:
            self._consecutive_sl += 1
        else:
            self._consecutive_sl = 0

        self._evaluate()

    def is_tripped(self) -> bool:
        """Return ``True`` when the circuit breaker is active."""
        return self._tripped

    def reset(self) -> None:
        """Manually reset the circuit breaker and clear all counters."""
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = None
        self._consecutive_sl = 0
        log.info("Circuit breaker reset manually.")

    def status_text(self) -> str:
        """Return a human-readable status string."""
        if self._tripped:
            tripped_ago = (
                f"{time.monotonic() - self._trip_time:.0f}s ago"
                if self._trip_time
                else "unknown"
            )
            return (
                f"⚠️ *Circuit Breaker TRIPPED* ({tripped_ago})\n"
                f"Reason: {self._trip_reason}\n"
                f"Use /reset_circuit_breaker to resume signal generation."
            )
        hourly = self._hourly_sl_count()
        daily_dd = self._daily_drawdown_pct()
        return (
            "✅ *Circuit Breaker: OK*\n"
            f"Consecutive SL hits: {self._consecutive_sl}/{self.max_consecutive_sl}\n"
            f"Hourly SL hits: {hourly}/{self.max_hourly_sl}\n"
            f"Daily drawdown: {daily_dd:.2f}% / {self.max_daily_drawdown_pct}%"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """Check thresholds and trip if any are exceeded."""
        if self._tripped:
            return

        # 1. Consecutive SL hits
        if self._consecutive_sl >= self.max_consecutive_sl:
            self._trip(
                f"{self._consecutive_sl} consecutive SL hits "
                f"(max={self.max_consecutive_sl})"
            )
            return

        # 2. Hourly SL rate
        hourly = self._hourly_sl_count()
        if hourly >= self.max_hourly_sl:
            self._trip(
                f"{hourly} SL hits in the last hour "
                f"(max={self.max_hourly_sl})"
            )
            return

        # 3. Daily drawdown
        daily_dd = self._daily_drawdown_pct()
        if daily_dd >= self.max_daily_drawdown_pct:
            self._trip(
                f"Daily drawdown {daily_dd:.2f}% exceeded "
                f"threshold {self.max_daily_drawdown_pct}%"
            )

    def _trip(self, reason: str) -> None:
        """Set the tripped state and fire the optional alert."""
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = time.monotonic()
        log.warning("Circuit breaker TRIPPED: %s", reason)

        if self._alert_callback is not None:
            import asyncio
            msg = (
                f"🚨 *Circuit Breaker TRIPPED*\n"
                f"Reason: {reason}\n"
                f"Signal generation paused.  Use /reset_circuit_breaker to resume."
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._alert_callback(msg))
            except Exception as exc:
                log.warning("Alert callback error (circuit breaker): %s", exc)

    def _hourly_sl_count(self) -> int:
        """Count SL hits in the last 3600 seconds."""
        cutoff = time.monotonic() - 3600.0
        return sum(1 for r in self._outcomes if r.hit_sl and r.timestamp >= cutoff)

    def _daily_drawdown_pct(self) -> float:
        """Sum of negative PnL % values over the last 86400 seconds."""
        cutoff = time.monotonic() - 86_400.0
        return abs(
            sum(r.pnl_pct for r in self._outcomes if r.pnl_pct < 0 and r.timestamp >= cutoff)
        )
