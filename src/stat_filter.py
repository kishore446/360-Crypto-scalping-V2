"""Statistical false-positive filter using rolling win-rate tracking.

Tracks per-(channel, pair, regime) rolling win rates and applies adaptive
confidence penalties or hard suppression when quality drops below thresholds.

Thresholds are configurable via environment variables:

  STAT_FILTER_WINDOW            Rolling window size (default: 30)
  STAT_FILTER_MIN_SAMPLES       Minimum outcomes before filtering (default: 15)
  STAT_FILTER_HARD_SUPPRESS_WR  Hard suppress threshold as % (default: 25)
  STAT_FILTER_SOFT_PENALTY_WR   Soft penalty threshold as % (default: 45)
  STAT_FILTER_SOFT_PENALTY_PTS  Confidence points deducted (default: 5.0)
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class SignalOutcome:
    """Record of a single resolved signal for win-rate tracking."""

    signal_id: str
    channel: str
    pair: str
    regime: str
    setup_class: str
    won: bool       # True if TP1 or higher was hit; False if SL hit or expired
    pnl_pct: float  # Actual PnL % achieved


@dataclass
class _OutcomeRecord:
    """Internal storage unit — won flag, PnL, and resolution timestamp."""

    won: bool
    pnl_pct: float
    timestamp: datetime


class RollingWinRateStore:
    """Thread-safe rolling win-rate store per (channel, pair, regime) key.

    Uses a fixed-size deque per key so memory is bounded regardless of
    how many signals are recorded.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> None:
        self._window: int = window if window is not None else _env_int("STAT_FILTER_WINDOW", 30)
        self._min_samples: int = (
            min_samples if min_samples is not None else _env_int("STAT_FILTER_MIN_SAMPLES", 15)
        )
        self._lock = threading.Lock()
        # Key: (channel, pair, regime) → deque of _OutcomeRecord
        self._records: Dict[Tuple[str, str, str], Deque[_OutcomeRecord]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    def record(self, outcome: SignalOutcome) -> None:
        """Record the outcome of a resolved signal."""
        key = (outcome.channel, outcome.pair, outcome.regime)
        rec = _OutcomeRecord(
            won=outcome.won,
            pnl_pct=outcome.pnl_pct,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            self._records[key].append(rec)

    def win_rate(self, channel: str, pair: str, regime: str) -> Optional[float]:
        """Return rolling win rate (0.0–1.0) or None if below min_samples.

        Returns None when there is insufficient history to make a judgment
        (fail-open behaviour).
        """
        key = (channel, pair, regime)
        with self._lock:
            records = self._records.get(key)
            if records is None or len(records) < self._min_samples:
                return None
            return sum(1 for r in records if r.won) / len(records)

    def stats(self, channel: str, pair: str, regime: str) -> Dict:
        """Return statistics dict for a specific (channel, pair, regime) key.

        Returns
        -------
        dict with keys: win_rate (float), n (int), avg_pnl (float),
        last_updated (datetime).  win_rate is 0.0 when n == 0.
        """
        key = (channel, pair, regime)
        with self._lock:
            records = self._records.get(key)
            if not records:
                return {
                    "win_rate": 0.0,
                    "n": 0,
                    "avg_pnl": 0.0,
                    "last_updated": None,
                }
            n = len(records)
            wr = sum(1 for r in records if r.won) / n
            avg_pnl = sum(r.pnl_pct for r in records) / n
            last_updated = max(r.timestamp for r in records)
            return {
                "win_rate": wr,
                "n": n,
                "avg_pnl": avg_pnl,
                "last_updated": last_updated,
            }

    def all_stats(self) -> Dict[str, Dict]:
        """Return stats for all tracked keys that have at least one outcome."""
        out: Dict[str, Dict] = {}
        with self._lock:
            for (ch, pair, regime), records in self._records.items():
                if not records:
                    continue
                n = len(records)
                wr = sum(1 for r in records if r.won) / n
                avg_pnl = sum(r.pnl_pct for r in records) / n
                last_updated = max(r.timestamp for r in records)
                key = f"{ch}/{pair}/{regime}"
                out[key] = {
                    "channel": ch,
                    "pair": pair,
                    "regime": regime,
                    "win_rate": wr,
                    "n": n,
                    "avg_pnl": avg_pnl,
                    "last_updated": last_updated,
                }
        return out

    def all_keys(self) -> List[Tuple[str, str, str]]:
        """Return all (channel, pair, regime) keys that have recorded outcomes."""
        with self._lock:
            return list(self._records.keys())


class StatisticalFilter:
    """Applies adaptive confidence gates based on rolling win-rate statistics.

    Gate logic (thresholds configurable via env vars; WR = win rate):
    ────────────────────────────────────────────────────────────
    WR >= SOFT_PENALTY_WR          → pass (no penalty)
    HARD_SUPPRESS_WR <= WR < SOFT_PENALTY_WR → soft penalty (-PENALTY_PTS confidence)
    WR < HARD_SUPPRESS_WR          → HARD SUPPRESS (signal dropped)
    None (no history)              → pass (fail-open)
    ────────────────────────────────────────────────────────────

    Default thresholds (overridable via env vars):
      Hard suppress: WR < 25%  (STAT_FILTER_HARD_SUPPRESS_WR)
      Soft penalty:  WR < 45%  (STAT_FILTER_SOFT_PENALTY_WR)
      Penalty pts:   -5.0      (STAT_FILTER_SOFT_PENALTY_PTS)
    """

    def __init__(self, store: Optional[RollingWinRateStore] = None) -> None:
        self._store = store or RollingWinRateStore()
        self._hard_suppress_wr: float = _env_float("STAT_FILTER_HARD_SUPPRESS_WR", 25.0) / 100.0
        self._soft_penalty_wr: float = _env_float("STAT_FILTER_SOFT_PENALTY_WR", 45.0) / 100.0
        self._soft_penalty_pts: float = _env_float("STAT_FILTER_SOFT_PENALTY_PTS", 5.0)

    @property
    def store(self) -> RollingWinRateStore:
        return self._store

    def check(
        self,
        channel: str,
        pair: str,
        regime: str,
        current_confidence: float,
    ) -> Tuple[bool, float, str]:
        """Check whether the signal should be emitted based on rolling win rate.

        Parameters
        ----------
        channel, pair, regime:
            Signal identifiers for win-rate lookup.
        current_confidence:
            Signal confidence score (0–100).

        Returns
        -------
        (allow: bool, adjusted_confidence: float, reason: str)
            allow: False means the signal should be suppressed.
            adjusted_confidence: confidence after penalty (may be unchanged).
            reason: human-readable explanation for logs.
        """
        win_rate = self._store.win_rate(channel, pair, regime)

        if win_rate is None:
            return True, current_confidence, "no_history"

        wr_pct = f"{win_rate:.1%}"

        if win_rate < self._hard_suppress_wr:
            return False, 0.0, f"hard_suppress:wr={wr_pct}"

        if win_rate < self._soft_penalty_wr:
            adj = max(0.0, current_confidence - self._soft_penalty_pts)
            return True, adj, f"soft_penalty:wr={wr_pct}"

        return True, current_confidence, f"ok:wr={wr_pct}"

    def record(self, outcome: SignalOutcome) -> None:
        """Forward a resolved signal outcome to the underlying win-rate store."""
        self._store.record(outcome)

    def format_statstats(self) -> str:
        """Format win-rate stats as a Telegram-friendly table for /statstats.

        Returns a string with a table showing Channel | Pair | Regime | WR% | N | Avg PnL.
        """
        all_stats = self._store.all_stats()
        if not all_stats:
            return "📊 *Statistical Filter Stats*\n\nNo outcomes recorded yet."

        # Column widths must match the header format string below.
        _COL_SEP_LEN: int = 61  # 14+1+10+1+14+1+6+1+4+1+8

        lines = ["📊 *Statistical Filter Stats*\n"]
        lines.append("```")
        lines.append(f"{'Channel':<14} {'Pair':<10} {'Regime':<14} {'WR%':>6} {'N':>4} {'AvgPnL%':>8}")
        lines.append("-" * _COL_SEP_LEN)
        for _key, s in sorted(all_stats.items()):
            wr_pct = f"{s['win_rate'] * 100:.1f}%"
            avg_pnl = f"{s['avg_pnl']:+.2f}%"
            lines.append(
                f"{s['channel']:<14} {s['pair']:<10} {s['regime']:<14} "
                f"{wr_pct:>6} {s['n']:>4} {avg_pnl:>8}"
            )
        lines.append("```")
        return "\n".join(lines)
