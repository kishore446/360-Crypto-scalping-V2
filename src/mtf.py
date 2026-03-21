"""Multi-Timeframe (MTF) Confluence Matrix.

Evaluates trend alignment across multiple timeframes (e.g. 1m, 15m, 1h).
A lower-timeframe signal is scored based on whether it aligns with the
higher-timeframe EMA/trend direction.

Typical usage
-------------
.. code-block:: python

    from src.mtf import compute_mtf_confluence, MTFResult

    # Each timeframe entry is a dict of {"ema_fast": float, "ema_slow": float,
    # "close": float}.  Timeframes should be ordered from lowest to highest.
    timeframes = {
        "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
        "15m": {"ema_fast": 102.0, "ema_slow": 101.0, "close": 102.0},
        "1h":  {"ema_fast": 103.0, "ema_slow": 101.5, "close": 103.5},
    }
    result = compute_mtf_confluence("LONG", timeframes)
    if result.is_aligned:
        print(f"All TFs aligned  score={result.score:.2f}")
    else:
        print(f"Misaligned: {result.reason}")

The module is **pure-function** – no I/O, no side-effects.  Wire it into
the signal validation pipeline after indicator calculations are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.utils import get_logger

log = get_logger("mtf")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum confluence score (0-1) required to pass the MTF gate.
#: 0.5 means at least half the supplied timeframes must agree.
MTF_MIN_SCORE: float = 0.5

#: Score threshold above which the confluence is considered *strong*.
MTF_STRONG_SCORE: float = 0.8


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeframeState:
    """Trend state derived for a single timeframe."""

    timeframe: str
    trend: str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    ema_fast: float
    ema_slow: float
    close: float


@dataclass
class MTFResult:
    """Output of :func:`compute_mtf_confluence`."""

    signal_direction: str               # "LONG" | "SHORT"
    score: float                        # 0.0 – 1.0  (aligned TFs / total TFs)
    aligned_count: int                  # number of TFs agreeing with signal
    total_count: int                    # total TFs evaluated
    is_aligned: bool                    # score >= MTF_MIN_SCORE
    is_strong: bool                     # score >= MTF_STRONG_SCORE
    timeframe_states: List[TimeframeState] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_trend(ema_fast: float, ema_slow: float, close: float) -> str:
    """Return "BULLISH", "BEARISH", or "NEUTRAL" for one timeframe."""
    if ema_fast > ema_slow and close > ema_fast:
        return "BULLISH"
    if ema_fast < ema_slow and close < ema_fast:
        return "BEARISH"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def compute_mtf_confluence(
    signal_direction: str,
    timeframes: Dict[str, Dict[str, float]],
    min_score: float = MTF_MIN_SCORE,
) -> MTFResult:
    """Evaluate trend alignment across multiple timeframes.

    Parameters
    ----------
    signal_direction:
        ``"LONG"`` or ``"SHORT"``.
    timeframes:
        Mapping of timeframe label → indicator dict.  Each dict **must**
        contain the keys ``"ema_fast"``, ``"ema_slow"``, and ``"close"``.
        Missing or malformed entries are skipped and logged.
    min_score:
        Minimum fraction of timeframes that must agree with the signal
        direction to be considered aligned.  Defaults to
        :data:`MTF_MIN_SCORE`.

    Returns
    -------
    :class:`MTFResult`
    """
    direction = signal_direction.upper()
    states: List[TimeframeState] = []
    aligned: float = 0.0

    for tf_label, data in timeframes.items():
        try:
            ema_fast = float(data["ema_fast"])
            ema_slow = float(data["ema_slow"])
            close = float(data["close"])
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("MTF: skipping timeframe {} – bad data: {}", tf_label, exc)
            continue

        trend = _classify_trend(ema_fast, ema_slow, close)
        states.append(TimeframeState(
            timeframe=tf_label,
            trend=trend,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            close=close,
        ))

        wanted = "BULLISH" if direction == "LONG" else "BEARISH"
        if trend == wanted:
            aligned += 1.0
        elif trend == "NEUTRAL":
            aligned += 0.5  # Partial credit — not opposing the direction

    total = len(states)
    if total == 0:
        return MTFResult(
            signal_direction=direction,
            score=0.0,
            aligned_count=0,
            total_count=0,
            is_aligned=False,
            is_strong=False,
            timeframe_states=states,
            reason="no valid timeframe data provided",
        )

    score = aligned / total
    is_aligned = score >= min_score
    is_strong = score >= MTF_STRONG_SCORE

    misaligned = [s.timeframe for s in states if s.trend != ("BULLISH" if direction == "LONG" else "BEARISH")]
    reason = ""
    if not is_aligned:
        reason = (
            f"MTF misaligned: {aligned}/{total} TFs agree with {direction}; "
            f"conflicting TFs: {misaligned}"
        )

    return MTFResult(
        signal_direction=direction,
        score=round(score, 4),
        aligned_count=aligned,
        total_count=total,
        is_aligned=is_aligned,
        is_strong=is_strong,
        timeframe_states=states,
        reason=reason,
    )


def check_mtf_gate(
    signal_direction: str,
    timeframes: Dict[str, Dict[str, float]],
    min_score: float = MTF_MIN_SCORE,
) -> tuple[bool, str]:
    """Pipeline hook: return ``(allowed, reason)`` for the MTF confluence gate.

    Fails open (returns ``True``) when no valid timeframe data is provided,
    matching the behaviour of the order book and CVD filters.

    Parameters
    ----------
    signal_direction:
        ``"LONG"`` or ``"SHORT"``.
    timeframes:
        Same format as :func:`compute_mtf_confluence`.
    min_score:
        Minimum passing score.

    Returns
    -------
    ``(allowed, reason)`` – ``allowed`` is ``False`` only when sufficient
    data exists *and* the confluence score falls below *min_score*.
    """
    if not timeframes:
        return True, ""

    result = compute_mtf_confluence(signal_direction, timeframes, min_score)
    if result.total_count == 0:
        return True, ""

    if not result.is_aligned:
        return False, result.reason

    return True, ""
