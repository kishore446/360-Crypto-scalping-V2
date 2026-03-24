"""Chart pattern detection — computationally cheap OHLCV pattern recognition.

Detects the following patterns from raw OHLCV numpy arrays:

* **Double Top** — two similar peaks separated by a valley (bearish reversal)
* **Double Bottom** — two similar troughs separated by a peak (bullish reversal)
* **Bollinger Band Squeeze Breakout** — contraction then expansion of BB width
* **Ascending / Descending Triangle** — converging high/low trendlines

All functions return ``None`` when the pattern is not detected, or a ``dict``
describing the pattern when it is.  No I/O, pure-compute.

Confidence values (0–1) in the returned dicts reflect how cleanly the price
action matches the idealised pattern.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.indicators import bollinger_bands, ema
from src.utils import get_logger

log = get_logger("chart_patterns")


# ---------------------------------------------------------------------------
# Double Top / Double Bottom
# ---------------------------------------------------------------------------

def detect_double_top(
    high: NDArray,
    lookback: int = 50,
    tolerance_pct: float = 1.0,
) -> Optional[Dict]:
    """Detect a double-top formation (bearish reversal).

    Parameters
    ----------
    high:
        Array of high prices.
    lookback:
        Number of most-recent candles to scan.
    tolerance_pct:
        Maximum percentage difference between the two peaks for them to be
        considered equal (e.g. 1.0 = peaks within 1% of each other).

    Returns
    -------
    dict or None
        ``{"pattern": "DOUBLE_TOP", "peak1_idx": int, "peak2_idx": int,
           "peak1": float, "peak2": float, "neckline": float,
           "confidence": float}``
        or ``None`` if the pattern is not detected.
    """
    arr = np.asarray(high, dtype=np.float64)
    n = len(arr)
    if n < lookback:
        return None

    window = arr[-lookback:]
    n_w = len(window)

    # Find all local maxima (peaks) in the window
    peaks: List[int] = []
    for i in range(1, n_w - 1):
        if window[i] >= window[i - 1] and window[i] >= window[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return None

    # Find the two highest peaks that are at least 10 candles apart
    peaks_sorted = sorted(peaks, key=lambda i: window[i], reverse=True)
    best: Optional[Dict] = None

    for idx_a in range(len(peaks_sorted)):
        for idx_b in range(idx_a + 1, len(peaks_sorted)):
            pa, pb = peaks_sorted[idx_a], peaks_sorted[idx_b]
            # Ensure minimum separation
            if abs(pa - pb) < 10:
                continue
            p1, p2 = (pa, pb) if pa < pb else (pb, pa)
            price1, price2 = window[p1], window[p2]

            # Check peaks are within tolerance
            diff_pct = abs(price1 - price2) / max(price1, price2) * 100.0
            if diff_pct > tolerance_pct:
                continue

            # Find valley between the two peaks (minimum between them)
            valley_val = float(np.min(window[p1:p2 + 1]))
            peak_avg = (price1 + price2) / 2.0
            valley_drop_pct = (peak_avg - valley_val) / peak_avg * 100.0
            if valley_drop_pct < 2.0:
                continue

            # Confidence: 1 minus the normalised peak difference (0→perfect)
            confidence = round(1.0 - diff_pct / tolerance_pct, 3)

            # Convert back to original array indices
            offset = n - lookback
            result = {
                "pattern": "DOUBLE_TOP",
                "peak1_idx": p1 + offset,
                "peak2_idx": p2 + offset,
                "peak1": float(price1),
                "peak2": float(price2),
                "neckline": valley_val,
                "confidence": confidence,
            }
            if best is None or confidence > best["confidence"]:
                best = result

    return best


def detect_double_bottom(
    low: NDArray,
    lookback: int = 50,
    tolerance_pct: float = 1.0,
) -> Optional[Dict]:
    """Detect a double-bottom formation (bullish reversal).

    Parameters
    ----------
    low:
        Array of low prices.
    lookback:
        Number of most-recent candles to scan.
    tolerance_pct:
        Maximum percentage difference between the two troughs.

    Returns
    -------
    dict or None
        ``{"pattern": "DOUBLE_BOTTOM", "trough1_idx": int, "trough2_idx": int,
           "trough1": float, "trough2": float, "neckline": float,
           "confidence": float}``
        or ``None`` if the pattern is not detected.
    """
    arr = np.asarray(low, dtype=np.float64)
    n = len(arr)
    if n < lookback:
        return None

    window = arr[-lookback:]
    n_w = len(window)

    # Find all local minima (troughs) in the window
    troughs: List[int] = []
    for i in range(1, n_w - 1):
        if window[i] <= window[i - 1] and window[i] <= window[i + 1]:
            troughs.append(i)

    if len(troughs) < 2:
        return None

    troughs_sorted = sorted(troughs, key=lambda i: window[i])
    best: Optional[Dict] = None

    for idx_a in range(len(troughs_sorted)):
        for idx_b in range(idx_a + 1, len(troughs_sorted)):
            ta, tb = troughs_sorted[idx_a], troughs_sorted[idx_b]
            if abs(ta - tb) < 10:
                continue
            t1, t2 = (ta, tb) if ta < tb else (tb, ta)
            price1, price2 = window[t1], window[t2]

            diff_pct = abs(price1 - price2) / max(abs(price1), abs(price2), 1e-10) * 100.0
            if diff_pct > tolerance_pct:
                continue

            # Peak between the troughs must be at least 2% above
            peak_val = float(np.max(window[t1:t2 + 1]))
            trough_avg = (price1 + price2) / 2.0
            peak_rise_pct = (peak_val - trough_avg) / max(abs(trough_avg), 1e-10) * 100.0
            if peak_rise_pct < 2.0:
                continue

            confidence = round(1.0 - diff_pct / tolerance_pct, 3)
            offset = n - lookback
            result = {
                "pattern": "DOUBLE_BOTTOM",
                "trough1_idx": t1 + offset,
                "trough2_idx": t2 + offset,
                "trough1": float(price1),
                "trough2": float(price2),
                "neckline": peak_val,
                "confidence": confidence,
            }
            if best is None or confidence > best["confidence"]:
                best = result

    return best


# ---------------------------------------------------------------------------
# Bollinger Band Squeeze Breakout
# ---------------------------------------------------------------------------

def detect_bollinger_squeeze(
    close: NDArray,
    period: int = 20,
    squeeze_threshold: float = 0.02,
) -> Optional[Dict]:
    """Detect a Bollinger Band squeeze followed by an expansion (breakout).

    Parameters
    ----------
    close:
        Array of closing prices.
    period:
        Bollinger Band period (default 20).
    squeeze_threshold:
        Maximum BB width (``(upper−lower)/middle``) that qualifies as
        squeezed.  Default 0.02 (2%).

    Returns
    -------
    dict or None
        ``{"pattern": "BB_SQUEEZE", "squeeze_width": float,
           "expansion_direction": "UP" | "DOWN", "confidence": float}``
        or ``None`` if the pattern is not detected.
    """
    arr = np.asarray(close, dtype=np.float64)
    if len(arr) < period + 10:
        return None

    upper, mid, lower = bollinger_bands(arr, period)

    # Compute BB width for all valid candles
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.where(mid != 0, (upper - lower) / mid, np.nan)

    valid_width = width[~np.isnan(width)]
    if len(valid_width) < 10:
        return None

    # Check that the width contracted below threshold in the last 10 candles
    recent_width = valid_width[-10:]
    min_width = float(np.min(recent_width))
    if min_width >= squeeze_threshold:
        return None

    # Check that width is now expanding (most recent > minimum in the window)
    current_width = float(valid_width[-1])
    squeeze_width = min_width

    # Direction: did price close above upper band or below lower band?
    last_close = float(arr[-1])
    last_upper = float(upper[-1]) if not np.isnan(upper[-1]) else float("inf")
    last_lower = float(lower[-1]) if not np.isnan(lower[-1]) else float("-inf")

    if current_width <= min_width * 1.05:
        # Width not yet expanding — still in squeeze, no breakout yet
        return None

    if last_close > last_upper:
        direction = "UP"
    elif last_close < last_lower:
        direction = "DOWN"
    else:
        direction = "UP" if arr[-1] > arr[-2] else "DOWN"

    # Confidence: how far did width expand relative to squeeze depth
    expansion_pct = (current_width - squeeze_width) / max(squeeze_threshold, 1e-10)
    confidence = round(min(expansion_pct, 1.0), 3)

    return {
        "pattern": "BB_SQUEEZE",
        "squeeze_width": round(squeeze_width, 5),
        "expansion_direction": direction,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Ascending / Descending Triangle
# ---------------------------------------------------------------------------

def detect_triangle(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    lookback: int = 50,
) -> Optional[Dict]:
    """Detect an ascending or descending triangle pattern.

    * **Ascending triangle**: flat resistance (highs converging horizontally)
      with rising support (lows trending upward).
    * **Descending triangle**: flat support with falling resistance.

    Parameters
    ----------
    high:
        Array of high prices.
    low:
        Array of low prices.
    close:
        Array of close prices.
    lookback:
        Number of most-recent candles to analyse.

    Returns
    -------
    dict or None
        ``{"pattern": "ASCENDING_TRIANGLE" | "DESCENDING_TRIANGLE",
           "resistance": float, "support": float, "confidence": float}``
        or ``None`` if neither pattern is detected.
    """
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    n = len(h)
    if n < lookback:
        return None

    h_w = h[-lookback:]
    lo_w = lo[-lookback:]
    x = np.arange(len(h_w), dtype=np.float64)

    # Fit linear regression to highs and lows separately
    high_slope, high_intercept = np.polyfit(x, h_w, 1)
    low_slope, low_intercept = np.polyfit(x, lo_w, 1)

    resistance = float(high_intercept + high_slope * (len(h_w) - 1))
    support = float(low_intercept + low_slope * (len(h_w) - 1))

    # Normalise slopes relative to price
    mid_price = float(np.mean(np.concatenate([h_w, lo_w])))
    if mid_price <= 0:
        return None
    norm_high_slope = high_slope / mid_price * lookback
    norm_low_slope = low_slope / mid_price * lookback

    # Ascending triangle: flat or slight resistance (|high_slope| < threshold)
    # and rising support (positive low_slope)
    flat_threshold = 0.005  # < 0.5% drift over lookback candles = "flat"
    if abs(norm_high_slope) < flat_threshold and norm_low_slope > flat_threshold:
        confidence = round(min(norm_low_slope / 0.02, 1.0), 3)
        return {
            "pattern": "ASCENDING_TRIANGLE",
            "resistance": resistance,
            "support": support,
            "confidence": confidence,
        }

    # Descending triangle: falling resistance and flat support
    if norm_high_slope < -flat_threshold and abs(norm_low_slope) < flat_threshold:
        confidence = round(min(abs(norm_high_slope) / 0.02, 1.0), 3)
        return {
            "pattern": "DESCENDING_TRIANGLE",
            "resistance": resistance,
            "support": support,
            "confidence": confidence,
        }

    return None


# ---------------------------------------------------------------------------
# Aggregate helper — detects all patterns in one pass
# ---------------------------------------------------------------------------

def detect_patterns(candles: Dict) -> List[Dict]:
    """Run all pattern detectors on a single-timeframe candle dict.

    Parameters
    ----------
    candles:
        Dict with ``"high"``, ``"low"``, ``"close"`` keys (numpy arrays or
        lists).  Missing keys are handled gracefully.

    Returns
    -------
    list of dict
        List of detected pattern dicts (may be empty).  Each dict contains
        at minimum ``"pattern"`` and ``"confidence"`` keys.
    """
    results: List[Dict] = []
    try:
        h = np.asarray(candles.get("high", []), dtype=np.float64).ravel()
        lo = np.asarray(candles.get("low", []), dtype=np.float64).ravel()
        c = np.asarray(candles.get("close", []), dtype=np.float64).ravel()

        for detector, args in [
            (detect_double_top, (h,)),
            (detect_double_bottom, (lo,)),
            (detect_bollinger_squeeze, (c,)),
            (detect_triangle, (h, lo, c)),
        ]:
            try:
                result = detector(*args)  # type: ignore[operator]
                if result is not None:
                    results.append(result)
            except Exception as exc:
                log.debug("Pattern detector error: {}", exc)
    except Exception as exc:
        log.debug("detect_patterns failed: {}", exc)
    return results


def pattern_confidence_bonus(patterns: List[Dict], direction: str) -> float:
    """Compute a confidence bonus based on detected confirming patterns.

    Parameters
    ----------
    patterns:
        List of detected pattern dicts from :func:`detect_patterns`.
    direction:
        ``"LONG"`` or ``"SHORT"`` — signal direction to align against.

    Returns
    -------
    float
        Confidence bonus in the range [0, 5].  Confirming patterns (e.g.
        double bottom for LONG, double top for SHORT) each contribute up to
        +3 points; contradicting patterns each contribute −1.5 points.
        The total is clamped to [0, 5].
    """
    bonus = 0.0
    bullish_patterns = {"DOUBLE_BOTTOM", "ASCENDING_TRIANGLE"}
    bearish_patterns = {"DOUBLE_TOP", "DESCENDING_TRIANGLE"}

    for p in patterns:
        name = p.get("pattern", "")
        conf = float(p.get("confidence", 0.5))
        if direction == "LONG":
            if name in bullish_patterns:
                bonus += conf * 3.0
            elif name in bearish_patterns:
                bonus -= 1.5
        elif direction == "SHORT":
            if name in bearish_patterns:
                bonus += conf * 3.0
            elif name in bullish_patterns:
                bonus -= 1.5
        # BB_SQUEEZE is direction-neutral (direction comes from expansion_direction)
        if name == "BB_SQUEEZE":
            exp_dir = p.get("expansion_direction", "")
            if (direction == "LONG" and exp_dir == "UP") or (
                direction == "SHORT" and exp_dir == "DOWN"
            ):
                bonus += conf * 2.0

    return round(max(0.0, min(bonus, 5.0)), 2)
