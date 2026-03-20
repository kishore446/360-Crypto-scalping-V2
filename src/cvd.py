"""Cumulative Volume Delta (CVD) calculation and divergence detection.

This module provides **standalone, pure-function** utilities for computing CVD
from raw trade-tick data and for detecting price/CVD divergences that may
signal a reversal.

Typical usage
-------------
.. code-block:: python

    import numpy as np
    from src.cvd import compute_cvd, detect_cvd_divergence

    # buy_volumes[i] and sell_volumes[i] are the aggressive buy/sell
    # USD volume for candle i.
    buy_vol  = np.array([200, 300, 150, 100, 180], dtype=float)
    sell_vol = np.array([100, 200, 250, 400, 120], dtype=float)

    cvd = compute_cvd(buy_vol, sell_vol)

    close = np.array([100, 101, 100.5, 99, 98.5], dtype=float)
    signal = detect_cvd_divergence(close, cvd)
    # Returns "BULLISH", "BEARISH", or None

The divergence detection logic is intentionally identical to the implementation
in :mod:`src.order_flow` so that both callers share the same behaviour.
:func:`detect_cvd_divergence` is re-exported here so downstream code only
needs a single import.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Re-export the divergence detector from order_flow so callers that only
# depend on this module get a complete API without a circular import.
from src.order_flow import detect_cvd_divergence as detect_cvd_divergence  # noqa: F401

__all__ = [
    "compute_cvd",
    "detect_cvd_divergence",
]


def compute_cvd(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    *,
    window: Optional[int] = None,
) -> np.ndarray:
    """Calculate the Cumulative Volume Delta (CVD) from per-candle volume arrays.

    CVD is the running (cumulative) sum of *Volume Delta*, where Volume Delta
    for a single candle is defined as::

        delta[i] = buy_volumes[i] - sell_volumes[i]

    A positive delta means aggressive buyers dominated that candle; a negative
    delta means aggressive sellers dominated.

    Parameters
    ----------
    buy_volumes:
        1-D array of quote-currency volume attributed to the aggressive buy
        side for each candle (``qty × price`` when ``isBuyerMaker == False``).
    sell_volumes:
        1-D array of quote-currency volume attributed to the aggressive sell
        side for each candle (``qty × price`` when ``isBuyerMaker == True``).
        Must be the same length as *buy_volumes*.
    window:
        Optional rolling window size.  When provided, the cumulative sum is
        reset (restarted) every *window* candles, yielding a **rolling** CVD
        instead of a single ever-growing line.  When ``None`` (default), the
        cumulative sum is computed across the entire input (standard CVD).

    Returns
    -------
    np.ndarray
        1-D array of CVD values with the same length as the inputs.

    Raises
    ------
    ValueError
        If *buy_volumes* and *sell_volumes* have different lengths, or if
        *window* is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from src.cvd import compute_cvd
    >>> buy = np.array([300.0, 200.0, 100.0, 400.0, 250.0])
    >>> sell = np.array([100.0, 300.0, 200.0, 100.0, 50.0])
    >>> compute_cvd(buy, sell)
    array([200., 100.,   0., 300., 500.])

    Rolling window example – the cumsum restarts every 3 candles:

    >>> buy2  = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    >>> sell2 = np.array([ 50.0,  50.0,  50.0,  50.0,  50.0,  50.0])
    >>> compute_cvd(buy2, sell2, window=3)
    array([ 50., 100., 150., 150., 300., 450.])
    """
    buy_arr = np.asarray(buy_volumes, dtype=np.float64).ravel()
    sell_arr = np.asarray(sell_volumes, dtype=np.float64).ravel()

    if buy_arr.shape != sell_arr.shape:
        raise ValueError(
            f"buy_volumes and sell_volumes must have the same length; "
            f"got {len(buy_arr)} and {len(sell_arr)}"
        )

    if window is not None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window must be a positive integer; got {window!r}"
            )

    delta = buy_arr - sell_arr

    if window is None or window >= len(delta):
        # Standard CVD: full cumulative sum
        return np.cumsum(delta)

    # Rolling CVD: restart the cumulative sum every *window* candles.
    # We compute cumsum over the full array and subtract the cumsum value
    # at the start of the current window to "reset" each block.
    result = np.empty_like(delta)
    cum = np.cumsum(delta)
    for i in range(len(delta)):
        window_start = (i // window) * window
        base = cum[window_start - 1] if window_start > 0 else 0.0
        result[i] = cum[i] - base
    return result
