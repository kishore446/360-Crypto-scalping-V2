"""Volume Weighted Average Price (VWAP) with Standard Deviation Bands.

Provides a rolling / daily-anchored VWAP calculation together with 1st,
2nd, and 3rd standard-deviation bands.  A helper function rejects trades
that are statistically overextended relative to VWAP.

Typical usage
-------------
.. code-block:: python

    import numpy as np
    from src.vwap import compute_vwap, check_vwap_extension

    highs  = np.array([101.5, 102.0, 101.8])
    lows   = np.array([ 99.5, 100.5, 100.0])
    closes = np.array([101.0, 101.5, 101.0])
    volumes = np.array([1000.0, 1500.0, 1200.0])

    result = compute_vwap(highs, lows, closes, volumes)
    print(result.vwap, result.upper_band_3)

    allowed, reason = check_vwap_extension("LONG", closes[-1], result)
    if not allowed:
        print(reason)   # e.g. "VWAP: price at +3 SD band – LONG rejected"

The module is **pure-function** – no I/O, no side-effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.utils import get_logger

log = get_logger("vwap")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of standard deviations for each band level.
VWAP_SD1: float = 1.0
VWAP_SD2: float = 2.0
VWAP_SD3: float = 3.0

#: Band beyond which a trade is considered statistically overextended and
#: should be rejected.  Default: touching or exceeding the ±3 SD band.
VWAP_EXTENSION_SD: float = 3.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VWAPResult:
    """Computed VWAP and standard deviation bands for a price series."""

    vwap: float            # Volume Weighted Average Price
    std_dev: float         # Standard deviation of (typical_price - vwap), volume-weighted

    upper_band_1: float    # VWAP + 1 SD
    upper_band_2: float    # VWAP + 2 SD
    upper_band_3: float    # VWAP + 3 SD

    lower_band_1: float    # VWAP - 1 SD
    lower_band_2: float    # VWAP - 2 SD
    lower_band_3: float    # VWAP - 3 SD


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def compute_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> Optional[VWAPResult]:
    """Compute VWAP and ±1/2/3 standard-deviation bands.

    Uses the HLC/3 typical price convention:
    ``typical_price = (high + low + close) / 3``

    The VWAP is anchored to the first bar in the input window (equivalent
    to a session-anchored VWAP when the input covers one trading session).

    Parameters
    ----------
    highs, lows, closes:
        Per-bar price arrays (same length).
    volumes:
        Per-bar volume array (same length as price arrays).

    Returns
    -------
    :class:`VWAPResult` or ``None`` when input is empty or all volumes are zero.

    Raises
    ------
    ValueError
        If the input arrays have mismatched lengths.
    """
    h = np.asarray(highs, dtype=np.float64).ravel()
    l = np.asarray(lows, dtype=np.float64).ravel()
    c = np.asarray(closes, dtype=np.float64).ravel()
    v = np.asarray(volumes, dtype=np.float64).ravel()

    if not (h.shape == l.shape == c.shape == v.shape):
        raise ValueError(
            "highs, lows, closes, volumes must all have the same length; "
            f"got shapes {h.shape}, {l.shape}, {c.shape}, {v.shape}"
        )

    if len(h) == 0:
        return None

    total_volume = v.sum()
    if total_volume <= 0:
        return None

    typical = (h + l + c) / 3.0

    # VWAP = Σ(typical × volume) / Σ(volume)
    vwap = float(np.dot(typical, v) / total_volume)

    # Volume-weighted standard deviation of typical price around VWAP
    variance = float(np.dot((typical - vwap) ** 2, v) / total_volume)
    std_dev = math.sqrt(variance)

    return VWAPResult(
        vwap=round(vwap, 8),
        std_dev=round(std_dev, 8),
        upper_band_1=round(vwap + VWAP_SD1 * std_dev, 8),
        upper_band_2=round(vwap + VWAP_SD2 * std_dev, 8),
        upper_band_3=round(vwap + VWAP_SD3 * std_dev, 8),
        lower_band_1=round(vwap - VWAP_SD1 * std_dev, 8),
        lower_band_2=round(vwap - VWAP_SD2 * std_dev, 8),
        lower_band_3=round(vwap - VWAP_SD3 * std_dev, 8),
    )


# ---------------------------------------------------------------------------
# Pipeline hook
# ---------------------------------------------------------------------------


def check_vwap_extension(
    direction: str,
    current_price: float,
    vwap_result: Optional[VWAPResult],
    extension_sd: float = VWAP_EXTENSION_SD,
) -> tuple[bool, str]:
    """Reject a trade when price is statistically overextended vs VWAP.

    Fails open (returns ``True``) when *vwap_result* is ``None`` so the
    filter never hard-blocks trades due to missing data.

    Parameters
    ----------
    direction:
        ``"LONG"`` or ``"SHORT"``.
    current_price:
        Latest close price to compare against the VWAP bands.
    vwap_result:
        Output of :func:`compute_vwap`.  ``None`` → fails open.
    extension_sd:
        The band level that triggers rejection.  Default: 3 (±3 SD).
        Can be set to ``2.0`` for a tighter filter.

    Returns
    -------
    ``(allowed, reason)`` where *allowed* is ``False`` only when price
    is clearly overextended above the configured SD band.

    Examples
    --------
    >>> # Price is far above VWAP at the +3 SD band – reject LONG
    >>> check_vwap_extension("LONG", 120.0, result)
    (False, 'VWAP: price 120.0 above +3.0 SD band 119.5 – LONG overextended, rejected')
    """
    if vwap_result is None:
        return True, ""

    sd = vwap_result.std_dev

    # Compute the relevant upper / lower band dynamically for any extension_sd
    upper_band = vwap_result.vwap + extension_sd * sd
    lower_band = vwap_result.vwap - extension_sd * sd

    if direction == "LONG" and current_price >= upper_band:
        return (
            False,
            (
                f"VWAP: price {current_price} above +{extension_sd:.1f} SD band "
                f"{upper_band:.4f} – LONG overextended, rejected"
            ),
        )

    if direction == "SHORT" and current_price <= lower_band:
        return (
            False,
            (
                f"VWAP: price {current_price} below -{extension_sd:.1f} SD band "
                f"{lower_band:.4f} – SHORT overextended, rejected"
            ),
        )

    return True, ""
