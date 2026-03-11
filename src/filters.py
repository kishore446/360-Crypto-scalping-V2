"""Centralized filter functions shared across all channel strategies.

Each function returns ``True`` when the condition passes (signal may proceed)
and ``False`` when it should be filtered out.
"""

from __future__ import annotations


def check_spread(spread_pct: float, max_spread: float) -> bool:
    """Return True if spread is within acceptable bounds.

    Parameters
    ----------
    spread_pct:
        Current bid-ask spread as a percentage of mid-price.
    max_spread:
        Maximum acceptable spread percentage (from channel config).
    """
    return spread_pct <= max_spread


def check_adx(adx_val: float | None, min_adx: float, max_adx: float = 100.0) -> bool:
    """Return True if ADX is within [min_adx, max_adx].

    A ``None`` value (not yet computed) is treated as a filter failure.
    """
    if adx_val is None:
        return False
    return min_adx <= adx_val <= max_adx


def check_ema_alignment(
    ema_fast: float | None,
    ema_slow: float | None,
    direction: str,
) -> bool:
    """Return True when fast/slow EMAs are aligned with *direction*.

    Parameters
    ----------
    ema_fast:
        Value of the fast EMA (e.g. EMA-9).
    ema_slow:
        Value of the slow EMA (e.g. EMA-21).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    """
    if ema_fast is None or ema_slow is None:
        return False
    if direction == "LONG":
        return ema_fast > ema_slow
    if direction == "SHORT":
        return ema_fast < ema_slow
    return False


def check_volume(volume_24h_usd: float, min_volume: float) -> bool:
    """Return True if 24-hour USD volume meets the minimum threshold.

    Parameters
    ----------
    volume_24h_usd:
        24-hour trading volume in USD.
    min_volume:
        Minimum required volume in USD.
    """
    return volume_24h_usd >= min_volume


def check_rsi(
    rsi_val: float | None,
    overbought: float,
    oversold: float,
    direction: str,
) -> bool:
    """Return True when RSI is not in an extreme zone conflicting with direction.

    For a ``LONG`` signal, RSI must be below the overbought threshold.
    For a ``SHORT`` signal, RSI must be above the oversold threshold.
    A ``None`` RSI value passes (no filter applied).

    Parameters
    ----------
    rsi_val:
        Current RSI value (0-100), or ``None`` if unavailable.
    overbought:
        Overbought threshold (e.g. 70).
    oversold:
        Oversold threshold (e.g. 30).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    """
    if rsi_val is None:
        return True  # no data, don't filter
    if direction == "LONG":
        return rsi_val < overbought
    if direction == "SHORT":
        return rsi_val > oversold
    return True
