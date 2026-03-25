"""Session-adaptive threshold manager for time-of-day filter adjustments."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from src.utils import get_logger

log = get_logger("session_thresholds")


class TradingSession:
    ASIAN = "ASIAN"          # 00:00 - 08:00 UTC
    EUROPEAN = "EUROPEAN"    # 08:00 - 14:00 UTC
    US = "US"                # 14:00 - 21:00 UTC
    OVERNIGHT = "OVERNIGHT"  # 21:00 - 00:00 UTC


def get_current_session(utc_now: Optional[datetime] = None) -> str:
    """Determine current trading session based on UTC hour."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    hour = utc_now.hour
    if 0 <= hour < 8:
        return TradingSession.ASIAN
    elif 8 <= hour < 14:
        return TradingSession.EUROPEAN
    elif 14 <= hour < 21:
        return TradingSession.US
    else:
        return TradingSession.OVERNIGHT


# Session-specific multipliers for spread_max and min_volume.
# These multiply the channel's configured thresholds.
_SESSION_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    TradingSession.ASIAN: {
        "spread_mult": 1.5,         # Allow 50% wider spreads in Asian session
        "volume_mult": 0.6,         # Require only 60% of configured volume
        "confidence_offset": -2.0,  # Slight penalty for lower-liquidity session
    },
    TradingSession.EUROPEAN: {
        "spread_mult": 1.2,         # Slightly wider spreads during EU open
        "volume_mult": 0.8,
        "confidence_offset": 0.0,
    },
    TradingSession.US: {
        "spread_mult": 1.0,         # Standard thresholds
        "volume_mult": 1.0,
        "confidence_offset": 1.0,   # Slight bonus for highest-liquidity session
    },
    TradingSession.OVERNIGHT: {
        "spread_mult": 1.3,
        "volume_mult": 0.7,
        "confidence_offset": -1.0,
    },
}


def get_session_adjusted_thresholds(
    channel_spread_max: float,
    channel_min_volume: float,
    session: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Return session-adjusted (spread_max, min_volume, confidence_offset).

    Args:
        channel_spread_max: The channel's configured spread_max.
        channel_min_volume: The channel's configured min_volume.
        session: Trading session name. If None, auto-detect from current UTC time.

    Returns:
        (adjusted_spread_max, adjusted_min_volume, confidence_offset)
    """
    if session is None:
        try:
            session = get_current_session()
        except Exception as exc:
            log.debug("Session detection failed, falling back to US defaults: {}", exc)
            session = TradingSession.US
    adj = _SESSION_ADJUSTMENTS.get(session, _SESSION_ADJUSTMENTS[TradingSession.US])
    adjusted_spread = round(channel_spread_max * adj["spread_mult"], 6)
    adjusted_volume = round(channel_min_volume * adj["volume_mult"], 2)
    return adjusted_spread, adjusted_volume, adj["confidence_offset"]
