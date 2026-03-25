"""Cross-channel confluence scoring for multi-channel signal confirmation."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

from src.utils import get_logger

log = get_logger("confluence")

# Confidence bonus when other channels confirm the signal direction.
# Keyed by number of confirming channels.
_CONFLUENCE_BONUS: Dict[int, float] = {
    1: 3.0,   # 1 other channel agrees
    2: 5.0,   # 2 other channels agree
    3: 7.0,   # 3 other channels agree
    4: 8.0,   # 4 other channels agree (all scalp channels)
}

# Maximum bonus cap
_MAX_CONFLUENCE_BONUS = 8.0

# Only scalp channels participate in cross-confluence
_CONFLUENCE_CHANNELS = {
    "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
    "360_SCALP_VWAP", "360_SCALP_OBI",
}


def compute_confluence_bonus(
    primary_channel: str,
    primary_direction: str,
    channels: List[Any],
    symbol: str,
    candles: Dict[str, dict],
    indicators: Dict[str, dict],
    smc_data: dict,
    spread_pct: float,
    volume_24h_usd: float,
    regime_result: Any = None,
) -> Tuple[float, List[str]]:
    """Check if other scalp channels would also generate a signal in the same direction.

    Returns:
        (bonus, confirming_channels): confidence bonus and list of confirming channel names.
    """
    if primary_channel not in _CONFLUENCE_CHANNELS:
        return 0.0, []

    confirming: List[str] = []
    for chan in channels:
        chan_name = chan.config.name
        if chan_name == primary_channel or chan_name not in _CONFLUENCE_CHANNELS:
            continue
        try:
            sig = chan.evaluate(
                symbol=symbol,
                candles=candles,
                indicators=indicators,
                smc_data=smc_data,
                spread_pct=spread_pct,
                volume_24h_usd=volume_24h_usd,
                regime_result=regime_result,
            )
            if sig is not None and sig.direction.value == primary_direction:
                confirming.append(chan_name)
        except Exception as exc:
            log.debug("Confluence check failed for {} {}: {}", chan_name, symbol, exc)

    count = len(confirming)
    bonus = min(_CONFLUENCE_BONUS.get(count, 0.0), _MAX_CONFLUENCE_BONUS)
    if bonus > 0.0:
        log.debug(
            "Confluence bonus for {} {} {}: +{:.1f} (confirmed by {})",
            symbol, primary_channel, primary_direction, bonus, confirming,
        )
    return bonus, confirming
