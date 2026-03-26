"""Regime-aware signal parameter tables.

Provides per-(channel, setup_class, regime) signal construction parameters
so that TP ratios, SL multipliers, entry zone bias, DCA policy, and
validity windows adapt to market conditions instead of being uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SignalParams:
    """Signal construction parameters for a specific context."""

    tp_ratios: Optional[Tuple[float, ...]] = None  # Override config tp_ratios if set
    sl_multiplier: float = 1.0       # Multiplier applied to base SL distance
    entry_zone_bias: float = 0.7     # 0.5 = symmetric, 0.7 = directional (current default)
    dca_enabled: bool = True         # Whether DCA zone should be computed
    validity_minutes: Optional[int] = None  # Override default validity window
    vol_stretch_factor: float = 1.3  # High-vol TP stretch multiplier
    vol_compress_factor: float = 0.7  # Low-vol TP compress multiplier


# Default params (matches current behaviour)
_DEFAULT = SignalParams()

# Lookup table: (channel_prefix, setup_class, regime) -> SignalParams
# channel_prefix: "SCALP", "SWING", "SPOT" (matched via startswith)
# setup_class: as string from SetupClass enum values
# regime: MarketRegime enum value string
PARAM_TABLE: Dict[Tuple[str, str, str], SignalParams] = {
    # --- SCALP + LIQUIDITY_SWEEP_REVERSAL ---
    ("SCALP", "LIQUIDITY_SWEEP_REVERSAL", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.0, 1.8, 2.5), sl_multiplier=1.2, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=8,
    ),
    ("SCALP", "LIQUIDITY_SWEEP_REVERSAL", "TRENDING_DOWN"): SignalParams(
        tp_ratios=(1.0, 1.8, 2.5), sl_multiplier=1.2, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=8,
    ),
    ("SCALP", "LIQUIDITY_SWEEP_REVERSAL", "RANGING"): SignalParams(
        tp_ratios=(0.7, 1.2, 1.8), sl_multiplier=0.9, entry_zone_bias=0.6,
        dca_enabled=True, validity_minutes=10,
    ),
    # --- SCALP + RANGE_FADE ---
    ("SCALP", "RANGE_FADE", "RANGING"): SignalParams(
        tp_ratios=(0.5, 0.8, 1.2), sl_multiplier=0.8, entry_zone_bias=0.5,
        dca_enabled=True, validity_minutes=12,
    ),
    ("SCALP", "RANGE_FADE", "QUIET"): SignalParams(
        tp_ratios=(0.4, 0.7, 1.0), sl_multiplier=0.7, entry_zone_bias=0.5,
        dca_enabled=True, validity_minutes=15,
    ),
    # --- SCALP + WHALE_MOMENTUM ---
    ("SCALP", "WHALE_MOMENTUM", "VOLATILE"): SignalParams(
        tp_ratios=(0.3, 0.7, 1.0), sl_multiplier=1.5, entry_zone_bias=0.8,
        dca_enabled=False, validity_minutes=3,
    ),
    ("SCALP", "WHALE_MOMENTUM", "TRENDING_UP"): SignalParams(
        tp_ratios=(0.5, 1.0, 1.5), sl_multiplier=1.3, entry_zone_bias=0.8,
        dca_enabled=False, validity_minutes=5,
    ),
    ("SCALP", "WHALE_MOMENTUM", "TRENDING_DOWN"): SignalParams(
        tp_ratios=(0.5, 1.0, 1.5), sl_multiplier=1.3, entry_zone_bias=0.8,
        dca_enabled=False, validity_minutes=5,
    ),
    # --- SCALP + BREAKOUT_RETEST ---
    ("SCALP", "BREAKOUT_RETEST", "TRENDING_UP"): SignalParams(
        tp_ratios=(0.8, 1.5, 2.2), sl_multiplier=1.1, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=8,
    ),
    ("SCALP", "BREAKOUT_RETEST", "TRENDING_DOWN"): SignalParams(
        tp_ratios=(0.8, 1.5, 2.2), sl_multiplier=1.1, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=8,
    ),
    # --- SCALP + MOMENTUM_EXPANSION ---
    ("SCALP", "MOMENTUM_EXPANSION", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.0, 2.0, 3.0), sl_multiplier=1.3, entry_zone_bias=0.75,
        dca_enabled=False, validity_minutes=5,
    ),
    ("SCALP", "MOMENTUM_EXPANSION", "VOLATILE"): SignalParams(
        tp_ratios=(0.5, 1.0, 1.5), sl_multiplier=1.5, entry_zone_bias=0.8,
        dca_enabled=False, validity_minutes=3,
    ),
    # --- SCALP + TREND_PULLBACK_CONTINUATION ---
    ("SCALP", "TREND_PULLBACK_CONTINUATION", "TRENDING_UP"): SignalParams(
        tp_ratios=(0.8, 1.5, 2.3), sl_multiplier=1.0, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=10,
    ),
    ("SCALP", "TREND_PULLBACK_CONTINUATION", "WEAK_TREND"): SignalParams(
        tp_ratios=(0.6, 1.0, 1.5), sl_multiplier=0.9, entry_zone_bias=0.6,
        dca_enabled=True, validity_minutes=12,
    ),
    # --- SWING setups ---
    ("SWING", "BREAKOUT_RETEST", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.2, 2.0, 3.0), sl_multiplier=1.2, entry_zone_bias=0.7,
        dca_enabled=True, validity_minutes=30,
    ),
    ("SWING", "LIQUIDITY_SWEEP_REVERSAL", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.0, 1.8, 2.8), sl_multiplier=1.1, entry_zone_bias=0.65,
        dca_enabled=True, validity_minutes=30,
    ),
    # --- SPOT setups ---
    ("SPOT", "BREAKOUT_RETEST", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.5, 2.5, 4.0), sl_multiplier=1.0, entry_zone_bias=0.65,
        dca_enabled=True, validity_minutes=60,
    ),
    ("SPOT", "BREAKOUT_INITIAL", "TRENDING_UP"): SignalParams(
        tp_ratios=(1.5, 2.5, 4.0), sl_multiplier=1.0, entry_zone_bias=0.65,
        dca_enabled=True, validity_minutes=60,
    ),
    # --- SPOT SHORT setups: DCA disabled for SHORT signals ---
    ("SPOT", "BREAKOUT_RETEST", "TRENDING_DOWN"): SignalParams(
        tp_ratios=(2.0, 5.0, 10.0), sl_multiplier=1.0, entry_zone_bias=0.65,
        dca_enabled=False, validity_minutes=60,
    ),
    ("SPOT", "BREAKOUT_INITIAL", "TRENDING_DOWN"): SignalParams(
        tp_ratios=(2.0, 5.0, 10.0), sl_multiplier=1.0, entry_zone_bias=0.65,
        dca_enabled=False, validity_minutes=60,
    ),
}


def lookup_signal_params(
    channel: str,
    setup_class: str,
    regime: str,
) -> SignalParams:
    """Look up regime-aware signal params with graceful fallback.

    Tries exact match first, then falls back to channel+setup with any regime,
    then returns defaults.
    """
    # Determine channel prefix
    channel_upper = channel.upper()
    if "SWING" in channel_upper:
        prefix = "SWING"
    elif "SPOT" in channel_upper:
        prefix = "SPOT"
    else:
        prefix = "SCALP"

    # Exact match
    key = (prefix, setup_class, regime)
    if key in PARAM_TABLE:
        return PARAM_TABLE[key]

    # Fallback: try any regime for this channel+setup
    for (p, s, _r), params in PARAM_TABLE.items():
        if p == prefix and s == setup_class:
            return params

    # Default
    return _DEFAULT
