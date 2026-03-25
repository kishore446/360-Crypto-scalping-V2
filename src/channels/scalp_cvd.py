"""360_SCALP_CVD – CVD Divergence Scalp ⚡

Trigger : Bullish or bearish CVD divergence detected on 5m timeframe.
Logic   : BULLISH divergence (price makes new low, CVD higher low) → LONG
          BEARISH divergence (price makes new high, CVD lower high) → SHORT
Filters : Must be near a support/resistance zone (recent 20-bar high/low),
          standard quality gates (ADX, spread, volume)
Risk    : SL 0.15-0.3%, TP1 1R, TP2 2R
Signal ID prefix: "SCVD-"
"""

from __future__ import annotations

from typing import Dict, Optional

from config import CHANNEL_SCALP_CVD
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_spread, check_volume
from src.smc import Direction

# Price must be within this percentage of recent 20-bar high/low to be
# considered "at support/resistance".
_SR_PROXIMITY_PCT: float = 0.8  # was 0.5; 0.5% is too tight for many valid setups


class ScalpCVDChannel(BaseChannel):
    """CVD Divergence scalp trigger."""

    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP_CVD)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 21:
            return None

        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        ind = indicators.get("5m", {})

        # ADX gate: CVD divergence is unreliable in strong trends (ADX > 35)
        # divergence can persist for 20+ candles without reverting
        adx_val = ind.get("adx_last")
        if adx_val is not None and adx_val > 35:
            return None

        # Use CVD divergence from smc_data (already detected by SMCDetector)
        cvd_div = smc_data.get("cvd_divergence")
        if cvd_div is None:
            return None

        closes = list(m5.get("close", []))
        if len(closes) < 20:
            return None

        close = float(closes[-1])
        recent_high = max(float(h) for h in list(m5.get("high", closes))[-20:])
        recent_low = min(float(l) for l in list(m5.get("low", closes))[-20:])

        if cvd_div == "BULLISH":
            direction = Direction.LONG
            # Must be near recent low (support)
            if close > recent_low * (1 + _SR_PROXIMITY_PCT / 100):
                return None
        elif cvd_div == "BEARISH":
            direction = Direction.SHORT
            # Must be near recent high (resistance)
            if close < recent_high * (1 - _SR_PROXIMITY_PCT / 100):
                return None
        else:
            return None

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        rsi_last = ind.get("rsi_last")
        if rsi_last is not None:
            if direction == Direction.LONG and rsi_last > 75:
                return None
            if direction == Direction.SHORT and rsi_last < 25:
                return None

        atr_val = ind.get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.8)

        if direction == Direction.LONG:
            sl = close - sl_dist
            tp1 = close + sl_dist * self.config.tp_ratios[0]
            tp2 = close + sl_dist * self.config.tp_ratios[1]
            tp3 = close + sl_dist * self.config.tp_ratios[2]
        else:
            sl = close + sl_dist
            tp1 = close - sl_dist * self.config.tp_ratios[0]
            tp2 = close - sl_dist * self.config.tp_ratios[1]
            tp3 = close - sl_dist * self.config.tp_ratios[2]

        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
            return None

        return build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SCVD",
            atr_val=atr_val,
            setup_class="CVD_DIVERGENCE",
        )
