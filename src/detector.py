"""SMC Detection Orchestrator.

Provides :class:`SMCDetector` which bundles all Smart Money Concepts detection
logic into a single, reusable component.  The result is returned as an
:class:`SMCResult` dataclass so ``main.py._scan_symbol()`` stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ai_engine import WhaleAlert, detect_volume_delta_spike, detect_whale_trade
from src.smc import FVGZone, LiquiditySweep, MSSSignal, detect_fvg, detect_liquidity_sweeps, detect_mss
from src.utils import get_logger

log = get_logger("detector")

# Lower-timeframe lookup used for MSS confirmation
_LTF_MAP: Dict[str, str] = {
    "4h": "1h",
    "1h": "15m",
    "15m": "5m",
    "5m": "1m",
}

# Ordered preference for SMC detection timeframes (most sensitive first)
_SMC_TIMEFRAMES = ("5m", "4h", "15m", "1m")


@dataclass
class SMCResult:
    """Unified output of SMC detection for a single symbol."""

    sweeps: List[LiquiditySweep] = field(default_factory=list)
    mss: Optional[MSSSignal] = None
    fvg: List[FVGZone] = field(default_factory=list)
    whale_alert: Optional[WhaleAlert] = None
    volume_delta_spike: bool = False
    recent_ticks: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a plain dict for backward-compat with channel evaluate() calls."""
        return {
            "sweeps": self.sweeps,
            "mss": self.mss,
            "fvg": self.fvg,
            "whale_alert": self.whale_alert,
            "volume_delta_spike": self.volume_delta_spike,
            "recent_ticks": self.recent_ticks,
        }


class SMCDetector:
    """Runs all SMC + whale/tape detection for a given symbol snapshot."""

    def detect(
        self,
        symbol: str,
        candles: Dict[str, Dict[str, Any]],
        ticks: List[Dict[str, Any]],
    ) -> SMCResult:
        """Run full SMC detection and return an :class:`SMCResult`.

        Parameters
        ----------
        symbol:
            Trading symbol (used only for logging).
        candles:
            Dict of timeframe → OHLCV arrays, e.g. ``{"5m": {"high": ..., ...}}``.
        ticks:
            Recent trade ticks from the data store.
        """
        result = SMCResult()

        # ------------------------------------------------------------------
        # SMC detection (sweeps, MSS, FVG) across preferred timeframes
        # ------------------------------------------------------------------
        for tf_key in _SMC_TIMEFRAMES:
            cd = candles.get(tf_key)
            if cd is None or len(cd.get("close", [])) < 51:
                continue

            sweeps = detect_liquidity_sweeps(cd["high"], cd["low"], cd["close"])
            if not sweeps:
                continue

            result.sweeps = sweeps

            ltf_key = _LTF_MAP.get(tf_key, "1m")
            ltf_cd = candles.get(ltf_key)
            if ltf_cd and len(ltf_cd.get("close", [])) > 1:
                mss_sig = detect_mss(sweeps[0], ltf_cd["close"])
                result.mss = mss_sig

            result.fvg = detect_fvg(cd["high"], cd["low"], cd["close"])
            break  # use first timeframe that has a sweep

        # ------------------------------------------------------------------
        # Whale / tape detection
        # ------------------------------------------------------------------
        if ticks:
            latest = ticks[-1]
            result.whale_alert = detect_whale_trade(
                latest.get("price", 0.0), latest.get("qty", 0.0)
            )
            recent = ticks[-100:]
            result.recent_ticks = recent

            buy_v = sum(
                t.get("qty", 0) * t.get("price", 0)
                for t in recent
                if not t.get("isBuyerMaker", True)
            )
            sell_v = sum(
                t.get("qty", 0) * t.get("price", 0)
                for t in recent
                if t.get("isBuyerMaker", True)
            )
            avg_delta = (buy_v + sell_v) / 2.0 if (buy_v + sell_v) > 0 else 0.0
            result.volume_delta_spike = detect_volume_delta_spike(
                buy_v - sell_v, avg_delta
            )

        return result
