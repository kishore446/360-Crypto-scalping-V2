"""Risk Management Module.

Provides :class:`RiskManager` for position sizing, risk labelling, and
concurrent-signal validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils import get_logger

log = get_logger("risk")

# Minimum required 24h volume (USD) to be considered adequately liquid
_MIN_VOLUME_LOW_RISK: float = 50_000_000    # > $50M  → Low risk
_MIN_VOLUME_MED_RISK: float = 10_000_000    # > $10M  → Medium risk

# ATR-to-price ratio thresholds (higher ATR = higher volatility risk)
_ATR_HIGH_RISK_PCT: float = 0.5    # ATR > 0.5 % of price → High
_ATR_VERY_HIGH_RISK_PCT: float = 1.0  # ATR > 1.0 % → Very High

# Default account risk per trade (percentage)
_DEFAULT_ACCOUNT_RISK_PCT: float = 1.0

# Max concurrent signals per symbol in the same direction
_MAX_CONCURRENT_SAME_DIRECTION: int = 1
_MAX_CONCURRENT_PER_SYMBOL: int = 2


@dataclass
class RiskAssessment:
    """Output of a risk calculation."""

    risk_label: str          # "Low" | "Medium" | "High" | "Very High"
    position_size_pct: float  # recommended position size as % of account
    risk_reward: float        # TP1 / SL distance ratio
    allowed: bool             # False if concurrent-signal limits exceeded
    reason: str = ""


class RiskManager:
    """Calculates risk, position sizing, and validates concurrent-signal limits."""

    def __init__(
        self,
        account_risk_pct: float = _DEFAULT_ACCOUNT_RISK_PCT,
        max_concurrent_same_direction: int = _MAX_CONCURRENT_SAME_DIRECTION,
        max_concurrent_per_symbol: int = _MAX_CONCURRENT_PER_SYMBOL,
    ) -> None:
        self.account_risk_pct = account_risk_pct
        self.max_concurrent_same_direction = max_concurrent_same_direction
        self.max_concurrent_per_symbol = max_concurrent_per_symbol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_risk(
        self,
        signal: Any,
        indicators: Dict[str, Any],
        volume_24h_usd: float,
        active_signals: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Evaluate risk for *signal* and return a :class:`RiskAssessment`.

        Parameters
        ----------
        signal:
            A :class:`src.channels.base.Signal` instance.
        indicators:
            Pre-computed indicator dict for the signal's primary timeframe.
        volume_24h_usd:
            24-hour USD volume for the symbol.
        active_signals:
            Current active signals dict (signal_id → Signal) for concurrent
            limit checking.  Pass ``None`` or an empty dict to skip.
        """
        entry: float = float(getattr(signal, "entry", 0.0))
        sl: float = float(getattr(signal, "stop_loss", entry))
        tp1: float = float(getattr(signal, "tp1", entry))
        direction_val: str = getattr(getattr(signal, "direction", None), "value", "LONG")
        symbol: str = getattr(signal, "symbol", "")

        # Risk-reward ratio
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

        # Position sizing
        position_size_pct = self._position_size(entry, sl)

        # Risk label
        atr_val: Optional[float] = indicators.get("atr_last")
        risk_label = self._classify_risk(entry, atr_val, volume_24h_usd, signal)

        # Concurrent-signal validation
        allowed, reason = self._validate_concurrent(
            symbol, direction_val, active_signals or {}
        )

        return RiskAssessment(
            risk_label=risk_label,
            position_size_pct=position_size_pct,
            risk_reward=round(rr, 2),
            allowed=allowed,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _position_size(self, entry: float, stop_loss: float) -> float:
        """Calculate position size as % of account using fixed-risk formula.

        position_size % = account_risk_pct / (|entry - sl| / entry * 100)
        """
        if entry == 0 or stop_loss == 0:
            return self.account_risk_pct
        sl_pct = abs(entry - stop_loss) / entry * 100.0
        if sl_pct == 0:
            return self.account_risk_pct
        return round(min(self.account_risk_pct / sl_pct * 100.0, 100.0), 2)

    @staticmethod
    def _classify_risk(
        entry: float,
        atr_val: Optional[float],
        volume_24h_usd: float,
        signal: Any,
    ) -> str:
        """Assign a risk label based on ATR, volume and spread."""
        spread_pct: float = float(getattr(signal, "spread_pct", 0.0))
        confidence: float = float(getattr(signal, "confidence", 50.0))

        # ATR-based volatility
        atr_pct = (atr_val / entry * 100.0) if (atr_val and entry > 0) else 0.0

        score = 0
        if atr_pct >= _ATR_VERY_HIGH_RISK_PCT:
            score += 3
        elif atr_pct >= _ATR_HIGH_RISK_PCT:
            score += 2
        elif atr_pct > 0:
            score += 1

        if volume_24h_usd < _MIN_VOLUME_MED_RISK:
            score += 2
        elif volume_24h_usd < _MIN_VOLUME_LOW_RISK:
            score += 1

        if spread_pct > 0.02:
            score += 1
        if confidence < 60:
            score += 1

        if score >= 5:
            return "Very High"
        if score >= 3:
            return "High"
        if score >= 2:
            return "Medium"
        return "Low"

    def _validate_concurrent(
        self,
        symbol: str,
        direction: str,
        active_signals: Dict[str, Any],
    ) -> tuple[bool, str]:
        """Return (allowed, reason) after checking concurrent-signal limits."""
        same_dir = 0
        same_sym = 0
        for sig in active_signals.values():
            if getattr(sig, "symbol", "") == symbol:
                same_sym += 1
                if getattr(getattr(sig, "direction", None), "value", "") == direction:
                    same_dir += 1

        if same_sym >= self.max_concurrent_per_symbol:
            return False, f"Max {self.max_concurrent_per_symbol} concurrent signals per symbol exceeded"
        if same_dir >= self.max_concurrent_same_direction:
            return False, f"Max {self.max_concurrent_same_direction} concurrent {direction} signals for {symbol}"
        return True, ""
