"""Select Mode – ultra-selective automated signal filtering.

When enabled, signals must pass stricter multi-layer filters before publishing.
Mimics manual analyst review but fully automated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

log = get_logger("select_mode")


@dataclass
class SelectModeConfig:
    """Configurable thresholds for select mode."""

    enabled: bool = False
    min_confidence: float = 80.0
    max_daily_signals: int = 5
    min_confluence_timeframes: int = 2
    max_spread_pct: float = 0.015
    min_volume_24h: float = 10_000_000.0
    min_adx: float = 25.0
    rsi_min: float = 30.0
    rsi_max: float = 70.0
    require_smc_event: bool = True
    require_ai_sentiment_match: bool = True
    require_cross_exchange: bool = True


@dataclass
class SelectModeStats:
    """Daily tracking stats for select mode."""

    signals_today: Dict[str, int] = field(default_factory=dict)  # channel -> count
    last_reset_date: str = ""
    signals_passed: int = 0
    signals_rejected: int = 0


class SelectModeFilter:
    """Applies ultra-selective filters when select mode is enabled."""

    def __init__(self, config: Optional[SelectModeConfig] = None) -> None:
        self.config = config or SelectModeConfig()
        self._stats = SelectModeStats()

    @property
    def enabled(self) -> bool:
        """Return ``True`` when select mode is currently active."""
        return self.config.enabled

    def enable(self) -> None:
        """Enable ultra-selective filtering."""
        self.config.enabled = True
        log.info("Select mode ENABLED")

    def disable(self) -> None:
        """Disable ultra-selective filtering (back to regular signal flow)."""
        self.config.enabled = False
        log.info("Select mode DISABLED")

    def should_publish(
        self,
        signal: Any,
        confidence: float,
        indicators: Dict[str, dict],
        smc_data: Any,
        ai_sentiment: Dict,
        cross_exchange_verified: Optional[bool],
        volume_24h: float,
        spread_pct: float,
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for whether the signal passes select mode filters.

        If select mode is OFF, always returns ``(True, "")``.
        """
        if not self.config.enabled:
            return True, ""

        # Reset daily counters at midnight
        self._maybe_reset_daily()

        reasons: List[str] = []
        passed_filters = 0
        total_filters = 0

        # Filter 1: Minimum confidence
        total_filters += 1
        if confidence >= self.config.min_confidence:
            passed_filters += 1
        else:
            reasons.append(f"Confidence {confidence:.0f} < {self.config.min_confidence:.0f}")

        # Filter 2: SMC event required
        if self.config.require_smc_event:
            total_filters += 1
            if isinstance(smc_data, dict):
                has_sweep = bool(smc_data.get("sweeps"))
                has_mss = smc_data.get("mss") is not None
                has_fvg = bool(smc_data.get("fvg"))
            else:
                has_sweep = bool(getattr(smc_data, "sweeps", None))
                has_mss = bool(getattr(smc_data, "mss", None))
                has_fvg = bool(getattr(smc_data, "fvg", None))
            if has_sweep or has_mss or has_fvg:
                passed_filters += 1
            else:
                reasons.append("No SMC event detected")

        # Filter 3: ADX minimum
        total_filters += 1
        primary_tf = indicators.get("5m", indicators.get("1m", {}))
        adx_val = primary_tf.get("adx_last") or 0
        if adx_val >= self.config.min_adx:
            passed_filters += 1
        else:
            reasons.append(f"ADX {adx_val:.1f} < {self.config.min_adx:.1f}")

        # Filter 4: Spread check
        total_filters += 1
        if spread_pct <= self.config.max_spread_pct:
            passed_filters += 1
        else:
            reasons.append(f"Spread {spread_pct:.4f} > {self.config.max_spread_pct:.4f}")

        # Filter 5: Volume check
        total_filters += 1
        if volume_24h >= self.config.min_volume_24h:
            passed_filters += 1
        else:
            reasons.append(f"Volume ${volume_24h:,.0f} < ${self.config.min_volume_24h:,.0f}")

        # Filter 6: Multi-timeframe EMA confluence
        total_filters += 1
        direction = signal.direction.value
        tf_agree_count = 0
        for tf_key, ind in indicators.items():
            ema9 = ind.get("ema9_last")
            ema21 = ind.get("ema21_last")
            if ema9 is not None and ema21 is not None:
                if direction == "LONG" and ema9 > ema21:
                    tf_agree_count += 1
                elif direction == "SHORT" and ema9 < ema21:
                    tf_agree_count += 1
        if tf_agree_count >= self.config.min_confluence_timeframes:
            passed_filters += 1
        else:
            reasons.append(
                f"EMA confluence {tf_agree_count}/{self.config.min_confluence_timeframes} TFs"
            )

        # Filter 7: Multi-TF momentum agreement
        total_filters += 1
        mom_agree = 0
        for tf_key, ind in indicators.items():
            mom = ind.get("momentum_last")
            if mom is not None:
                if direction == "LONG" and mom > 0:
                    mom_agree += 1
                elif direction == "SHORT" and mom < 0:
                    mom_agree += 1
        if mom_agree >= 2:
            passed_filters += 1
        else:
            reasons.append(f"Momentum agreement {mom_agree}/2 TFs")

        # Filter 8: RSI not extreme
        total_filters += 1
        rsi_val = primary_tf.get("rsi_last")
        if rsi_val is None:
            passed_filters += 1  # no RSI data available – treat as neutral and allow signal to pass
        elif self.config.rsi_min <= rsi_val <= self.config.rsi_max:
            passed_filters += 1
        else:
            reasons.append(f"RSI {rsi_val:.1f} outside {self.config.rsi_min:.0f}-{self.config.rsi_max:.0f}")

        # Filter 9: AI sentiment matches direction
        if self.config.require_ai_sentiment_match:
            total_filters += 1
            ai_label = ai_sentiment.get("label", "Neutral").lower()
            if ai_label == "neutral":
                passed_filters += 1
            elif ai_label == "bullish" and direction == "LONG":
                passed_filters += 1
            elif ai_label == "bearish" and direction == "SHORT":
                passed_filters += 1
            else:
                reasons.append(f"AI sentiment '{ai_label}' vs signal {direction}")

        # Filter 10: Cross-exchange verified
        if self.config.require_cross_exchange:
            total_filters += 1
            if cross_exchange_verified is True:
                passed_filters += 1
            elif cross_exchange_verified is None:
                passed_filters += 1  # no second exchange configured – treat as neutral and allow signal to pass
            else:
                reasons.append("Cross-exchange verification failed")

        # Filter 11: Daily cap per channel
        total_filters += 1
        channel_name = signal.channel
        daily_count = self._stats.signals_today.get(channel_name, 0)
        if daily_count < self.config.max_daily_signals:
            passed_filters += 1
        else:
            reasons.append(
                f"Daily cap reached ({daily_count}/{self.config.max_daily_signals})"
            )

        allowed = len(reasons) == 0

        if allowed:
            self._stats.signals_passed += 1
            self._stats.signals_today[channel_name] = daily_count + 1
        else:
            self._stats.signals_rejected += 1

        reason_str = (
            "; ".join(reasons) if reasons else f"Passed {passed_filters}/{total_filters} filters"
        )

        log.info(
            "Select mode %s: %s %s — %s",
            "PASS" if allowed else "REJECT",
            signal.symbol,
            direction,
            reason_str,
        )

        return allowed, reason_str

    def _maybe_reset_daily(self) -> None:
        """Reset per-channel daily signal counters at midnight."""
        today = time.strftime("%Y-%m-%d")
        if self._stats.last_reset_date != today:
            self._stats.signals_today = {}
            self._stats.last_reset_date = today

    def status_text(self) -> str:
        """Return a human-readable status string for use in Telegram replies."""
        self._maybe_reset_daily()
        if not self.config.enabled:
            return "🔘 Select Mode: OFF (regular signals)"

        today_total = sum(self._stats.signals_today.values())
        lines = [
            "🌹 *Select Mode: ON (ultra-selective)*",
            f"Min Confidence: {self.config.min_confidence:.0f}",
            f"Min ADX: {self.config.min_adx:.0f}",
            f"Max Spread: {self.config.max_spread_pct:.4f}",
            f"Min Volume: ${self.config.min_volume_24h:,.0f}",
            f"Daily Cap: {self.config.max_daily_signals}/channel",
            f"Min TF Confluence: {self.config.min_confluence_timeframes}",
            f"RSI Band: {self.config.rsi_min:.0f}-{self.config.rsi_max:.0f}",
            f"Require SMC: {'Yes' if self.config.require_smc_event else 'No'}",
            f"Require AI Match: {'Yes' if self.config.require_ai_sentiment_match else 'No'}",
            f"Require Cross-Exch: {'Yes' if self.config.require_cross_exchange else 'No'}",
            "",
            f"Today: {today_total} signals passed | {self._stats.signals_rejected} rejected",
        ]
        for ch, count in self._stats.signals_today.items():
            lines.append(f"  {ch}: {count}/{self.config.max_daily_signals}")

        return "\n".join(lines)

    def update_config(self, key: str, value: str) -> tuple[bool, str]:
        """Update a config threshold by name.

        Returns ``(success, message)``.
        """
        mapping: Dict[str, tuple] = {
            "confidence": ("min_confidence", float),
            "min_confidence": ("min_confidence", float),
            "daily_cap": ("max_daily_signals", int),
            "max_daily": ("max_daily_signals", int),
            "max_daily_signals": ("max_daily_signals", int),
            "min_confluence": ("min_confluence_timeframes", int),
            "min_tf": ("min_confluence_timeframes", int),
            "spread": ("max_spread_pct", float),
            "max_spread": ("max_spread_pct", float),
            "volume": ("min_volume_24h", float),
            "min_volume": ("min_volume_24h", float),
            "adx": ("min_adx", float),
            "min_adx": ("min_adx", float),
            "rsi_min": ("rsi_min", float),
            "rsi_max": ("rsi_max", float),
        }

        if key not in mapping:
            valid_attrs = sorted({attr for attr, _ in mapping.values()})
            return False, f"Unknown config key: {key}. Valid: {', '.join(valid_attrs)}"

        attr, type_fn = mapping[key]
        try:
            typed_value = type_fn(value)
            setattr(self.config, attr, typed_value)
            return True, f"✅ Select mode {attr} set to {typed_value}"
        except ValueError:
            return False, f"❌ Invalid value: {value}"
