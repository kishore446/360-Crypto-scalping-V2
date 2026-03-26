"""Portfolio-Level Drawdown Circuit Breaker.

Monitors aggregate realized + unrealized P&L across all channels and
applies tiered throttling to protect the account from catastrophic losses.

Tiers:
  NORMAL   — no restrictions
  YELLOW   — daily drawdown >= 3%: reduce new position sizes by 50%
  RED      — daily drawdown >= 5%: halt ALL new signals for 4 hours
  BLACK    — daily drawdown >= 8%: halt ALL new signals for 24 hours + admin alert
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from src.utils import get_logger

log = get_logger("portfolio_guard")


class DrawdownTier(str, Enum):
    """Portfolio drawdown severity tiers."""
    NORMAL = "NORMAL"
    YELLOW = "YELLOW"
    RED = "RED"
    BLACK = "BLACK"


# Configurable thresholds
_DEFAULT_YELLOW_DRAWDOWN_PCT: float = 3.0
_DEFAULT_RED_DRAWDOWN_PCT: float = 5.0
_DEFAULT_BLACK_DRAWDOWN_PCT: float = 8.0
_DEFAULT_RED_HALT_SECONDS: int = 4 * 3600      # 4 hours
_DEFAULT_BLACK_HALT_SECONDS: int = 24 * 3600    # 24 hours
_DEFAULT_YELLOW_SIZE_MULTIPLIER: float = 0.5    # 50% position size reduction
_ROLLING_WINDOW_SECONDS: float = 24 * 3600.0    # 24-hour rolling window
_MAX_PNL_RECORDS: int = 5000


@dataclass
class PnLRecord:
    """A single P&L event (trade close or periodic mark-to-market)."""
    signal_id: str
    channel: str
    symbol: str
    pnl_pct: float
    is_realized: bool           # True for closed trades, False for mark-to-market
    timestamp: float = field(default_factory=time.monotonic)


class PortfolioGuard:
    """Monitors aggregate portfolio drawdown and enforces tiered protection.

    Parameters
    ----------
    yellow_pct : float
        Daily drawdown % to trigger YELLOW (position size reduction).
    red_pct : float
        Daily drawdown % to trigger RED (4-hour signal halt).
    black_pct : float
        Daily drawdown % to trigger BLACK (24-hour halt + admin alert).
    red_halt_seconds : int
        Duration of RED halt.
    black_halt_seconds : int
        Duration of BLACK halt.
    yellow_size_multiplier : float
        Position size multiplier during YELLOW tier (0.5 = 50% reduction).
    alert_callback : Optional[Callable]
        Async callable for sending admin alerts (Telegram).
    """

    def __init__(
        self,
        yellow_pct: float = _DEFAULT_YELLOW_DRAWDOWN_PCT,
        red_pct: float = _DEFAULT_RED_DRAWDOWN_PCT,
        black_pct: float = _DEFAULT_BLACK_DRAWDOWN_PCT,
        red_halt_seconds: int = _DEFAULT_RED_HALT_SECONDS,
        black_halt_seconds: int = _DEFAULT_BLACK_HALT_SECONDS,
        yellow_size_multiplier: float = _DEFAULT_YELLOW_SIZE_MULTIPLIER,
        alert_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.yellow_pct = yellow_pct
        self.red_pct = red_pct
        self.black_pct = black_pct
        self.red_halt_seconds = red_halt_seconds
        self.black_halt_seconds = black_halt_seconds
        self.yellow_size_multiplier = yellow_size_multiplier
        self._alert_callback = alert_callback

        # Rolling P&L records
        self._pnl_records: Deque[PnLRecord] = deque(maxlen=_MAX_PNL_RECORDS)

        # Current state
        self._current_tier: DrawdownTier = DrawdownTier.NORMAL
        self._halt_until: Optional[float] = None
        self._last_tier_change: float = time.monotonic()
        self._peak_equity_pct: float = 100.0   # Track peak for drawdown calc
        self._current_equity_pct: float = 100.0

        # Per-channel P&L tracking for attribution
        self._channel_pnl: Dict[str, float] = {}

        # Tier transition history for reporting
        self._tier_history: List[dict] = []

    # ---- Public API ----

    def record_pnl(
        self,
        signal_id: str,
        channel: str,
        symbol: str,
        pnl_pct: float,
        is_realized: bool = True,
    ) -> None:
        """Record a P&L event and re-evaluate the drawdown tier."""
        record = PnLRecord(
            signal_id=signal_id,
            channel=channel,
            symbol=symbol,
            pnl_pct=pnl_pct,
            is_realized=is_realized,
        )
        self._pnl_records.append(record)

        # Update channel P&L
        self._channel_pnl[channel] = self._channel_pnl.get(channel, 0.0) + pnl_pct

        # Update equity tracking
        self._current_equity_pct += pnl_pct
        if self._current_equity_pct > self._peak_equity_pct:
            self._peak_equity_pct = self._current_equity_pct

        self._evaluate_tier()

    def check_signal_allowed(self) -> tuple[bool, str, float]:
        """Check if a new signal is allowed under current drawdown state.

        Returns
        -------
        tuple of (allowed, reason, position_size_multiplier)
            - allowed: False during RED/BLACK halt periods
            - reason: human-readable explanation
            - position_size_multiplier: 1.0 for NORMAL, 0.5 for YELLOW, 0.0 for RED/BLACK
        """
        self._check_halt_expiry()

        tier = self._current_tier
        dd_pct = self._rolling_drawdown_pct()

        if tier == DrawdownTier.BLACK:
            return (
                False,
                f"🔴 BLACK drawdown alert: {dd_pct:.2f}% daily loss. "
                f"All signals halted for {self.black_halt_seconds // 3600}h.",
                0.0,
            )

        if tier == DrawdownTier.RED:
            return (
                False,
                f"🟠 RED drawdown alert: {dd_pct:.2f}% daily loss. "
                f"All signals halted for {self.red_halt_seconds // 3600}h.",
                0.0,
            )

        if tier == DrawdownTier.YELLOW:
            return (
                True,
                f"🟡 YELLOW drawdown alert: {dd_pct:.2f}% daily loss. "
                f"Position sizes reduced to {self.yellow_size_multiplier * 100:.0f}%.",
                self.yellow_size_multiplier,
            )

        return (True, "", 1.0)

    @property
    def current_tier(self) -> DrawdownTier:
        """Return the current drawdown tier."""
        self._check_halt_expiry()
        return self._current_tier

    def rolling_drawdown_pct(self) -> float:
        """Return the current rolling 24h drawdown percentage."""
        return self._rolling_drawdown_pct()

    def status_text(self) -> str:
        """Return a human-readable status string for Telegram/dashboard."""
        self._check_halt_expiry()
        dd = self._rolling_drawdown_pct()
        tier = self._current_tier

        tier_emoji = {
            DrawdownTier.NORMAL: "✅",
            DrawdownTier.YELLOW: "🟡",
            DrawdownTier.RED: "🟠",
            DrawdownTier.BLACK: "🔴",
        }

        lines = [
            f"{tier_emoji[tier]} *Portfolio Guard: {tier.value}*",
            f"Rolling 24h drawdown: {dd:.2f}%",
            f"Thresholds: Yellow {self.yellow_pct}% / Red {self.red_pct}% / Black {self.black_pct}%",
        ]

        if self._halt_until is not None:
            remaining = max(0, self._halt_until - time.monotonic())
            lines.append(f"Halt remaining: {remaining / 60:.0f} minutes")

        if self._channel_pnl:
            lines.append("\n*Per-Channel P&L (24h):*")
            for ch, pnl in sorted(self._channel_pnl.items()):
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {ch}: {sign}{pnl:.2f}%")

        return "\n".join(lines)

    def reset(self) -> None:
        """Manual reset — clear all state and return to NORMAL."""
        self._pnl_records.clear()
        self._current_tier = DrawdownTier.NORMAL
        self._halt_until = None
        self._peak_equity_pct = 100.0
        self._current_equity_pct = 100.0
        self._channel_pnl.clear()
        self._tier_history.clear()
        log.info("PortfolioGuard reset to NORMAL manually.")

    # ---- Internal ----

    def _rolling_drawdown_pct(self) -> float:
        """Compute the rolling 24h drawdown from peak equity."""
        now = time.monotonic()
        cutoff = now - _ROLLING_WINDOW_SECONDS

        # Sum P&L within rolling window
        window_pnl = sum(
            r.pnl_pct for r in self._pnl_records
            if r.timestamp >= cutoff
        )

        # Drawdown = max loss from peak within window
        # Simple: if cumulative window P&L is negative, that's our drawdown
        return abs(min(0.0, window_pnl))

    def _evaluate_tier(self) -> None:
        """Evaluate and transition drawdown tier based on current metrics."""
        dd = self._rolling_drawdown_pct()
        old_tier = self._current_tier

        if dd >= self.black_pct:
            new_tier = DrawdownTier.BLACK
            if old_tier != DrawdownTier.BLACK:
                self._halt_until = time.monotonic() + self.black_halt_seconds
                self._emit_alert(
                    f"🔴 *PORTFOLIO GUARD: BLACK ALERT*\n"
                    f"Daily drawdown: {dd:.2f}% (threshold: {self.black_pct}%)\n"
                    f"ALL signals halted for {self.black_halt_seconds // 3600} hours.\n"
                    f"Manual override: /portfolio_reset"
                )
        elif dd >= self.red_pct:
            new_tier = DrawdownTier.RED
            if old_tier not in (DrawdownTier.RED, DrawdownTier.BLACK):
                self._halt_until = time.monotonic() + self.red_halt_seconds
                self._emit_alert(
                    f"🟠 *PORTFOLIO GUARD: RED ALERT*\n"
                    f"Daily drawdown: {dd:.2f}% (threshold: {self.red_pct}%)\n"
                    f"ALL signals halted for {self.red_halt_seconds // 3600} hours."
                )
        elif dd >= self.yellow_pct:
            new_tier = DrawdownTier.YELLOW
            if old_tier == DrawdownTier.NORMAL:
                self._emit_alert(
                    f"🟡 *PORTFOLIO GUARD: YELLOW ALERT*\n"
                    f"Daily drawdown: {dd:.2f}% (threshold: {self.yellow_pct}%)\n"
                    f"Position sizes reduced to {self.yellow_size_multiplier * 100:.0f}%."
                )
        else:
            new_tier = DrawdownTier.NORMAL
            if old_tier != DrawdownTier.NORMAL:
                self._emit_alert(
                    f"✅ *PORTFOLIO GUARD: RECOVERED*\n"
                    f"Daily drawdown back to {dd:.2f}%. Normal operations resumed."
                )

        if new_tier != old_tier:
            self._tier_history.append({
                "from": old_tier.value,
                "to": new_tier.value,
                "drawdown_pct": dd,
                "timestamp": time.monotonic(),
            })
            log.info(
                "Portfolio guard tier change: %s → %s (drawdown: %.2f%%)",
                old_tier.value, new_tier.value, dd,
            )

        self._current_tier = new_tier

    def _check_halt_expiry(self) -> None:
        """Check if a RED/BLACK halt has expired and re-evaluate."""
        if self._halt_until is not None and time.monotonic() >= self._halt_until:
            self._halt_until = None
            log.info("Portfolio guard halt period expired, re-evaluating tier.")
            self._evaluate_tier()

    def _emit_alert(self, message: str) -> None:
        """Send an alert via the configured callback."""
        log.warning(message)
        if self._alert_callback is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._alert_callback(message))
                else:
                    loop.run_until_complete(self._alert_callback(message))
            except Exception as exc:
                log.error("Failed to send portfolio guard alert: %s", exc)
