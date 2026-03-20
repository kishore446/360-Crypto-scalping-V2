"""Macro Watchdog — async background task for global market-event alerts.

This module replaces the AI's role in the trade-signal hot path.  Instead of
scoring individual trade signals (which added 2-5 s latency), the AI is now
used exclusively for *global market awareness*.  The ``MacroWatchdog`` polls
news feeds, the Fear & Greed Index, and an OpenAI-powered classifier to detect
macro-level events that could materially move crypto markets:

* 📰 Breaking news (wars, regulatory bans, exchange hacks)
* 🏛️ FOMC meetings / interest-rate decisions
* 📈 Major macroeconomic data releases (CPI, NFP)
* 🔰 New token listings on major exchanges
* ⚠️  Fear & Greed index extremes

When a significant event is detected it is formatted as a high-priority alert
and pushed directly to the configured Telegram admin/alerts channel, completely
bypassing the trade-signal queue so it never delays live signals.

Configuration (via environment variables):
  MACRO_WATCHDOG_ENABLED                  – "true" to enable (default: "true")
  MACRO_WATCHDOG_POLL_INTERVAL            – seconds between polls (default: "300")
  MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_LOW  – F&G below this → extreme fear alert (default: "20")
  MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_HIGH – F&G above this → extreme greed alert (default: "80")
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp

from config import (
    MACRO_WATCHDOG_ENABLED,
    MACRO_WATCHDOG_POLL_INTERVAL,
    MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_LOW,
    MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_HIGH,
)
from src.ai_engine import (
    fetch_fear_greed_index,
    fetch_news_sentiment,
)
from src.openai_evaluator import MacroEventResult, OpenAIEvaluator
from src.utils import get_logger

log = get_logger("macro_watchdog")

# Severity → emoji mapping for Telegram alerts
_SEVERITY_EMOJI: Dict[str, str] = {
    "LOW": "🟡",
    "MEDIUM": "🟠",
    "HIGH": "🔴",
    "CRITICAL": "🚨",
}

# Minimum Fear & Greed values for automatic alerts (before OpenAI classification)
_FG_EXTREME_LOW: int = MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_LOW
_FG_EXTREME_HIGH: int = MACRO_WATCHDOG_FEAR_GREED_THRESHOLD_HIGH

# Tracked symbols for news monitoring – uses the most liquid pairs by default
_DEFAULT_WATCH_SYMBOLS: List[str] = ["BTC", "ETH", "SOL", "BNB"]

# Minimum seconds between duplicate-headline alerts (per headline hash)
_DEDUP_TTL: float = 3600.0  # 1 hour


class MacroWatchdog:
    """Async background service that monitors macro-events and sends Telegram alerts.

    Parameters
    ----------
    send_alert:
        Async callable that sends a string message to the admin Telegram channel.
        Typically ``TelegramBot.send_admin_alert``.
    openai_evaluator:
        Optional :class:`~src.openai_evaluator.OpenAIEvaluator` instance.
        When ``None`` or disabled, events are still detected via the Fear & Greed
        index and CryptoPanic headlines but skipped through the AI classifier.
    poll_interval:
        Seconds between each polling cycle.  Defaults to the
        ``MACRO_WATCHDOG_POLL_INTERVAL`` config value.
    watch_symbols:
        Base coin names (e.g. ``["BTC", "ETH"]``) to monitor for news.
    """

    def __init__(
        self,
        send_alert: Callable[[str], Coroutine[Any, Any, bool]],
        openai_evaluator: Optional[OpenAIEvaluator] = None,
        poll_interval: Optional[float] = None,
        watch_symbols: Optional[List[str]] = None,
    ) -> None:
        self._send_alert = send_alert
        self._openai = openai_evaluator
        self._poll_interval: float = poll_interval if poll_interval is not None else MACRO_WATCHDOG_POLL_INTERVAL
        self._watch_symbols: List[str] = watch_symbols or _DEFAULT_WATCH_SYMBOLS
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._seen_headlines: Dict[str, float] = {}  # headline_hash → timestamp
        self._last_fg_value: Optional[int] = None    # track F&G changes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if not MACRO_WATCHDOG_ENABLED:
            log.info("MacroWatchdog disabled by configuration – skipping start")
            return
        if self._task is not None and not self._task.done():
            log.debug("MacroWatchdog already running")
            return
        self._task = asyncio.create_task(self._poll_loop(), name="macro_watchdog")
        log.info(
            "MacroWatchdog started (poll_interval={}s, symbols={})",
            self._poll_interval,
            self._watch_symbols,
        )

    async def stop(self) -> None:
        """Cancel the polling loop and close HTTP session."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        log.info("MacroWatchdog stopped")

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Infinite loop: poll events, send alerts, sleep."""
        while True:
            try:
                await self._check_macro_events()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("MacroWatchdog poll error: {}", exc)
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    async def _check_macro_events(self) -> None:
        """Run one full polling cycle: Fear & Greed + news headlines."""
        session = await self._get_session()

        # 1) Fear & Greed Index extremes
        await self._check_fear_greed(session)

        # 2) News headlines for each watched symbol
        for symbol in self._watch_symbols:
            try:
                await self._check_news(symbol, session)
            except Exception as exc:
                log.debug("News check failed for {}: {}", symbol, exc)

    async def _check_fear_greed(self, session: aiohttp.ClientSession) -> None:
        """Alert when the Fear & Greed index hits extreme territory."""
        try:
            fg = await fetch_fear_greed_index(session)
        except Exception as exc:
            log.debug("Fear & Greed fetch failed in MacroWatchdog: {}", exc)
            return

        value: int = fg.get("value", 50)
        classification: str = fg.get("classification", "Neutral")

        # Only alert on meaningful changes (avoid repeating the same extreme value)
        prev = self._last_fg_value
        self._last_fg_value = value
        if prev is not None and abs(value - prev) < 5:
            return  # no significant change

        if value <= _FG_EXTREME_LOW:
            emoji = "😱"
            severity = "HIGH" if value <= 10 else "MEDIUM"
            msg = (
                f"{emoji} *Macro Alert – Extreme Fear*\n\n"
                f"*Fear & Greed Index:* {value} ({classification})\n"
                f"*Severity:* {severity}\n\n"
                "_Extreme fear often precedes capitulation events or short-squeeze "
                "bounces.  Review open positions and tighten stop losses._"
            )
            await self._send_alert(msg)
            log.info("MacroWatchdog: Fear & Greed extreme fear alert (value={})", value)

        elif value >= _FG_EXTREME_HIGH:
            emoji = "🤑"
            severity = "HIGH" if value >= 90 else "MEDIUM"
            msg = (
                f"{emoji} *Macro Alert – Extreme Greed*\n\n"
                f"*Fear & Greed Index:* {value} ({classification})\n"
                f"*Severity:* {severity}\n\n"
                "_Extreme greed historically precedes sharp corrections.  "
                "Consider tightening trailing stops on long positions._"
            )
            await self._send_alert(msg)
            log.info("MacroWatchdog: Fear & Greed extreme greed alert (value={})", value)

    async def _check_news(self, coin: str, session: aiohttp.ClientSession) -> None:
        """Fetch recent headlines for *coin* and evaluate via OpenAI if available."""
        result = await fetch_news_sentiment(coin, session)
        headline = result.summary.strip()
        if not headline:
            return

        # De-duplicate: skip if we've sent this headline recently
        h_hash = hashlib.sha256(headline.encode()).hexdigest()
        last_seen = self._seen_headlines.get(h_hash, 0.0)
        now = time.monotonic()
        if now - last_seen < _DEDUP_TTL:
            return
        self._seen_headlines[h_hash] = now
        self._prune_seen_headlines()

        # Try to classify via OpenAI
        macro_result: Optional[MacroEventResult] = None
        if self._openai is not None and self._openai.enabled:
            try:
                macro_result = await asyncio.wait_for(
                    self._openai.evaluate_macro_event(headline, event_type="NEWS"),
                    timeout=8.0,
                )
            except Exception as exc:
                log.debug("OpenAI macro eval failed for '{}': {}", headline[:60], exc)

        # If OpenAI says it's not significant (or is unavailable and score is neutral), skip
        if macro_result is not None and not macro_result.is_significant:
            log.debug("MacroWatchdog: non-significant news skipped – {}", headline[:60])
            return

        # Determine severity (fall back to sentiment-based heuristic when no OpenAI)
        if macro_result is not None:
            severity = macro_result.severity
            impact = macro_result.impact
            ai_summary = macro_result.summary
        else:
            # Simple heuristic: very bearish or bullish sentiment → MEDIUM alert
            if abs(result.score) < 0.5:
                return
            severity = "MEDIUM"
            impact = "Significant sentiment movement detected – monitor closely."
            ai_summary = headline

        emoji = _SEVERITY_EMOJI.get(severity, "🔔")
        direction_label = "📈 Bullish" if result.score > 0 else "📉 Bearish"

        msg = (
            f"{emoji} *Macro Alert – {coin} News*\n\n"
            f"*Headline:* {ai_summary}\n"
            f"*Sentiment:* {direction_label} ({result.score:+.2f})\n"
            f"*Severity:* {severity}\n"
        )
        if impact:
            msg += f"*Expected Impact:* {impact}\n"
        msg += "\n_This is an AI-generated macro alert.  No trade signal has been generated._"

        await self._send_alert(msg)
        log.info(
            "MacroWatchdog: {} news alert ({}, score={:.2f})",
            coin, severity, result.score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _prune_seen_headlines(self) -> None:
        """Remove stale deduplicated entries to prevent memory growth."""
        now = time.monotonic()
        stale = [k for k, ts in self._seen_headlines.items() if now - ts >= _DEDUP_TTL]
        for k in stale:
            self._seen_headlines.pop(k, None)
