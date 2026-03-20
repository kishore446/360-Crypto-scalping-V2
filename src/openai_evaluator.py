"""OpenAI GPT-4 macro-event evaluator — repurposed for market-wide alerts.

The OpenAI integration is no longer used for individual trade-signal scoring
(which is now 100 % quantitative / zero latency).  Instead, this module
evaluates crypto macro-events fetched by the MacroWatchdog (news, FOMC
decisions, interest-rate changes, major geopolitical events, new token
listings) and decides whether they are significant enough to push a
high-priority alert to the Telegram admin channel.

Degrades gracefully: if ``OPENAI_API_KEY`` is not set every call returns a
neutral result immediately so the MacroWatchdog still runs (it just won't
AI-classify events).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import aiohttp

from src.utils import get_logger

log = get_logger("openai_evaluator")

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_CACHE_TTL: float = 120.0  # seconds
_TIMEOUT: float = 5.0       # HTTP timeout
_MAX_ADJUSTMENT: float = 15.0
_CACHE_MAX_ITEMS: int = 256


@dataclass
class EvalResult:
    """Result of an OpenAI trade evaluation (kept for backward compatibility)."""
    adjustment: float = 0.0   # -15 to +15 confidence adjustment
    recommended: bool = True  # False = AI says skip this trade
    reasoning: str = ""
    model: str = ""


@dataclass
class MacroEventResult:
    """Result of an OpenAI macro-event classification."""
    is_significant: bool = False   # True → push alert to Telegram
    severity: str = "LOW"          # LOW / MEDIUM / HIGH / CRITICAL
    summary: str = ""              # One-line human-readable summary
    impact: str = ""               # Expected market impact description
    model: str = ""


class OpenAIEvaluator:
    """Async wrapper around the OpenAI Chat Completions API.

    Sends a structured prompt describing the current signal and returns a
    :class:`EvalResult` with a confidence adjustment and trade recommendation.

    All calls are cached per evaluation fingerprint for :data:`_CACHE_TTL`
    seconds to avoid spamming the API on every scan cycle.
    """

    def __init__(self) -> None:
        self._api_key: str = os.getenv("OPENAI_API_KEY", "")
        self._model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._enabled: bool = bool(self._api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Union[EvalResult, MacroEventResult]]] = {}

    @property
    def enabled(self) -> bool:
        """Return ``True`` when the API key is configured."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> EvalResult:
        """Evaluate a trade signal via GPT-4o-mini.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTCUSDT"``.
        direction:
            ``"LONG"`` or ``"SHORT"``.
        channel:
            Channel name, e.g. ``"360_SCALP"``.
        entry_price, stop_loss, tp1, tp2:
            Signal price levels.
        indicators:
            Dict of computed indicators (``ema9_last``, ``ema21_last``,
            ``adx_last``, ``rsi_last``, ``atr_last``, …).
        smc_summary:
            Human-readable SMC event description.
        ai_sentiment_summary:
            Combined news/social/fear-greed summary string.
        market_phase:
            Market regime label, e.g. ``"TRENDING_UP"``.
        confidence_before:
            Confidence score (0–100) before this evaluation.

        Returns
        -------
        EvalResult
        """
        if not self._enabled:
            return EvalResult(
                adjustment=0.0,
                reasoning="OpenAI not configured",
                recommended=True,
            )

        self._prune_cache()
        cache_key = self._build_cache_key(
            symbol=symbol,
            direction=direction,
            channel=channel,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            indicators=indicators,
            smc_summary=smc_summary,
            ai_sentiment_summary=ai_sentiment_summary,
            market_phase=market_phase,
            confidence_before=confidence_before,
        )
        cached = self._cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL:
            cached_val = cached[1]
            if isinstance(cached_val, EvalResult):
                return cached_val

        prompt = self._build_prompt(
            symbol=symbol,
            direction=direction,
            channel=channel,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            indicators=indicators,
            smc_summary=smc_summary,
            ai_sentiment_summary=ai_sentiment_summary,
            market_phase=market_phase,
            confidence_before=confidence_before,
        )

        try:
            result = await self._call_api(prompt)
        except Exception as exc:
            log.debug("OpenAI evaluation failed for {}: {}", symbol, exc)
            return EvalResult(adjustment=0.0, reasoning="OpenAI error", recommended=True)

        self._cache[cache_key] = (time.monotonic(), result)
        return result

    async def evaluate_macro_event(
        self,
        headline: str,
        event_type: str = "NEWS",
    ) -> MacroEventResult:
        """Classify a macro event headline and decide if it warrants a Telegram alert.

        Parameters
        ----------
        headline:
            Short description of the event (e.g. a news headline or calendar entry).
        event_type:
            Category hint: ``"NEWS"``, ``"FOMC"``, ``"LISTING"``, ``"WAR"``,
            ``"INTEREST_RATE"``, or ``"OTHER"``.

        Returns
        -------
        MacroEventResult
            ``is_significant=True`` when the event is material enough for an alert.
        """
        if not self._enabled:
            return MacroEventResult(
                is_significant=False,
                summary=headline[:120],
                model="",
            )

        cache_key = f"macro:{hashlib.sha1(headline.encode()).hexdigest()}"
        cached_entry = self._cache.get(cache_key)
        if cached_entry is not None and (time.monotonic() - cached_entry[0]) < _CACHE_TTL:
            cached_val = cached_entry[1]
            if isinstance(cached_val, MacroEventResult):
                return cached_val

        prompt = (
            "You are a crypto macro analyst monitoring global events that could move markets.\n\n"
            f"Event type: {event_type}\n"
            f"Headline: {headline}\n\n"
            "Classify this event for a crypto trading system. Respond ONLY with valid JSON:\n"
            '{"is_significant": <true or false>, '
            '"severity": "<LOW|MEDIUM|HIGH|CRITICAL>", '
            '"summary": "<one sentence summary>", '
            '"impact": "<brief expected market impact>"}'
        )

        macro_result: MacroEventResult
        try:
            raw = await self._call_api_raw(prompt)
            parsed = self._parse_response_content(raw)
            macro_result = MacroEventResult(
                is_significant=bool(parsed.get("is_significant", False)),
                severity=str(parsed.get("severity", "LOW")).upper(),
                summary=str(parsed.get("summary", headline[:120])),
                impact=str(parsed.get("impact", "")),
                model=self._model,
            )
        except Exception as exc:
            log.debug("OpenAI macro evaluation failed: {}", exc)
            macro_result = MacroEventResult(
                is_significant=False,
                summary=headline[:120],
                model=self._model,
            )

        self._cache[cache_key] = (time.monotonic(), macro_result)
        return macro_result

    async def close(self) -> None:
        """Close the underlying :class:`aiohttp.ClientSession` if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_cache(self) -> None:
        now = time.monotonic()
        stale_keys = [
            key for key, (ts, _) in self._cache.items()
            if (now - ts) >= _CACHE_TTL
        ]
        for key in stale_keys:
            self._cache.pop(key, None)
        if len(self._cache) <= _CACHE_MAX_ITEMS:
            return
        for key, _ in sorted(self._cache.items(), key=lambda item: item[1][0])[: len(self._cache) - _CACHE_MAX_ITEMS]:
            self._cache.pop(key, None)

    def _build_cache_key(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> str:
        payload = {
            "symbol": symbol,
            "direction": direction,
            "channel": channel,
            "entry_price": round(entry_price, 8),
            "stop_loss": round(stop_loss, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "market_phase": market_phase,
            "confidence_before": round(confidence_before, 4),
            "smc_summary": smc_summary.strip(),
            "ai_sentiment_summary": ai_sentiment_summary.strip(),
            "indicators": {
                key: indicators.get(key)
                for key in sorted(indicators)
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return f"{symbol}:{channel}:{digest}"

    def _parse_response_content(self, content: str) -> Dict[str, Any]:
        raw = content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            parsed = json.loads(raw[start: end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response must be a JSON object")
        return parsed

    @staticmethod
    def _coerce_recommended(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "skip"}
        return bool(value)

    def _build_prompt(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> str:
        ema9 = indicators.get("ema9_last", "N/A")
        ema21 = indicators.get("ema21_last", "N/A")
        adx = indicators.get("adx_last", "N/A")
        rsi = indicators.get("rsi_last", "N/A")
        atr = indicators.get("atr_last", "N/A")

        def _fmt(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        return (
            "You are an expert crypto trading analyst. Evaluate this signal and provide your assessment.\n\n"
            "Signal Details:\n"
            f"- Pair: {symbol} {direction}\n"
            f"- Channel: {channel}\n"
            f"- Entry: {_fmt(entry_price)}\n"
            f"- Stop Loss: {_fmt(stop_loss)}\n"
            f"- TP1: {_fmt(tp1)}, TP2: {_fmt(tp2)}\n"
            f"- Current Confidence: {confidence_before:.1f}%\n\n"
            "Technical Indicators:\n"
            f"- EMA9: {_fmt(ema9)}, EMA21: {_fmt(ema21)}\n"
            f"- ADX: {_fmt(adx)}, RSI: {_fmt(rsi)}\n"
            f"- ATR: {_fmt(atr)}\n\n"
            f"Smart Money Concepts: {smc_summary}\n"
            f"AI Sentiment: {ai_sentiment_summary}\n"
            f"Market Phase: {market_phase}\n\n"
            'Respond ONLY with valid JSON (no markdown, no code fences):\n'
            '{"confidence_adjustment": <number -15 to 15>, "recommended": <true or false>, "reasoning": "<short explanation>"}'
        )

    async def _call_api(self, prompt: str) -> EvalResult:
        """POST to the OpenAI chat completions endpoint and parse the result."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a crypto trading analyst. Respond only with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 150,
        }

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
        async with self._session.post(
            _OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout
        ) as resp:
            if resp.status != 200:
                log.warning("OpenAI API returned status {}", resp.status)
                return EvalResult(
                    adjustment=0.0, reasoning="OpenAI API error", recommended=True
                )
            data = await resp.json(content_type=None)

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_response_content(str(content))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.debug("Failed to parse OpenAI response: {}", exc)
            return EvalResult(
                adjustment=0.0,
                reasoning="Invalid OpenAI response",
                recommended=True,
                model=self._model,
            )

        try:
            raw_adj = float(parsed.get("confidence_adjustment", 0.0))
        except (TypeError, ValueError):
            raw_adj = 0.0
        adjustment = max(-_MAX_ADJUSTMENT, min(_MAX_ADJUSTMENT, raw_adj))
        recommended = self._coerce_recommended(parsed.get("recommended", True))
        reasoning = str(parsed.get("reasoning", ""))

        return EvalResult(
            adjustment=adjustment,
            recommended=recommended,
            reasoning=reasoning,
            model=self._model,
        )

    async def _call_api_raw(self, prompt: str) -> str:
        """POST to the OpenAI chat completions endpoint and return the raw text."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a crypto macro analyst. Respond only with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
        async with self._session.post(
            _OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout
        ) as resp:
            if resp.status != 200:
                log.warning("OpenAI API returned status {} for macro evaluation", resp.status)
                raise RuntimeError(f"OpenAI API error: {resp.status}")
            data = await resp.json(content_type=None)

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenAI response structure: {exc}") from exc
