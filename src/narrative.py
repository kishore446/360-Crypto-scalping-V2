"""Narrative builder — generates human-readable trade analysis for SPOT/GEM signals.

Constructs a "WHY THIS TRADE" narrative from structured signal metadata:
indicators, SMC events, regime context, sector data, and on-chain metrics.

Two modes:
1. Template-based (default): deterministic, no API calls
2. AI-enhanced (optional): uses OpenAI GPT-4o-mini for natural prose (if OPENAI_API_KEY set)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.channels.base import Signal
from src.utils import get_logger

log = get_logger("narrative")


class NarrativeBuilder:
    """Builds human-readable trade narratives for SPOT/GEM signals."""

    def __init__(self, openai_client: Optional[Any] = None) -> None:
        self._openai = openai_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_narrative(self, signal: Signal, context: Dict[str, Any]) -> str:
        """Build the WHY THIS TRADE narrative (template-based, deterministic).

        Parameters
        ----------
        signal:
            The signal being narrated.
        context:
            Dict with optional keys:
            - regime: str  (e.g. "RANGING", "TRENDING_UP")
            - indicators: dict with ema9, ema21, ema200, rsi, adx, atr
            - smc_events: list[str]  (e.g. ["swept $20.5 weekly low"])
            - volume_ratio: float  (current vs 20-period avg)
            - sector: str  (e.g. "DeFi", "L1", "AI")
            - sector_7d_change: float  (sector 7d performance %)
            - symbol_7d_change: float  (this token's 7d %)
            - drawdown_from_ath: float  (% from ATH, for GEM)
            - accumulation_days: int  (for GEM)
            - funding_rate: Optional[float]
            - onchain_summary: Optional[str]

        Returns
        -------
        str
            A multi-line narrative (≤5 lines, ~60-80 chars each).
        """
        try:
            return self._build_template_narrative(signal, context)
        except Exception as exc:
            log.warning("Narrative template failed: {}", exc)
            return ""

    async def build_narrative_ai(self, signal: Signal, context: Dict[str, Any]) -> str:
        """AI-enhanced narrative using OpenAI. Falls back to template on failure."""
        if not self._openai:
            return self.build_narrative(signal, context)

        prompt = self._build_ai_prompt(signal, context)
        try:
            response = await self._openai.chat_completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional crypto analyst. Write a concise 3-5 sentence "
                            "trade thesis explaining why this setup is worth taking. Be specific "
                            "about price levels and technical reasons. No hype, no emojis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.strip()
        except Exception as exc:
            log.warning("AI narrative failed, using template: {}", exc)
            return self.build_narrative(signal, context)

    def build_education_narrative(self, signal: Signal, context: Dict[str, Any]) -> str:
        """Build an educational narrative explaining the signal reasoning.

        Outputs a 5-8 sentence explanation suitable for Telegram, covering:
        1. Setup classification and what it means
        2. Quality gates passed and failed
        3. Current market regime context
        4. Confidence breakdown (which factors scored highest/lowest)
        5. Risk management rationale

        Parameters
        ----------
        signal:
            The signal being explained.
        context:
            Dict with optional keys:
            - regime: str
            - setup_class: str
            - gates_passed: list[str]
            - gates_failed: list[tuple[str, float]] — (gate_name, penalty)
            - confidence_breakdown: dict[str, float]
            - indicators: dict
            - entry_zone: str (optional)

        Returns
        -------
        str
            Multi-line educational narrative with Telegram formatting.
        """
        try:
            return self._build_education_template(signal, context)
        except Exception as exc:
            log.warning("Education narrative failed: {}", exc)
            return ""

    def _build_education_template(self, signal: Signal, context: Dict[str, Any]) -> str:
        """Compose the structured education narrative from context data."""
        lines: list[str] = []
        symbol = signal.symbol
        direction = signal.direction.value

        # Header
        lines.append(f"📚 *LEARNING MODE — {symbol} {direction}*")
        lines.append("")

        # 1. Setup classification
        setup_class = context.get("setup_class") or getattr(signal, "setup_class", "") or "MOMENTUM"
        setup_label = setup_class.replace("_", " ").title()
        setup_explanations: Dict[str, str] = {
            "TREND PULLBACK": (
                "Price pulled back to EMA support in an uptrend. "
                "High-probability entry — trend intact but price temporarily discounted."
            ),
            "BREAKOUT RETEST": (
                "Price broke a key level and is retesting it as support. "
                "Volume confirmation validates the break."
            ),
            "RANGE FADE": (
                "Price is at the range boundary. "
                "Best in RANGING regime — fades to the opposite boundary."
            ),
            "MOMENTUM": (
                "Strong momentum detected with SMC confluence. "
                "Aligns with institutional order flow."
            ),
        }
        explanation = setup_explanations.get(setup_label, f"Setup class: {setup_label}.")
        lines.append(f"🔍 *Setup: {setup_label}* — {explanation}")

        # 2. Gates passed/failed
        gates_passed: list = context.get("gates_passed") or []
        gates_failed: list = context.get("gates_failed") or []

        if gates_passed:
            passed_str = ", ".join(str(g) for g in gates_passed[:5])
            lines.append(f"✅ *Gates Passed:* {passed_str}.")
        if gates_failed:
            failed_parts = []
            for item in gates_failed[:3]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    failed_parts.append(f"{item[0]} (−{item[1]:.0f} pts)")
                else:
                    failed_parts.append(str(item))
            lines.append(f"❌ *Gates Failed:* {', '.join(failed_parts)}.")

        # 3. Market regime
        regime = context.get("regime", "UNKNOWN")
        regime_label = regime.replace("_", " ").title()
        regime_notes: Dict[str, str] = {
            "Trending Up": "Strong uptrend. Favours trend-following LONG setups.",
            "Trending Down": "Strong downtrend. Favours trend-following SHORT setups.",
            "Ranging": "Sideways market. Best for mean-reversion / range-fade entries.",
            "Volatile": "High volatility. Wider stops required; signals need extra confluence.",
            "Quiet": "Low activity. Breakouts imminent but direction uncertain.",
        }
        regime_note = regime_notes.get(regime_label, "Regime context not classified.")
        lines.append(f"📊 *Regime: {regime_label}* — {regime_note}")

        # 4. Confidence breakdown
        breakdown: Dict[str, float] = context.get("confidence_breakdown") or {}
        total_conf = signal.confidence
        if breakdown:
            top_items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:4]
            breakdown_str = ", ".join(
                f"{k.replace('_', ' ').title()} ({v:.0f})" for k, v in top_items
            )
            lines.append(
                f"💯 *Confidence: {total_conf:.0f}/100* — {breakdown_str}."
            )
        else:
            lines.append(f"💯 *Confidence: {total_conf:.0f}/100*")

        # 5. Risk management
        entry = signal.entry
        sl = getattr(signal, "stop_loss", None)
        tp1 = getattr(signal, "tp1", None)
        risk_parts: list[str] = []
        if sl is not None and entry > 0:
            sl_pct = abs(entry - sl) / entry * 100.0
            risk_parts.append(f"SL at {sl:.4g} ({sl_pct:.1f}% risk)")
        if tp1 is not None and sl is not None and entry > 0:
            sl_dist = abs(entry - sl)
            tp1_dist = abs(tp1 - entry)
            rr = tp1_dist / sl_dist if sl_dist > 0 else 0.0
            risk_parts.append(f"TP1 at {tp1:.4g} ({rr:.1f}:1 RR)")
        if risk_parts:
            lines.append(f"⚠️ *Risk:* {' — '.join(risk_parts)}.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_template_narrative(self, signal: Signal, context: Dict[str, Any]) -> str:
        """Compose 3-5 sentences from structured data."""
        sentences: list[str] = []

        channel = signal.channel
        symbol = signal.symbol
        indicators: Dict[str, Any] = context.get("indicators") or {}

        # Sentence 1 — Setup description
        drawdown = context.get("drawdown_from_ath")
        accumulation_days = context.get("accumulation_days")
        regime = context.get("regime", "")
        setup_class = getattr(signal, "setup_class", "") or ""

        if channel == "360_GEM" and drawdown is not None and accumulation_days is not None:
            sentences.append(
                f"{symbol} is down {abs(drawdown):.0f}% from ATH and has been "
                f"accumulating for {accumulation_days} days in a tight range."
            )
        else:
            timeframe = "4h" if channel == "360_SPOT" else "daily"
            regime_label = regime.replace("_", " ").title() if regime else "neutral"
            setup_label = setup_class.replace("_", " ").title() if setup_class else "momentum"
            sentences.append(
                f"{symbol} showing {regime_label} regime with "
                f"{setup_label} setup on the {timeframe} chart."
            )

        # Sentence 2 — Technical confluence
        rsi = indicators.get("rsi")
        ema200 = indicators.get("ema200")
        smc_events: list[str] = context.get("smc_events") or []

        tech_parts: list[str] = []
        if rsi is not None:
            tech_parts.append(f"RSI at {rsi:.0f}")
        if ema200 is not None:
            side = "above" if signal.entry > ema200 else "below"
            tech_parts.append(f"price {side} EMA200 ({ema200:.2f})")
        if tech_parts:
            sentences.append(". ".join(tech_parts) + ".")
        if smc_events:
            event = smc_events[0]
            sentences.append(f"Smart money {event}.")

        # Sentence 3 — Volume / momentum
        volume_ratio = context.get("volume_ratio")
        funding_rate = context.get("funding_rate")

        if volume_ratio is not None:
            direction_word = "accumulation" if signal.direction.value == "LONG" else "distribution"
            sentences.append(
                f"Volume {volume_ratio:.1f}x above 20-period average, "
                f"confirming {direction_word}."
            )
        if funding_rate is not None:
            funding_desc = (
                "crowded longs" if funding_rate > 0 else "shorts overleveraged"
            )
            sentences.append(
                f"Funding rate at {funding_rate:.3f}% ({funding_desc})."
            )

        # Sentence 4 — Sector context
        sector = context.get("sector")
        sector_7d = context.get("sector_7d_change")
        symbol_7d = context.get("symbol_7d_change")

        if sector and sector_7d is not None and symbol_7d is not None:
            sector_dir = "up" if sector_7d >= 0 else "down"
            if symbol_7d > sector_7d + 2:
                rel_label = "leading"
                rel_note = "momentum confirmation"
            elif symbol_7d < sector_7d - 2:
                rel_label = "lagging"
                rel_note = "catch-up potential"
            else:
                rel_label = "in-line"
                rel_note = "sector aligned"
            sentences.append(
                f"{sector} sector {sector_dir} {abs(sector_7d):.1f}% (7d). "
                f"{symbol} {rel_label} at {symbol_7d:+.1f}% — {rel_note}."
            )

        # Sentence 5 — On-chain (if available)
        onchain_summary = context.get("onchain_summary")
        if onchain_summary:
            sentences.append(f"{onchain_summary}.")

        # Trim to 5 sentences maximum
        sentences = sentences[:5]

        if not sentences:
            return ""

        return "\n".join(sentences)

    def _build_ai_prompt(self, signal: Signal, context: Dict[str, Any]) -> str:
        """Build the prompt string sent to OpenAI."""
        parts = [
            f"Symbol: {signal.symbol}",
            f"Channel: {signal.channel}",
            f"Direction: {signal.direction.value}",
            f"Entry: {signal.entry}",
            f"SL: {signal.stop_loss}",
            f"TP1: {signal.tp1}",
            f"Confidence: {signal.confidence:.1f}",
        ]
        for key, val in context.items():
            parts.append(f"{key}: {val}")
        return "\n".join(parts)
