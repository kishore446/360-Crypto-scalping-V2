"""Tests for NarrativeBuilder in src/narrative.py."""

from __future__ import annotations

import pytest

from src.channels.base import Signal
from src.narrative import NarrativeBuilder
from src.smc import Direction
from src.utils import utcnow


def _make_signal(**kwargs) -> Signal:
    defaults = dict(
        channel="360_SPOT",
        symbol="INJUSDT",
        direction=Direction.LONG,
        entry=22.80,
        stop_loss=19.80,
        tp1=28.50,
        tp2=35.00,
        tp3=42.00,
        confidence=78.2,
        risk_label="MEDIUM",
        quality_tier="A+",
        setup_class="SWEEP_RECLAIM",
        liquidity_info="swept $20.5 weekly low",
        timestamp=utcnow(),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _full_context() -> dict:
    return {
        "regime": "RANGING",
        "indicators": {
            "ema9": 23.1,
            "ema21": 22.5,
            "ema200": 20.0,
            "rsi": 42.0,
            "adx": 18.5,
            "atr": 1.2,
        },
        "smc_events": ["swept $20.5 weekly low"],
        "volume_ratio": 2.3,
        "sector": "DeFi",
        "sector_7d_change": 2.1,
        "symbol_7d_change": -4.2,
        "drawdown_from_ath": 78.0,
        "accumulation_days": 18,
        "funding_rate": -0.012,
        "onchain_summary": "Net inflow of $12M in the last 24h",
    }


class TestNarrativeBuilderBasic:
    def test_returns_non_empty_string(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        result = builder.build_narrative(signal, _full_context())
        assert isinstance(result, str)
        assert len(result) > 20

    def test_contains_key_data_points(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        result = builder.build_narrative(signal, _full_context())
        # Should mention symbol name, RSI, or sector at minimum
        assert "INJUSDT" in result or "DeFi" in result or "RSI" in result

    def test_at_most_five_lines(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        result = builder.build_narrative(signal, _full_context())
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) <= 5


class TestNarrativeMissingFields:
    def test_missing_sector_still_works(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        ctx = _full_context()
        del ctx["sector"]
        del ctx["sector_7d_change"]
        del ctx["symbol_7d_change"]
        result = builder.build_narrative(signal, ctx)
        assert isinstance(result, str)

    def test_missing_onchain_still_works(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        ctx = _full_context()
        del ctx["onchain_summary"]
        result = builder.build_narrative(signal, ctx)
        assert isinstance(result, str)

    def test_empty_context_returns_string(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        result = builder.build_narrative(signal, {})
        assert isinstance(result, str)

    def test_minimal_context_no_crash(self):
        builder = NarrativeBuilder()
        signal = _make_signal()
        result = builder.build_narrative(signal, {"regime": "TRENDING_UP"})
        assert isinstance(result, str)


class TestNarrativeGemSpecific:
    def test_gem_channel_uses_drawdown_language(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_GEM")
        ctx = _full_context()
        ctx["drawdown_from_ath"] = 78.0
        ctx["accumulation_days"] = 18
        result = builder.build_narrative(signal, ctx)
        # GEM narrative should mention ATH or accumulation
        assert "ATH" in result or "accumulating" in result or "78" in result

    def test_gem_no_drawdown_data_falls_back(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_GEM")
        ctx = _full_context()
        ctx.pop("drawdown_from_ath", None)
        ctx.pop("accumulation_days", None)
        result = builder.build_narrative(signal, ctx)
        assert isinstance(result, str)


class TestNarrativeSpotSpecific:
    def test_spot_channel_uses_regime_language(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_SPOT")
        ctx = _full_context()
        ctx.pop("drawdown_from_ath", None)
        ctx.pop("accumulation_days", None)
        result = builder.build_narrative(signal, ctx)
        # Should mention regime or chart or setup
        assert "Ranging" in result or "chart" in result or "setup" in result or "regime" in result.lower()

    def test_spot_smc_events_mentioned(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_SPOT")
        ctx = _full_context()
        ctx["smc_events"] = ["swept $20.5 weekly low"]
        result = builder.build_narrative(signal, ctx)
        assert "swept" in result or "Smart money" in result


class TestNarrativeSectorContext:
    def test_lagging_sector_note(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_SPOT")
        ctx = {
            "sector": "DeFi",
            "sector_7d_change": 5.0,
            "symbol_7d_change": -3.0,  # lagging (below avg - 2)
        }
        result = builder.build_narrative(signal, ctx)
        assert "lagging" in result or "catch-up" in result

    def test_leading_sector_note(self):
        builder = NarrativeBuilder()
        signal = _make_signal(channel="360_SPOT")
        ctx = {
            "sector": "DeFi",
            "sector_7d_change": 2.0,
            "symbol_7d_change": 8.0,  # leading (above avg + 2)
        }
        result = builder.build_narrative(signal, ctx)
        assert "leading" in result or "momentum" in result


class TestNarrativeAIFallback:
    @pytest.mark.asyncio
    async def test_ai_mode_without_client_uses_template(self):
        builder = NarrativeBuilder(openai_client=None)
        signal = _make_signal()
        result = await builder.build_narrative_ai(signal, _full_context())
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_ai_mode_client_exception_falls_back(self):
        class _FailingClient:
            async def chat_completion(self, **kwargs):  # type: ignore[override]
                raise RuntimeError("API down")

        builder = NarrativeBuilder(openai_client=_FailingClient())
        signal = _make_signal()
        result = await builder.build_narrative_ai(signal, _full_context())
        # Should fall back to template without raising
        assert isinstance(result, str)
