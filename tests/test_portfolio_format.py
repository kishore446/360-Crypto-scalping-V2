"""Tests for format_portfolio_signal() in src/telegram_bot.py."""

from __future__ import annotations

from src.channels.base import Signal
from src.sector import SectorContext
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow


def _make_signal(**kwargs) -> Signal:
    defaults = dict(
        channel="360_GEM",
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
        liquidity_info="BTC sweep at 22.5",
        timestamp=utcnow(),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _make_sector_context(**kwargs) -> SectorContext:
    defaults = dict(
        sector_name="DeFi",
        sector_7d_pct=2.1,
        symbol_7d_pct=-4.2,
        peers=[("UNIUSDT", 3.1), ("AAVEUSDT", 1.8), ("RUNEUSDT", 4.5)],
        correlated_major=("ETHUSDT", 5.2),
        relative_strength="lagging",
    )
    defaults.update(kwargs)
    return SectorContext(**defaults)


class TestFormatPortfolioSignalSections:
    def test_entry_present(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "Entry" in text

    def test_sl_present(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "SL" in text

    def test_tp1_present(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "TP1" in text

    def test_tp2_present(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "TP2" in text

    def test_tp3_present_when_set(self):
        sig = _make_signal(tp3=42.00)
        text = TelegramBot.format_portfolio_signal(sig)
        assert "TP3" in text

    def test_tp3_absent_when_none(self):
        sig = _make_signal(tp3=None)
        text = TelegramBot.format_portfolio_signal(sig)
        assert "TP3" not in text

    def test_key_levels_section(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "KEY LEVELS" in text

    def test_confidence_shown(self):
        sig = _make_signal(confidence=78.2)
        text = TelegramBot.format_portfolio_signal(sig)
        assert "78.2" in text

    def test_hold_time_shown(self):
        sig = _make_signal(channel="360_GEM")
        text = TelegramBot.format_portfolio_signal(sig)
        assert "2-4w" in text

    def test_rr_ratio_shown(self):
        # TP1 dist = 28.50 - 22.80 = 5.70; SL dist = 22.80 - 19.80 = 3.0; R:R ≈ 1:1.9
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig)
        assert "R:R" in text


class TestFormatPortfolioSignalNarrative:
    def test_narrative_section_shown_when_present(self):
        sig = _make_signal()
        narrative = "INJ is down 78% from ATH and accumulating.\nRSI at 42."
        text = TelegramBot.format_portfolio_signal(sig, narrative=narrative)
        assert "WHY THIS TRADE" in text
        assert "78%" in text

    def test_narrative_section_omitted_when_empty(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig, narrative="")
        assert "WHY THIS TRADE" not in text


class TestFormatPortfolioSignalSector:
    def test_sector_section_shown_when_present(self):
        sig = _make_signal()
        ctx = _make_sector_context()
        text = TelegramBot.format_portfolio_signal(sig, sector_context=ctx)
        assert "SECTOR COMPARISON" in text
        assert "DeFi" in text

    def test_sector_section_omitted_when_none(self):
        sig = _make_signal()
        text = TelegramBot.format_portfolio_signal(sig, sector_context=None)
        assert "SECTOR COMPARISON" not in text

    def test_peers_shown_in_sector_section(self):
        sig = _make_signal()
        ctx = _make_sector_context()
        text = TelegramBot.format_portfolio_signal(sig, sector_context=ctx)
        assert "UNIUSDT" in text

    def test_correlated_major_shown(self):
        sig = _make_signal()
        ctx = _make_sector_context(correlated_major=("ETHUSDT", 5.2))
        text = TelegramBot.format_portfolio_signal(sig, sector_context=ctx)
        assert "ETHUSDT" in text

    def test_relative_strength_label(self):
        sig = _make_signal()
        ctx = _make_sector_context(relative_strength="lagging")
        text = TelegramBot.format_portfolio_signal(sig, sector_context=ctx)
        assert "lagging" in text or "catch-up" in text


class TestFormatPortfolioSignalChannelEmojis:
    def test_gem_emoji(self):
        sig = _make_signal(channel="360_GEM")
        text = TelegramBot.format_portfolio_signal(sig)
        assert "💎" in text
        assert "GEM" in text

    def test_spot_emoji(self):
        sig = _make_signal(channel="360_SPOT", stop_loss=19.80, tp1=28.50, tp2=35.00)
        text = TelegramBot.format_portfolio_signal(sig)
        assert "📈" in text
        assert "SPOT" in text


class TestFormatPortfolioSignalPercentages:
    def test_sl_percentage_is_negative_for_long(self):
        sig = _make_signal(direction=Direction.LONG, entry=22.80, stop_loss=19.80)
        text = TelegramBot.format_portfolio_signal(sig)
        # SL below entry → negative %
        assert "-" in text

    def test_tp_percentage_is_positive_for_long(self):
        sig = _make_signal(direction=Direction.LONG, entry=22.80, tp1=28.50)
        text = TelegramBot.format_portfolio_signal(sig)
        # TP1 above entry → positive %
        assert "+" in text


class TestFormatPortfolioSignalRiskLabel:
    def test_risk_label_in_footer(self):
        sig = _make_signal(risk_label="MEDIUM")
        text = TelegramBot.format_portfolio_signal(sig)
        assert "MEDIUM" in text

    def test_override_risk_label(self):
        sig = _make_signal(risk_label="LOW")
        text = TelegramBot.format_portfolio_signal(sig, risk_label="HIGH")
        assert "HIGH" in text
