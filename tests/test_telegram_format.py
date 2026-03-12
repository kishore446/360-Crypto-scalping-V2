"""Tests for src.telegram_bot – signal formatting."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow


class TestFormatSignal:
    def test_scalp_long_format(self):
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32150,
            stop_loss=32120,
            tp1=32200,
            tp2=32300,
            tp3=32400,
            trailing_active=True,
            trailing_desc="1.5×ATR",
            confidence=87,
            ai_sentiment_label="Positive",
            ai_sentiment_summary="Whale Activity",
            risk_label="Aggressive",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "⚡" in text
        assert "360_SCALP" in text
        assert "BTCUSDT" in text
        assert "LONG" in text
        assert "32,150" in text
        assert "87%" in text
        assert "Whale Activity" in text
        assert "Aggressive" in text
        assert "Trailing Active" in text

    def test_swing_short_format(self):
        sig = Signal(
            channel="360_SWING",
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            entry=2350,
            stop_loss=2380,
            tp1=2320,
            tp2=2300,
            tp3=2270,
            trailing_active=True,
            trailing_desc="2.5×ATR",
            confidence=92,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="Moderate Volume Spike",
            risk_label="Medium",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "🏛️" in text
        assert "SHORT" in text
        assert "⬇️" in text
        assert "92%" in text

    def test_tape_format_with_ai_adaptive(self):
        sig = Signal(
            channel="360_THE_TAPE",
            symbol="ETHUSDT",
            direction=Direction.LONG,
            entry=2355,
            stop_loss=2340,
            tp1=2370,
            tp2=2390,
            tp3=None,
            trailing_active=True,
            trailing_desc="AI Adaptive",
            confidence=95,
            ai_sentiment_label="Bullish",
            ai_sentiment_summary="Whale Trade Confirmed",
            risk_label="Medium-High",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "🐋" in text
        assert "Dynamic/trailing" in text
        assert "AI Adaptive" in text
        assert "95%" in text

    def test_range_format(self):
        sig = Signal(
            channel="360_RANGE",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32100,
            stop_loss=32050,
            tp1=32150,
            tp2=32200,
            tp3=None,
            trailing_active=True,
            trailing_desc="1×ATR",
            confidence=80,
            ai_sentiment_label="Positive",
            ai_sentiment_summary="",
            risk_label="Conservative",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "⚖️" in text
        assert "Conservative" in text
        assert "80%" in text


class TestEscapeMd:
    def test_escape_backtick(self):
        assert TelegramBot._escape_md("price `0.175700`") == "price \\`0.175700\\`"

    def test_escape_asterisk(self):
        assert TelegramBot._escape_md("*bold*") == "\\*bold\\*"

    def test_escape_underscore(self):
        assert TelegramBot._escape_md("_italic_") == "\\_italic\\_"

    def test_escape_bracket(self):
        assert TelegramBot._escape_md("[link]") == "\\[link]"

    def test_escape_backslash(self):
        assert TelegramBot._escape_md("a\\b") == "a\\\\b"

    def test_escape_combined(self):
        result = TelegramBot._escape_md("Sweep SHORT at 0.3572 | FVG 0.3543-0.3538")
        # Pipe and digits should pass through unchanged; no special MD chars here
        assert result == "Sweep SHORT at 0.3572 | FVG 0.3543-0.3538"

    def test_escape_with_backtick_in_liquidity(self):
        raw = "Sweep `SHORT` at 0.3572 | FVG 0.3543-0.3538"
        escaped = TelegramBot._escape_md(raw)
        assert "\\`" in escaped
        assert "`" not in escaped.replace("\\`", "")

    def test_plain_text_unmodified(self):
        assert TelegramBot._escape_md("No special chars here") == "No special chars here"


class TestFormatSignalEscaping:
    def test_liquidity_info_with_pipe_and_fvg(self):
        """Liquidity info containing | and decimals should appear escaped in output."""
        sig = Signal(
            channel="360_SCALP",
            symbol="PIPPINUSDT",
            direction=Direction.SHORT,
            entry=0.35599,
            stop_loss=0.35642,
            tp1=0.35671,
            tp2=0.35592,
            tp3=0.35512,
            trailing_active=True,
            trailing_desc="1.5×ATR",
            confidence=78,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="No API key",
            risk_label="Low",
            market_phase="VOLATILE",
            liquidity_info="Sweep SHORT at 0.3572 | FVG 0.3543-0.3538",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        # The raw pipe and text should still appear (no MD special chars in this string)
        assert "Sweep SHORT at 0.3572 | FVG 0.3543-0.3538" in text

    def test_liquidity_info_with_backtick_escaped(self):
        """Backticks in liquidity_info must be escaped to prevent Markdown parse errors."""
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=69640,
            stop_loss=69508,
            tp1=69848,
            tp2=69880,
            tp3=69911,
            trailing_active=False,
            confidence=73,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="",
            risk_label="Low",
            market_phase="QUIET",
            liquidity_info="Sweep LONG at `69594` | FVG 69790-69786",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "\\`69594\\`" in text

    def test_ai_sentiment_summary_escaped(self):
        """AI sentiment summary with Markdown chars should be escaped."""
        sig = Signal(
            channel="360_SCALP",
            symbol="ETHUSDT",
            direction=Direction.LONG,
            entry=2042,
            stop_loss=2037,
            tp1=2049,
            tp2=2050,
            tp3=2051,
            trailing_active=False,
            confidence=73,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="Price *near* support_level",
            risk_label="Low",
            market_phase="QUIET",
            liquidity_info="Standard",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "\\*near\\*" in text
        assert "support\\_level" in text

    def test_trailing_desc_escaped(self):
        """trailing_desc with × should pass through; * would be escaped."""
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000,
            stop_loss=31900,
            tp1=32100,
            tp2=32200,
            trailing_active=True,
            trailing_desc="1.5*ATR",
            confidence=80,
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "1.5\\*ATR" in text


class TestSendMessageFallback:
    def test_plain_text_retry_on_markdown_parse_error(self):
        """send_message retries without parse_mode when Telegram returns 400 parse error."""
        bot = TelegramBot()
        bot._token = "fake-token"

        call_count = 0

        async def _run():
            nonlocal call_count

            # First response: 400 with parse entities error
            first_resp = MagicMock()
            first_resp.status = 400
            first_resp.text = AsyncMock(
                return_value='{"ok":false,"error_code":400,"description":"Bad Request: can\'t parse entities: Can\'t find end of the entity starting at byte offset 309"}'
            )
            first_resp.__aenter__ = AsyncMock(return_value=first_resp)
            first_resp.__aexit__ = AsyncMock(return_value=False)

            # Second response (retry): 200 OK
            second_resp = MagicMock()
            second_resp.status = 200
            second_resp.__aenter__ = AsyncMock(return_value=second_resp)
            second_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.post = MagicMock(side_effect=[first_resp, second_resp])

            bot._session = mock_session

            result = await bot.send_message("123456", "test *message* `broken")
            call_count = mock_session.post.call_count
            return result

        result = asyncio.new_event_loop().run_until_complete(_run())
        assert result is True
        assert call_count == 2  # initial attempt + retry

    def test_no_retry_on_other_400_errors(self):
        """send_message does NOT retry on 400 errors unrelated to Markdown parsing."""
        bot = TelegramBot()
        bot._token = "fake-token"

        call_count = 0

        async def _run():
            nonlocal call_count

            resp = MagicMock()
            resp.status = 400
            resp.text = AsyncMock(
                return_value='{"ok":false,"error_code":400,"description":"Bad Request: chat not found"}'
            )
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=resp)

            bot._session = mock_session

            result = await bot.send_message("123456", "hello")
            call_count = mock_session.post.call_count
            return result

        result = asyncio.new_event_loop().run_until_complete(_run())
        assert result is False
        assert call_count == 1  # no retry
