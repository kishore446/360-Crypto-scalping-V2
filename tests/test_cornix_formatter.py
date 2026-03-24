"""Tests for src.cornix_formatter — Cornix-compatible signal formatting."""

from __future__ import annotations

from src.channels.base import Signal
from src.smc import Direction
from src.cornix_formatter import format_cornix_signal, _fmt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    channel: str = "360_SPOT",
    entry: float = 65000.0,
    stop_loss: float = 63500.0,
    tp1: float = 66500.0,
    tp2: float = 68000.0,
    tp3: float = 70000.0,
) -> Signal:
    sig = Signal(
        symbol=symbol,
        channel=channel,
        direction=Direction.LONG if direction == "LONG" else Direction.SHORT,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
    )
    return sig


# ---------------------------------------------------------------------------
# format_cornix_signal
# ---------------------------------------------------------------------------

class TestFormatCornixSignal:
    def test_returns_string(self):
        sig = _make_signal()
        result = format_cornix_signal(sig)
        assert isinstance(result, str)

    def test_contains_symbol_hashtag(self):
        sig = _make_signal(symbol="ETHUSDT")
        result = format_cornix_signal(sig)
        assert "#ETHUSDT" in result

    def test_contains_signal_type_long(self):
        sig = _make_signal(direction="LONG")
        result = format_cornix_signal(sig)
        assert "Long" in result

    def test_contains_signal_type_short(self):
        sig = _make_signal(direction="SHORT")
        result = format_cornix_signal(sig)
        assert "Short" in result

    def test_contains_entry_targets(self):
        sig = _make_signal(entry=65000.0)
        result = format_cornix_signal(sig)
        assert "Entry Targets:" in result
        assert "65000" in result

    def test_contains_tp_targets(self):
        sig = _make_signal(tp1=66500.0, tp2=68000.0, tp3=70000.0)
        result = format_cornix_signal(sig)
        assert "Take-Profit Targets:" in result
        assert "66500" in result
        assert "68000" in result
        assert "70000" in result

    def test_contains_stop_target(self):
        sig = _make_signal(stop_loss=63500.0)
        result = format_cornix_signal(sig)
        assert "Stop Targets:" in result
        assert "63500" in result

    def test_contains_leverage_spot(self):
        sig = _make_signal(channel="360_SPOT")
        result = format_cornix_signal(sig)
        assert "1x" in result

    def test_contains_leverage_scalp(self):
        sig = _make_signal(channel="360_SCALP")
        result = format_cornix_signal(sig)
        assert "20x" in result

    def test_contains_leverage_swing(self):
        sig = _make_signal(channel="360_SWING")
        result = format_cornix_signal(sig)
        assert "5x" in result

    def test_returns_empty_when_no_entry(self):
        sig = _make_signal(entry=0.0)
        result = format_cornix_signal(sig)
        assert result == ""

    def test_returns_empty_when_no_stop_loss(self):
        """format_cornix_signal returns empty string when stop_loss is falsy."""
        class NoStopSignal:
            symbol = "BTCUSDT"
            direction = type("D", (), {"value": "LONG"})()
            entry = 65000.0
            stop_loss = None
            tp1 = tp2 = tp3 = 0.0
            channel = "360_SPOT"
            entry_zone = None
        result = format_cornix_signal(NoStopSignal())  # type: ignore
        assert result == ""

    def test_dca_zone_long_two_entries(self):
        sig = _make_signal(direction="LONG")
        sig.entry_zone = "64800 - 65200"
        result = format_cornix_signal(sig)
        assert "Entry Targets:" in result
        assert "64800" in result or "65200" in result

    def test_signal_without_tp2_tp3(self):
        """Only tp1 present — should still produce valid output."""
        sig = _make_signal(tp2=0.0, tp3=None)
        result = format_cornix_signal(sig)
        assert "Take-Profit Targets:" in result
        assert str(int(sig.tp1)) in result

    def test_signal_without_any_tp(self):
        """No TPs — TP section omitted."""
        sig = _make_signal(tp1=0.0, tp2=0.0, tp3=None)
        result = format_cornix_signal(sig)
        # TP section should be absent
        assert "Take-Profit Targets:" not in result

    def test_format_does_not_raise_on_exception(self):
        """format_cornix_signal should never raise — returns empty on error."""
        # Pass a malformed object
        class BadSignal:
            symbol = "BTC"
            direction = None  # missing .value attribute
            entry = None
            stop_loss = None
            channel = "360_SPOT"
            tp1 = tp2 = tp3 = None
            entry_zone = None
        result = format_cornix_signal(BadSignal())  # type: ignore
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _fmt helper
# ---------------------------------------------------------------------------

class TestFmt:
    def test_integer_value(self):
        assert _fmt(65000.0) == "65000"

    def test_decimal_value(self):
        assert _fmt(0.00034567) == "0.00034567"

    def test_trailing_zeros_stripped(self):
        assert _fmt(65000.100) == "65000.1"

    def test_non_numeric_falls_back(self):
        result = _fmt("abc")
        assert isinstance(result, str)
