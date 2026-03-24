"""Cornix-compatible signal formatter for auto-execution via Telegram.

Cornix is the most popular Telegram trading bot for crypto signal auto-execution.
When signals are formatted in Cornix's expected syntax, subscribers can
auto-execute trades directly from the Telegram channel without any manual steps.

Format specification: https://docs.cornix.io/signal-format

Example output
--------------
📊 #BTCUSDT

Signal Type: Regular (Long)

Entry Targets:
1) 65000
2) 64800

Take-Profit Targets:
1) 66000
2) 67000
3) 68000

Stop Targets:
1) 63500

Leverage: Cross 10x
"""

from __future__ import annotations

from typing import Any

from src.channels.base import Signal
from src.utils import get_logger

log = get_logger("cornix_formatter")

# Default leverage by channel type (configurable via env in a future iteration)
_CHANNEL_LEVERAGE: dict[str, str] = {
    "360_SCALP":      "Cross 20x",
    "360_SCALP_FVG":  "Cross 15x",
    "360_SCALP_CVD":  "Cross 15x",
    "360_SCALP_VWAP": "Cross 15x",
    "360_SCALP_OBI":  "Cross 15x",
    "360_SWING":      "Cross 5x",
    "360_SPOT":       "Cross 1x",
    "360_GEM":        "Cross 1x",
}

# Signal type labels per direction
_DIRECTION_LABEL: dict[str, str] = {
    "LONG": "Long",
    "SHORT": "Short",
}


def format_cornix_signal(signal: Signal) -> str:
    """Format a :class:`~src.channels.base.Signal` into Cornix-compatible text.

    Parameters
    ----------
    signal:
        The signal to format.  Uses ``signal.entry``, ``signal.stop_loss``,
        ``signal.tp1``, ``signal.tp2``, ``signal.tp3``, ``signal.channel``,
        ``signal.direction``, and ``signal.symbol``.

    Returns
    -------
    str
        A formatted string ready to append to a Telegram message.  Returns
        an empty string if essential fields (entry, stop_loss) are missing.
    """
    try:
        return _build_cornix_block(signal)
    except Exception as exc:
        log.warning("Cornix formatter error for {}: {}", signal.symbol, exc)
        return ""


def _build_cornix_block(signal: Signal) -> str:
    """Build the Cornix-formatted signal block."""
    symbol = signal.symbol.upper().replace("/", "")
    direction = getattr(signal.direction, "value", str(signal.direction))
    direction_label = _DIRECTION_LABEL.get(direction, direction.capitalize())
    channel = signal.channel or ""
    leverage = _CHANNEL_LEVERAGE.get(channel, "Cross 5x")

    entry = signal.entry
    stop_loss = getattr(signal, "stop_loss", None)
    tp1 = getattr(signal, "tp1", None)
    tp2 = getattr(signal, "tp2", None)
    tp3 = getattr(signal, "tp3", None)
    entry_zone = getattr(signal, "entry_zone", None)

    if not entry or not stop_loss:
        return ""

    # Treat zero-value TPs as absent (dataclass defaults may use 0.0)
    tp1 = tp1 if tp1 else None
    tp2 = tp2 if tp2 else None
    tp3 = tp3 if tp3 else None

    lines: list[str] = []
    lines.append(f"📊 #{symbol}")
    lines.append("")
    lines.append(f"Signal Type: Regular ({direction_label})")
    lines.append("")

    # Entry targets — use DCA zone if available, otherwise single entry
    lines.append("Entry Targets:")
    if entry_zone and " - " in str(entry_zone):
        # entry_zone is formatted like "64500 - 64800"
        try:
            parts = str(entry_zone).split(" - ")
            entry_low = float(parts[0].strip())
            entry_high = float(parts[1].strip())
            # DCA: two entry targets (primary = closer to current price)
            if direction == "LONG":
                lines.append(f"1) {_fmt(entry_high)}")
                lines.append(f"2) {_fmt(entry_low)}")
            else:
                lines.append(f"1) {_fmt(entry_low)}")
                lines.append(f"2) {_fmt(entry_high)}")
        except (ValueError, IndexError):
            lines.append(f"1) {_fmt(entry)}")
    else:
        lines.append(f"1) {_fmt(entry)}")
    lines.append("")

    # Take-profit targets
    lines.append("Take-Profit Targets:")
    tp_idx = 1
    for tp in (tp1, tp2, tp3):
        if tp is not None:
            lines.append(f"{tp_idx}) {_fmt(tp)}")
            tp_idx += 1
    if tp_idx == 1:
        # No TPs provided — skip TP section
        lines.pop()  # remove "Take-Profit Targets:"
    lines.append("")

    # Stop target
    lines.append("Stop Targets:")
    lines.append(f"1) {_fmt(stop_loss)}")
    lines.append("")

    lines.append(f"Leverage: {leverage}")

    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a price value for Cornix — strip unnecessary trailing zeros."""
    try:
        f = float(value)
        # Use up to 8 significant digits but no more trailing zeros than needed
        formatted = f"{f:.8f}".rstrip("0").rstrip(".")
        return formatted
    except (TypeError, ValueError):
        return str(value)
