"""Paper Trading Portfolio Simulator.

Manages per-user, per-channel virtual portfolios that shadow real signals.
Portfolios are updated silently whenever a signal completes (TP or SL hit).
All output is returned as formatted strings — this module never sends any
Telegram messages directly.

Storage: ``data/paper_portfolios.json`` (created automatically on first write).
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import get_logger

log = get_logger("paper_portfolio")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PaperTrade:
    """A single completed paper trade."""

    signal_id: str
    channel: str
    symbol: str
    direction: str          # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    leverage: int
    position_size_usdt: float
    fee_paid: float         # entry + exit fees
    pnl_usdt: float         # net PnL in USDT (after fees)
    pnl_pct: float          # percentage PnL from the signal
    status: str             # "WIN" / "LOSS" / "BREAKEVEN"
    timestamp: float


@dataclass
class ChannelPortfolio:
    """Per-channel portfolio for a single user."""

    channel: str
    initial_balance: float = 1000.0
    current_balance: float = 1000.0
    leverage: int = 1
    risk_per_trade_pct: float = 2.0  # Risk 2% of balance per trade
    trades: List[PaperTrade] = field(default_factory=list)
    total_fees: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    reset_count: int = 0
    peak_balance: float = 1000.0
    max_drawdown_pct: float = 0.0


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _portfolio_to_dict(portfolio: ChannelPortfolio) -> dict:
    """Serialize a ChannelPortfolio to a plain dict for JSON storage."""
    d = dataclasses.asdict(portfolio)
    return d


def _portfolio_from_dict(d: dict) -> ChannelPortfolio:
    """Deserialize a ChannelPortfolio from a plain dict."""
    trades_raw = d.pop("trades", [])
    trades = [PaperTrade(**t) for t in trades_raw]
    portfolio = ChannelPortfolio(**d)
    portfolio.trades = trades
    return portfolio


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PaperPortfolioManager:
    """Manages virtual paper trading portfolios for all users.

    Thread-safety: this class is designed to be used from a single asyncio
    event loop.  ``_save`` is wrapped in a try/except so persistence errors
    never propagate to callers.
    """

    INITIAL_BALANCE: float = 1000.0
    DEFAULT_LEVERAGE: int = 1
    DEFAULT_RISK_PCT: float = 2.0
    FEE_RATE: float = 0.001  # 0.1% per side (Binance taker)

    CHANNELS = ["360_SCALP", "360_SWING", "360_RANGE", "360_THE_TAPE"]

    def __init__(self, storage_path: str = "data/paper_portfolios.json") -> None:
        self._path = Path(storage_path)
        # chat_id (str) → channel (str) → ChannelPortfolio
        self._portfolios: Dict[str, Dict[str, ChannelPortfolio]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record_trade(
        self,
        channel: str,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        hit_tp: int,
        hit_sl: bool,
        pnl_pct: float,
    ) -> None:
        """Record a completed signal outcome across ALL user portfolios.

        Called by ``TradeMonitor._record_outcome()`` for every completed signal.
        Updates every registered user's portfolio for the given channel silently.
        Only channels in :attr:`CHANNELS` are tracked (SELECT is excluded).
        """
        if channel not in self.CHANNELS:
            return

        for chat_id in list(self._portfolios.keys()):
            portfolio = self._portfolios[chat_id].get(channel)
            if portfolio is None:
                continue

            # Position sizing
            risk_amount = portfolio.current_balance * (portfolio.risk_per_trade_pct / 100.0)
            position_size = risk_amount * portfolio.leverage

            # Raw PnL based on signal's actual pnl_pct
            trade_pnl_raw = position_size * (pnl_pct / 100.0)

            # Fees: entry + exit (both sides)
            fee = position_size * self.FEE_RATE * 2

            # Net PnL after fees
            net_pnl = trade_pnl_raw - fee

            # Determine outcome status
            if abs(pnl_pct) < 0.05:
                status = "BREAKEVEN"
                portfolio.breakeven_count += 1
            elif pnl_pct > 0:
                status = "WIN"
                portfolio.win_count += 1
            else:
                status = "LOSS"
                portfolio.loss_count += 1

            # Update balance (floor at 0)
            portfolio.current_balance += net_pnl
            portfolio.current_balance = max(portfolio.current_balance, 0.0)
            portfolio.total_pnl += net_pnl
            portfolio.total_fees += fee

            # Track peak and max drawdown
            if portfolio.current_balance > portfolio.peak_balance:
                portfolio.peak_balance = portfolio.current_balance
            if portfolio.peak_balance > 0:
                dd = (
                    (portfolio.peak_balance - portfolio.current_balance)
                    / portfolio.peak_balance
                    * 100
                )
                portfolio.max_drawdown_pct = max(portfolio.max_drawdown_pct, dd)

            # Record the individual trade
            trade = PaperTrade(
                signal_id=signal_id,
                channel=channel,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                leverage=portfolio.leverage,
                position_size_usdt=position_size,
                fee_paid=fee,
                pnl_usdt=net_pnl,
                pnl_pct=pnl_pct,
                status=status,
                timestamp=time.time(),
            )
            portfolio.trades.append(trade)

        self._save()

    def ensure_user(self, chat_id: str) -> None:
        """Initialize portfolios for a new user if they don't already exist."""
        if chat_id not in self._portfolios:
            self._portfolios[chat_id] = {}
            for channel in self.CHANNELS:
                self._portfolios[chat_id][channel] = ChannelPortfolio(
                    channel=channel,
                    initial_balance=self.INITIAL_BALANCE,
                    current_balance=self.INITIAL_BALANCE,
                    peak_balance=self.INITIAL_BALANCE,
                )
            self._save()

    # ------------------------------------------------------------------
    # Query / formatting
    # ------------------------------------------------------------------

    def get_portfolio_summary(self, chat_id: str) -> str:
        """Return a formatted Telegram message with all channel balances."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_RANGE": "⚖️",
            "360_THE_TAPE": "🐋",
        }

        lines = ["💼 *Paper Trading Portfolio*\n"]
        total_balance = 0.0
        total_pnl = 0.0
        total_fees = 0.0
        total_resets = 0

        for channel in self.CHANNELS:
            p = portfolios.get(channel)
            if p is None:
                continue
            emoji = chan_emojis.get(channel, "📡")
            pnl_pct_display = (
                (p.current_balance - p.initial_balance) / p.initial_balance * 100
                if p.initial_balance > 0
                else 0.0
            )
            win_rate = (
                p.win_count / (p.win_count + p.loss_count) * 100
                if (p.win_count + p.loss_count) > 0
                else 0.0
            )

            lines.append(f"{emoji} *{channel.replace('360_', '')}*")
            lines.append(f"   Balance: ${p.current_balance:,.2f} ({pnl_pct_display:+.1f}%)")
            lines.append(f"   Leverage: {p.leverage}×  |  Risk: {p.risk_per_trade_pct:.0f}%")
            lines.append(
                f"   W/L/BE: {p.win_count}/{p.loss_count}/{p.breakeven_count}"
                f"  |  WR: {win_rate:.0f}%"
            )
            lines.append(f"   Fees: ${p.total_fees:,.2f}  |  DD: {p.max_drawdown_pct:.1f}%")
            lines.append("")

            total_balance += p.current_balance
            total_pnl += p.total_pnl
            total_fees += p.total_fees
            total_resets += p.reset_count

        initial_total = self.INITIAL_BALANCE * len(self.CHANNELS)
        total_pnl_pct = (
            (total_balance - initial_total) / initial_total * 100
            if initial_total > 0
            else 0.0
        )

        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append(f"📊 Total: ${total_balance:,.2f} ({total_pnl_pct:+.1f}%)")
        lines.append(f"💰 Net PnL: ${total_pnl:+,.2f}")
        lines.append(f"💸 Total Fees: ${total_fees:,.2f}")
        lines.append(f"🔄 Resets: {total_resets}")

        return "\n".join(lines)

    def get_channel_detail(self, chat_id: str, channel: str) -> str:
        """Return detailed view of a single channel portfolio."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios.get(chat_id, {})
        p = portfolios.get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_RANGE": "⚖️",
            "360_THE_TAPE": "🐋",
        }
        emoji = chan_emojis.get(channel, "📡")
        pnl_pct = (
            (p.current_balance - p.initial_balance) / p.initial_balance * 100
            if p.initial_balance > 0
            else 0.0
        )
        total_trades = p.win_count + p.loss_count + p.breakeven_count
        win_rate = (
            p.win_count / (p.win_count + p.loss_count) * 100
            if (p.win_count + p.loss_count) > 0
            else 0.0
        )

        lines = [
            f"{emoji} *{channel} Portfolio Detail*\n",
            f"💰 Balance: ${p.current_balance:,.2f} ({pnl_pct:+.1f}%)",
            f"📊 Initial: ${p.initial_balance:,.2f}",
            f"⚙️ Leverage: {p.leverage}×  |  Risk: {p.risk_per_trade_pct:.0f}%",
            f"📈 Trades: {total_trades} (W:{p.win_count} L:{p.loss_count} BE:{p.breakeven_count})",
            f"🏆 Win Rate: {win_rate:.1f}%",
            f"💸 Total Fees: ${p.total_fees:,.2f}",
            f"📉 Max Drawdown: {p.max_drawdown_pct:.1f}%",
            f"🏔️ Peak Balance: ${p.peak_balance:,.2f}",
            f"🔄 Resets: {p.reset_count}",
        ]

        if p.trades:
            lines.append("\n📋 *Last 5 Trades:*")
            for t in p.trades[-5:]:
                status_emoji = (
                    "✅" if t.status == "WIN" else ("❌" if t.status == "LOSS" else "➖")
                )
                lines.append(
                    f"  {status_emoji} {t.symbol} {t.direction} | "
                    f"PnL: ${t.pnl_usdt:+.2f} ({t.pnl_pct:+.2f}%) | Fee: ${t.fee_paid:.2f}"
                )

        return "\n".join(lines)

    def get_trade_history(
        self,
        chat_id: str,
        channel: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Return formatted trade history for a user."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        all_trades: List[PaperTrade] = []
        if channel:
            p = portfolios.get(channel)
            if p is None:
                return f"❌ Channel `{channel}` not found."
            all_trades = list(p.trades)
        else:
            for ch in self.CHANNELS:
                p = portfolios.get(ch)
                if p:
                    all_trades.extend(p.trades)

        if not all_trades:
            return "📋 No trades yet. Trades are recorded when signals complete."

        # Most recent first
        all_trades.sort(key=lambda t: t.timestamp, reverse=True)
        recent = all_trades[:limit]

        label = channel or "All Channels"
        lines = [f"📋 *Trade History — {label}*\n"]
        for t in recent:
            status_emoji = (
                "✅" if t.status == "WIN" else ("❌" if t.status == "LOSS" else "➖")
            )
            chan_short = t.channel.replace("360_", "")
            lines.append(
                f"{status_emoji} *{t.symbol}* {t.direction} ({chan_short})\n"
                f"   PnL: ${t.pnl_usdt:+.2f} ({t.pnl_pct:+.2f}%) | "
                f"Lev: {t.leverage}× | Fee: ${t.fee_paid:.2f}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mutation commands
    # ------------------------------------------------------------------

    def reset_portfolio(self, chat_id: str, channel: Optional[str] = None) -> str:
        """Reset portfolio to $1,000. Returns confirmation message."""
        self.ensure_user(chat_id)
        portfolios = self._portfolios[chat_id]

        if channel:
            p = portfolios.get(channel)
            if p is None:
                return f"❌ Channel `{channel}` not found."
            p.current_balance = self.INITIAL_BALANCE
            p.total_pnl = 0.0
            p.total_fees = 0.0
            p.win_count = 0
            p.loss_count = 0
            p.breakeven_count = 0
            p.peak_balance = self.INITIAL_BALANCE
            p.max_drawdown_pct = 0.0
            p.trades.clear()
            p.reset_count += 1
            self._save()
            return (
                f"🔄 Portfolio reset for *{channel}*. "
                f"Balance: $1,000.00 (Reset #{p.reset_count})"
            )
        else:
            for ch in self.CHANNELS:
                p = portfolios.get(ch)
                if p:
                    p.current_balance = self.INITIAL_BALANCE
                    p.total_pnl = 0.0
                    p.total_fees = 0.0
                    p.win_count = 0
                    p.loss_count = 0
                    p.breakeven_count = 0
                    p.peak_balance = self.INITIAL_BALANCE
                    p.max_drawdown_pct = 0.0
                    p.trades.clear()
                    p.reset_count += 1
            self._save()
            return "🔄 All portfolios reset to $1,000.00 each."

    def set_leverage(self, chat_id: str, channel: str, leverage: int) -> str:
        """Set leverage for a channel. Returns confirmation message."""
        self.ensure_user(chat_id)
        if leverage < 1 or leverage > 20:
            return "❌ Leverage must be between 1 and 20."
        p = self._portfolios[chat_id].get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."
        p.leverage = leverage
        self._save()
        return f"✅ Leverage for *{channel}* set to *{leverage}×*"

    def set_risk(self, chat_id: str, channel: str, risk_pct: float) -> str:
        """Set risk percentage per trade. Returns confirmation message."""
        self.ensure_user(chat_id)
        if risk_pct < 0.5 or risk_pct > 10.0:
            return "❌ Risk must be between 0.5% and 10%."
        p = self._portfolios[chat_id].get(channel)
        if p is None:
            return f"❌ Channel `{channel}` not found."
        p.risk_per_trade_pct = risk_pct
        self._save()
        return f"✅ Risk for *{channel}* set to *{risk_pct:.1f}%* per trade"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist all portfolios to disk (JSON). Errors are logged, not raised."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            for chat_id, channels in self._portfolios.items():
                data[chat_id] = {
                    ch: _portfolio_to_dict(p) for ch, p in channels.items()
                }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to save paper portfolios: %s", exc)

    def _load(self) -> None:
        """Load portfolios from disk. Missing file is treated as empty state."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for chat_id, channels in raw.items():
                self._portfolios[chat_id] = {
                    ch: _portfolio_from_dict(p_dict)
                    for ch, p_dict in channels.items()
                }
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load paper portfolios: %s", exc)
