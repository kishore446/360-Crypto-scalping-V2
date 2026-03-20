"""Tests for PaperPortfolioManager (src/paper_portfolio.py)."""

from __future__ import annotations

from pathlib import Path

from src.paper_portfolio import PaperPortfolioManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> PaperPortfolioManager:
    return PaperPortfolioManager(storage_path=str(tmp_path / "portfolios.json"))


def _record_win(mgr: PaperPortfolioManager, channel: str = "360_SCALP") -> None:
    mgr.record_trade(
        channel=channel,
        signal_id="sig-win",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=30000.0,
        exit_price=30450.0,
        hit_tp=3,
        hit_sl=False,
        pnl_pct=1.5,
    )


def _record_loss(mgr: PaperPortfolioManager, channel: str = "360_SCALP") -> None:
    mgr.record_trade(
        channel=channel,
        signal_id="sig-loss",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=30000.0,
        exit_price=29700.0,
        hit_tp=0,
        hit_sl=True,
        pnl_pct=-1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaperPortfolioManager:
    def test_ensure_user_creates_4_channels(self, tmp_path):
        """New user gets portfolios for all 4 channels with $1000 each."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        portfolios = mgr._portfolios["user1"]
        assert set(portfolios.keys()) == set(PaperPortfolioManager.CHANNELS)
        for ch, p in portfolios.items():
            assert p.current_balance == 1000.0
            assert p.initial_balance == 1000.0

    def test_ensure_user_idempotent(self, tmp_path):
        """Calling ensure_user twice doesn't reset portfolios."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        # Record a trade to mutate state
        _record_win(mgr)
        balance_before = mgr._portfolios["user1"]["360_SCALP"].current_balance
        mgr.ensure_user("user1")
        balance_after = mgr._portfolios["user1"]["360_SCALP"].current_balance
        assert balance_before == balance_after

    def test_record_winning_trade(self, tmp_path):
        """A winning trade increases balance and increments win_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial = mgr._portfolios["user1"]["360_SCALP"].current_balance

        _record_win(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance > initial
        assert p.win_count == 1
        assert p.loss_count == 0

    def test_record_losing_trade(self, tmp_path):
        """A losing trade decreases balance and increments loss_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial = mgr._portfolios["user1"]["360_SCALP"].current_balance

        _record_loss(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance < initial
        assert p.loss_count == 1
        assert p.win_count == 0

    def test_record_breakeven_trade(self, tmp_path):
        """A breakeven trade (~0 PnL) increments breakeven_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-be",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,  # exactly 0 → BREAKEVEN
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.breakeven_count == 1
        assert p.win_count == 0
        assert p.loss_count == 0

    def test_fees_deducted_correctly(self, tmp_path):
        """Fees = position_size × 0.001 × 2 (entry + exit)."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-fee",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        # risk_amount = 1000 * 2% = 20; position = 20 * 1 = 20; fee = 20 * 0.001 * 2 = 0.04
        expected_fee = 1000.0 * 0.02 * 1 * 0.001 * 2
        assert abs(p.total_fees - expected_fee) < 1e-9

    def test_leverage_amplifies_pnl(self, tmp_path):
        """Setting leverage to 5x should amplify PnL by 5x."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        # Default leverage (1x) — record a win
        _record_win(mgr)
        balance_1x = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # Reset and set 5x leverage
        mgr.reset_portfolio("user1", "360_SCALP")
        mgr.set_leverage("user1", "360_SCALP", 5)
        _record_win(mgr)
        balance_5x = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # With 5x leverage, net PnL change from initial should be ~5x bigger
        delta_1x = balance_1x - 1000.0
        delta_5x = balance_5x - 1000.0
        # Not exactly 5x due to fees scaling too, but should be close
        assert delta_5x > delta_1x * 4

    def test_balance_cannot_go_below_zero(self, tmp_path):
        """A huge loss should cap balance at 0, not go negative."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-huge-loss",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=0.0,
            hit_tp=0,
            hit_sl=True,
            pnl_pct=-10000.0,  # Catastrophic loss
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.current_balance >= 0.0

    def test_reset_portfolio_single_channel(self, tmp_path):
        """Reset one channel back to $1000, others unchanged."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr, channel="360_SCALP")
        _record_win(mgr, channel="360_SWING")

        swing_before = mgr._portfolios["user1"]["360_SWING"].current_balance
        mgr.reset_portfolio("user1", "360_SCALP")

        assert mgr._portfolios["user1"]["360_SCALP"].current_balance == 1000.0
        assert mgr._portfolios["user1"]["360_SWING"].current_balance == swing_before

    def test_reset_portfolio_all_channels(self, tmp_path):
        """Reset all channels back to $1000."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        for ch in PaperPortfolioManager.CHANNELS:
            _record_win(mgr, channel=ch)

        mgr.reset_portfolio("user1")

        for ch in PaperPortfolioManager.CHANNELS:
            assert mgr._portfolios["user1"][ch].current_balance == 1000.0

    def test_reset_increments_reset_count(self, tmp_path):
        """Each reset increments the reset_count."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.reset_portfolio("user1", "360_SCALP")
        assert mgr._portfolios["user1"]["360_SCALP"].reset_count == 1
        mgr.reset_portfolio("user1", "360_SCALP")
        assert mgr._portfolios["user1"]["360_SCALP"].reset_count == 2

    def test_set_leverage_valid(self, tmp_path):
        """Setting leverage within 1-20 works."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_leverage("user1", "360_SCALP", 10)
        assert "✅" in result
        assert mgr._portfolios["user1"]["360_SCALP"].leverage == 10

    def test_set_leverage_invalid(self, tmp_path):
        """Leverage outside 1-20 is rejected."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result_low = mgr.set_leverage("user1", "360_SCALP", 0)
        result_high = mgr.set_leverage("user1", "360_SCALP", 21)
        assert "❌" in result_low
        assert "❌" in result_high
        assert mgr._portfolios["user1"]["360_SCALP"].leverage == 1  # unchanged

    def test_set_risk_valid(self, tmp_path):
        """Setting risk within 0.5-10% works."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_risk("user1", "360_SCALP", 5.0)
        assert "✅" in result
        assert mgr._portfolios["user1"]["360_SCALP"].risk_per_trade_pct == 5.0

    def test_set_risk_invalid(self, tmp_path):
        """Risk outside 0.5-10% is rejected."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result_low = mgr.set_risk("user1", "360_SCALP", 0.1)
        result_high = mgr.set_risk("user1", "360_SCALP", 11.0)
        assert "❌" in result_low
        assert "❌" in result_high
        assert mgr._portfolios["user1"]["360_SCALP"].risk_per_trade_pct == 2.0  # unchanged

    def test_max_drawdown_tracked(self, tmp_path):
        """Max drawdown is updated when balance drops from peak."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        # Win first to raise peak, then lose
        _record_win(mgr)
        _record_loss(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.max_drawdown_pct > 0.0

    def test_peak_balance_tracked(self, tmp_path):
        """Peak balance updates when balance reaches new high."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial_peak = mgr._portfolios["user1"]["360_SCALP"].peak_balance

        _record_win(mgr)

        p = mgr._portfolios["user1"]["360_SCALP"]
        assert p.peak_balance > initial_peak

    def test_persistence_save_and_load(self, tmp_path):
        """Portfolios persist to JSON and reload correctly."""
        path = str(tmp_path / "portfolios.json")
        mgr = PaperPortfolioManager(storage_path=path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        saved_balance = mgr._portfolios["user1"]["360_SCALP"].current_balance

        # Reload from disk
        mgr2 = PaperPortfolioManager(storage_path=path)
        assert "user1" in mgr2._portfolios
        assert abs(mgr2._portfolios["user1"]["360_SCALP"].current_balance - saved_balance) < 1e-9
        assert mgr2._portfolios["user1"]["360_SCALP"].win_count == 1

    def test_get_portfolio_summary_format(self, tmp_path):
        """Summary message contains key fields: balance, PnL, fees, leverage."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)

        summary = mgr.get_portfolio_summary("user1")
        assert "Paper Trading Portfolio" in summary
        assert "Balance" in summary
        assert "Leverage" in summary
        assert "Total" in summary
        assert "Fees" in summary

    def test_get_channel_detail_format(self, tmp_path):
        """Channel detail contains trades, win rate, drawdown."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)

        detail = mgr.get_channel_detail("user1", "360_SCALP")
        assert "Portfolio Detail" in detail
        assert "Win Rate" in detail
        assert "Max Drawdown" in detail
        assert "Last 5 Trades" in detail

    def test_get_trade_history_format(self, tmp_path):
        """Trade history shows recent trades with PnL and fees."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr)
        _record_loss(mgr)

        history = mgr.get_trade_history("user1")
        assert "Trade History" in history
        assert "BTCUSDT" in history
        assert "PnL" in history
        assert "Fee" in history

    def test_get_trade_history_empty(self, tmp_path):
        """Trade history returns helpful message when no trades exist."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        history = mgr.get_trade_history("user1")
        assert "No trades yet" in history

    def test_record_trade_updates_all_users(self, tmp_path):
        """When a signal completes, all users' portfolios are updated."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.ensure_user("user2")

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-multi",
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=2000.0,
            exit_price=1960.0,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=2.0,
        )

        assert mgr._portfolios["user1"]["360_SCALP"].win_count == 1
        assert mgr._portfolios["user2"]["360_SCALP"].win_count == 1

    def test_record_trade_only_updates_matching_channel(self, tmp_path):
        """A SCALP signal only affects the SCALP portfolio, not SWING."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        initial_swing = mgr._portfolios["user1"]["360_SWING"].current_balance

        _record_win(mgr, channel="360_SCALP")

        swing_balance = mgr._portfolios["user1"]["360_SWING"].current_balance
        assert swing_balance == initial_swing  # SWING should be untouched

    def test_custom_risk_affects_position_size(self, tmp_path):
        """Setting risk to 5% should use 5% of balance per trade."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        mgr.set_risk("user1", "360_SCALP", 5.0)

        mgr.record_trade(
            channel="360_SCALP",
            signal_id="sig-risk",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30000.0,
            hit_tp=0,
            hit_sl=False,
            pnl_pct=0.0,  # breakeven so we only see fee impact
        )

        p = mgr._portfolios["user1"]["360_SCALP"]
        # position_size = 1000 * 5% * 1 = 50; fee = 50 * 0.001 * 2 = 0.10
        expected_fee = 1000.0 * 0.05 * 1 * 0.001 * 2
        assert abs(p.total_fees - expected_fee) < 1e-9

    def test_select_channel_ignored(self, tmp_path):
        """Trades for the SELECT channel are silently ignored."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")

        mgr.record_trade(
            channel="360_SELECT",
            signal_id="sig-select",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=30000.0,
            exit_price=30450.0,
            hit_tp=3,
            hit_sl=False,
            pnl_pct=1.5,
        )

        # No portfolio mutation should have happened since SELECT is not tracked
        for ch in PaperPortfolioManager.CHANNELS:
            assert mgr._portfolios["user1"][ch].win_count == 0

    def test_get_channel_detail_unknown_channel(self, tmp_path):
        """Requesting detail for an unknown channel returns an error string."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.get_channel_detail("user1", "360_UNKNOWN")
        assert "❌" in result

    def test_reset_portfolio_unknown_channel(self, tmp_path):
        """Resetting an unknown channel returns an error string."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.reset_portfolio("user1", "360_UNKNOWN")
        assert "❌" in result

    def test_set_leverage_unknown_channel(self, tmp_path):
        """Setting leverage on unknown channel returns error."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_leverage("user1", "360_UNKNOWN", 5)
        assert "❌" in result

    def test_set_risk_unknown_channel(self, tmp_path):
        """Setting risk on unknown channel returns error."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        result = mgr.set_risk("user1", "360_UNKNOWN", 3.0)
        assert "❌" in result

    def test_trade_history_single_channel(self, tmp_path):
        """Trade history filtered to a single channel only shows that channel."""
        mgr = _make_manager(tmp_path)
        mgr.ensure_user("user1")
        _record_win(mgr, channel="360_SCALP")
        _record_win(mgr, channel="360_SWING")

        history = mgr.get_trade_history("user1", channel="360_SCALP")
        assert "Trade History" in history
        assert "360_SCALP" in history
