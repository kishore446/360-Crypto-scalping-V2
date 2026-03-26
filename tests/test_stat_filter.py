"""Unit tests for the PR_12 statistical false-positive filter.

Tests cover:
* Fail-open behaviour when no history exists
* Hard suppress when win rate < 25%
* Soft penalty when win rate is in the 25–45% range
* Rolling window bounds (deque stays at window size)
* win_rate returns None when below min_samples
* stats() returns correct structure
* format_statstats() returns non-empty string
"""

from __future__ import annotations

import pytest

from src.stat_filter import (
    RollingWinRateStore,
    SignalOutcome,
    StatisticalFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcome(
    signal_id: str = "sig1",
    channel: str = "360_SCALP",
    pair: str = "BTCUSDT",
    regime: str = "TRENDING_UP",
    won: bool = True,
    pnl_pct: float = 1.0,
) -> SignalOutcome:
    return SignalOutcome(
        signal_id=signal_id,
        channel=channel,
        pair=pair,
        regime=regime,
        setup_class="LIQUIDITY_SWEEP_REVERSAL",
        won=won,
        pnl_pct=pnl_pct,
    )


def _fill_store(
    store: RollingWinRateStore,
    channel: str,
    pair: str,
    regime: str,
    wins: int,
    losses: int,
) -> None:
    """Helper: record a specified number of wins and losses."""
    for i in range(wins):
        store.record(
            SignalOutcome(
                signal_id=f"win{i}",
                channel=channel,
                pair=pair,
                regime=regime,
                setup_class="",
                won=True,
                pnl_pct=1.5,
            )
        )
    for i in range(losses):
        store.record(
            SignalOutcome(
                signal_id=f"loss{i}",
                channel=channel,
                pair=pair,
                regime=regime,
                setup_class="",
                won=False,
                pnl_pct=-1.0,
            )
        )


# ===========================================================================
# 1. Fail-open behaviour
# ===========================================================================


def test_stat_filter_allows_when_no_history():
    """Filter must pass signals through when no history exists (fail-open)."""
    sf = StatisticalFilter()
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "TRENDING_UP", 75.0)
    assert allow is True
    assert conf == 75.0
    assert "no_history" in reason


def test_stat_filter_allows_when_below_min_samples():
    """Filter must pass when fewer than min_samples outcomes are recorded."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # Record 14 losses — below min_samples threshold
    _fill_store(store, "360_SCALP", "BTCUSDT", "RANGING", wins=0, losses=14)
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "RANGING", 80.0)
    assert allow is True
    assert conf == 80.0
    assert "no_history" in reason


# ===========================================================================
# 2. Hard suppress
# ===========================================================================


def test_stat_filter_suppresses_below_threshold():
    """Hard suppress must fire when win rate is below 25%."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # 3 wins / 23 total ≈ 13% WR — well below the 25% suppress threshold
    _fill_store(store, "360_SCALP", "BTCUSDT", "RANGING", wins=3, losses=20)
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "RANGING", 80.0)
    assert allow is False
    assert conf == 0.0
    assert "hard_suppress" in reason


def test_stat_filter_suppress_reason_contains_win_rate():
    """Suppress reason string must contain the observed win rate."""
    store = RollingWinRateStore(window=30, min_samples=15)
    _fill_store(store, "SWING", "ETHUSDT", "VOLATILE", wins=2, losses=18)
    sf = StatisticalFilter(store)
    _allow, _conf, reason = sf.check("SWING", "ETHUSDT", "VOLATILE", 70.0)
    assert "wr=" in reason


# ===========================================================================
# 3. Soft penalty
# ===========================================================================


def test_stat_filter_soft_penalty_at_40pct_win_rate():
    """Soft penalty must fire and deduct 5 pts when WR is in the 25–45% range."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # 12 wins + 18 losses = 40% WR (below the 45% soft-penalty threshold)
    _fill_store(store, "SWING", "ETHUSDT", "VOLATILE", wins=12, losses=18)
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("SWING", "ETHUSDT", "VOLATILE", 70.0)
    assert allow is True
    assert conf == pytest.approx(65.0)  # –5 pt soft penalty
    assert "soft_penalty" in reason


def test_stat_filter_soft_penalty_clamps_at_zero():
    """Soft penalty must never produce a negative confidence value."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # 35% WR — inside soft-penalty zone; start with very low confidence
    _fill_store(store, "360_SCALP", "SOLUSDT", "RANGING", wins=7, losses=13)
    sf = StatisticalFilter(store)
    _allow, conf, _reason = sf.check("360_SCALP", "SOLUSDT", "RANGING", 2.0)
    assert conf >= 0.0


# ===========================================================================
# 4. Pass-through (good WR)
# ===========================================================================


def test_stat_filter_passes_good_win_rate():
    """Signals from combinations with WR ≥ 45% must pass unchanged."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # 15 wins / 20 total = 75% WR
    _fill_store(store, "360_SCALP", "BTCUSDT", "TRENDING_UP", wins=15, losses=5)
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "TRENDING_UP", 82.0)
    assert allow is True
    assert conf == pytest.approx(82.0)
    assert "ok" in reason


# ===========================================================================
# 5. RollingWinRateStore — window bounds
# ===========================================================================


def test_rolling_win_rate_store_bounds():
    """Deque must not exceed window size regardless of how many outcomes are recorded."""
    store = RollingWinRateStore(window=30, min_samples=15)
    # Record 40 outcomes — only the last 30 should be retained
    for i in range(40):
        won = i % 2 == 0
        store.record(
            SignalOutcome(
                signal_id=f"s{i}",
                channel="CH",
                pair="SYM",
                regime="REGIME",
                setup_class="",
                won=won,
                pnl_pct=0.0,
            )
        )
    # Verify win_rate is available (≥ min_samples records present)
    wr = store.win_rate("CH", "SYM", "REGIME")
    assert wr is not None
    assert 0.0 <= wr <= 1.0

    # Verify internal deque is capped at window=30
    key = ("CH", "SYM", "REGIME")
    with store._lock:
        assert len(store._records[key]) == 30


def test_rolling_win_rate_store_returns_none_below_min_samples():
    """win_rate() must return None when fewer than min_samples outcomes exist."""
    store = RollingWinRateStore(window=30, min_samples=15)
    _fill_store(store, "SPOT", "ADAUSDT", "QUIET", wins=5, losses=5)
    assert store.win_rate("SPOT", "ADAUSDT", "QUIET") is None


def test_rolling_win_rate_store_returns_float_at_min_samples():
    """win_rate() must return a float once exactly min_samples outcomes exist."""
    store = RollingWinRateStore(window=30, min_samples=15)
    _fill_store(store, "SPOT", "BNBUSDT", "RANGING", wins=8, losses=7)
    wr = store.win_rate("SPOT", "BNBUSDT", "RANGING")
    assert wr is not None
    assert isinstance(wr, float)
    assert wr == pytest.approx(8 / 15)


# ===========================================================================
# 6. stats() method
# ===========================================================================


def test_rolling_win_rate_store_stats_structure():
    """stats() must return the expected dict keys even for an unknown key."""
    store = RollingWinRateStore()
    result = store.stats("UNKNOWN", "XYZUSDT", "VOLATILE")
    assert set(result.keys()) == {"win_rate", "n", "avg_pnl", "last_updated"}
    assert result["n"] == 0


def test_rolling_win_rate_store_stats_values():
    """stats() must return accurate win_rate, n, and avg_pnl."""
    store = RollingWinRateStore(window=30, min_samples=5)
    _fill_store(store, "GEM", "LTCUSDT", "TRENDING_DOWN", wins=6, losses=4)
    s = store.stats("GEM", "LTCUSDT", "TRENDING_DOWN")
    assert s["n"] == 10
    assert s["win_rate"] == pytest.approx(0.6)
    # avg_pnl: 6 wins at 1.5% + 4 losses at -1.0% = (9 - 4) / 10 = 0.5
    assert s["avg_pnl"] == pytest.approx(0.5)
    assert s["last_updated"] is not None


# ===========================================================================
# 7. StatisticalFilter.record() delegation
# ===========================================================================


def test_stat_filter_record_delegates_to_store():
    """StatisticalFilter.record() must forward outcomes to the underlying store."""
    store = RollingWinRateStore(window=30, min_samples=5)
    sf = StatisticalFilter(store)
    outcome = _make_outcome(channel="360_SCALP", pair="DOTUSDT", regime="QUIET", won=True)
    sf.record(outcome)
    # The store should now have one entry
    s = store.stats("360_SCALP", "DOTUSDT", "QUIET")
    assert s["n"] == 1
    assert s["win_rate"] == pytest.approx(1.0)


# ===========================================================================
# 8. format_statstats()
# ===========================================================================


def test_format_statstats_empty():
    """format_statstats() must return a non-empty message when no data exists."""
    sf = StatisticalFilter()
    msg = sf.format_statstats()
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_format_statstats_with_data():
    """format_statstats() must include the channel/pair/regime in the output."""
    store = RollingWinRateStore(window=30, min_samples=5)
    _fill_store(store, "360_SCALP", "BTCUSDT", "TRENDING_UP", wins=10, losses=5)
    sf = StatisticalFilter(store)
    msg = sf.format_statstats()
    assert "BTCUSDT" in msg
    assert "360_SCALP" in msg
