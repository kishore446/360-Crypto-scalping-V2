from __future__ import annotations

from types import SimpleNamespace

from src.signal_quality import (
    MarketState,
    QualityTier,
    SetupClass,
    assess_pair_quality,
    build_risk_plan,
    classify_market_state,
    classify_setup,
    execution_quality_check,
    passes_select_filter,
    score_signal_components,
)
from src.smc import Direction


def _candles(base: float = 100.0, trend: float = 1.0, n: int = 60) -> dict:
    close = [base + trend * i * 0.2 for i in range(n)]
    high = [c + 0.4 for c in close]
    low = [c - 0.4 for c in close]
    return {"high": high, "low": low, "close": close, "volume": [1000.0] * n}


def _signal(channel: str = "360_SCALP", direction: Direction = Direction.LONG):
    return SimpleNamespace(
        channel=channel,
        direction=direction,
        entry=100.0,
        stop_loss=97.0,
        tp1=104.0,
        tp2=108.0,
        tp3=112.0,
    )


def _indicators() -> dict:
    return {
        "1m": {"ema9_last": 100.8, "ema21_last": 100.2, "atr_last": 1.1, "momentum_last": 0.7},
        "5m": {
            "ema9_last": 101.0,
            "ema21_last": 100.4,
            "atr_last": 1.4,
            "momentum_last": 0.6,
            "bb_upper_last": 104.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 98.0,
        },
        "15m": {
            "ema9_last": 101.2,
            "ema21_last": 100.6,
            "atr_last": 1.8,
            "momentum_last": 0.5,
            "bb_upper_last": 105.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 97.0,
        },
        "1h": {
            "ema9_last": 102.0,
            "ema21_last": 101.0,
            "atr_last": 2.0,
            "momentum_last": 0.4,
            "bb_upper_last": 106.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 96.0,
        },
    }


def _smc(direction: Direction = Direction.LONG) -> dict:
    return {
        "sweeps": [SimpleNamespace(direction=direction, sweep_level=98.0)],
        "mss": SimpleNamespace(direction=direction, midpoint=99.2),
        "fvg": [],
        "whale_alert": {"usd": 1_500_000},
        "volume_delta_spike": True,
    }


class TestRegimeSetupCompatibility:
    def test_range_setup_rejected_in_strong_trend(self):
        signal = _signal(channel="360_RANGE")
        setup = classify_setup("360_RANGE", signal, _indicators(), _smc(), MarketState.STRONG_TREND)
        assert setup.channel_compatible is True
        assert setup.regime_compatible is False

    def test_continuation_rejected_in_dirty_range(self):
        signal = _signal(channel="360_SCALP")
        setup = classify_setup("360_SCALP", signal, _indicators(), {"sweeps": [], "mss": None, "fvg": []}, MarketState.DIRTY_RANGE)
        assert setup.setup_class == SetupClass.RANGE_REJECTION
        assert setup.channel_compatible is False

    def test_breakout_setup_allowed_in_breakout_expansion(self):
        signal = _signal(channel="360_SCALP")
        setup = classify_setup("360_SCALP", signal, _indicators(), _smc(), MarketState.BREAKOUT_EXPANSION)
        assert setup.setup_class in {SetupClass.BREAKOUT_RETEST, SetupClass.LIQUIDITY_SWEEP_REVERSAL}
        assert setup.channel_compatible is True
        assert setup.regime_compatible is True


class TestExecutionAndRiskChecks:
    def test_overextended_entry_is_rejected(self):
        signal = _signal(channel="360_THE_TAPE")
        signal.entry = 105.0
        indicators = _indicators()
        indicators["1m"]["ema9_last"] = 100.0
        indicators["1m"]["atr_last"] = 1.0
        result = execution_quality_check(signal, indicators, _smc(), SetupClass.MOMENTUM_EXPANSION, MarketState.BREAKOUT_EXPANSION)
        assert result.passed is False
        assert "overextended" in result.reason

    def test_reclaim_required_for_sweep_reversal(self):
        signal = _signal(channel="360_SCALP")
        signal.entry = 97.5
        result = execution_quality_check(signal, _indicators(), _smc(), SetupClass.LIQUIDITY_SWEEP_REVERSAL, MarketState.CLEAN_RANGE)
        assert result.passed is False
        assert "trigger" in result.reason

    def test_structure_first_risk_plan_updates_targets(self):
        signal = _signal(channel="360_RANGE")
        risk = build_risk_plan(signal, _indicators(), {"15m": _candles()}, _smc(), SetupClass.RANGE_REJECTION, 0.008)
        assert risk.passed is True
        assert risk.stop_loss < signal.entry
        assert risk.tp2 > risk.tp1
        assert "structure" in risk.invalidation_summary


class TestScoringAndSelectTier:
    def test_stronger_quality_scores_higher_than_weaker(self):
        pair_strong = assess_pair_quality(20_000_000.0, 0.008, _indicators()["5m"], _candles())
        pair_weak = assess_pair_quality(1_500_000.0, 0.025, {"atr_last": 6.0}, _candles())
        strong = score_signal_components(
            pair_quality=pair_strong,
            setup=SimpleNamespace(
                setup_class=SetupClass.BREAKOUT_RETEST,
                channel_compatible=True,
                regime_compatible=True,
            ),
            execution=SimpleNamespace(trigger_confirmed=True, extension_ratio=0.5),
            risk=SimpleNamespace(r_multiple=1.6),
            legacy_confidence=78.0,
            cross_verified=True,
        )
        weak = score_signal_components(
            pair_quality=pair_weak,
            setup=SimpleNamespace(
                setup_class=SetupClass.EXHAUSTION_FADE,
                channel_compatible=True,
                regime_compatible=False,
            ),
            execution=SimpleNamespace(trigger_confirmed=False, extension_ratio=1.6),
            risk=SimpleNamespace(r_multiple=0.9),
            legacy_confidence=52.0,
            cross_verified=False,
        )
        assert strong.total > weak.total
        assert strong.quality_tier in {QualityTier.A, QualityTier.A_PLUS}
        assert weak.quality_tier in {QualityTier.B, QualityTier.C}

    def test_select_filter_only_allows_top_tier(self):
        allowed, _ = passes_select_filter(
            setup_class=SetupClass.BREAKOUT_RETEST.value,
            market_state=MarketState.STRONG_TREND.value,
            pair_quality_score=88.0,
            quality_tier="A",
            confidence=88.0,
            r_multiple=1.5,
            component_scores={"market": 20.0, "execution": 16.0},
            higher_timeframe_aligned=True,
        )
        blocked, reason = passes_select_filter(
            setup_class=SetupClass.BREAKOUT_RETEST.value,
            market_state=MarketState.STRONG_TREND.value,
            pair_quality_score=74.0,
            quality_tier="B",
            confidence=79.0,
            r_multiple=1.2,
            component_scores={"market": 17.0, "execution": 13.0},
            higher_timeframe_aligned=True,
        )
        assert allowed is True
        assert blocked is False
        assert "threshold" in reason


class TestMarketStateClassification:
    def test_dirty_and_clean_range_distinguished(self):
        clean = classify_market_state(
            regime_result=SimpleNamespace(regime="RANGING", bb_width_pct=2.0),
            indicators={"adx_last": 14.0, "atr_last": 0.9, "momentum_last": 0.05},
            candles=_candles(base=100.0, trend=0.0),
            spread_pct=0.008,
        )
        dirty = classify_market_state(
            regime_result=SimpleNamespace(regime="RANGING", bb_width_pct=4.0),
            indicators={"adx_last": 19.0, "atr_last": 1.8, "momentum_last": 0.15},
            candles={
                "high": [101.0, 102.5, 101.2, 102.8, 101.4, 102.9],
                "low": [99.0, 98.0, 99.1, 97.8, 99.2, 97.7],
                "close": [100.0, 100.1, 100.0, 100.2, 100.1, 100.0],
            },
            spread_pct=0.012,
        )
        assert clean == MarketState.CLEAN_RANGE
        assert dirty == MarketState.DIRTY_RANGE
