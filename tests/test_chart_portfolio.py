"""Tests for generate_portfolio_chart() in src/chart_generator.py."""

from __future__ import annotations

import pytest

from src.chart_generator import generate_portfolio_chart


def _make_data(n: int = 60, base: float = 22.0) -> dict:
    """Generate synthetic OHLCV data with n candles."""
    import random
    random.seed(42)
    closes = [base + random.uniform(-2, 2) for _ in range(n)]
    highs = [c + random.uniform(0, 1) for c in closes]
    lows = [c - random.uniform(0, 1) for c in closes]
    volumes = [random.uniform(1_000_000, 5_000_000) for _ in range(n)]
    return {"closes": closes, "highs": highs, "lows": lows, "volumes": volumes}


class TestGeneratePortfolioChartValid:
    def test_returns_bytes_for_valid_input(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50, 35.00, 42.00],
            channel_name="GEM",
        )
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_png_magic_bytes(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50, 35.00, 42.00],
            channel_name="GEM",
        )
        assert result is not None
        # PNG files start with \x89PNG
        assert result[:4] == b"\x89PNG"

    def test_spot_channel(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="ETHUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50, 35.00],
            channel_name="SPOT",
        )
        assert result is not None
        assert isinstance(result, bytes)

    def test_single_tp_level(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="BTCUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50],
            channel_name="SPOT",
        )
        assert result is not None

    def test_two_tp_levels(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="BTCUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50, 35.00],
            channel_name="SPOT",
        )
        assert result is not None

    def test_three_tp_levels(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="BTCUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50, 35.00, 42.00],
            channel_name="GEM",
        )
        assert result is not None

    def test_custom_ema_periods(self):
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50],
            channel_name="GEM",
            ema_periods=(9, 21),
        )
        assert result is not None

    def test_exactly_ten_candles(self):
        data = _make_data(10)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50],
            channel_name="GEM",
        )
        assert result is not None


class TestGeneratePortfolioChartInsufficient:
    def test_returns_none_for_fewer_than_10_candles(self):
        data = _make_data(5)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50],
            channel_name="GEM",
        )
        assert result is None

    def test_returns_none_for_empty_input(self):
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=[],
            highs=[],
            lows=[],
            volumes=[],
            entry=22.80,
            sl=19.80,
            tp_levels=[28.50],
            channel_name="GEM",
        )
        assert result is None

    def test_returns_none_for_no_tp_levels(self):
        """With an empty tp_levels list the chart should still generate (no TP lines)."""
        data = _make_data(60)
        result = generate_portfolio_chart(
            symbol="INJUSDT",
            closes=data["closes"],
            highs=data["highs"],
            lows=data["lows"],
            volumes=data["volumes"],
            entry=22.80,
            sl=19.80,
            tp_levels=[],
            channel_name="SPOT",
        )
        # Chart is valid even with no TP levels — only entry/SL lines are drawn.
        assert result is not None
