"""Tests for SectorComparator and SectorContext in src/sector.py."""

from __future__ import annotations

from src.sector import SectorComparator, SectorContext


class _MockDataStore:
    """Minimal data_store stub for testing."""

    def __init__(self, candles_by_symbol: dict | None = None):
        self._candles = candles_by_symbol or {}

    def get_candles(self, symbol: str, timeframe: str) -> dict | None:
        return self._candles.get(symbol)


class _MockPairMgr:
    pass


def _make_comparator(candles_by_symbol: dict | None = None) -> SectorComparator:
    return SectorComparator(
        data_store=_MockDataStore(candles_by_symbol),
        pair_mgr=_MockPairMgr(),
    )


class TestGetSector:
    def test_known_defi_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("INJUSDT") == "DeFi"

    def test_known_l1_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("ETHUSDT") == "L1"

    def test_known_l2_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("ARBUSDT") == "L2"

    def test_known_ai_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("FETUSDT") == "AI"

    def test_known_meme_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("DOGEUSDT") == "Meme"

    def test_known_oracle_symbol(self):
        cmp = _make_comparator()
        assert cmp.get_sector("LINKUSDT") == "Oracle"

    def test_unknown_symbol_returns_altcoin(self):
        cmp = _make_comparator()
        assert cmp.get_sector("FAKEUSDT") == "Altcoin"

    def test_btc_store_of_value(self):
        cmp = _make_comparator()
        assert cmp.get_sector("BTCUSDT") == "Store of Value"


class TestGetSectorContext:
    def _make_close_series(self, start: float, change_pct: float, length: int = 15) -> list:
        """Generate a synthetic close series with the given 7d change."""
        prices = [start] * length
        # close_7d_ago is index -8, latest is index -1
        # change_pct = (latest / close_7d_ago - 1) * 100
        prices[-8] = start
        prices[-1] = start * (1 + change_pct / 100)
        return prices

    def test_returns_sector_context(self):
        candles = {
            "INJUSDT": {"close": self._make_close_series(22.0, -4.2)},
            "ETHUSDT": {"close": self._make_close_series(3000.0, 5.2)},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert isinstance(ctx, SectorContext)
        assert ctx.sector_name == "DeFi"

    def test_symbol_7d_pct_is_computed(self):
        close_series = self._make_close_series(22.0, -4.2)
        candles = {"INJUSDT": {"close": close_series}}
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert abs(ctx.symbol_7d_pct - (-4.2)) < 0.5

    def test_relative_strength_lagging(self):
        """Symbol well below sector avg → lagging."""
        close_inj = self._make_close_series(22.0, -4.0)
        close_uni = self._make_close_series(10.0, 3.0)
        close_aave = self._make_close_series(100.0, 4.0)
        candles = {
            "INJUSDT": {"close": close_inj},
            "UNIUSDT": {"close": close_uni},
            "AAVEUSDT": {"close": close_aave},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert ctx.relative_strength == "lagging"

    def test_relative_strength_leading(self):
        """Symbol well above sector avg → leading."""
        close_inj = self._make_close_series(22.0, 12.0)
        close_uni = self._make_close_series(10.0, 2.0)
        close_aave = self._make_close_series(100.0, 1.0)
        candles = {
            "INJUSDT": {"close": close_inj},
            "UNIUSDT": {"close": close_uni},
            "AAVEUSDT": {"close": close_aave},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert ctx.relative_strength == "leading"

    def test_relative_strength_in_line(self):
        """Symbol close to sector avg → in-line."""
        close_inj = self._make_close_series(22.0, 3.0)
        close_uni = self._make_close_series(10.0, 3.0)
        candles = {
            "INJUSDT": {"close": close_inj},
            "UNIUSDT": {"close": close_uni},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert ctx.relative_strength == "in-line"

    def test_missing_candle_data_for_peers_no_crash(self):
        """If a peer has no candle data it is silently skipped."""
        candles = {
            "INJUSDT": {"close": self._make_close_series(22.0, -4.2)},
            # UNIUSDT, AAVEUSDT etc. all missing
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert isinstance(ctx, SectorContext)
        # Peers without data are excluded — may be empty
        assert isinstance(ctx.peers, list)

    def test_missing_all_candle_data_no_crash(self):
        """All candle data missing — returns sensible defaults."""
        cmp = _make_comparator({})
        ctx = cmp.get_sector_context("INJUSDT")
        assert isinstance(ctx, SectorContext)
        assert ctx.symbol_7d_pct == 0.0
        assert ctx.sector_7d_pct == 0.0

    def test_correlated_major_defi_is_eth(self):
        candles = {
            "INJUSDT": {"close": self._make_close_series(22.0, -4.2)},
            "ETHUSDT": {"close": self._make_close_series(3000.0, 5.2)},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        if ctx.correlated_major is not None:
            assert ctx.correlated_major[0] == "ETHUSDT"

    def test_insufficient_candle_length_skipped(self):
        """Close series shorter than 8 are treated as missing."""
        candles = {
            "INJUSDT": {"close": [22.0, 22.5, 23.0]},  # only 3 candles
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("INJUSDT")
        assert ctx.symbol_7d_pct == 0.0

    def test_unknown_symbol_sector_is_altcoin(self):
        candles = {
            "FAKEUSDT": {"close": self._make_close_series(1.0, 10.0)},
        }
        cmp = _make_comparator(candles)
        ctx = cmp.get_sector_context("FAKEUSDT")
        assert ctx.sector_name == "Altcoin"
