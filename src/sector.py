"""Sector comparison — peer performance context for SPOT/GEM signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils import get_logger

log = get_logger("sector")


@dataclass
class SectorContext:
    """Sector comparison context for a single symbol."""

    sector_name: str
    sector_7d_pct: float
    symbol_7d_pct: float
    peers: List[Tuple[str, float]] = field(default_factory=list)  # [(peer_symbol, 7d_pct), ...]
    correlated_major: Optional[Tuple[str, float]] = None  # (major_symbol, 7d_pct)
    relative_strength: str = "in-line"  # "leading", "lagging", "in-line"
    market_cap_rank: Optional[int] = None
    market_cap_usd: Optional[float] = None


class SectorComparator:
    """Provides sector / peer comparison data for a given symbol."""

    def __init__(self, data_store: Any, pair_mgr: Any) -> None:
        self._data_store = data_store
        self._pair_mgr = pair_mgr

        # Hard-coded sector classifications (extend as needed)
        self._sector_map: Dict[str, str] = {
            "BTCUSDT": "Store of Value",
            "ETHUSDT": "L1",
            "SOLUSDT": "L1",
            "BNBUSDT": "Exchange",
            "ADAUSDT": "L1",
            "DOTUSDT": "L0",
            "AVAXUSDT": "L1",
            "MATICUSDT": "L2",
            "LINKUSDT": "Oracle",
            "UNIUSDT": "DeFi",
            "AAVEUSDT": "DeFi",
            "MKRUSDT": "DeFi",
            "INJUSDT": "DeFi",
            "RUNEUSDT": "DeFi",
            "SUSHIUSDT": "DeFi",
            "ARBUSDT": "L2",
            "OPUSDT": "L2",
            "NEARUSDT": "L1",
            "ATOMUSDT": "L0",
            "APTUSDT": "L1",
            "SUIUSDT": "L1",
            "SEIUSDT": "L1",
            "FETUSDT": "AI",
            "RENDERUSDT": "AI",
            "RNDR": "AI",
            "TAOUSDT": "AI",
            "AGIXUSDT": "AI",
            "FILUSDT": "Storage",
            "ARUSDT": "Storage",
            "PEPEUSDT": "Meme",
            "DOGEUSDT": "Meme",
            "SHIBUSDT": "Meme",
            "WIFUSDT": "Meme",
            "FLOKIUSDT": "Meme",
        }

        # Which major asset correlates to each sector
        self._sector_major: Dict[str, str] = {
            "Store of Value": "BTCUSDT",
            "L1": "ETHUSDT",
            "L2": "ETHUSDT",
            "DeFi": "ETHUSDT",
            "L0": "ETHUSDT",
            "Exchange": "BTCUSDT",
            "Oracle": "ETHUSDT",
            "AI": "SOLUSDT",
            "Meme": "SOLUSDT",
            "Storage": "ETHUSDT",
            "Altcoin": "BTCUSDT",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sector(self, symbol: str) -> str:
        """Return the sector classification for *symbol*.  Falls back to ``"Altcoin"``."""
        return self._sector_map.get(symbol, "Altcoin")

    def get_sector_context(self, symbol: str) -> SectorContext:
        """Compute sector comparison context for the given symbol.

        Uses daily candles from ``data_store`` to compute 7-day % changes.
        Missing data for individual peers is silently skipped (fail-open).
        """
        sector_name = self.get_sector(symbol)

        # 7d change for the requested symbol
        symbol_7d = self._compute_7d_change(symbol)

        # All same-sector peers (excluding the symbol itself)
        peer_symbols = [
            sym for sym, sec in self._sector_map.items()
            if sec == sector_name and sym != symbol
        ]

        peers: List[Tuple[str, float]] = []
        sector_changes: List[float] = []

        if symbol_7d is not None:
            sector_changes.append(symbol_7d)

        for peer in peer_symbols:
            change = self._compute_7d_change(peer)
            if change is not None:
                peers.append((peer, change))
                sector_changes.append(change)

        sector_7d_pct = (
            sum(sector_changes) / len(sector_changes) if sector_changes else 0.0
        )
        symbol_7d_pct = symbol_7d if symbol_7d is not None else 0.0

        # Relative strength
        if symbol_7d_pct > sector_7d_pct + 2:
            relative_strength = "leading"
        elif symbol_7d_pct < sector_7d_pct - 2:
            relative_strength = "lagging"
        else:
            relative_strength = "in-line"

        # Correlated major
        major_symbol = self._sector_major.get(sector_name, "BTCUSDT")
        major_7d = self._compute_7d_change(major_symbol)
        correlated_major: Optional[Tuple[str, float]] = None
        if major_7d is not None:
            correlated_major = (major_symbol, major_7d)

        # Sort peers by absolute change (most extreme first, for readability)
        peers.sort(key=lambda x: abs(x[1]), reverse=True)

        return SectorContext(
            sector_name=sector_name,
            sector_7d_pct=sector_7d_pct,
            symbol_7d_pct=symbol_7d_pct,
            peers=peers[:4],  # top 4 peers for display
            correlated_major=correlated_major,
            relative_strength=relative_strength,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_7d_change(self, symbol: str) -> Optional[float]:
        """Return 7-day price change % for *symbol* using daily candles.

        Returns ``None`` if insufficient candle data is available.
        """
        try:
            candles = self._data_store.get_candles(symbol, "1d")
            if candles is None:
                return None
            closes = candles.get("close", [])
            if len(closes) < 8:
                return None
            latest_close = float(closes[-1])
            close_7d_ago = float(closes[-8])
            if close_7d_ago == 0:
                return None
            return (latest_close / close_7d_ago - 1) * 100
        except Exception as exc:
            log.debug("_compute_7d_change({}) failed: {}", symbol, exc)
            return None
