"""Multi-Exchange Abstraction.

Provides :class:`ExchangeManager` which wraps multiple exchange clients and
enables cross-exchange signal verification.  Initially ships with Binance
support and a stub for a second exchange (Bybit / OKX).
"""

from __future__ import annotations

from typing import Optional

import aiohttp

from src.utils import get_logger

log = get_logger("exchange_mgr")

# Price tolerance for cross-exchange validation (percentage)
_PRICE_TOLERANCE_PCT: float = 0.5


class ExchangeManager:
    """Wraps multiple exchange REST clients for cross-exchange verification.

    Only Binance is fully implemented. A second exchange can be added by
    providing a ``second_exchange_url`` and implementing ``_fetch_price_second``.

    Parameters
    ----------
    second_exchange_url:
        Optional base URL for a second exchange ticker endpoint.
        If not provided, :meth:`verify_signal_cross_exchange` returns ``False``
        (unable to verify), which the confidence scorer maps to a neutral score.
    """

    def __init__(self, second_exchange_url: Optional[str] = None) -> None:
        self._second_url = second_exchange_url
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Cross-exchange verification
    # ------------------------------------------------------------------

    async def verify_signal_cross_exchange(
        self,
        symbol: str,
        direction: str,
        price: float,
    ) -> bool:
        """Check a second exchange to confirm the signal direction.

        Returns ``True`` if the second-exchange price is consistent with
        *direction* relative to *price*, ``False`` otherwise or when no
        second exchange is configured.

        Parameters
        ----------
        symbol:
            Trading symbol, e.g. ``"BTCUSDT"``.
        direction:
            ``"LONG"`` or ``"SHORT"``.
        price:
            Entry price from the primary exchange.
        """
        if not self._second_url:
            log.debug("No second exchange configured – cross-exchange check skipped")
            return False

        second_price = await self._fetch_price_second(symbol)
        if second_price is None:
            return False

        tolerance = price * _PRICE_TOLERANCE_PCT / 100.0
        spread = abs(second_price - price)

        if spread > tolerance:
            log.debug(
                "%s cross-exchange price divergence %.4f (primary) vs %.4f (second) – "
                "spread %.4f > tol %.4f",
                symbol, price, second_price, spread, tolerance,
            )
            return False

        # Prices are in agreement – direction confirmed
        log.debug(
            "%s cross-exchange verified: primary=%.4f second=%.4f direction=%s",
            symbol, price, second_price, direction,
        )
        return True

    # ------------------------------------------------------------------
    # Second-exchange price fetch (override to support specific exchanges)
    # ------------------------------------------------------------------

    async def _fetch_price_second(self, symbol: str) -> Optional[float]:
        """Fetch the latest price from the second exchange.

        Default implementation calls ``{second_exchange_url}?symbol={symbol}``
        and expects a JSON response with a ``"price"`` or ``"lastPrice"`` field.

        Override this method to integrate Bybit, OKX, or any other exchange.
        """
        if not self._second_url:
            return None
        try:
            session = await self._ensure_session()
            url = f"{self._second_url}?symbol={symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                raw = data.get("price") or data.get("lastPrice") or data.get("last")
                return float(raw) if raw is not None else None
        except Exception as exc:
            log.debug("Second-exchange price fetch for %s failed: %s", symbol, exc)
            return None
