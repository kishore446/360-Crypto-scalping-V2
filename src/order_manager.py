"""Order manager – structural foundation for direct exchange execution (V3).

This module provides the :class:`OrderManager` interface so that the rest of
the engine can already call ``await order_manager.place_limit_order(sig)``
without any live exchange logic being wired in yet.

When ``AUTO_EXECUTION_ENABLED=true`` and a real :class:`~src.exchange_client.CCXTClient`
is passed, the manager executes orders directly on the exchange using CCXT.

The stubs log the intent and return ``None`` when the CCXT client is absent.
The calling code in :class:`src.trade_monitor.TradeMonitor` does not need to change.

Design notes
------------
* Limit orders are used for DCA / swing strategies (``360_SWING``, ``360_SPOT``)
  to capture maker-fee rebates and reduce slippage on fills.
* Market orders are used for high-frequency scalp strategies
  (``360_SCALP``) where immediate fill is more important
  than the maker/taker fee delta.
* Auto-execution is **off by default** (``AUTO_EXECUTION_ENABLED=false``).
  The engine still publishes to Telegram as normal; the order stubs simply
  no-op until the feature flag is enabled.
* Position sizing: ``POSITION_SIZE_PCT`` (default 2%) of available USDT balance,
  capped at ``MAX_POSITION_USD`` (default $100).
* Partial take-profit: ``close_partial()`` sells a fraction of the open
  position at each TP level (TP1: 33%, TP2: 33%, TP3: 34%).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import MAX_POSITION_USD, POSITION_SIZE_PCT
from src.utils import get_logger

log = get_logger("order_manager")

# Channels for which limit orders should be preferred (maker-fee strategy).
_LIMIT_ORDER_CHANNELS = {"360_SWING", "360_SPOT"}

# Partial TP fractions (must sum to 1.0)
_TP_FRACTIONS: Dict[int, float] = {1: 0.33, 2: 0.33, 3: 0.34}


def _symbol_to_ccxt(symbol: str) -> str:
    """Convert Binance-style symbol (BTCUSDT) to CCXT format (BTC/USDT)."""
    for quote in ("USDT", "BTC", "ETH", "BNB", "BUSD"):
        if symbol.upper().endswith(quote):
            base = symbol.upper()[: -len(quote)]
            return f"{base}/{quote}"
    return symbol


class OrderManager:
    """Manages direct exchange order placement.

    Parameters
    ----------
    auto_execution_enabled:
        Master toggle.  When ``False`` all methods are no-ops; signals are
        still routed to Telegram as usual.
    exchange_client:
        A :class:`~src.exchange_client.CCXTClient` instance (or any object with
        ``create_limit_order``, ``create_market_order``, ``cancel_order``, and
        ``fetch_balance`` coroutines).  Pass ``None`` until the real client is
        available.
    position_size_pct:
        Percentage of available balance to risk per trade (default 2.0).
    max_position_usd:
        Hard cap on position size in USD (default 100.0).
    """

    def __init__(
        self,
        auto_execution_enabled: bool = False,
        exchange_client: Optional[Any] = None,
        position_size_pct: float = POSITION_SIZE_PCT,
        max_position_usd: float = MAX_POSITION_USD,
    ) -> None:
        self._enabled = auto_execution_enabled
        self._client = exchange_client
        self._position_size_pct = position_size_pct
        self._max_position_usd = max_position_usd
        # Track open position sizes for partial TP execution: signal_id → quantity
        self._open_quantities: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when auto-execution is active."""
        return self._enabled

    async def _compute_quantity(self, entry_price: float) -> float:
        """Compute order quantity based on available balance and position sizing.

        Uses ``POSITION_SIZE_PCT`` of free USDT balance, capped at
        ``MAX_POSITION_USD``.  Falls back to ``MAX_POSITION_USD / entry_price``
        when balance fetch fails.
        """
        if self._client is None or entry_price <= 0:
            return self._max_position_usd / max(entry_price, 1e-12)

        try:
            balance = await self._client.fetch_balance()
            free_usdt = float(
                (balance.get("USDT") or balance.get("usdt") or {}).get("free", 0.0)
            )
            position_usd = min(
                free_usdt * self._position_size_pct / 100.0,
                self._max_position_usd,
            )
        except Exception as exc:
            log.warning("Balance fetch failed, using MAX_POSITION_USD: %s", exc)
            position_usd = self._max_position_usd

        return position_usd / entry_price

    async def place_limit_order(
        self,
        signal: Any,
        *,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Place a limit (maker) order on the exchange.

        Used by spot accumulation and swing strategies (``360_SPOT``, ``360_SWING``) to post
        resting bids/offers and capture maker-fee rebates.

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` driving the order.
        price:
            Explicit limit price.  When ``None`` the signal's ``entry``
            price is used.
        quantity:
            Order size in base currency.  When ``None`` computed from balance.

        Returns
        -------
        str or None
            Exchange order-ID on success; ``None`` when execution is disabled.
        """
        if not self._enabled:
            return None

        limit_price = price if price is not None else signal.entry
        direction = getattr(signal.direction, "value", str(signal.direction))
        side = "buy" if direction == "LONG" else "sell"

        if quantity is None:
            quantity = await self._compute_quantity(limit_price)

        if self._client is not None:
            try:
                ccxt_symbol = _symbol_to_ccxt(signal.symbol)
                order = await self._client.create_limit_order(
                    ccxt_symbol, side, quantity, limit_price
                )
                order_id: str = str(order.get("id", ""))
                self._open_quantities[signal.signal_id] = quantity
                log.info(
                    "[OrderManager] limit order placed: %s %s %s @ %s qty=%s id=%s",
                    signal.symbol, signal.channel, side, limit_price, quantity, order_id,
                )
                return order_id
            except Exception as exc:
                log.error(
                    "[OrderManager] limit order failed for %s: %s",
                    signal.symbol, exc,
                )
                return None

        log.info(
            "[OrderManager] STUB place_limit_order: {} {} {} @ {} (qty={})",
            signal.symbol,
            signal.channel,
            side,
            limit_price,
            quantity,
        )
        return None

    async def place_market_order(
        self,
        signal: Any,
        *,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Place a market (taker) order on the exchange.

        Used by high-frequency strategies (``360_SCALP``) where immediate fill certainty outweighs the taker-fee cost.

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` driving the order.
        quantity:
            Order size in base currency.

        Returns
        -------
        str or None
            Exchange order-ID on success; ``None`` when disabled / stub.
        """
        if not self._enabled:
            return None

        direction = getattr(signal.direction, "value", str(signal.direction))
        side = "buy" if direction == "LONG" else "sell"

        if quantity is None:
            quantity = await self._compute_quantity(signal.entry)

        if self._client is not None:
            try:
                ccxt_symbol = _symbol_to_ccxt(signal.symbol)
                order = await self._client.create_market_order(
                    ccxt_symbol, side, quantity
                )
                order_id = str(order.get("id", ""))
                self._open_quantities[signal.signal_id] = quantity
                log.info(
                    "[OrderManager] market order placed: %s %s %s qty=%s id=%s",
                    signal.symbol, signal.channel, side, quantity, order_id,
                )
                return order_id
            except Exception as exc:
                log.error(
                    "[OrderManager] market order failed for %s: %s",
                    signal.symbol, exc,
                )
                return None

        log.info(
            "[OrderManager] STUB place_market_order: {} {} {} (qty={})",
            signal.symbol,
            signal.channel,
            side,
            quantity,
        )
        return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open exchange order.

        Parameters
        ----------
        order_id:
            The exchange-assigned order identifier returned by
            :meth:`place_limit_order` or :meth:`place_market_order`.
        symbol:
            Trading-pair symbol (e.g. ``"BTCUSDT"``).

        Returns
        -------
        bool
            ``True`` when the cancellation was confirmed; ``False`` when
            execution is disabled or the operation failed.
        """
        if not self._enabled:
            return False

        if self._client is not None:
            try:
                ccxt_symbol = _symbol_to_ccxt(symbol)
                result = await self._client.cancel_order(order_id, ccxt_symbol)
                cancelled = result.get("status") == "canceled"
                log.info(
                    "[OrderManager] cancel_order: id=%s symbol=%s status=%s",
                    order_id, symbol, result.get("status"),
                )
                return cancelled
            except Exception as exc:
                log.error(
                    "[OrderManager] cancel_order failed for %s id=%s: %s",
                    symbol, order_id, exc,
                )
                return False

        log.info(
            "[OrderManager] STUB cancel_order: order_id={} symbol={}",
            order_id,
            symbol,
        )
        return False

    async def close_partial(self, signal: Any, fraction: float) -> Optional[str]:
        """Close a fraction of an open position (partial take-profit execution).

        Called by :class:`~src.trade_monitor.TradeMonitor` on TP1/TP2/TP3 hits:

        * TP1: ``fraction=0.33``
        * TP2: ``fraction=0.33``
        * TP3: ``fraction=0.34``

        Parameters
        ----------
        signal:
            The active :class:`~src.channels.base.Signal`.
        fraction:
            Fraction of the original position to close (0.0–1.0).

        Returns
        -------
        str or None
            Exchange order-ID on success; ``None`` when disabled or stub.
        """
        if not self._enabled:
            return None

        original_qty = self._open_quantities.get(signal.signal_id, 0.0)
        if original_qty <= 0:
            log.debug(
                "[OrderManager] close_partial: no tracked quantity for %s",
                signal.signal_id,
            )
            return None

        close_qty = original_qty * fraction
        direction = getattr(signal.direction, "value", str(signal.direction))
        # To close a LONG we sell; to close a SHORT we buy.
        side = "sell" if direction == "LONG" else "buy"

        if self._client is not None:
            try:
                ccxt_symbol = _symbol_to_ccxt(signal.symbol)
                order = await self._client.create_market_order(
                    ccxt_symbol, side, close_qty
                )
                order_id = str(order.get("id", ""))
                log.info(
                    "[OrderManager] partial close: %s %s %.2f%% of %s qty=%.6f id=%s",
                    signal.symbol, side, fraction * 100, signal.signal_id,
                    close_qty, order_id,
                )
                return order_id
            except Exception as exc:
                log.error(
                    "[OrderManager] close_partial failed for %s: %s",
                    signal.symbol, exc,
                )
                return None

        log.info(
            "[OrderManager] STUB close_partial: {} {} {}% (qty={})",
            signal.symbol, side, fraction * 100, close_qty,
        )
        return None

    async def execute_signal(self, signal: Any) -> Optional[str]:
        """Dispatch an order for *signal* using the appropriate order type.

        Convenience wrapper that selects limit vs. market order based on the
        signal's channel:

        * ``360_SPOT`` / ``360_SWING`` → :meth:`place_limit_order` (maker)
        * All other channels → :meth:`place_market_order` (taker)

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` to execute.

        Returns
        -------
        str or None
            Exchange order-ID, or ``None`` when disabled / stub.
        """
        if not self._enabled:
            return None

        if signal.channel in _LIMIT_ORDER_CHANNELS:
            return await self.place_limit_order(signal)
        return await self.place_market_order(signal)
