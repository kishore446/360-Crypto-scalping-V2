"""On-Chain Intelligence — exchange flow data as a confidence sub-score.

Fetches net exchange flow data (coins entering / leaving exchanges) from
Glassnode or CryptoQuant and converts the signal into a 0–10 confidence
sub-score:

* Large net **outflows** (coins leaving exchanges) → bullish → score near 10
* Large net **inflows** (coins entering exchanges) → bearish → score near 0
* Neutral / unavailable → 5.0

All API calls degrade gracefully.  If ``ONCHAIN_API_KEY`` is not configured
the client returns a neutral score of ``5.0`` for every symbol.

Whale Alert integration extends tracking to all tokens beyond the BTC/ETH
Glassnode free tier, using the optional ``WHALE_ALERT_API_KEY``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from src.utils import get_logger

log = get_logger("onchain")

_CACHE_TTL: float = 300.0   # 5 minutes – on-chain data updates slowly
_NEUTRAL_SCORE: float = 5.0   # neutral mid-point on the 0-10 scale
_MAX_SCORE: float = 10.0

# Assets supported by Glassnode's free tier.  All other coins will get a
# neutral score immediately without making any API calls.
_SUPPORTED_ONCHAIN_ASSETS: frozenset = frozenset({"BTC", "ETH"})

# Glassnode endpoint for BTC exchange net-flow (free tier)
_GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics/transactions"

# Whale Alert — free tier allows 10 req/min (no key for status, key for feed)
_WHALE_ALERT_BASE = "https://api.whale-alert.io/v1"
# Minimum USD transfer size to fetch from Whale Alert
_WHALE_MIN_USD: int = 500_000


@dataclass
class OnChainData:
    """On-chain exchange flow snapshot for a single asset."""
    symbol: str = ""
    net_flow_usd: float = 0.0    # positive = inflow (bearish), negative = outflow (bullish)
    source: str = ""
    score: float = _NEUTRAL_SCORE  # 0–5 confidence contribution


class OnChainClient:
    """Async client for on-chain exchange flow data.

    Supports Glassnode (``ONCHAIN_API_KEY`` env-var).  When the key is absent
    every call immediately returns a neutral :class:`OnChainData` so the rest
    of the pipeline is unaffected.

    Results are cached per asset for :data:`_CACHE_TTL` seconds.
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key: str = api_key
        self._enabled: bool = bool(api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, OnChainData]] = {}

    @property
    def enabled(self) -> bool:
        """Return ``True`` when an API key is configured."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_exchange_flow(self, symbol: str) -> OnChainData:
        """Return on-chain exchange flow data for *symbol*.

        Parameters
        ----------
        symbol:
            Trading pair such as ``"BTCUSDT"`` or just ``"BTC"``.

        Returns
        -------
        OnChainData
            Always returns a valid object; score is :data:`_NEUTRAL_SCORE`
            when data is unavailable.
        """
        coin = _strip_quote_currency(symbol)
        neutral = OnChainData(symbol=coin, source="none", score=_NEUTRAL_SCORE)

        if not self._enabled:
            return neutral

        # Glassnode free-tier only supports BTC and ETH.
        # Skip the API call entirely for other assets to avoid errors and latency.
        if coin.upper() not in _SUPPORTED_ONCHAIN_ASSETS:
            log.debug("On-chain data not supported for %s – returning neutral score", coin)
            return OnChainData(symbol=coin, source="unsupported", score=_NEUTRAL_SCORE)

        cached = self._cache.get(coin)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL:
            return cached[1]

        try:
            result = await self._fetch_glassnode(coin)
        except Exception as exc:
            log.debug("On-chain fetch failed for %s: %s", symbol, exc)
            return neutral

        self._cache[coin] = (time.monotonic(), result)
        return result

    async def close(self) -> None:
        """Close the underlying :class:`aiohttp.ClientSession` if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_glassnode(self, coin: str) -> OnChainData:
        """Fetch net exchange flow from Glassnode for *coin*."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        asset = coin.lower()
        url = f"{_GLASSNODE_BASE}/transfers_to_exchanges_sum"
        params: Dict[str, Any] = {
            "a": asset,
            "api_key": self._api_key,
            "i": "24h",
            "limit": 1,
        }

        timeout = aiohttp.ClientTimeout(total=10)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                log.debug(
                    "Glassnode returned %d for %s exchange inflow",
                    resp.status, coin,
                )
                return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)
            inflow_data: Any = await resp.json(content_type=None)

        # Fetch outflow
        url_out = f"{_GLASSNODE_BASE}/transfers_from_exchanges_sum"
        async with self._session.get(
            url_out, params=params, timeout=timeout
        ) as resp2:
            if resp2.status != 200:
                return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)
            outflow_data: Any = await resp2.json(content_type=None)

        inflow = _parse_glassnode_latest(inflow_data)
        outflow = _parse_glassnode_latest(outflow_data)

        if inflow is None or outflow is None:
            return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)

        net_flow = inflow - outflow  # positive = net inflow (bearish)
        score = _net_flow_to_score(net_flow, inflow, outflow)

        log.debug(
            "On-chain %s: inflow=%.2f outflow=%.2f net=%.2f score=%.2f",
            coin, inflow, outflow, net_flow, score,
        )
        return OnChainData(
            symbol=coin,
            net_flow_usd=net_flow,
            source="glassnode",
            score=score,
        )


# ---------------------------------------------------------------------------
# Whale Alert Client
# ---------------------------------------------------------------------------

@dataclass
class WhaleTransaction:
    """A single large-value on-chain transaction from Whale Alert."""
    symbol: str = ""
    amount_usd: float = 0.0
    direction: str = "unknown"   # "exchange_inflow" | "exchange_outflow" | "unknown"
    from_owner: str = "unknown"
    to_owner: str = "unknown"
    timestamp: float = 0.0


class WhaleAlertClient:
    """Async client for Whale Alert large-transaction data (free tier).

    Fetches recent whale transactions (≥ $500K) for any token.  The free
    tier allows 10 requests/min.  An optional ``WHALE_ALERT_API_KEY``
    (obtainable free at https://whale-alert.io/) unlocks the full
    transaction feed.  Without a key the client returns neutral scores.

    Scoring logic
    -------------
    * Exchange **outflow** (whale withdraws from exchange) → bullish
    * Exchange **inflow** (whale deposits to exchange) → bearish
    * Wallet-to-wallet (no exchange involved) → neutral

    The resulting score (0–10) is blended into the on-chain sub-score.

    Results are cached per coin for :data:`_CACHE_TTL` seconds.
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key: str = api_key
        self._enabled: bool = bool(api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, float]] = {}  # coin → (ts, score)

    @property
    def enabled(self) -> bool:
        """Return ``True`` when a Whale Alert API key is configured."""
        return self._enabled

    async def get_whale_score(self, symbol: str) -> float:
        """Return a 0–10 whale-activity score for *symbol*.

        Returns
        -------
        float
            5.0 (neutral) when the key is absent or the call fails.
        """
        if not self._enabled:
            return _NEUTRAL_SCORE

        coin = _strip_quote_currency(symbol)
        cached = self._cache.get(coin)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL:
            return cached[1]

        try:
            score = await self._fetch_whale_score(coin)
        except Exception as exc:
            log.debug("Whale Alert fetch failed for %s: %s", symbol, exc)
            return _NEUTRAL_SCORE

        self._cache[coin] = (time.monotonic(), score)
        return score

    async def close(self) -> None:
        """Close the underlying :class:`aiohttp.ClientSession` if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _fetch_whale_score(self, coin: str) -> float:
        """Fetch whale transactions and compute a directional score."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        since = int(time.time()) - 3600  # last 1 hour
        params: Dict[str, Any] = {
            "api_key": self._api_key,
            "min_value": _WHALE_MIN_USD,
            "start": since,
            "limit": 100,
            "currency": coin.lower(),
        }
        url = f"{_WHALE_ALERT_BASE}/transactions"
        timeout = aiohttp.ClientTimeout(total=10)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                log.debug("Whale Alert returned %d for %s", resp.status, coin)
                return _NEUTRAL_SCORE
            data: Any = await resp.json(content_type=None)

        transactions: List[dict] = data.get("transactions") or []
        if not transactions:
            return _NEUTRAL_SCORE

        bullish_usd = 0.0
        bearish_usd = 0.0
        for tx in transactions:
            amount_usd = float(tx.get("amount_usd", 0))
            from_owner = (tx.get("from", {}) or {}).get("owner_type", "")
            to_owner = (tx.get("to", {}) or {}).get("owner_type", "")
            if from_owner == "exchange" and to_owner != "exchange":
                bullish_usd += amount_usd   # leaving exchange → bullish
            elif to_owner == "exchange" and from_owner != "exchange":
                bearish_usd += amount_usd   # entering exchange → bearish

        total = bullish_usd + bearish_usd
        if total <= 0:
            return _NEUTRAL_SCORE

        normalised = max(-1.0, min(1.0, (bullish_usd - bearish_usd) / total))
        score = round((normalised + 1.0) / 2.0 * _MAX_SCORE, 2)
        log.debug(
            "Whale Alert %s: bullish=%.0f bearish=%.0f score=%.2f",
            coin, bullish_usd, bearish_usd, score,
        )
        return score


# ---------------------------------------------------------------------------
# Module-level scoring helper (used by confidence.py import)
# ---------------------------------------------------------------------------

def score_onchain(onchain_data: Optional["OnChainData"]) -> float:
    """Convert an :class:`OnChainData` snapshot to a 0–10 confidence score.

    Parameters
    ----------
    onchain_data:
        Result from :meth:`OnChainClient.get_exchange_flow`, or ``None`` when
        on-chain intelligence is unavailable.

    Returns
    -------
    float
        0 (bearish on-chain) → 10 (bullish on-chain); 5.0 is neutral.
    """
    if onchain_data is None:
        return _NEUTRAL_SCORE
    return float(min(max(onchain_data.score, 0.0), _MAX_SCORE))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_quote_currency(symbol: str) -> str:
    """Strip common quote-currency suffixes to get the base coin name."""
    for suffix in ("USDT", "BUSD", "USDC"):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)].upper()
    return symbol.upper()


def _parse_glassnode_latest(data: Any) -> Optional[float]:
    """Extract the most recent value from a Glassnode API response list."""
    if not isinstance(data, list) or len(data) == 0:
        return None
    entry = data[-1]
    if isinstance(entry, dict):
        v = entry.get("v")
        return float(v) if v is not None else None
    return None


def _net_flow_to_score(net_flow: float, inflow: float, outflow: float) -> float:
    """Map net exchange flow to a 0–10 confidence score.

    A large net outflow (coins leaving exchanges) is bullish → score near 10.
    A large net inflow (coins entering exchanges) is bearish → score near 0.
    Near-zero net flow is neutral → score near 5.0.
    """
    total = inflow + outflow
    if total <= 0:
        return _NEUTRAL_SCORE

    # Normalise net flow relative to total volume: range [-1, +1]
    normalised = max(-1.0, min(1.0, -net_flow / total))  # invert: outflow → positive
    # Map [-1, +1] → [0, 10]
    return round((normalised + 1.0) / 2.0 * _MAX_SCORE, 2)
