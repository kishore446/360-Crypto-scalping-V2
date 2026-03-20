"""360_THE_TAPE – Tick / Data Whale Tracking 🐋

Trigger : Trade > 1 M USD **or** Volume Delta > 2× + Min 2× flow delta ratio
Filters : Order-book imbalance (1.5×), whale detection, AI sentiment, spread < 0.02 %
Risk    : SL 0.1–0.3 % AI-adaptive, TP1 1.5R partial, TP2 3R partial,
          TP3 5R, trailing AI-adaptive
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from config import CHANNEL_TAPE
from src.channels.base import BaseChannel, Signal
from src.filters import check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# Minimum ratio of dominant side to weak side (e.g. buy_vol >= 2× sell_vol)
WHALE_DELTA_MIN_RATIO: float = 2.0

# Minimum total tick volume in USD to avoid firing on thin tick windows
WHALE_MIN_TICK_VOLUME_USD: float = 500_000.0

# Minimum order book imbalance ratio in signal's direction (optional filter)
ORDER_BOOK_IMBALANCE_MIN: float = 1.5


class TapeChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_TAPE)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        # --- Whale trigger ---
        whale = smc_data.get("whale_alert")
        delta_spike = smc_data.get("volume_delta_spike", False)

        if whale is None and not delta_spike:
            return None

        if not check_spread(spread_pct, self.config.spread_max):
            return None

        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        m1 = candles.get("1m")
        if m1 is None or len(m1.get("close", [])) < 10:
            return None

        close = float(m1["close"][-1])

        # Direction from net whale flow or delta (requires strong imbalance)
        ticks: List[Dict[str, Any]] = smc_data.get("recent_ticks", [])
        buy_vol = sum(t.get("qty", 0) * t.get("price", 0) for t in ticks if not t.get("isBuyerMaker", True))
        sell_vol = sum(t.get("qty", 0) * t.get("price", 0) for t in ticks if t.get("isBuyerMaker", True))

        total_vol = buy_vol + sell_vol
        if total_vol < WHALE_MIN_TICK_VOLUME_USD:
            return None

        if buy_vol >= sell_vol * WHALE_DELTA_MIN_RATIO:
            direction = Direction.LONG
        elif sell_vol >= buy_vol * WHALE_DELTA_MIN_RATIO:
            direction = Direction.SHORT
        else:
            return None  # Flow is ambiguous — skip

        # Order book imbalance check (if available)
        order_book = smc_data.get("order_book")
        if order_book is not None:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10])
            ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10])

            if bid_depth > 0 and ask_depth > 0:
                if direction == Direction.LONG:
                    imbalance_ratio = bid_depth / ask_depth
                else:
                    imbalance_ratio = ask_depth / bid_depth
                if imbalance_ratio < ORDER_BOOK_IMBALANCE_MIN:
                    return None  # Order book doesn't support the direction

        atr_val = indicators.get("1m", {}).get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val)

        if direction == Direction.LONG:
            sl = close - sl_dist
            tp1 = close + sl_dist * self.config.tp_ratios[0]
            tp2 = close + sl_dist * self.config.tp_ratios[1]
            tp3 = close + sl_dist * self.config.tp_ratios[2]
        else:
            sl = close + sl_dist
            tp1 = close - sl_dist * self.config.tp_ratios[0]
            tp2 = close - sl_dist * self.config.tp_ratios[1]
            tp3 = close - sl_dist * self.config.tp_ratios[2]

        return Signal(
            channel=self.config.name,
            symbol=symbol,
            direction=direction,
            entry=close,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            trailing_active=True,
            trailing_desc="AI Adaptive",
            confidence=0.0,
            ai_sentiment_label="",
            ai_sentiment_summary="",
            risk_label="Medium-High",
            timestamp=utcnow(),
            signal_id=f"TAPE-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )
