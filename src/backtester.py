"""Backtesting framework.

Replays historical candle data through channel strategies and computes
performance metrics including win rate, average R:R, and max drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.channels.base import Signal
from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.range_channel import RangeChannel
from src.channels.tape import TapeChannel
from src.detector import SMCDetector
from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi
from src.utils import get_logger

log = get_logger("backtester")


@dataclass
class BacktestResult:
    """Summary metrics for a single backtest run."""

    channel: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    total_pnl_pct: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    signal_details: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Backtest: {self.channel}\n"
            f"Signals: {self.total_signals} | Wins: {self.wins} | Losses: {self.losses}\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Avg R:R: {self.avg_rr:.2f}\n"
            f"Total PnL: {self.total_pnl_pct:+.2f}%\n"
            f"Max Drawdown: {self.max_drawdown:.2f}%\n"
            f"Best: {self.best_trade:+.2f}% | Worst: {self.worst_trade:+.2f}%"
        )


def _compute_indicators(candles: Dict) -> Dict:
    """Compute the standard set of technical indicators from candle arrays."""
    h = candles.get("high", np.array([]))
    lo = candles.get("low", np.array([]))
    c = candles.get("close", np.array([]))
    ind: Dict = {}

    if len(c) >= 21:
        ind["ema9_last"] = float(ema(c, 9)[-1])
        ind["ema21_last"] = float(ema(c, 21)[-1])
    if len(c) >= 200:
        ind["ema200_last"] = float(ema(c, 200)[-1])
    if len(c) >= 30:
        a = adx(h, lo, c, 14)
        valid = a[~np.isnan(a)]
        ind["adx_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 15:
        a = atr(h, lo, c, 14)
        valid = a[~np.isnan(a)]
        ind["atr_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 15:
        r = rsi(c, 14)
        valid = r[~np.isnan(r)]
        ind["rsi_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 20:
        u, m, lo_b = bollinger_bands(c, 20)
        ind["bb_upper_last"] = float(u[-1]) if not np.isnan(u[-1]) else None
        ind["bb_mid_last"] = float(m[-1]) if not np.isnan(m[-1]) else None
        ind["bb_lower_last"] = float(lo_b[-1]) if not np.isnan(lo_b[-1]) else None
    if len(c) >= 4:
        mom = momentum(c, 3)
        ind["momentum_last"] = float(mom[-1]) if not np.isnan(mom[-1]) else None

    return ind


def _simulate_trade(
    signal: Signal,
    future_candles: Dict,
    sl_multiplier: float = 1.0,
) -> Tuple[bool, float, int]:
    """Simulate a signal against future price data.

    Returns
    -------
    (won, pnl_pct, tp_level_hit)
        ``won`` is True if TP1 was hit before SL.
        ``pnl_pct`` is the estimated PnL percentage.
        ``tp_level_hit`` is 0 (SL), 1 (TP1), 2 (TP2), or 3 (TP3).
    """
    highs = future_candles.get("high", np.array([]))
    lows = future_candles.get("low", np.array([]))

    if len(highs) == 0:
        return False, 0.0, 0

    is_long = signal.direction.value == "LONG"
    sl = signal.stop_loss * sl_multiplier
    targets = [signal.tp1, signal.tp2]
    if signal.tp3 is not None:
        targets.append(signal.tp3)

    for i in range(min(len(highs), len(lows))):
        h, lo = float(highs[i]), float(lows[i])

        if is_long:
            if lo <= sl:
                pnl = (sl - signal.entry) / signal.entry * 100.0
                return False, pnl, 0
            for level_idx, tp in enumerate(targets, start=1):
                if h >= tp:
                    pnl = (tp - signal.entry) / signal.entry * 100.0
                    return True, pnl, level_idx
        else:
            if h >= sl:
                pnl = (signal.entry - sl) / signal.entry * 100.0
                return False, -abs(pnl), 0
            for level_idx, tp in enumerate(targets, start=1):
                if lo <= tp:
                    pnl = (signal.entry - tp) / signal.entry * 100.0
                    return True, pnl, level_idx

    # No TP or SL hit in the lookahead window
    last_close = float(future_candles.get("close", [signal.entry])[-1])
    if is_long:
        pnl = (last_close - signal.entry) / signal.entry * 100.0
    else:
        pnl = (signal.entry - last_close) / signal.entry * 100.0
    return pnl > 0, pnl, 0


class Backtester:
    """Replays historical candle data through channel strategies.

    Parameters
    ----------
    channels:
        List of channel strategy objects to test.  Defaults to all four
        standard channels (SCALP, SWING, RANGE, TAPE).
    lookahead_candles:
        Number of future candles to use for simulating trade outcomes.
    min_window:
        Minimum number of candles required before evaluation starts.
    """

    def __init__(
        self,
        channels: Optional[List] = None,
        lookahead_candles: int = 20,
        min_window: int = 50,
    ) -> None:
        if channels is None:
            channels = [
                ScalpChannel(),
                SwingChannel(),
                RangeChannel(),
                TapeChannel(),
            ]
        self._channels = channels
        self._lookahead = lookahead_candles
        self._min_window = min_window
        self._smc_detector = SMCDetector()

    def run(
        self,
        candles_by_tf: Dict[str, Dict],
        symbol: str = "BTCUSDT",
        channel_name: Optional[str] = None,
        spread_pct: float = 0.01,
        volume_24h_usd: float = 10_000_000.0,
    ) -> List[BacktestResult]:
        """Run backtest across all (or one) channel(s).

        Parameters
        ----------
        candles_by_tf:
            Dict mapping timeframe → OHLCV dict with numpy arrays.
            E.g. ``{"5m": {"high": ..., "low": ..., "close": ..., ...}}``.
        symbol:
            Symbol name (used for SMC detection).
        channel_name:
            If provided, only backtest this specific channel.
        spread_pct:
            Simulated spread percentage.
        volume_24h_usd:
            Simulated 24h volume.

        Returns
        -------
        List of :class:`BacktestResult`, one per channel tested.
        """
        channels = self._channels
        if channel_name:
            channels = [c for c in channels if c.config.name == channel_name]

        results: List[BacktestResult] = []
        for chan in channels:
            result = self._backtest_channel(
                chan, candles_by_tf, symbol, spread_pct, volume_24h_usd
            )
            results.append(result)
            log.info("Backtest %s: %s", chan.config.name, result.summary())

        return results

    def _backtest_channel(
        self,
        channel,
        candles_by_tf: Dict[str, Dict],
        symbol: str,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> BacktestResult:
        """Run a single channel backtest across all candle windows."""
        result = BacktestResult(channel=channel.config.name)
        pnl_history: List[float] = []

        # Use the primary timeframe for the channel
        primary_tf = channel.config.timeframes[0]
        if primary_tf not in candles_by_tf:
            log.warning(
                "Backtest: timeframe %s not available for %s",
                primary_tf,
                channel.config.name,
            )
            return result

        primary_candles = candles_by_tf[primary_tf]
        total_candles = len(primary_candles.get("close", []))

        for i in range(self._min_window, total_candles - self._lookahead):
            # Slice candles up to index i for the evaluation window
            window: Dict[str, Dict] = {}
            for tf, cd in candles_by_tf.items():
                window[tf] = {
                    k: v[:i] if hasattr(v, "__getitem__") else v
                    for k, v in cd.items()
                }

            # Compute indicators for each timeframe
            indicators: Dict[str, Dict] = {}
            for tf, cd in window.items():
                indicators[tf] = _compute_indicators(cd)

            # SMC detection
            smc_result = self._smc_detector.detect(symbol, window, [])
            smc_data = smc_result.as_dict()

            try:
                sig = channel.evaluate(
                    symbol=symbol,
                    candles=window,
                    indicators=indicators,
                    smc_data=smc_data,
                    ai_insight={"label": "Neutral", "summary": "", "score": 0.0},
                    spread_pct=spread_pct,
                    volume_24h_usd=volume_24h_usd,
                )
            except Exception as exc:
                log.debug("Channel eval error at candle %d: %s", i, exc)
                continue

            if sig is None:
                continue

            # Simulate against future candles
            future: Dict[str, np.ndarray] = {}
            for k, v in primary_candles.items():
                if hasattr(v, "__getitem__"):
                    future[k] = v[i: i + self._lookahead]
            won, pnl, tp_level = _simulate_trade(sig, future)

            result.total_signals += 1
            if won:
                result.wins += 1
            else:
                result.losses += 1
            pnl_history.append(pnl)
            result.signal_details.append({
                "candle_index": i,
                "direction": sig.direction.value,
                "entry": sig.entry,
                "won": won,
                "pnl_pct": round(pnl, 4),
                "tp_level": tp_level,
            })

        # Aggregate statistics
        if result.total_signals > 0:
            total = result.wins + result.losses
            result.win_rate = result.wins / total * 100.0 if total > 0 else 0.0
            result.total_pnl_pct = sum(pnl_history)
            result.avg_rr = (
                sum(p for p in pnl_history if p > 0) / result.wins
                if result.wins > 0
                else 0.0
            )
            result.best_trade = max(pnl_history) if pnl_history else 0.0
            result.worst_trade = min(pnl_history) if pnl_history else 0.0

            # Max drawdown
            cum = 0.0
            peak = 0.0
            dd = 0.0
            for p in pnl_history:
                cum += p
                if cum > peak:
                    peak = cum
                drop = peak - cum
                if drop > dd:
                    dd = drop
            result.max_drawdown = dd

        return result
