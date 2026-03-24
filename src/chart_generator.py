"""Chart generation for 360_GEM signals.

Generates TradingView-style dark-theme candlestick charts using ``mplfinance``.
Chart generation is fully optional — if ``mplfinance`` is not installed the
module degrades gracefully and every public function returns ``None``.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

from src.utils import get_logger

log = get_logger("chart_generator")

try:
    import matplotlib
    matplotlib.use("Agg")
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import pandas as pd
    _MPF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPF_AVAILABLE = False
    log.info("mplfinance not installed – gem chart generation disabled")


def generate_gem_chart(
    symbol: str,
    daily_candles: Dict[str, list],
    ath: float,
    current_price: float,
    ema_20: List[float],
    ema_50: List[float],
) -> Optional[bytes]:
    """Generate a dark-theme daily candlestick chart for a gem signal.

    Parameters
    ----------
    symbol:
        Trading pair symbol (e.g. ``"LYNUSDT"``).
    daily_candles:
        Dict with keys ``"open"``, ``"high"``, ``"low"``, ``"close"``,
        ``"volume"`` — daily OHLCV lists.
    ath:
        All-time high price drawn as a horizontal dashed line.
    current_price:
        Current price (used to compute x-potential for the chart title).
    ema_20:
        EMA(20) values aligned to the *last* N candles.
    ema_50:
        EMA(50) values aligned to the *last* N candles.

    Returns
    -------
    Optional[bytes]
        PNG image bytes if generation succeeds, ``None`` otherwise.
    """
    if not _MPF_AVAILABLE:
        return None

    try:
        opens = [float(v) for v in daily_candles.get("open", [])]
        highs = [float(v) for v in daily_candles.get("high", [])]
        lows = [float(v) for v in daily_candles.get("low", [])]
        closes = [float(v) for v in daily_candles.get("close", [])]
        volumes = [float(v) for v in daily_candles.get("volume", [])]

        n = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
        if n < 10:
            return None

        # Use last 90–120 candles for the chart
        window = min(120, n)
        opens = opens[-window:]
        highs = highs[-window:]
        lows = lows[-window:]
        closes = closes[-window:]
        volumes = volumes[-window:]

        # Build a DatetimeIndex (synthetic daily dates ending today)
        import datetime
        end_date = datetime.date.today()
        dates = [end_date - datetime.timedelta(days=window - 1 - i) for i in range(window)]
        idx = pd.DatetimeIndex(dates)

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }, index=idx)

        # Align EMA arrays to the chart window
        ema20_window = list(ema_20[-window:]) if len(ema_20) >= window else list(ema_20)
        ema50_window = list(ema_50[-window:]) if len(ema_50) >= window else list(ema_50)

        # Pad shorter EMA arrays with NaN at the start
        def _pad(arr: list, target: int) -> list:
            if len(arr) < target:
                return [float("nan")] * (target - len(arr)) + arr
            return arr[-target:]

        ema20_padded = _pad(ema20_window, window)
        ema50_padded = _pad(ema50_window, window)

        x_potential = ath / current_price if current_price > 0 else 0.0
        x_label = f"x{x_potential:.0f}" if x_potential > 0 else "N/A"
        title = f"{symbol} — Daily | 💎 GEM {x_label}"

        # ATH horizontal line
        ath_line = [ath] * window

        addplots = [
            mpf.make_addplot(ema20_padded, color="#26a69a", width=1.2, label="EMA 20"),
            mpf.make_addplot(ema50_padded, color="#ef5350", width=1.2, label="EMA 50"),
            mpf.make_addplot(ath_line, color="#ffd700", width=1.0, linestyle="--", label="ATH"),
        ]

        # Dark TradingView-like style
        mc = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
            volume={"up": "#26a69a", "down": "#ef5350"},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor="#131722",
            figcolor="#131722",
            gridcolor="#1e222d",
            gridstyle="--",
            gridaxis="both",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#787b86",
                "xtick.color": "#787b86",
                "ytick.color": "#787b86",
                "text.color": "#d1d4dc",
            },
        )

        buf = io.BytesIO()
        mpf.plot(
            df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            addplot=addplots,
            figsize=(14, 8),
            savefig=dict(fname=buf, format="png", dpi=150, bbox_inches="tight"),
        )
        plt.close("all")
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        log.error("generate_gem_chart error for %s: %s", symbol, exc)
        return None


def generate_portfolio_chart(
    symbol: str,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    entry: float,
    sl: float,
    tp_levels: List[float],
    channel_name: str = "SPOT",
    ema_periods: Tuple[int, ...] = (9, 21, 200),
) -> Optional[bytes]:
    """Generate a dark-theme portfolio signal chart with EMA overlays and key levels.

    Parameters
    ----------
    symbol:
        Trading pair symbol (e.g. ``"INJUSDT"``).
    closes:
        Close prices (most recent last).
    highs:
        High prices aligned to *closes*.
    lows:
        Low prices aligned to *closes*.
    volumes:
        Volume values aligned to *closes*.
    entry:
        Entry price — drawn as a blue dashed horizontal line.
    sl:
        Stop-loss price — drawn as a red horizontal line.
    tp_levels:
        List of TP prices (TP1, TP2, TP3, …) — drawn as green horizontal lines.
    channel_name:
        Channel label used in the chart title (``"SPOT"`` or ``"GEM"``).
    ema_periods:
        EMA periods to overlay.  Defaults to (9, 21, 200).

    Returns
    -------
    Optional[bytes]
        PNG image bytes if generation succeeds, ``None`` otherwise (fail-open).
    """
    if not _MPF_AVAILABLE:
        return None

    try:
        closes_f = [float(v) for v in closes]
        highs_f = [float(v) for v in highs]
        lows_f = [float(v) for v in lows]
        volumes_f = [float(v) for v in volumes]

        n = min(len(closes_f), len(highs_f), len(lows_f), len(volumes_f))
        if n < 10:
            return None

        # Use last 90 candles for readability
        window = min(90, n)
        closes_f = closes_f[-window:]
        highs_f = highs_f[-window:]
        lows_f = lows_f[-window:]
        volumes_f = volumes_f[-window:]

        # Build a synthetic DatetimeIndex
        import datetime
        end_date = datetime.date.today()
        dates = [end_date - datetime.timedelta(days=window - 1 - i) for i in range(window)]
        idx = pd.DatetimeIndex(dates)

        # We need open prices — use previous close as a proxy when not provided
        opens_f = [closes_f[0]] + closes_f[:-1]

        df = pd.DataFrame(
            {
                "Open": opens_f,
                "High": highs_f,
                "Low": lows_f,
                "Close": closes_f,
                "Volume": volumes_f,
            },
            index=idx,
        )

        # Compute EMAs
        import numpy as np  # already a project dependency

        def _ema(prices: List[float], period: int) -> List[float]:
            arr = np.array(prices, dtype=float)
            result = np.full(len(arr), float("nan"))
            # Guard: need at least `period` data points to compute the seed SMA
            if period <= 0 or len(arr) < period:
                return result.tolist()
            k = 2.0 / (period + 1)
            result[period - 1] = float(np.mean(arr[:period]))
            for i in range(period, len(arr)):
                result[i] = arr[i] * k + result[i - 1] * (1 - k)
            return result.tolist()

        ema_colors = {9: "#26a69a", 21: "#ef5350", 200: "#ffd700"}
        addplots = []
        for period in ema_periods:
            vals = _ema(closes_f, period)
            # Skip EMA addplots where all values are NaN (period > window) —
            # mplfinance raises on all-NaN arrays.
            vals_arr = np.array(vals)
            if np.all(np.isnan(vals_arr)):
                continue
            color = ema_colors.get(period, "#9c27b0")
            addplots.append(
                mpf.make_addplot(
                    vals,
                    color=color,
                    width=1.2,
                    label=f"EMA {period}",
                )
            )

        # Horizontal level lines
        entry_line = [entry] * window
        sl_line = [sl] * window
        addplots.append(
            mpf.make_addplot(entry_line, color="#2196f3", linestyle="--", width=1.0)
        )
        addplots.append(
            mpf.make_addplot(sl_line, color="#ef5350", linestyle="-", width=1.0)
        )

        tp_plot_colors = ["#4caf50", "#66bb6a", "#a5d6a7"]
        for i, tp_price in enumerate(tp_levels[:3]):
            color = tp_plot_colors[i] if i < len(tp_plot_colors) else "#4caf50"
            tp_line = [tp_price] * window
            addplots.append(
                mpf.make_addplot(tp_line, color=color, linestyle="-", width=0.8)
            )

        channel_emoji = "💎" if channel_name == "GEM" else "📈"
        title = f"{symbol} — {channel_emoji} {channel_name}"

        # Dark TradingView-like style
        mc = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
            volume={"up": "#26a69a", "down": "#ef5350"},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor="#131722",
            figcolor="#131722",
            gridcolor="#1e222d",
            gridstyle="--",
            gridaxis="both",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#787b86",
                "xtick.color": "#787b86",
                "ytick.color": "#787b86",
                "text.color": "#d1d4dc",
            },
        )

        # Accumulation zone shading for GEM signals
        fig_kwargs: dict = {}
        if channel_name == "GEM" and tp_levels:
            # Shade the accumulation zone between SL and entry
            fig_kwargs["returnfig"] = True

        buf = io.BytesIO()
        if fig_kwargs.get("returnfig"):
            fig, axes = mpf.plot(
                df,
                type="candle",
                style=style,
                title=title,
                volume=True,
                addplot=addplots,
                figsize=(12, 8),
                returnfig=True,
            )
            # Shade accumulation zone on the price axis
            ax_price = axes[0]
            ax_price.axhspan(sl, entry, alpha=0.08, color="#2196f3", zorder=0)
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            mpf.plot(
                df,
                type="candle",
                style=style,
                title=title,
                volume=True,
                addplot=addplots,
                figsize=(12, 8),
                savefig=dict(fname=buf, format="png", dpi=100, bbox_inches="tight"),
            )
            plt.close("all")

        buf.seek(0)
        return buf.read()

    except Exception as exc:
        log.error("generate_portfolio_chart error for %s: %s", symbol, exc)
        return None
