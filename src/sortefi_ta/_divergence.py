"""Shared divergence detection logic used by all calc_*_divergence functions.

Two detection modes:

- method="tv"  (default) — fixed-window method, the convention most charting
  platforms use: pivots are found on the *indicator* series and consecutive
  pivots must sit within a fixed bar distance (range_lower/range_upper).

- method="swing" — dynamic, volatility-adaptive divergence. Pivots are found on
  *price* (fractal local extrema) and filtered to *significant* swings using an
  ATR-scaled zigzag (a new swing is registered only when price reverses by at
  least atr_mult x ATR). There is NO fixed bar-distance cap, so divergences can
  span however far apart the significant swings are (e.g. multi-month on daily).
  Requires an `atr` array aligned with the indicator.

Both modes return the full history of confirmations (0.0/1.0 arrays); callers
decide "active" (recent slice) vs "historical" (whole array) downstream.
"""

import numpy as np


def _calc_divergence(
    price_low: np.ndarray,
    price_high: np.ndarray,
    indicator: np.ndarray,
    lookback_left: int = 5,
    lookback_right: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
    *,
    method: str = "tv",
    atr: np.ndarray | None = None,
    atr_mult: float = 1.0,
    fractal_period: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect regular bullish and bearish divergence between price and an indicator.

    Args:
        price_low:      numpy array of bar lows (aligned with indicator)
        price_high:     numpy array of bar highs (aligned with indicator)
        indicator:      numpy array of indicator values to compare against price
        lookback_left:  bars to the left of pivot required (default 5)   [tv only]
        lookback_right: bars to the right of pivot required (default 5)  [tv only]
        range_lower:    min bars between consecutive pivots (default 5)  [tv only]
        range_upper:    max bars between consecutive pivots (default 60) [tv only]
        method:         "tv" (fixed-window, pivots on indicator) or
                        "swing" (ATR-significant price swings, no bar cap)
        atr:            ATR array aligned with indicator — required for method="swing"
        atr_mult:       significance threshold; a swing counts only if price reverses
                        by >= atr_mult x ATR from the previous swing (swing only)
        fractal_period: bars either side for the price fractal window (swing only)

    Returns:
        (bull_div, bear_div, bull_strength, bear_strength) — bull_div/bear_div are
        0.0/1.0 (1.0 at confirmation bar); bull_strength/bear_strength carry the
        indicator's counter-move magnitude at that same bar (e.g. RSI points), 0.0
        elsewhere. Strength is >= 0; larger = a wider, higher-conviction divergence.

    Regular Bullish: price lower low + indicator higher low  → reversal up
    Regular Bearish: price higher high + indicator lower high → reversal down
    """
    if method == "swing":
        return _calc_divergence_swing(
            price_low, price_high, indicator,
            atr=atr, atr_mult=atr_mult, fractal_period=fractal_period,
            min_bar_gap=range_lower,
        )
    if method != "tv":
        raise ValueError(f"unknown divergence method: {method!r} (expected 'tv' or 'swing')")
    return _calc_divergence_pivot_window(
        price_low, price_high, indicator,
        lookback_left, lookback_right, range_lower, range_upper,
    )


def _window_pivots(values: np.ndarray, left: int, right: int) -> tuple[list[int], list[int]]:
    """Turning points of `values`: index j is a pivot low (high) when values[j] is the minimum
    (maximum) of the window from `left` bars before to `right` bars after it. Windows that
    contain a missing value are skipped."""
    lows: list[int] = []
    highs: list[int] = []
    for j in range(left, len(values) - right):
        win = values[j - left: j + right + 1]
        if np.isnan(win).any():
            continue
        if values[j] == win.min():
            lows.append(j)
        if values[j] == win.max():
            highs.append(j)
    return lows, highs


def _calc_divergence_pivot_window(
    price_low: np.ndarray,
    price_high: np.ndarray,
    indicator: np.ndarray,
    left: int,
    right: int,
    min_gap: int,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-window method (method="tv"; the widely used charting-platform convention).

    1. Turning points are found on the INDICATOR with a `left`/`right` bar window.
    2. Each pivot is compared with the previous pivot of the same kind, if the two are
       between `min_gap` and `max_gap` bars apart.
    3. Bullish: indicator makes a higher low while price makes a lower low.
       Bearish: indicator makes a lower high while price makes a higher high.
    4. The signal is marked `right` bars after the pivot — the first bar on which the pivot
       is known — and its strength is the indicator's counter-move between the two pivots.
    """
    n = len(indicator)
    out = {k: np.zeros(n) for k in ("bull", "bear", "bull_s", "bear_s")}
    lows, highs = _window_pivots(indicator, left, right)

    def _mark(pivots, is_div, kind):
        for prev_j, j in zip(pivots, pivots[1:]):
            if not (min_gap <= j - prev_j <= max_gap) or not is_div(prev_j, j):
                continue
            seen = j + right
            if seen < n:
                out[kind][seen] = 1.0
                out[kind + "_s"][seen] = abs(indicator[j] - indicator[prev_j])

    _mark(lows, lambda a, b: indicator[b] > indicator[a] and price_low[b] < price_low[a], "bull")
    _mark(highs, lambda a, b: indicator[b] < indicator[a] and price_high[b] > price_high[a], "bear")
    return out["bull"], out["bear"], out["bull_s"], out["bear_s"]


def _calc_divergence_swing(
    price_low: np.ndarray,
    price_high: np.ndarray,
    indicator: np.ndarray,
    atr: np.ndarray | None,
    atr_mult: float = 1.0,
    fractal_period: int = 2,
    min_bar_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divergence between ATR-significant price swings (no fixed bar-distance cap).

    Steps:
      1. Find candidate fractal highs/lows on price (local extremum over
         +/- fractal_period bars).
      2. Reduce them to an alternating zigzag of *significant* swings: a swing of
         the opposite type is accepted only when price has reversed by at least
         atr_mult x ATR from the last accepted swing (a same-type candidate simply
         extends/replaces the running extreme).
      3. Compare consecutive significant highs (bearish) and lows (bullish); mark
         a confirmation at swing_bar + fractal_period when price makes a
         higher-high / lower-low while the indicator diverges.

    min_bar_gap: the two connected swing pivots must be at least this many bars
    apart (the same minimum gap as the fixed-window method). Drops noise-level divergences
    between near-adjacent swings; 0 disables the check.
    """
    n = len(indicator)
    bull_div = np.zeros(n)
    bear_div = np.zeros(n)
    bull_strength = np.zeros(n)
    bear_strength = np.zeros(n)

    if atr is None:
        raise ValueError("method='swing' requires an `atr` array")

    fp = max(1, int(fractal_period))
    if n < 2 * fp + 2:
        return bull_div, bear_div, bull_strength, bear_strength

    # 1. Candidate fractals on price.
    cand: list[tuple[int, str]] = []
    for j in range(fp, n - fp):
        hw = price_high[j - fp: j + fp + 1]
        lw = price_low[j - fp: j + fp + 1]
        if np.any(np.isnan(hw)) or np.any(np.isnan(lw)):
            continue
        if price_high[j] == np.max(hw):
            cand.append((j, "H"))
        if price_low[j] == np.min(lw):
            cand.append((j, "L"))
    cand.sort(key=lambda x: x[0])

    # 2. ATR-significant alternating zigzag. piv entries: [index, kind, price].
    piv: list[list] = []
    for j, kind in cand:
        a = atr[j]
        price = price_high[j] if kind == "H" else price_low[j]
        if not piv:
            piv.append([j, kind, price])
            continue
        last = piv[-1]
        if kind == last[1]:
            # Same type — keep the more extreme swing as the running pivot.
            if (kind == "H" and price > last[2]) or (kind == "L" and price < last[2]):
                piv[-1] = [j, kind, price]
        else:
            # Opposite type — accept only if the reversal is ATR-significant.
            thr = atr_mult * a if (a is not None and not np.isnan(a) and a > 0) else 0.0
            if abs(price - last[2]) >= thr:
                piv.append([j, kind, price])

    highs = [p[0] for p in piv if p[1] == "H"]
    lows = [p[0] for p in piv if p[1] == "L"]

    # 3a. Regular Bearish: consecutive significant highs, price HH + indicator LH.
    for idx in range(1, len(highs)):
        j, prev_j = highs[idx], highs[idx - 1]
        if (j - prev_j) < min_bar_gap:
            continue
        if np.isnan(indicator[j]) or np.isnan(indicator[prev_j]):
            continue
        if price_high[j] > price_high[prev_j] and indicator[j] < indicator[prev_j]:
            confirm = j + fp
            if confirm < n:
                bear_div[confirm] = 1.0
                bear_strength[confirm] = indicator[prev_j] - indicator[j]

    # 3b. Regular Bullish: consecutive significant lows, price LL + indicator HL.
    for idx in range(1, len(lows)):
        j, prev_j = lows[idx], lows[idx - 1]
        if (j - prev_j) < min_bar_gap:
            continue
        if np.isnan(indicator[j]) or np.isnan(indicator[prev_j]):
            continue
        if price_low[j] < price_low[prev_j] and indicator[j] > indicator[prev_j]:
            confirm = j + fp
            if confirm < n:
                bull_div[confirm] = 1.0
                bull_strength[confirm] = indicator[j] - indicator[prev_j]

    return bull_div, bear_div, bull_strength, bear_strength
