import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array
from sortefi_ta.support_resistance.fractals import calc_fractals


def calc_sfp(df: pd.DataFrame, range_period: int = 0, fractal_period: int = 2) -> dict:
    """
    Swing Failure Pattern (SFP) — failed sweep of a prior swing high/low.

    Uses Williams Fractals to identify confirmed swing points within the range,
    then scans for the 3-candle SFP pattern that sweeps and rejects the level.

    For each fractal swing high in the range, check if any subsequent 3-candle
    pattern sweeps it (wick above, close below). Similarly for swing lows.

    Bearish SFP at swing high X:
        candle 1: close < X
        candle 2: high > X AND close < X  (sweep + rejection)
        candle 3: close < X               (confirmation)

    Bullish SFP at swing low Y:
        candle 1: close > Y
        candle 2: low < Y AND close > Y   (sweep + rejection)
        candle 3: close > Y               (confirmation)

    Parameters:
        range_period   — number of bars to look back for swing levels (REQUIRED).
                         If 0, returns empty arrays.
        fractal_period — Williams Fractal period for swing detection (default 2).

    Returns:
        bear_sfp         — 1.0 at candle 3 (confirmation bar) of a confirmed bearish SFP
        bull_sfp         — 1.0 at candle 3 of a confirmed bullish SFP
        bear_sfp_forming — 1.0 at current bar if bearish SFP is forming (candles 1+2 valid)
        bull_sfp_forming — 1.0 at current bar if bullish SFP is forming
        swing_high       — the swing high level that was swept (float, most recent SFP)
        swing_low        — the swing low level that was swept (float, most recent SFP)
    """
    n = len(df)
    empty = {
        "bear_sfp": np.zeros(n),
        "bull_sfp": np.zeros(n),
        "bear_sfp_forming": np.zeros(n),
        "bull_sfp_forming": np.zeros(n),
        "swing_high": np.nan,
        "swing_low": np.nan,
    }

    if range_period <= 0 or n < range_period or n < 5:
        return empty

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # Get fractal swing points
    frac = calc_fractals(df, period=fractal_period)
    up_frac = frac["up_fractal"]
    down_frac = frac["down_fractal"]

    range_start = n - range_period

    bear_sfp = np.zeros(n)
    bull_sfp = np.zeros(n)
    bear_sfp_forming = np.zeros(n)
    bull_sfp_forming = np.zeros(n)
    last_bear_level = np.nan
    last_bull_level = np.nan

    # --- Bearish SFP: for each fractal swing high, look for failed sweep ---
    for sh_idx in range(range_start, n):
        if up_frac[sh_idx] != 1.0:
            continue
        level = float(high[sh_idx])

        # Scan for 3-candle pattern AFTER this swing high
        # i is candle 3 (confirmation)
        for i in range(sh_idx + 3, n):
            c1 = close[i - 2]
            c2_h = high[i - 1]
            c2_c = close[i - 1]
            c3 = close[i]

            if (c1 < level
                    and c2_h > level
                    and c2_c < level
                    and c3 < level):
                bear_sfp[i] = 1.0
                last_bear_level = level

        # Check forming at current bar
        if n >= 3 and n - 2 > sh_idx + 1:
            c1 = close[n - 3]
            c2_h = high[n - 2]
            c2_c = close[n - 2]
            if (c1 < level
                    and c2_h > level
                    and c2_c < level):
                if bear_sfp[n - 1] != 1.0:
                    bear_sfp_forming[n - 1] = 1.0
                    last_bear_level = level

    # --- Bullish SFP: for each fractal swing low, look for failed sweep ---
    for sl_idx in range(range_start, n):
        if down_frac[sl_idx] != 1.0:
            continue
        level = float(low[sl_idx])

        for i in range(sl_idx + 3, n):
            c1 = close[i - 2]
            c2_l = low[i - 1]
            c2_c = close[i - 1]
            c3 = close[i]

            if (c1 > level
                    and c2_l < level
                    and c2_c > level
                    and c3 > level):
                bull_sfp[i] = 1.0
                last_bull_level = level

        # Check forming
        if n >= 3 and n - 2 > sl_idx + 1:
            c1 = close[n - 3]
            c2_l = low[n - 2]
            c2_c = close[n - 2]
            if (c1 > level
                    and c2_l < level
                    and c2_c > level):
                if bull_sfp[n - 1] != 1.0:
                    bull_sfp_forming[n - 1] = 1.0
                    last_bull_level = level

    return {
        "bear_sfp": bear_sfp,
        "bull_sfp": bull_sfp,
        "bear_sfp_forming": bear_sfp_forming,
        "bull_sfp_forming": bull_sfp_forming,
        "swing_high": last_bear_level,
        "swing_low": last_bull_level,
    }
