import numpy as np
import pandas as pd


def calc_fvg(
    df: pd.DataFrame,
    min_gap_pct: float = 0.004,
) -> dict:
    """
    Fair Value Gap (FVG) — Smart Money Concepts price imbalance detector.

    A FVG forms when 3 consecutive candles leave a gap that price hasn't
    returned to fill. Only unfilled FVGs are flagged.

    Bullish FVG (candle i):
        - candle[i-2].high < candle[i].low  (gap exists)
        - gap size >= min_gap_pct of candle[i-2].high
        - No subsequent candle's LOW entered below candle[i].low

    Bearish FVG (candle i):
        - candle[i-2].low > candle[i].high  (gap exists)
        - gap size >= min_gap_pct of candle[i].high
        - No subsequent candle's HIGH entered above candle[i].high

    Returns:
        bull_fvg        — 1.0 at bar i where an unfilled bullish FVG formed
        bear_fvg        — 1.0 at bar i where an unfilled bearish FVG formed
        bull_fvg_top    — candle[i].low of most recent unfilled bull FVG (float)
        bull_fvg_bottom — candle[i-2].high of most recent unfilled bull FVG (float)
        bear_fvg_top    — candle[i-2].low of most recent unfilled bear FVG (float)
        bear_fvg_bottom — candle[i].high of most recent unfilled bear FVG (float)
    """
    n = len(df)
    bull_fvg = np.zeros(n)
    bear_fvg = np.zeros(n)
    bull_fvg_top = np.nan
    bull_fvg_bottom = np.nan
    bear_fvg_top = np.nan
    bear_fvg_bottom = np.nan

    if n < 3:
        return {
            "bull_fvg": bull_fvg, "bear_fvg": bear_fvg,
            "bull_fvg_top": bull_fvg_top, "bull_fvg_bottom": bull_fvg_bottom,
            "bear_fvg_top": bear_fvg_top, "bear_fvg_bottom": bear_fvg_bottom,
        }

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    for i in range(2, n):
        # --- Bullish FVG ---
        gap_bull = low[i] - high[i - 2]
        if gap_bull > 0 and gap_bull / high[i - 2] >= min_gap_pct:
            # Check unfilled: no subsequent candle's low went below candle[i].low
            fvg_top = low[i]
            filled = np.any(low[i + 1:] < fvg_top) if i + 1 < n else False
            if not filled:
                bull_fvg[i] = 1.0

        # --- Bearish FVG ---
        gap_bear = low[i - 2] - high[i]
        if gap_bear > 0 and gap_bear / high[i] >= min_gap_pct:
            # Check unfilled: no subsequent candle's high went above candle[i].high
            fvg_bottom = high[i]
            filled = np.any(high[i + 1:] > fvg_bottom) if i + 1 < n else False
            if not filled:
                bear_fvg[i] = 1.0

    # Most recent unfilled FVG levels
    bull_indices = np.where(bull_fvg == 1.0)[0]
    if len(bull_indices) > 0:
        j = bull_indices[-1]
        bull_fvg_top = float(low[j])
        bull_fvg_bottom = float(high[j - 2])

    bear_indices = np.where(bear_fvg == 1.0)[0]
    if len(bear_indices) > 0:
        j = bear_indices[-1]
        bear_fvg_top = float(low[j - 2])
        bear_fvg_bottom = float(high[j])

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bull_fvg_top": bull_fvg_top,
        "bull_fvg_bottom": bull_fvg_bottom,
        "bear_fvg_top": bear_fvg_top,
        "bear_fvg_bottom": bear_fvg_bottom,
    }
