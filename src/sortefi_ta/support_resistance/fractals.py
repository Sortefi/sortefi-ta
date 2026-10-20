import numpy as np
import pandas as pd


def calc_fractals(df: pd.DataFrame, period: int = 2) -> dict:
    """
    Williams Fractals — pivot high/low detection matching TradingView's built-in.

    An up fractal at bar i means high[i] is the local maximum:
      - All `period` bars before have strictly lower highs
      - Bars after have lower highs (allows up to 4 equal highs immediately
        after the peak before requiring strictly lower — matches TV behavior)

    A down fractal at bar i means low[i] is the local minimum:
      - All `period` bars before have strictly higher lows
      - Bars after have higher lows (same equal-bar tolerance)

    Returns:
        up_fractal   — 1.0 at bars where an up fractal (swing high) is confirmed
        down_fractal — 1.0 at bars where a down fractal (swing low) is confirmed
        up_fractal_price   — price (high) at most recent up fractal, NaN if none
        down_fractal_price — price (low) at most recent down fractal, NaN if none
    """
    period = max(period, 2)
    n = len(df)
    up_fractal = np.zeros(n)
    down_fractal = np.zeros(n)

    if n < 2 * period + 1:
        return {
            "up_fractal": up_fractal,
            "down_fractal": down_fractal,
            "up_fractal_price": np.nan,
            "down_fractal_price": np.nan,
        }

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    # We need `period` bars before and enough bars after to confirm.
    # TV allows up to `period + 4` bars after (equal-bar tolerance up to 4).
    # We can only confirm fractals where we have enough future bars.
    max_equal = 4  # TV allows up to 4 equal bars after the peak

    for center in range(period, n):
        # --- Up fractal (swing high) ---
        # Check left side: all period bars before must have strictly lower highs
        left_ok = True
        for i in range(1, period + 1):
            if high[center - i] >= high[center]:
                left_ok = False
                break

        if left_ok:
            # Check right side with equal-bar tolerance (TV logic)
            # Try frontiers 0..4: allow 0..k equal highs immediately after,
            # then require `period` strictly lower highs after those
            right_ok = False
            for k in range(max_equal + 1):
                # Need center + k + period bars to exist
                if center + k + period >= n:
                    break
                # Check k equal bars immediately after
                eq_ok = True
                for j in range(1, k + 1):
                    if high[center + j] > high[center]:
                        eq_ok = False
                        break
                if not eq_ok:
                    continue
                # Check period strictly lower bars after the equal stretch
                strict_ok = True
                for j in range(1, period + 1):
                    if high[center + k + j] >= high[center]:
                        strict_ok = False
                        break
                if strict_ok:
                    right_ok = True
                    break
            if right_ok:
                up_fractal[center] = 1.0

        # --- Down fractal (swing low) ---
        left_ok = True
        for i in range(1, period + 1):
            if low[center - i] <= low[center]:
                left_ok = False
                break

        if left_ok:
            right_ok = False
            for k in range(max_equal + 1):
                if center + k + period >= n:
                    break
                eq_ok = True
                for j in range(1, k + 1):
                    if low[center + j] < low[center]:
                        eq_ok = False
                        break
                if not eq_ok:
                    continue
                strict_ok = True
                for j in range(1, period + 1):
                    if low[center + k + j] <= low[center]:
                        strict_ok = False
                        break
                if strict_ok:
                    right_ok = True
                    break
            if right_ok:
                down_fractal[center] = 1.0

    # Most recent fractal prices
    up_indices = np.where(up_fractal == 1.0)[0]
    up_price = float(high[up_indices[-1]]) if len(up_indices) > 0 else np.nan

    down_indices = np.where(down_fractal == 1.0)[0]
    down_price = float(low[down_indices[-1]]) if len(down_indices) > 0 else np.nan

    return {
        "up_fractal": up_fractal,
        "down_fractal": down_fractal,
        "up_fractal_price": up_price,
        "down_fractal_price": down_price,
    }
