import numpy as np
import pandas as pd


# Retracement ratios (0 = swing extreme where the move ended, 1 = the origin),
# plus the two common extension ratios used for continuation targets.
_RATIOS = {
    "fib_0": 0.0,
    "fib_236": 0.236,
    "fib_382": 0.382,
    "fib_500": 0.500,
    "fib_618": 0.618,
    "fib_786": 0.786,
    "fib_1": 1.0,
    "fib_1272": 1.272,
    "fib_1618": 1.618,
}


def calc_fibonacci_levels(
    df: pd.DataFrame, period: int = 100, direction: str = "auto"
) -> dict[str, float]:
    """
    Fibonacci retracement + extension levels over the last `period` bars.

    Levels are anchored to the highest high (HH) and lowest low (LL) of the window.
    `direction` decides which extreme the retracement is measured FROM:
      - "auto" (default): whichever extreme is more recent is the end of the swing.
        HH more recent -> up-swing (retrace down from HH); LL more recent ->
        down-swing (retrace up from LL).
      - "up":   force an up-swing (0% at HH, 100% at LL, extensions below LL).
      - "down": force a down-swing (0% at LL, 100% at HH, extensions above HH).

    Returns {"fib_0","fib_236","fib_382","fib_500","fib_618","fib_786","fib_1",
             "fib_1272","fib_1618"} as floats. fib_0 = 0% (no retracement, at the
    swing extreme where price currently is), fib_1 = 100% (full retrace to origin),
    fib_1272/fib_1618 = extensions beyond the origin (continuation targets).
    """
    nan_result = {k: np.nan for k in _RATIOS}
    if len(df) < 2:
        return nan_result

    lookback = df.tail(period)
    highs = lookback["high"].to_numpy(dtype=float)
    lows = lookback["low"].to_numpy(dtype=float)
    hh = float(highs.max())
    ll = float(lows.min())
    diff = hh - ll

    if diff == 0 or np.isnan(diff):
        return nan_result

    # Which extreme ended the swing? The more recent one.
    idx_hh = int(np.argmax(highs))
    idx_ll = int(np.argmin(lows))
    if direction == "up":
        up_swing = True
    elif direction == "down":
        up_swing = False
    else:  # auto
        up_swing = idx_hh >= idx_ll

    if up_swing:
        # 0% at the high, retrace downward; extensions below the low.
        return {k: hh - r * diff for k, r in _RATIOS.items()}
    else:
        # 0% at the low, retrace upward; extensions above the high.
        return {k: ll + r * diff for k, r in _RATIOS.items()}
