import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array
from sortefi_ta.trend.ema import calc_ema


def calc_first_pullback(
    df: pd.DataFrame,
    ma_period: int = 20,
    trend_lookback: int = 60,
) -> dict[str, np.ndarray]:
    """
    First-Pullback setup — the FIRST pullback to a rising MA after a new trend leg.

    A fresh uptrend leg begins when close crosses ABOVE a rising `ma_period` EMA.
    The first time price then dips back to that MA (bar low <= MA) while the MA is
    still rising — and no earlier touch has occurred since the cross — is the
    first-pullback trigger (a classic lower-risk continuation entry).

    - first_pullback: 1.0 at the bar of the first pullback touch after each new
                      cross-up. Re-arms only after a new cross-up.

    Returns arrays where [-1] = current bar. `trend_lookback` bounds how stale a
    cross-up may be and still count (default 60 bars).
    """
    n = len(df)
    trig = np.zeros(n)
    if n < ma_period + 2:
        return {"first_pullback": trig}

    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ma = calc_ema(df, period=ma_period)["ema"]

    armed = False           # a cross-up happened and we're waiting for the first dip
    cross_bar = -1
    for i in range(1, n):
        if not (np.isfinite(ma[i]) and np.isfinite(ma[i - 1])):
            continue
        rising = ma[i] > ma[i - 1]
        crossed_up = close[i - 1] <= ma[i - 1] and close[i] > ma[i]

        if crossed_up and rising:
            armed = True
            cross_bar = i
            continue

        if armed and rising and (i - cross_bar) <= trend_lookback:
            # first bar since the cross whose low reaches back to the MA
            if low[i] <= ma[i] and close[i] >= ma[i] * 0.97:
                trig[i] = 1.0
                armed = False
        elif armed and (not rising or (i - cross_bar) > trend_lookback):
            armed = False  # trend rolled over / cross went stale before a clean pullback

    return {"first_pullback": trig}
