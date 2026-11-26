import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array
from sortefi_ta.momentum.rsi import calc_rsi


def calc_stochrsi(
    df: pd.DataFrame, period: int = 14, rsi_period: int = 14, k: int = 3, d: int = 3
) -> dict[str, np.ndarray]:
    """Stochastic RSI, as TradingView draws it. Returns {"stochrsi_k", "stochrsi_d"}.

    RSI(rsi_period) — seeded like TradingView (calc_rsi) → where it sits in its range over the last `period`
    candles (0–100) → %K = SMA(k) of that → %D = SMA(d) of %K; a smoothing length of 1 = no smoothing. A line
    without enough history is blank (NaN). In 0.1.x stochrsi used the old RSI seed (the first change
    alone), so short histories read differently; and the stochastic window was `rsi_period` even when `period`
    was given — it is now `period`, as documented (the same number unless both are stated differently)."""
    n = len(df)
    empty = {"stochrsi_k": _nan_array(n), "stochrsi_d": _nan_array(n)}
    period, rsi_period, k, d = int(period), int(rsi_period), max(int(k), 1), max(int(d), 1)
    if period < 1 or rsi_period < 1 or n < rsi_period + period:
        return empty
    rsi = pd.Series(calc_rsi(df, period=rsi_period)["rsi"])
    lo, hi = rsi.rolling(period).min(), rsi.rolling(period).max()
    rng = hi - lo
    if rng.eq(0).any():                     # a flat RSI range: nudged by machine epsilon (avoids 0 ÷ 0), as before
        rng = rng + np.finfo(float).eps
    stoch = 100 * (rsi - lo) / rng
    k_line = stoch if k == 1 else stoch.rolling(k).mean()
    d_line = k_line if d == 1 else k_line.rolling(d).mean()
    return {"stochrsi_k": k_line.to_numpy(dtype=float), "stochrsi_d": d_line.to_numpy(dtype=float)}
