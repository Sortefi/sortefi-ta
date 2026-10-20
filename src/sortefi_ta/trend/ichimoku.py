import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_ichimoku(
    df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52,
    displacement: int = 26,
) -> dict[str, np.ndarray]:
    """Ichimoku Cloud.
    Returns {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
             "senkou_a_future", "senkou_b_future"}.

    Computed directly from OHLC to match TradingView exactly:
      tenkan           = (highest_high_9  + lowest_low_9)  / 2
      kijun            = (highest_high_26 + lowest_low_26) / 2
      senkou_a         = (tenkan + kijun) / 2,  shifted forward (displacement-1) bars
      senkou_b         = (highest_high_52 + lowest_low_52) / 2,  shifted forward (displacement-1) bars
      chikou           = close shifted back displacement bars
      senkou_a_future  = unshifted Span A (projects displacement bars forward from now)
      senkou_b_future  = unshifted Span B (projects displacement bars forward from now)

    senkou_a[-1] / senkou_b[-1]  → currently visible cloud (what TV shows now).
    senkou_a_future[-1] > senkou_b_future[-1]  → future cloud turns bullish.
    """
    n = len(df)
    empty = {
        "tenkan": _nan_array(n), "kijun": _nan_array(n),
        "senkou_a": _nan_array(n), "senkou_b": _nan_array(n),
        "chikou": _nan_array(n),
        "senkou_a_future": _nan_array(n), "senkou_b_future": _nan_array(n),
    }
    if n < senkou:
        return empty

    high  = pd.Series(df["high"].values,  dtype=float)
    low   = pd.Series(df["low"].values,   dtype=float)
    close = pd.Series(df["close"].values, dtype=float)

    # Tenkan-sen and Kijun-sen: donchian midpoints
    tenkan_s = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_s  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2

    # Senkou Span A (unshifted): (tenkan + kijun) / 2
    sa_raw = ((tenkan_s + kijun_s) / 2).to_numpy(dtype=float)

    # Senkou Span B (unshifted): donchian(52)
    sb_raw = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).to_numpy(dtype=float)

    # Chikou Span: close shifted back displacement bars
    chikou = close.shift(displacement).to_numpy(dtype=float)

    # Apply TradingView's offset = displacement-1 forward shift.
    # Prepend NaN so that [-1] = value calculated (displacement-1) bars ago = current visible cloud.
    disp = displacement - 1  # matches Pine Script `offset = displacement - 1`
    if n > disp:
        senkou_a = np.concatenate([_nan_array(disp), sa_raw[:-disp]])
        senkou_b = np.concatenate([_nan_array(disp), sb_raw[:-disp]])
    else:
        senkou_a = _nan_array(n)
        senkou_b = _nan_array(n)

    return {
        "tenkan":           tenkan_s.to_numpy(dtype=float),
        "kijun":            kijun_s.to_numpy(dtype=float),
        "senkou_a":         senkou_a,
        "senkou_b":         senkou_b,
        "chikou":           chikou,
        "senkou_a_future":  sa_raw,
        "senkou_b_future":  sb_raw,
    }
