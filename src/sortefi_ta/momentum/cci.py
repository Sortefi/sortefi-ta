import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_cci(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """CCI matching TradingView's formula exactly.

    TV formula:
        src  = hlc3 = (high + low + close) / 3
        ma   = sma(src, length)
        cci  = (src - ma) / (0.015 * ta.dev(src, length))

    ta.dev() in Pine Script is mean absolute deviation (MAD), not std.
    pandas_ta's CCI has a parenthesis bug on the non-talib path so we
    implement it directly here.
    """
    n = len(df)
    if n < period:
        return {"cci": _nan_array(n)}

    hlc3 = (df["high"].to_numpy(dtype=float)
            + df["low"].to_numpy(dtype=float)
            + df["close"].to_numpy(dtype=float)) / 3.0

    cci = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = hlc3[i - period + 1: i + 1]
        mean = window.mean()
        mad = np.abs(window - mean).mean()  # mean absolute deviation
        if mad == 0:
            cci[i] = 0.0
        else:
            cci[i] = (hlc3[i] - mean) / (0.015 * mad)

    return {"cci": cci}
