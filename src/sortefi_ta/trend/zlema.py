import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_zlema(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Zero Lag EMA — reduces EMA lag by compensating with price momentum.

    Formula:
      lag    = (period - 1) / 2
      src*   = 2 * src - src[lag]   # lag-adjusted source
      zlema  = EMA(src*, period)

    Returns {"zlema": ndarray}.
    """
    n = len(df)
    if n < period:
        return {"zlema": _nan_array(n)}

    src = df[col].to_numpy(dtype=float)
    lag = int((period - 1) / 2)

    # Build lag-adjusted source
    adjusted = np.full(n, np.nan)
    for i in range(lag, n):
        if not np.isnan(src[i]) and not np.isnan(src[i - lag]):
            adjusted[i] = 2 * src[i] - src[i - lag]

    # EMA on adjusted source
    k = 2.0 / (period + 1)
    out = np.full(n, np.nan)

    # Seed with first valid adjusted value
    for i in range(n):
        if not np.isnan(adjusted[i]):
            out[i] = adjusted[i]
            for j in range(i + 1, n):
                if np.isnan(adjusted[j]):
                    out[j] = out[j - 1]
                else:
                    out[j] = adjusted[j] * k + out[j - 1] * (1 - k)
            break

    return {"zlema": out}
