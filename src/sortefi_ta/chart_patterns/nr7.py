import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_nr7(df: pd.DataFrame, period: int = 7) -> dict[str, np.ndarray]:
    """
    NR7 (Narrowest Range in N) + Inside-Day compression triggers.

    - nr7:        1.0 where the bar's high-low range is the narrowest of the
                  trailing `period` bars (default 7 → classic NR7). A volatility
                  contraction that often precedes an expansion move.
    - inside_bar: 1.0 where the bar is fully inside the prior bar
                  (high <= prior high AND low >= prior low).
    - signal:     1.0 where the bar is BOTH NR7 and an inside bar (strongest
                  compression trigger).

    Returns arrays where [-1] = current bar. 0.0 elsewhere; leading bars 0.0.
    """
    n = len(df)
    nr = np.zeros(n)
    inside = np.zeros(n)
    signal = np.zeros(n)
    if n < 2:
        return {"nr7": nr, "inside_bar": inside, "signal": signal}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    rng = high - low

    for i in range(1, n):
        if high[i] <= high[i - 1] and low[i] >= low[i - 1]:
            inside[i] = 1.0
        if i >= period - 1:
            window = rng[i - period + 1 : i + 1]
            if np.all(np.isfinite(window)) and rng[i] == window.min():
                nr[i] = 1.0
        if nr[i] and inside[i]:
            signal[i] = 1.0

    return {"nr7": nr, "inside_bar": inside, "signal": signal}
