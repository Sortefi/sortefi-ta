import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_laguerre_rsi(df: pd.DataFrame, gamma: float = 0.5) -> dict[str, np.ndarray]:
    """
    Laguerre RSI — RSI based on Laguerre transform filter.

    More responsive to price changes than standard RSI due to shorter
    effective data length. Uses four cascaded Laguerre filter stages.

    Args:
        df:    DataFrame with 'close' column.
        gamma: Damping factor (0.0–1.0). Higher = smoother/slower.
               Default 0.5. Common values: 0.5, 0.7, 0.75.

    Returns:
        lrsi — Laguerre RSI values in [0, 1] range (ndarray).
               0.8+ = overbought, 0.2- = oversold.
    """
    n = len(df)
    if n < 1:
        return {"lrsi": _nan_array(n)}

    close = df["close"].to_numpy(dtype=float)
    lrsi = np.full(n, np.nan)

    L0 = 0.0
    L1 = 0.0
    L2 = 0.0
    L3 = 0.0

    g = gamma

    for i in range(n):
        c = close[i]
        if not np.isfinite(c):
            continue

        prev_L0 = L0
        prev_L1 = L1
        prev_L2 = L2

        L0 = (1 - g) * c + g * L0
        L1 = -g * L0 + prev_L0 + g * L1
        L2 = -g * L1 + prev_L1 + g * L2
        L3 = -g * L2 + prev_L2 + g * L3

        cu = 0.0
        cd = 0.0
        if L0 >= L1:
            cu += L0 - L1
        else:
            cd += L1 - L0
        if L1 >= L2:
            cu += L1 - L2
        else:
            cd += L2 - L1
        if L2 >= L3:
            cu += L2 - L3
        else:
            cd += L3 - L2

        denom = cu + cd
        lrsi[i] = cu / denom if denom != 0 else 0.0

    return {"lrsi": lrsi}
