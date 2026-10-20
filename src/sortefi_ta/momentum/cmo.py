import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_cmo(df: pd.DataFrame, period: int = 9, col: str = "close") -> dict[str, np.ndarray]:
    """Chande Momentum Oscillator matching TradingView's formula exactly.

    TV formula:
        momm = change(close)           # 1-bar diff
        sm1  = sum(max(momm, 0), length)   # sum of positive changes
        sm2  = sum(max(-momm, 0), length)  # sum of abs negative changes
        cmo  = 100 * (sm1 - sm2) / (sm1 + sm2)

    Uses plain rolling sum — NOT Wilder's RMA.
    """
    n = len(df)
    if n < period + 1:
        return {"cmo": _nan_array(n)}

    close = df[col].to_numpy(dtype=float)
    diffs = np.diff(close)  # length n-1

    cmo = np.full(n, np.nan)
    for i in range(period, n):
        window = diffs[i - period: i]  # exactly `period` diffs
        pos = window[window > 0].sum()
        neg = np.abs(window[window < 0]).sum()
        total = pos + neg
        cmo[i] = 100.0 * (pos - neg) / total if total != 0 else 0.0

    return {"cmo": cmo}
