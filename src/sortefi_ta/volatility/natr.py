import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_natr(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Normalized ATR = ATR(RMA) / close * 100. Consistent with calc_atr. Returns {"natr"}."""
    from sortefi_ta.volatility.atr import calc_atr
    n = len(df)
    if n < period + 1:
        return {"natr": _nan_array(n)}
    atr = calc_atr(df, period=period)["atr"]
    natr = atr / df["close"].to_numpy(dtype=float) * 100
    return {"natr": natr}
