import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_keltner(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0, atr_length: int = 10
) -> dict[str, np.ndarray]:
    """Keltner Channels. Matches TV default: EMA(close, period) ± multiplier * ATR(atr_length, RMA).
    Returns {"kc_upper", "kc_middle", "kc_lower"}."""
    from sortefi_ta.volatility.atr import calc_atr
    n = len(df)
    empty = {"kc_upper": _nan_array(n), "kc_middle": _nan_array(n), "kc_lower": _nan_array(n)}
    if n < max(period, atr_length) + 1:
        return empty
    basis = df["close"].ewm(span=period, adjust=False).mean()
    atr   = pd.Series(calc_atr(df, period=atr_length)["atr"])
    upper = basis + multiplier * atr
    lower = basis - multiplier * atr
    return {
        "kc_upper":  upper.to_numpy(dtype=float),
        "kc_middle": basis.to_numpy(dtype=float),
        "kc_lower":  lower.to_numpy(dtype=float),
    }
