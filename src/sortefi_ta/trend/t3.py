import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_t3(
    df: pd.DataFrame, period: int = 5, vfactor: float = 0.7, col: str = "close"
) -> dict[str, np.ndarray]:
    """Tillson T3 Moving Average. Returns {"t3"}."""
    n = len(df)
    if ta is None or n < period * 6:
        return {"t3": _nan_array(n)}
    result = ta.t3(df[col], length=period, a=vfactor)
    if result is None:
        return {"t3": _nan_array(n)}
    return {"t3": result.to_numpy(dtype=float)}
