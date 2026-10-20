import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_mom(df: pd.DataFrame, period: int = 10, col: str = "close") -> dict[str, np.ndarray]:
    """Momentum (price difference). Returns {"mom"}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"mom": _nan_array(n)}
    result = ta.mom(df[col], length=period)
    if result is None:
        return {"mom": _nan_array(n)}
    return {"mom": result.to_numpy(dtype=float)}
