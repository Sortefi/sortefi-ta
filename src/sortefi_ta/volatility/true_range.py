import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_true_range(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """True Range (single-bar). Returns {"true_range"}."""
    n = len(df)
    if ta is None or n < 2:
        return {"true_range": _nan_array(n)}
    result = ta.true_range(df["high"], df["low"], df["close"])
    if result is None:
        return {"true_range": _nan_array(n)}
    return {"true_range": result.to_numpy(dtype=float)}
