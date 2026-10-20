import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_bop(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Balance of Power. Returns {"bop"}."""
    n = len(df)
    if ta is None or n < 1:
        return {"bop": _nan_array(n)}
    result = ta.bop(df["open"], df["high"], df["low"], df["close"])
    if result is None:
        return {"bop": _nan_array(n)}
    return {"bop": result.to_numpy(dtype=float)}
