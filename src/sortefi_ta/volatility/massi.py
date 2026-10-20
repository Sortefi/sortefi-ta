import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_massi(df: pd.DataFrame, fast: int = 9, slow: int = 10) -> dict[str, np.ndarray]:
    """Mass Index. TV default: sum(EMA(HL,9)/EMA(EMA(HL,9),9), 10). Returns {"massi"}."""
    n = len(df)
    if ta is None or n < slow + fast:
        return {"massi": _nan_array(n)}
    result = ta.massi(df["high"], df["low"], fast=fast, slow=slow)
    if result is None:
        return {"massi": _nan_array(n)}
    return {"massi": result.to_numpy(dtype=float)}
