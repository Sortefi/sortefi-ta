import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_ao(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> dict[str, np.ndarray]:
    """Awesome Oscillator. Returns {"ao"}."""
    n = len(df)
    if ta is None or n < slow:
        return {"ao": _nan_array(n)}
    result = ta.ao(df["high"], df["low"], fast=fast, slow=slow)
    if result is None:
        return {"ao": _nan_array(n)}
    return {"ao": result.to_numpy(dtype=float)}
