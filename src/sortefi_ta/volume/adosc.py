import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_adosc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> dict[str, np.ndarray]:
    """Chaikin A/D Oscillator. Returns {"adosc"}."""
    n = len(df)
    if ta is None or n < slow + 1:
        return {"adosc": _nan_array(n)}
    result = ta.adosc(df["high"], df["low"], df["close"], df["volume"], fast=fast, slow=slow)
    if result is None:
        return {"adosc": _nan_array(n)}
    return {"adosc": result.to_numpy(dtype=float)}
