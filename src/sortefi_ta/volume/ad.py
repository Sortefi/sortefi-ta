import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_ad(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Accumulation/Distribution Line. Returns {"ad": ndarray}."""
    n = len(df)
    if ta is None or n < 2:
        return {"ad": _nan_array(n)}
    result = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    if result is None:
        return {"ad": _nan_array(n)}
    return {"ad": result.to_numpy(dtype=float)}
