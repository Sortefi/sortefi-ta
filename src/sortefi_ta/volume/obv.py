import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_obv(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """On-Balance Volume. Returns {"obv": ndarray}."""
    n = len(df)
    if ta is None or n < 2:
        return {"obv": _nan_array(n)}
    result = ta.obv(df["close"], df["volume"])
    if result is None:
        return {"obv": _nan_array(n)}
    return {"obv": result.to_numpy(dtype=float)}
