import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_cmf(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """Chaikin Money Flow. Returns {"cmf": ndarray}."""
    n = len(df)
    if ta is None or n < period:
        return {"cmf": _nan_array(n)}
    result = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=period)
    if result is None:
        return {"cmf": _nan_array(n)}
    return {"cmf": result.to_numpy(dtype=float)}
