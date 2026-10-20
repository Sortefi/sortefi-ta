import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_body_size(df: pd.DataFrame, pct: bool = True) -> dict[str, np.ndarray]:
    """abs(close - open), optionally as % of open. Returns {"body": ndarray}."""
    n = len(df)
    if n < 1:
        return {"body": _nan_array(n)}
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    body = np.abs(c - o)
    if pct:
        body = np.where(o != 0, body / o * 100.0, np.nan)
    return {"body": body}
