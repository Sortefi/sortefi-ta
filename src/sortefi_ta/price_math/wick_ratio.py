import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_wick_ratio(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Upper/lower wick size relative to body.
    Returns {"upper_wick", "lower_wick", "wick_ratio"}."""
    n = len(df)
    if n < 1:
        return {"upper_wick": _nan_array(n), "lower_wick": _nan_array(n),
                "wick_ratio": _nan_array(n)}
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    body = body_top - body_bot

    upper_wick = h - body_top
    lower_wick = body_bot - l
    total_wick = upper_wick + lower_wick
    wick_ratio = np.where(body > 0, total_wick / body, np.nan)

    return {"upper_wick": upper_wick, "lower_wick": lower_wick, "wick_ratio": wick_ratio}
