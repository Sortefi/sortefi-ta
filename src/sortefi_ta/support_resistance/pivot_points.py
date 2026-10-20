import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_pivot_points(df: pd.DataFrame) -> dict[str, float]:
    """
    Classic Pivot Points from last completed candle's H/L/C.
    Returns {"pivot", "r1", "r2", "r3", "s1", "s2", "s3"} as floats.
    """
    nan_result = {k: np.nan for k in ("pivot", "r1", "r2", "r3", "s1", "s2", "s3")}
    if len(df) < 1:
        return nan_result

    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    pivot = (h + l + c) / 3.0
    r1 = 2.0 * pivot - l
    s1 = 2.0 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2.0 * (pivot - l)
    s3 = l - 2.0 * (h - pivot)

    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}
