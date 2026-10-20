import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_linreg(
    df: pd.DataFrame, period: int = 14, col: str = "close"
) -> dict[str, np.ndarray]:
    """Linear Regression Value. Returns {"linreg", "linreg_slope", "linreg_angle", "linreg_intercept"}."""
    n = len(df)
    empty = {"linreg": _nan_array(n), "linreg_slope": _nan_array(n),
             "linreg_angle": _nan_array(n), "linreg_intercept": _nan_array(n)}
    if ta is None or n < period:
        return empty
    val = ta.linreg(df[col], length=period)
    slope = ta.linreg(df[col], length=period, slope=True)
    angle = ta.linreg(df[col], length=period, angle=True)
    intercept = ta.linreg(df[col], length=period, intercept=True)
    return {
        "linreg": val.to_numpy(dtype=float) if val is not None else _nan_array(n),
        "linreg_slope": slope.to_numpy(dtype=float) if slope is not None else _nan_array(n),
        "linreg_angle": angle.to_numpy(dtype=float) if angle is not None else _nan_array(n),
        "linreg_intercept": intercept.to_numpy(dtype=float) if intercept is not None else _nan_array(n),
    }
