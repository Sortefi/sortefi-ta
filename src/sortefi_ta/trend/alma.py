import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_alma(
    df: pd.DataFrame, period: int = 9, sigma: float = 6.0, offset_pct: float = 0.85,
    col: str = "close"
) -> dict[str, np.ndarray]:
    """Arnaud Legoux Moving Average. Returns {"alma"}.

    offset_pct places the centre of the Gaussian weights within the window (0 = oldest candle,
    1 = newest; 0.85 by default)."""
    n = len(df)
    if ta is None or n < period:
        return {"alma": _nan_array(n)}
    result = ta.alma(df[col], length=period, sigma=sigma, dist_offset=float(offset_pct))
    if result is None:
        return {"alma": _nan_array(n)}
    return {"alma": result.to_numpy(dtype=float)}
