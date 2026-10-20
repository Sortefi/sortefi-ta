import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_vortex(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Vortex Indicator. Returns {"vortex_pos", "vortex_neg"}."""
    n = len(df)
    empty = {"vortex_pos": _nan_array(n), "vortex_neg": _nan_array(n)}
    if ta is None or n < period + 1:
        return empty
    result = ta.vortex(df["high"], df["low"], df["close"], length=period)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    pos_col = f"VTXP_{period}"
    neg_col = f"VTXM_{period}"
    if pos_col not in result.columns:
        return empty
    return {
        "vortex_pos": result[pos_col].to_numpy(dtype=float),
        "vortex_neg": result[neg_col].to_numpy(dtype=float),
    }
