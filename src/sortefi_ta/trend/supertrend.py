import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> dict[str, np.ndarray]:
    """Supertrend. Returns {"supertrend", "direction"} (1=bull, -1=bear)."""
    n = len(df)
    empty = {"supertrend": _nan_array(n), "direction": _nan_array(n)}
    if ta is None or n < period:
        return empty
    result = ta.supertrend(df["high"], df["low"], df["close"],
                           length=period, multiplier=multiplier)
    if result is None or result.empty:
        return empty
    cols = result.columns.tolist()
    st_col = [c for c in cols if "SUPERT" in c.upper() and "D" not in c.upper().split("_")[-1]]
    d_col = [c for c in cols if c.upper().startswith("SUPERTD")]
    return {
        "supertrend": result[st_col[0]].to_numpy(dtype=float) if st_col else _nan_array(n),
        "direction": result[d_col[0]].to_numpy(dtype=float) if d_col else _nan_array(n),
    }
