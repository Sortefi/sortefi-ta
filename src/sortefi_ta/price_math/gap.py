import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_gap(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Gap between current open and previous close. Returns {"gap", "gap_pct"}."""
    n = len(df)
    if n < 2:
        return {"gap": _nan_array(n), "gap_pct": _nan_array(n)}
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    gap = np.concatenate([[np.nan], o[1:] - c[:-1]])
    gap_pct = np.concatenate([[np.nan],
                              np.where(c[:-1] != 0, (o[1:] - c[:-1]) / c[:-1] * 100.0, np.nan)])
    return {"gap": gap, "gap_pct": gap_pct}
