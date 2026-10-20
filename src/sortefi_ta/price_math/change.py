import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_change(
    df: pd.DataFrame, periods: int = 1, col: str = "close"
) -> dict[str, np.ndarray]:
    """Close-to-close change (absolute + %). Returns {"change", "change_pct"}."""
    n = len(df)
    if n < periods + 1:
        return {"change": _nan_array(n), "change_pct": _nan_array(n)}
    vals = df[col].to_numpy(dtype=float)
    change = np.concatenate([_nan_array(periods), vals[periods:] - vals[:-periods]])
    prev = np.concatenate([_nan_array(periods), vals[:-periods]])
    change_pct = np.where(prev != 0, change / prev * 100.0, np.nan)
    return {"change": change, "change_pct": change_pct}
