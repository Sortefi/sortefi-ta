import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_midpoint(
    df: pd.DataFrame, col1: str = "open", col2: str = "close"
) -> dict[str, np.ndarray]:
    """(col1 + col2) / 2 per row. Returns {"midpoint": ndarray}."""
    n = len(df)
    if n < 1:
        return {"midpoint": _nan_array(n)}
    a = df[col1].to_numpy(dtype=float)
    b = df[col2].to_numpy(dtype=float)
    return {"midpoint": (a + b) / 2.0}
