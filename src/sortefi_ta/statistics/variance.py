import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_variance(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Rolling Variance (population, ddof=0). Matches TV variance(). Returns {"variance"}."""
    n = len(df)
    if n < period:
        return {"variance": _nan_array(n)}
    result = df[col].rolling(period).var(ddof=0)
    return {"variance": result.to_numpy(dtype=float)}
