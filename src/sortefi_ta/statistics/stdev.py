import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_stdev(df: pd.DataFrame, period: int = 20, col: str = "close") -> dict[str, np.ndarray]:
    """Rolling Standard Deviation (population, ddof=0). Matches TV stdev(). Returns {"stdev"}."""
    n = len(df)
    if n < period:
        return {"stdev": _nan_array(n)}
    result = df[col].rolling(period).std(ddof=0)
    return {"stdev": result.to_numpy(dtype=float)}
