import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_entropy(df: pd.DataFrame, period: int = 10, col: str = "close") -> dict[str, np.ndarray]:
    """Rolling Shannon Entropy: -sum(p * log2(p)), p = close / sum(close). Returns {"entropy"}."""
    n = len(df)
    if n < period:
        return {"entropy": _nan_array(n)}

    def _entropy(window: np.ndarray) -> float:
        p = window / window.sum()
        return float(-np.sum(p * np.log2(p + 1e-10)))

    result = df[col].rolling(period).apply(_entropy, raw=True)
    return {"entropy": result.to_numpy(dtype=float)}
