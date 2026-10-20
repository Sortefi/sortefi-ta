import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_efi(df: pd.DataFrame, period: int = 13) -> dict[str, np.ndarray]:
    """Elder Force Index. Returns {"efi"}."""
    n = len(df)
    if ta is None or n < period + 1:
        return {"efi": _nan_array(n)}
    result = ta.efi(df["close"], df["volume"], length=period)
    if result is None:
        return {"efi": _nan_array(n)}
    return {"efi": result.to_numpy(dtype=float)}
