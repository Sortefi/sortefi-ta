import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_kvo(
    df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13
) -> dict[str, np.ndarray]:
    """Klinger Volume Oscillator. Returns {"kvo", "kvo_signal"}."""
    n = len(df)
    empty = {"kvo": _nan_array(n), "kvo_signal": _nan_array(n)}
    if ta is None or n < slow + signal:
        return empty
    result = ta.kvo(df["high"], df["low"], df["close"], df["volume"],
                    fast=fast, slow=slow, signal=signal)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    if isinstance(result, pd.DataFrame):
        return {
            "kvo": result.iloc[:, 0].to_numpy(dtype=float),
            "kvo_signal": result.iloc[:, 1].to_numpy(dtype=float) if result.shape[1] > 1 else _nan_array(n),
        }
    return {"kvo": result.to_numpy(dtype=float), "kvo_signal": _nan_array(n)}
