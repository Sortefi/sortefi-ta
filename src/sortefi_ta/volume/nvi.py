import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_nvi(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Negative Volume Index. TV formula: multiplicative update only on volume-down bars.
    TV: ta.nvi starts at 1.0, script multiplies by 1000 → we start at 1000 directly.
    pandas_ta uses additive formula (NVI += pct*100) — replaced with direct TV formula.
    Returns {"nvi": ndarray}.
    """
    n = len(df)
    if n < 2:
        return {"nvi": _nan_array(n)}

    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    nvi = np.empty(n)
    nvi[0] = 1000.0
    for i in range(1, n):
        if vol[i] < vol[i - 1]:
            nvi[i] = nvi[i - 1] * (1.0 + (close[i] - close[i - 1]) / close[i - 1])
        else:
            nvi[i] = nvi[i - 1]

    return {"nvi": nvi}
