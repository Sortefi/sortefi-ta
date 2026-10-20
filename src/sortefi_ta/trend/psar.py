import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_psar(
    df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2
) -> dict[str, np.ndarray]:
    """Parabolic SAR. Returns {"psar", "psar_direction"} (1=bull, -1=bear)."""
    n = len(df)
    empty = {"psar": _nan_array(n), "psar_direction": _nan_array(n)}
    if ta is None or n < 2:
        return empty
    result = ta.psar(df["high"], df["low"], df["close"], af0=af, af=af, max_af=max_af)
    if result is None or result.empty:
        return empty
    cols = result.columns.tolist()
    # pandas-ta returns PSARl (long), PSARs (short), PSARaf, PSARr
    long_col = [c for c in cols if "PSARl" in c]
    short_col = [c for c in cols if "PSARs" in c]

    psar = _nan_array(n)
    direction = _nan_array(n)
    if long_col and short_col:
        long_arr = result[long_col[0]].to_numpy(dtype=float)
        short_arr = result[short_col[0]].to_numpy(dtype=float)
        for i in range(n):
            if not np.isnan(long_arr[i]):
                psar[i] = long_arr[i]
                direction[i] = 1.0
            elif not np.isnan(short_arr[i]):
                psar[i] = short_arr[i]
                direction[i] = -1.0

    return {"psar": psar, "psar_direction": direction}
