import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_rvol(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """Relative Volume (RVOL) — current volume as a RATIO to its recent average.

    RVOL[i] = volume[i] / mean(volume[i-period : i])   (prior `period` bars,
    EXCLUDING the current bar — the usual prior-N baseline).

    RVOL of 2.0 = twice the average volume; 0.5 = half. RVOL is a ratio, not a
    z-score; for a standardised measure, use a z-score of volume.

    Returns {"rvol": ndarray} with [-1] = current bar. NaN where insufficient
    history, no volume column, or a zero/NaN baseline.
    """
    n = len(df)
    if n == 0 or "volume" not in df.columns:
        return {"rvol": _nan_array(n)}

    s = pd.Series(df["volume"].to_numpy(dtype=float))
    # Prior-N average, excluding the current bar: rolling mean over [i-p+1 .. i]
    # then shift(1) → [i-p .. i-1].
    base = s.rolling(int(period)).mean().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = np.array((s / base).to_numpy(dtype=float), copy=True)
    rvol[~np.isfinite(rvol)] = np.nan
    return {"rvol": rvol}
