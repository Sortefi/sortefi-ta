import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_adr(df: pd.DataFrame, period: int = 20) -> dict[str, np.ndarray]:
    """
    ADR% — Average Daily Range as a percent of price (Minervini-style).

    ADR% = 100 * rolling_mean(high / low - 1, period)

    A pure intraday high-to-low range measure. Distinct from NATR (calc_natr),
    which is ATR/close and therefore includes overnight gaps via true range.
    ADR% ignores gaps and measures only the average within-bar swing.

    Returns {"adr"} — an array where [-1] = current bar. NaN until `period` bars.
    """
    n = len(df)
    if n < period:
        return {"adr": _nan_array(n)}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = high / low - 1.0
    ratio[~np.isfinite(ratio)] = np.nan  # guard zero/negative lows

    adr = pd.Series(ratio).rolling(window=period).mean().to_numpy() * 100.0
    return {"adr": adr}
