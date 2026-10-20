import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_eom(df: pd.DataFrame, period: int = 14, divisor: float = 10000.0) -> dict[str, np.ndarray]:
    """
    Ease of Movement. TV formula: SMA(period, divisor * change(hl2) * (high-low) / volume).
    Default divisor=10000 matches TradingView's built-in EOM indicator.
    Returns {"eom": ndarray}.
    """
    n = len(df)
    if n < period + 1:
        return {"eom": _nan_array(n)}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    hl2 = (high + low) / 2.0
    hl2_change = np.diff(hl2, prepend=hl2[0])
    hl2_change[0] = 0.0

    raw = np.where(vol > 0, divisor * hl2_change * (high - low) / vol, 0.0)

    eom = _nan_array(n)
    for i in range(period, n + 1):
        eom[i - 1] = float(np.mean(raw[i - period:i]))

    return {"eom": eom}
