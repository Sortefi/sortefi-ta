import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_aroon(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """Aroon Up/Down + Oscillator. Returns {"aroon_up", "aroon_down", "aroon_osc"}."""
    n = len(df)
    empty = {"aroon_up": _nan_array(n), "aroon_down": _nan_array(n), "aroon_osc": _nan_array(n)}
    if ta is None or n < period + 1:
        return empty
    result = ta.aroon(df["high"], df["low"], length=period)
    if result is None or result.empty:
        return empty
    up_col  = f"AROONU_{period}"
    dn_col  = f"AROOND_{period}"
    osc_col = f"AROONOSC_{period}"
    if up_col not in result.columns:
        return empty
    up  = result[up_col].to_numpy(dtype=float)
    down = result[dn_col].to_numpy(dtype=float)
    osc  = result[osc_col].to_numpy(dtype=float) if osc_col in result.columns else up - down
    return {"aroon_up": up, "aroon_down": down, "aroon_osc": osc}
