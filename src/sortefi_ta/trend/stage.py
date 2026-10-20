import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array
from sortefi_ta.trend.sma import calc_sma


def calc_stage(
    df: pd.DataFrame,
    ma_period: int = 200,
    slope_period: int = 20,
    flat_pct: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Weinstein Stage Analysis — categorical 1–4 classifier.

      Stage 1  Basing / accumulation  — flat MA after a decline
      Stage 2  Advancing / markup     — price above a RISING MA
      Stage 3  Topping / distribution — flat MA after an advance
      Stage 4  Declining / markdown   — price below a FALLING MA

    Method (per-bar, stateful):
      - MA = SMA(close, ma_period).
      - slope% = (MA[i] / MA[i-slope_period] - 1) * 100. |slope%| < flat_pct → "flat".
      - Stage 2 = rising MA and close above it; Stage 4 = falling MA and close below it.
      - When neither trend holds (transitional), the stage is decided by the most
        recent *directional* stage: after a Stage 2 you are Topping (3); after a
        Stage 4 you are Basing (1). Before any trend exists, price vs MA seeds it
        (below → 1, above → 3).

    Returns {"stage": ndarray} of floats in {1,2,3,4}; NaN until the MA is valid.
    [-1] = current stage. Reuses calc_sma; no new low-level math.
    """
    n = len(df)
    stage = _nan_array(n)
    if n < ma_period + slope_period:
        return {"stage": stage}

    close = df["close"].to_numpy(dtype=float)
    ma = calc_sma(df, period=ma_period)["sma"]

    last_dir = 0  # most recent directional stage: 2 (up) or 4 (down)
    for i in range(n):
        m = ma[i]
        if not np.isfinite(m) or i < slope_period or not np.isfinite(ma[i - slope_period]) or ma[i - slope_period] == 0:
            continue
        slope = (m / ma[i - slope_period] - 1.0) * 100.0
        above = close[i] > m
        rising = slope > flat_pct
        falling = slope < -flat_pct

        if rising and above:
            s = 2
            last_dir = 2
        elif falling and not above:
            s = 4
            last_dir = 4
        elif last_dir == 2:
            s = 3
        elif last_dir == 4:
            s = 1
        else:
            s = 3 if above else 1
        stage[i] = float(s)

    return {"stage": stage}
