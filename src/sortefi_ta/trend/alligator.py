import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_alligator(
    df: pd.DataFrame,
    jaw_length: int = 13,
    teeth_length: int = 8,
    lips_length: int = 5,
    jaw_offset: int = 8,
    teeth_offset: int = 5,
    lips_offset: int = 3,
) -> dict[str, np.ndarray]:
    """
    Williams Alligator — three offset SMMA lines on HL2.

    Matches TradingView's built-in Williams Alligator exactly:
      Jaw   = SMMA(HL2, 13) shifted forward 8 bars  (blue)
      Teeth = SMMA(HL2, 8)  shifted forward 5 bars  (red)
      Lips  = SMMA(HL2, 5)  shifted forward 3 bars  (green)

    Returns:
        jaw   — Jaw line (SMMA 13 of HL2, shifted forward by jaw_offset)
        teeth — Teeth line (SMMA 8 of HL2, shifted forward by teeth_offset)
        lips  — Lips line (SMMA 5 of HL2, shifted forward by lips_offset)
        jaw_raw   — Jaw without offset (for crossover detection at current bar)
        teeth_raw — Teeth without offset
        lips_raw  — Lips without offset
    """
    n = len(df)
    nan = _nan_array(n)
    empty = {
        "jaw": nan.copy(), "teeth": nan.copy(), "lips": nan.copy(),
        "jaw_raw": nan.copy(), "teeth_raw": nan.copy(), "lips_raw": nan.copy(),
    }

    if n < jaw_length:
        return empty

    hl2 = (df["high"].to_numpy(dtype=float) + df["low"].to_numpy(dtype=float)) / 2.0

    def _smma(src: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(src), np.nan)
        seed = np.nanmean(src[:period])
        if np.isnan(seed):
            return out
        out[period - 1] = seed
        for i in range(period, len(src)):
            if np.isnan(src[i]):
                out[i] = out[i - 1]
            else:
                out[i] = (out[i - 1] * (period - 1) + src[i]) / period
        return out

    jaw_raw = _smma(hl2, jaw_length)
    teeth_raw = _smma(hl2, teeth_length)
    lips_raw = _smma(hl2, lips_length)

    def _shift_forward(arr: np.ndarray, offset: int) -> np.ndarray:
        """Shift array forward by offset bars (like TV's plot offset)."""
        if offset <= 0:
            return arr.copy()
        out = np.full(len(arr), np.nan)
        if offset < len(arr):
            out[offset:] = arr[:-offset]
        return out

    return {
        "jaw": _shift_forward(jaw_raw, jaw_offset),
        "teeth": _shift_forward(teeth_raw, teeth_offset),
        "lips": _shift_forward(lips_raw, lips_offset),
        "jaw_raw": jaw_raw,
        "teeth_raw": teeth_raw,
        "lips_raw": lips_raw,
    }
