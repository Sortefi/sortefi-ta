"""Renko bricks from closing prices (traditional construction).

A Renko chart ignores time: a new brick is drawn only when the close moves a full box beyond the
last brick. Continuing in the same direction needs one box past the last brick's far edge; a
reversal needs one box past its near edge (two boxes from its far edge).

- `box`: brick size in price units. When None, the box is the ATR(`atr_period`) on the last
  candle — the same box for the whole series, as charting platforms do for an "ATR" box.
- The first brick is anchored at the first close.
- One candle can make several bricks; bricks are counted on the candle whose close made them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sortefi_ta._common import _nan_array
from sortefi_ta.volatility.atr import calc_atr

_EPS = 1e-9   # a close landing exactly on a box edge (in box units) counts as reaching it


def _box(df: pd.DataFrame, box, atr_period: int) -> float:
    if box is not None:
        b = float(box)
    else:
        atr = calc_atr(df, period=atr_period)["atr"]
        atr = atr[np.isfinite(atr)]
        b = float(atr[-1]) if atr.size else float("nan")
    return b if np.isfinite(b) and b > 0 else float("nan")


def _walk(closes: np.ndarray, box: float):
    """Yield (candle index, brick open, brick close, direction) for every brick, oldest first."""
    start = next((i for i, x in enumerate(closes) if np.isfinite(x)), None)
    if start is None:
        return
    lo = hi = float(closes[start])
    for i in range(start + 1, len(closes)):
        x = closes[i]
        if not np.isfinite(x):
            continue
        while True:
            if (x - hi) / box >= 1.0 - _EPS:          # one box above the last brick's top
                yield i, hi, hi + box, 1
                lo, hi = hi, hi + box
            elif (lo - x) / box >= 1.0 - _EPS:        # one box below the last brick's bottom
                yield i, lo, lo - box, -1
                hi, lo = lo, lo - box
            else:
                break


def renko_bricks(df: pd.DataFrame, box: float | None = None, atr_period: int = 14) -> pd.DataFrame:
    """Every brick, oldest first: columns candle (the row position whose close made it), open, close,
    direction (+1 up / -1 down). Empty when there is no valid box or no brick."""
    b = _box(df, box, atr_period)
    cols = ["candle", "open", "close", "direction"]
    if not np.isfinite(b):
        return pd.DataFrame(columns=cols)
    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(list(_walk(closes, b)), columns=cols)


def calc_renko(df: pd.DataFrame, box: float | None = None, atr_period: int = 14) -> dict[str, np.ndarray]:
    """Renko state per candle ([-1] = the latest), after that candle's close.

    Returns {"renko_dir": direction of the latest brick (+1 / -1; 0 before the first brick),
             "renko_new": signed number of bricks this candle made (0 = none),
             "renko_level": close of the latest brick (NaN before the first brick),
             "renko_run": bricks in a row in the current direction,
             "box": the brick size (the same value on every candle)}."""
    n = len(df)
    out = {"renko_dir": np.zeros(n), "renko_new": np.zeros(n), "renko_level": _nan_array(n),
           "renko_run": np.zeros(n), "box": _nan_array(n)}
    b = _box(df, box, atr_period)
    if not np.isfinite(b) or n == 0:
        return out
    out["box"][:] = b
    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    direction, level, run = 0, np.nan, 0
    bricks = list(_walk(closes, b))
    j = 0
    for i in range(n):
        while j < len(bricks) and bricks[j][0] == i:
            _, _, close, d = bricks[j]
            run = run + 1 if d == direction else 1
            direction, level = d, close
            out["renko_new"][i] += d
            j += 1
        out["renko_dir"][i], out["renko_level"][i], out["renko_run"][i] = direction, level, run
    return out
