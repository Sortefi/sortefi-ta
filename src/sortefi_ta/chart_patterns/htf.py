import numpy as np
import pandas as pd


def calc_htf(
    df: pd.DataFrame,
    pole_pct: float = 100.0,
    pole_bars: int = 40,
    flag_bars: int = 15,
    max_pullback_pct: float = 25.0,
) -> dict[str, np.ndarray]:
    """
    High-Tight-Flag — a rare, powerful continuation pattern.

    Two parts:
      1. Pole: price roughly DOUBLES (>= `pole_pct`%) within a short span
         (<= `pole_bars` bars), measured low → subsequent high.
      2. Flag: a tight, shallow consolidation since the pole's high — the pullback
         off the high stays within `max_pullback_pct`% over the last `flag_bars`,
         and the current close is still inside that shallow flag.

    - htf: 1.0 at bars where a valid pole is followed by an intact tight flag.

    Returns arrays where [-1] = current bar. Thresholds are defaults from the
    documented pattern; validate against known setups before relying on output.
    """
    n = len(df)
    trig = np.zeros(n)
    need = pole_bars + 2
    if n < need:
        return {"htf": trig}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    for i in range(need, n):
        # Flag = last `flag_bars`; pole must complete just before the flag.
        flag_hi = high[i - flag_bars + 1 : i + 1].max()
        flag_lo = low[i - flag_bars + 1 : i + 1].min()
        if not (np.isfinite(flag_hi) and np.isfinite(flag_lo)) or flag_hi <= 0:
            continue

        # Pole: within the pole_bars window ending at the flag start, low → high doubled.
        p_start = max(0, i - flag_bars - pole_bars + 1)
        p_end = i - flag_bars + 1
        if p_end - p_start < 2:
            continue
        pole_low = low[p_start:p_end].min()
        pole_high = high[p_start : i + 1].max()  # high can occur into the flag start
        if not (np.isfinite(pole_low) and pole_low > 0):
            continue
        gain = (pole_high / pole_low - 1.0) * 100.0
        if gain < pole_pct:
            continue

        # Flag tightness: pullback off the top stays shallow and price still in the flag.
        pullback = (pole_high - flag_lo) / pole_high * 100.0
        if pullback > max_pullback_pct:
            continue
        if close[i] < pole_high * (1.0 - max_pullback_pct / 100.0):
            continue

        trig[i] = 1.0

    return {"htf": trig}
