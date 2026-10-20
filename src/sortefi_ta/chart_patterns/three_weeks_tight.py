import numpy as np
import pandas as pd


def calc_three_weeks_tight(
    df: pd.DataFrame,
    min_bars: int = 3,
    max_bars: int = 8,
    tolerance_pct: float = 1.5,
    require_volume_contraction: bool = True,
) -> dict[str, np.ndarray]:
    """
    Three-Weeks-Tight (O'Neil) — a run of consecutive bars closing in a narrow
    band, signalling institutional support / a low-risk add point.

    Original pattern is weekly (each bar = one week); for the classic version run
    it on weekly bars. Generalised here to bars: at bar i, look back for the
    longest run of `min_bars`..`max_bars` consecutive closes that all sit within
    ±`tolerance_pct` of the run's mean close. Optionally require the run's volume
    to be contracting (last-bar volume below the run's first-bar volume).

    - three_weeks_tight: 1.0 at the last bar of a qualifying tight run.

    Returns arrays where [-1] = current bar.
    """
    n = len(df)
    trig = np.zeros(n)
    if n < min_bars:
        return {"three_weeks_tight": trig}

    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else None

    for i in range(min_bars - 1, n):
        # try the longest run first so the tightest/most-complete pattern wins
        for length in range(min(max_bars, i + 1), min_bars - 1, -1):
            seg = close[i - length + 1 : i + 1]
            if not np.all(np.isfinite(seg)):
                continue
            mean = seg.mean()
            if mean <= 0:
                continue
            spread = (np.abs(seg - mean) / mean * 100.0).max()
            if spread > tolerance_pct:
                continue
            if require_volume_contraction and vol is not None:
                vseg = vol[i - length + 1 : i + 1]
                if np.all(np.isfinite(vseg)) and vseg[0] > 0 and vseg[-1] >= vseg[0]:
                    continue
            trig[i] = 1.0
            break

    return {"three_weeks_tight": trig}
