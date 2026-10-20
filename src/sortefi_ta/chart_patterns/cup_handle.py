import numpy as np
import pandas as pd


def calc_cup_handle(
    df: pd.DataFrame,
    cup_min_bars: int = 25,
    cup_max_bars: int = 130,
    depth_min_pct: float = 12.0,
    depth_max_pct: float = 35.0,
    rim_tolerance_pct: float = 6.0,
    handle_bars: int = 15,
    handle_max_pct: float = 12.0,
) -> dict[str, np.ndarray]:
    """
    Cup-with-Handle (O'Neil) — a rounded base (the cup) followed by a short,
    shallow pullback (the handle) that sets up a breakout above the rim.

    Heuristic structural detection at bar i:
      - Cup region = the window ending `handle_bars` ago, spanning cup_min..cup_max.
      - Left rim  = the high early in the cup region; Right rim = the high late in
        the cup region. Both rims must sit within `rim_tolerance_pct` of each other
        (a level base, not a descending one).
      - Cup low   = the lowest low of the cup region; depth (rim→low) must be
        `depth_min_pct`..`depth_max_pct`, and the low must fall in the middle of the
        region (a rounded U, not a V at either edge).
      - Handle    = the last `handle_bars`: a shallow pullback (depth <=
        `handle_max_pct`) that holds in the upper half of the cup, current close
        still below the rim.

    - cup_handle: 1.0 at bars where an intact cup-with-handle is present.

    Returns arrays where [-1] = current bar. Thresholds are documented defaults;
    validate against known setups before relying on output.
    """
    n = len(df)
    trig = np.zeros(n)
    if n < cup_min_bars + handle_bars:
        return {"cup_handle": trig}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    for i in range(cup_min_bars + handle_bars, n):
        cup_end = i - handle_bars
        cup_start = max(0, i - cup_max_bars)
        cup_len = cup_end - cup_start
        if cup_len < cup_min_bars:
            continue

        seg_hi = high[cup_start : cup_end + 1]
        seg_lo = low[cup_start : cup_end + 1]
        if not (np.all(np.isfinite(seg_hi)) and np.all(np.isfinite(seg_lo))):
            continue

        q = max(2, cup_len // 4)
        left_rim = seg_hi[:q].max()
        right_rim = seg_hi[-q:].max()
        if left_rim <= 0 or right_rim <= 0:
            continue
        # rims level with each other
        if abs(right_rim - left_rim) / left_rim * 100.0 > rim_tolerance_pct:
            continue

        rim = min(left_rim, right_rim)
        cup_low_i = cup_start + int(np.argmin(seg_lo))
        cup_low = seg_lo.min()
        depth = (rim - cup_low) / rim * 100.0
        if not (depth_min_pct <= depth <= depth_max_pct):
            continue
        # roundedness: low sits in the middle 25%-75% of the cup, not at an edge
        rel = (cup_low_i - cup_start) / cup_len
        if not (0.25 <= rel <= 0.75):
            continue

        # Handle: shallow pullback in the upper half of the cup.
        h_hi = high[cup_end + 1 : i + 1].max()
        h_lo = low[cup_end + 1 : i + 1].min()
        if not (np.isfinite(h_hi) and np.isfinite(h_lo)) or h_hi <= 0:
            continue
        if (h_hi - h_lo) / h_hi * 100.0 > handle_max_pct:
            continue
        if h_lo < cup_low + depth / 100.0 * rim * 0.5:  # handle must stay in upper half
            continue
        if close[i] > rim:  # not yet broken out (still in the handle)
            continue

        trig[i] = 1.0

    return {"cup_handle": trig}
