import numpy as np
import pandas as pd
from sortefi_ta.support_resistance.fractals import calc_fractals


def calc_vcp(
    df: pd.DataFrame,
    min_contractions: int = 2,
    max_contractions: int = 6,
    final_contraction_pct: float = 10.0,
    fractal_period: int = 2,
    near_high_pct: float = 10.0,
    require_volume_contraction: bool = True,
) -> dict[str, np.ndarray]:
    """
    VCP — Minervini Volatility Contraction Pattern.

    A base that tightens through a series of successive pullbacks, each shallower
    than the last (e.g. 25% → 12% → 6%), on declining volume, coiling toward a
    pivot near the highs.

    Detection: swing highs/lows are found with `calc_fractals`. A contraction is a
    swing-high → next-swing-low drawdown; its depth = (high - low) / high * 100.
    At bar i the most recent `max_contractions` contractions must:
      - number at least `min_contractions`,
      - be monotonically shrinking (each depth < the previous),
      - end tight (last depth <= `final_contraction_pct`),
      - leave price within `near_high_pct`% of the base's highest swing high,
      - (optional) show contracting volume across the base.

    - vcp: 1.0 at bars where a valid contraction sequence is in force.

    Returns arrays where [-1] = current bar. Heuristic thresholds — validate
    against known setups before relying on output.
    """
    n = len(df)
    trig = np.zeros(n)
    if n < 4 * fractal_period + 3:
        return {"vcp": trig}

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else None

    fr = calc_fractals(df, period=fractal_period)
    up = fr["up_fractal"]
    down = fr["down_fractal"]

    # Ordered list of confirmed pivots: (center_index, price, is_high)
    pivots: list[tuple[int, float, bool]] = []
    for idx in range(n):
        if up[idx] == 1.0:
            pivots.append((idx, high[idx], True))
        if down[idx] == 1.0:
            pivots.append((idx, low[idx], False))
    pivots.sort(key=lambda p: p[0])

    if len(pivots) < 2 * min_contractions:
        return {"vcp": trig}

    for i in range(n):
        # pivots knowable by bar i (fractal confirms `fractal_period` bars after center)
        avail = [p for p in pivots if p[0] + fractal_period <= i]
        if len(avail) < 2:
            continue

        # Build contraction depths from high→(following low) pairs, most recent last.
        depths: list[float] = []
        base_high = 0.0
        j = 0
        seq_highs = []
        # walk pairs
        k = 0
        while k < len(avail) - 1:
            a_idx, a_price, a_high = avail[k]
            b_idx, b_price, b_high = avail[k + 1]
            if a_high and not b_high and a_price > 0:
                depths.append((a_price - b_price) / a_price * 100.0)
                seq_highs.append(a_price)
                k += 2
            else:
                k += 1
        if len(depths) < min_contractions:
            continue

        depths = depths[-max_contractions:]
        seq_highs = seq_highs[-len(depths):]
        base_high = max(seq_highs) if seq_highs else 0.0

        # strictly shrinking contractions
        if any(depths[m] >= depths[m - 1] for m in range(1, len(depths))):
            continue
        if depths[-1] > final_contraction_pct:
            continue
        # price coiled near the base high
        if base_high <= 0 or (base_high - close[i]) / base_high * 100.0 > near_high_pct:
            continue
        # volume contraction across the base (first-third avg vs last-third avg)
        if require_volume_contraction and vol is not None:
            start = avail[max(0, len(avail) - 2 * len(depths))][0]
            vseg = vol[start : i + 1]
            vseg = vseg[np.isfinite(vseg)]
            if len(vseg) >= 6:
                third = len(vseg) // 3
                if third > 0 and vseg[:third].mean() > 0 and vseg[-third:].mean() >= vseg[:third].mean():
                    continue

        trig[i] = 1.0

    return {"vcp": trig}
