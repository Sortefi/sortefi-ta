"""
calc_equal_levels: Detect "equal highs" / "equal lows" (multiple candle wicks
parked at ~the same price) and whether that level is still UNTESTED.

Only untested levels are returned; levels price has already traded back through
are dropped.

Definition — pure wick-extreme clustering, NO swing/pivot detection:
  - Equal highs: >= `min_touches` candles whose HIGHS fall within `tolerance_pct`
                 of each other. The shared price is a resistance level.
  - Equal lows:  same, on candle LOWS -> a support level.

Untested = after the MOST RECENT touching candle, price has not come WITHIN
`untested_buffer_pct` of the level — measured on the wick extreme, in every aspect
(no wick through it, no close through it, not even an approach inside the buffer):
  - Equal highs: invalidated if any later candle's HIGH reaches level*(1 - buffer).
  - Equal lows:  invalidated if any later candle's LOW  reaches level*(1 + buffer).
buffer = 0 is the strict "never crossed" rule; buffer = 1% (default) additionally
rejects levels price has crept back up/down toward. There is no wick-vs-close mode —
the wick extreme is the single measure, so a level stays untested only if price kept
a clean >= buffer margin from it the whole time since it formed.

Prominence (require_prominence, DEFAULT ON) — the discriminating gate. A cluster of
equal wicks is common mid-trend; what makes it a *tradeable* equal-high/low is that
the level is the CEILING (or FLOOR) of the whole lookback window passed in.
A level qualifies only if nothing in the window traded meaningfully past it:
  - Equal highs: rejected unless max(high over window) <= level*(1 + tolerance).
  - Equal lows:  rejected unless min(low  over window) >= level*(1 - tolerance).
This is what unifies the two shapes both called "equal highs": a flat DISTRIBUTION
SHELF (many near-equal touches parked at the top, then a drop — wick clustering sees
this; swing/fractal detection does NOT, because a flat top has no strictly-lower
neighbours) AND a separated DOUBLE/TRIPLE TOP (distinct pivots at the same price —
prominent by construction). The window is the df you pass in — no fixed lookback.
Set require_prominence=False to also accept equal wicks that are NOT the extreme
of the period.

The level line is the MEAN of the member wick extremes. The band is <= tolerance_pct
wide, so mean vs. edge differ negligibly.
"""

import numpy as np
import pandas as pd


def _cluster_bands(values: np.ndarray, tol: float, min_touches: int) -> list[list[int]]:
    """
    Group indices whose `values` sit within `tol` (fractional) of each other into
    disjoint price bands. Ascending sweep; each value joins the current band only if
    within `tol` of the band's LOWEST member, so band width <= tol (prevents a slow
    ramp from chaining into one giant band).

    Returns a list of member-index lists, each with len >= min_touches.
    """
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values, kind="mergesort")  # stable, NaNs sorted last
    bands: list[list[int]] = []
    cur_members: list[int] = []
    cur_ref = None  # lowest value in the current band
    for idx in order:
        v = float(values[idx])
        if cur_ref is not None and cur_ref > 0 and (v - cur_ref) <= tol * cur_ref:
            cur_members.append(int(idx))
        else:
            if len(cur_members) >= min_touches:
                bands.append(cur_members)
            cur_members = [int(idx)]
            cur_ref = v
    if len(cur_members) >= min_touches:
        bands.append(cur_members)
    return bands


def calc_equal_levels(
    df: pd.DataFrame,
    tolerance_pct: float = 0.5,
    min_touches: int = 2,
    min_separation_hours: float = 0.0,
    untested_buffer_pct: float = 1.0,
    require_prominence: bool = True,
) -> dict:
    """
    Detect untested equal highs / equal lows.

    Args:
        df:            OHLCV DataFrame for the lookback window (already the right TF).
        tolerance_pct: How close two wicks must be to count as "equal", in PERCENT.
                       0.5 => within 0.5%.
        min_touches:   Minimum candles that must share the level (default 2, floor 2).
        min_separation_hours: Require the wicks forming a level to span at least this
                       many hours (first touch -> last touch), measured from open_time.
                       Over long windows most series contain some pair of equal
                       wicks; requiring a minimum separation keeps the result
                       meaningful (e.g. 48 =
                       touches at least two days apart). 0 = no constraint (default).
        untested_buffer_pct: Clean-margin, in PERCENT, that price must keep from the
                       level after it forms for it to count as untested. 1.0 (default)
                       => a later high within 1% below an equal-high level (or a later
                       low within 1% above an equal-low level) invalidates it. 0 = the
                       strict "never crossed" rule. Measured on the wick extreme; there
                       is no wick/close mode.
        require_prominence: When True (default), a level qualifies only if it is the
                       extreme of the window within `tolerance` — the highest high (for
                       equal highs) / lowest low (for equal lows) of the df passed in. This
                       is the primary discriminator that turns a common wick-cluster into
                       a real untested range high/low, and it catches both a flat
                       distribution shelf and a separated double top. Set False to also
                       accept equal wicks inside a range.

    Returns dict:
        eq_high_untested   1.0 if >= 1 untested equal-HIGH level exists, else 0.0
        eq_high_level      price of the MOST RECENT untested equal-high level (NaN if none)
        eq_high_touches    touch count on that reported level (0 if none)
        eq_high_levels     sorted list[float] of ALL untested equal-high levels
        eq_high_touch_bars np.array, 1.0 at the touching bars of the reported level

        eq_low_untested / eq_low_level / eq_low_touches / eq_low_levels / eq_low_touch_bars
                           mirror, on candle lows (support)
    """
    n = len(df)
    empty = {
        "eq_high_untested": 0.0,
        "eq_high_level": np.nan,
        "eq_high_touches": 0,
        "eq_high_levels": [],
        "eq_high_touch_bars": np.zeros(n),
        "eq_low_untested": 0.0,
        "eq_low_level": np.nan,
        "eq_low_touches": 0,
        "eq_low_levels": [],
        "eq_low_touch_bars": np.zeros(n),
    }

    min_touches = max(int(min_touches), 2)
    if n < min_touches or tolerance_pct <= 0:
        return empty

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    tol = tolerance_pct / 100.0
    buf = max(float(untested_buffer_pct), 0.0) / 100.0

    # open_time for the optional time-based separation constraint between touches.
    ot = None
    if min_separation_hours > 0.0 and "open_time" in df.columns:
        ot = pd.to_datetime(df["open_time"], utc=True).dt.tz_localize(None).to_numpy()

    # Suffix extrema -> O(1) "did price approach the level after bar i" checks.
    # suffix_max_high[i] = max(high[i:]) ; suffix_min_low[i] = min(low[i:]).
    suffix_max_high = np.maximum.accumulate(high[::-1])[::-1]
    suffix_min_low = np.minimum.accumulate(low[::-1])[::-1]

    # Prominence reference: the extreme of the whole window passed in.
    window_max_high = float(np.max(high))
    window_min_low = float(np.min(low))

    def _resolve(values: np.ndarray, bands: list[list[int]], side: str) -> list[dict]:
        """side: 'high' (resistance) or 'low' (support). Returns untested bands only."""
        out = []
        for members in bands:
            # Separation: touches must span >= min_separation_hours (first -> last).
            if ot is not None:
                span_hours = (ot[max(members)] - ot[min(members)]) / np.timedelta64(1, "h")
                if span_hours < min_separation_hours:
                    continue
            anchor = max(members)              # most recent touching candle
            level = float(np.mean(values[members]))
            # Prominence: the level must be the ceiling/floor of the window (within tol),
            # i.e. nothing traded meaningfully past it anywhere in the lookback.
            if require_prominence:
                if side == "high" and window_max_high > level * (1.0 + tol):
                    continue
                if side == "low" and window_min_low < level * (1.0 - tol):
                    continue
            after = anchor + 1
            if after >= n:
                tested = False                 # just formed, nothing after it -> untested
            elif side == "high":
                # invalidated if price later came within `buf` below the level
                tested = suffix_max_high[after] >= level * (1.0 - buf)
            else:  # low
                # invalidated if price later came within `buf` above the level
                tested = suffix_min_low[after] <= level * (1.0 + buf)
            if not tested:
                out.append({"level": level, "anchor": anchor,
                            "touches": len(members), "members": members})
        return out

    high_untested = _resolve(high, _cluster_bands(high, tol, min_touches), "high")
    low_untested = _resolve(low, _cluster_bands(low, tol, min_touches), "low")

    result = dict(empty)

    if high_untested:
        # "most recent" = latest anchor; tie-break on more touches
        primary = max(high_untested, key=lambda d: (d["anchor"], d["touches"]))
        touch_bars = np.zeros(n)
        touch_bars[primary["members"]] = 1.0
        result.update({
            "eq_high_untested": 1.0,
            "eq_high_level": primary["level"],
            "eq_high_touches": primary["touches"],
            "eq_high_levels": sorted(d["level"] for d in high_untested),
            "eq_high_touch_bars": touch_bars,
        })

    if low_untested:
        primary = max(low_untested, key=lambda d: (d["anchor"], d["touches"]))
        touch_bars = np.zeros(n)
        touch_bars[primary["members"]] = 1.0
        result.update({
            "eq_low_untested": 1.0,
            "eq_low_level": primary["level"],
            "eq_low_touches": primary["touches"],
            "eq_low_levels": sorted(d["level"] for d in low_untested),
            "eq_low_touch_bars": touch_bars,
        })

    return result
