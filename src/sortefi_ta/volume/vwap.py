import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array

# Anchor period aliases — maps user-friendly names to pandas offset aliases
_ANCHOR_MAP = {
    "D": "D", "daily": "D", "day": "D",
    "W": "W", "weekly": "W", "week": "W",
    "M": "MS", "monthly": "MS", "month": "MS",
    "Q": "QS", "quarterly": "QS", "quarter": "QS",
    "Y": "YS", "yearly": "YS", "year": "YS", "annual": "YS",
}


def calc_vwap(df: pd.DataFrame, anchor: str | None = None) -> dict[str, np.ndarray | float]:
    """
    VWAP (Volume Weighted Average Price) with std-dev bands and volume profile.

    Args:
        df:     DataFrame with open_time, high, low, close, volume columns.
        anchor: Reset period — "D", "W", "M", "Q", "Y" (or long forms like
                "daily", "weekly", "monthly", "quarterly", "yearly").
                None = no reset, cumulate over entire window (default).
                Matches TradingView's VWAP anchor behaviour: cumulative sums
                reset at the start of each anchor period.

    Returns:
        vwap          — cumulative VWAP, resetting per anchor period (ndarray)
        vwap_std      — volume-weighted std dev of hl2 around VWAP (ndarray)
        vwap_upper_1 / vwap_lower_1 — VWAP +/- 1 std dev (ndarray)
        vwap_upper_2 / vwap_lower_2 — VWAP +/- 2 std dev (ndarray)
        vwap_upper_3 / vwap_lower_3 — VWAP +/- 3 std dev (ndarray)
        poc           — Point of Control: price level with highest volume (float)
        val           — Value Area Low: lower bound of 70% volume zone (float)
        vah           — Value Area High: upper bound of 70% volume zone (float)

    POC/VAL/VAH are computed over the current (most recent) anchor period only
    when anchor is set, or over the entire window when anchor is None.
    """
    n = len(df)
    nan_result: dict[str, np.ndarray | float] = {
        "vwap": _nan_array(n),
        "vwap_std": _nan_array(n),
        "vwap_upper_1": _nan_array(n),
        "vwap_lower_1": _nan_array(n),
        "vwap_upper_2": _nan_array(n),
        "vwap_lower_2": _nan_array(n),
        "vwap_upper_3": _nan_array(n),
        "vwap_lower_3": _nan_array(n),
        "poc": np.nan,
        "val": np.nan,
        "vah": np.nan,
    }
    if n < 1:
        return nan_result

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    # TV VWAP uses hl2 = (high+low)/2
    hl2 = (high + low) / 2.0

    # ---- Determine reset boundaries ----
    if anchor is not None:
        freq = _ANCHOR_MAP.get(anchor, _ANCHOR_MAP.get(anchor.upper(), None))
        if freq is None:
            freq = _ANCHOR_MAP.get(anchor.lower(), None)
        if freq is None:
            # Unknown anchor — fall back to no reset
            resets = np.zeros(n, dtype=bool)
            resets[0] = True
        else:
            # Detect period boundaries from open_time
            times = pd.to_datetime(df["open_time"])
            period_labels = times.dt.to_period(freq.replace("S", ""))  # MS->M, QS->Q, YS->Y, D->D, W->W
            resets = np.zeros(n, dtype=bool)
            resets[0] = True
            labels_arr = period_labels.values
            for i in range(1, n):
                if labels_arr[i] != labels_arr[i - 1]:
                    resets[i] = True
    else:
        resets = np.zeros(n, dtype=bool)
        resets[0] = True

    # ---- Cumulative sums with resets (matches TV formula exactly) ----
    cum_tp_vol = np.empty(n, dtype=float)
    cum_vol = np.empty(n, dtype=float)
    cum_sq_vol = np.empty(n, dtype=float)

    running_tp_vol = 0.0
    running_vol = 0.0
    running_sq_vol = 0.0

    for i in range(n):
        if resets[i]:
            running_tp_vol = 0.0
            running_vol = 0.0
            running_sq_vol = 0.0

        v = vol[i] if np.isfinite(vol[i]) else 0.0
        h = hl2[i] if np.isfinite(hl2[i]) else 0.0

        running_tp_vol += h * v
        running_vol += v
        running_sq_vol += v * h * h

        cum_tp_vol[i] = running_tp_vol
        cum_vol[i] = running_vol
        cum_sq_vol[i] = running_sq_vol

    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)

    # ---- Volume-weighted std dev: TV formula = sqrt(sum(vol*hl2^2)/sum(vol) - vwap^2) ----
    variance = np.where(cum_vol > 0, np.maximum(cum_sq_vol / cum_vol - vwap ** 2, 0.0), np.nan)
    vwap_std = np.sqrt(variance)

    # ---- VWAP bands ----
    vwap_upper_1 = vwap + vwap_std
    vwap_lower_1 = vwap - vwap_std
    vwap_upper_2 = vwap + 2.0 * vwap_std
    vwap_lower_2 = vwap - 2.0 * vwap_std
    vwap_upper_3 = vwap + 3.0 * vwap_std
    vwap_lower_3 = vwap - 3.0 * vwap_std

    # ---- Volume Profile: POC, VAL, VAH (current anchor period only) ----
    poc_val: float = np.nan
    val_val: float = np.nan
    vah_val: float = np.nan

    # Find start of the last anchor period
    last_reset_idx = 0
    for i in range(n - 1, -1, -1):
        if resets[i]:
            last_reset_idx = i
            break

    period_high = high[last_reset_idx:]
    period_low = low[last_reset_idx:]
    period_vol = vol[last_reset_idx:]
    period_hl2 = hl2[last_reset_idx:]
    period_n = len(period_high)

    price_min = np.nanmin(period_low)
    price_max = np.nanmax(period_high)
    total_vol = np.nansum(period_vol)

    if np.isfinite(price_min) and np.isfinite(price_max) and price_max > price_min and total_vol > 0:
        num_bins = 100
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_volumes = np.zeros(num_bins, dtype=float)

        bin_indices = np.searchsorted(bin_edges[1:], period_hl2, side="left")
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        for i in range(period_n):
            if np.isfinite(period_vol[i]) and np.isfinite(period_hl2[i]):
                bin_volumes[bin_indices[i]] += period_vol[i]

        # POC: midpoint of the bin with highest volume
        poc_bin = int(np.argmax(bin_volumes))
        poc_val = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)

        # Value Area: expand outward from POC until 70% of total volume captured
        target_vol = 0.70 * total_vol
        area_vol = bin_volumes[poc_bin]
        lo_idx = poc_bin
        hi_idx = poc_bin

        while area_vol < target_vol and (lo_idx > 0 or hi_idx < num_bins - 1):
            look_down = bin_volumes[lo_idx - 1] if lo_idx > 0 else -1.0
            look_up = bin_volumes[hi_idx + 1] if hi_idx < num_bins - 1 else -1.0

            if look_down >= look_up:
                lo_idx -= 1
                area_vol += bin_volumes[lo_idx]
            else:
                hi_idx += 1
                area_vol += bin_volumes[hi_idx]

        val_val = float(bin_edges[lo_idx])
        vah_val = float(bin_edges[hi_idx + 1])

    return {
        "vwap": vwap,
        "vwap_std": vwap_std,
        "vwap_upper_1": vwap_upper_1,
        "vwap_lower_1": vwap_lower_1,
        "vwap_upper_2": vwap_upper_2,
        "vwap_lower_2": vwap_lower_2,
        "vwap_upper_3": vwap_upper_3,
        "vwap_lower_3": vwap_lower_3,
        "poc": poc_val,
        "val": val_val,
        "vah": vah_val,
    }


def calc_anchored_vwap(df: pd.DataFrame, anchor) -> dict[str, np.ndarray | float]:
    """
    VWAP anchored to an arbitrary point — an event datetime or a bar index —
    rather than calc_vwap's periodic calendar resets. The standard tool for
    "VWAP since the Fed announcement / earnings / the 52-week low / listing day".

    Args:
        df:     DataFrame with open_time, high, low, close, volume columns
                (single symbol, sorted by open_time).
        anchor: Where the VWAP starts. Accepts:
                - ISO datetime string ("2024-01-31T19:00:00Z") / pd.Timestamp /
                  datetime → first bar with open_time >= anchor
                - int → bar index (e.g. int(np.argmax(highs)) to anchor at a
                  swing high; 0 = full history / listing day)
                Negative ints index from the end, numpy-style.

    Returns:
        Same keys as calc_vwap. Arrays are full df length with NaN before the
        anchor bar ([-1] = current bar, per the calc_* contract); poc/val/vah
        scalars are computed over the anchored window only. Anchor beyond the
        last bar (or unparseable) → all-NaN result.
    """
    n = len(df)
    nan_result = calc_vwap(df.iloc[0:0])  # canonical all-NaN shape, then resize
    for k, v in nan_result.items():
        if isinstance(v, np.ndarray):
            nan_result[k] = _nan_array(n)
    if n < 1:
        return nan_result

    # ---- Resolve the anchor to a positional bar index ----
    if isinstance(anchor, (int, np.integer)) and not isinstance(anchor, bool):
        start = int(anchor) if anchor >= 0 else n + int(anchor)
    else:
        try:
            ts = pd.Timestamp(anchor)
        except (TypeError, ValueError):
            return nan_result
        times = pd.to_datetime(df["open_time"])
        # Align tz-awareness both ways so comparison never raises
        tz = getattr(times.dt, "tz", None)
        if ts.tzinfo is None and tz is not None:
            ts = ts.tz_localize(tz)
        elif ts.tzinfo is not None and tz is None:
            ts = ts.tz_localize(None)
        mask = (times >= ts).to_numpy()
        if not mask.any():
            return nan_result
        start = int(np.argmax(mask))
    if start < 0 or start >= n:
        return nan_result

    # ---- Delegate to the vetted VWAP math over the anchored window ----
    sub = calc_vwap(df.iloc[start:].reset_index(drop=True), anchor=None)
    out: dict[str, np.ndarray | float] = {}
    for k, v in sub.items():
        if isinstance(v, np.ndarray):
            full = _nan_array(n)
            full[start:] = v
            out[k] = full
        else:
            out[k] = v  # poc/val/vah scalars — already anchored-window-only
    return out
