"""
calc_range_state: Classify current price relative to an explicit reference range.

Used for: Monday range analysis, weekly range, session range, any user-defined period range.

The 5 states:
  "above"        — current price > range_high  (sustained breakout, still holding)
  "below"        — current price < range_low   (sustained breakdown, still holding)
  "deviate_high" — observation window wicked above range_high, but current price ≤ range_high
                   (failed breakout)
  "deviate_low"  — observation window wicked below range_low,  but current price ≥ range_low
                   (reclaim / failed breakdown)
  "inside"       — price never broke outside the reference range boundaries
"""

import numpy as np
import pandas as pd


def calc_range_state(
    df: "pd.DataFrame",
    range_high: float,
    range_low: float,
) -> dict:
    """
    Classify price relative to a reference range [range_low, range_high].

    Args:
        df:          OHLCV DataFrame for the **observation window** — the bars that occurred
                     AFTER the reference period ended (e.g. Tuesday-to-now for a Monday range).
                     Must have columns: open_time, high, low, close.
                     May include the reference candle itself if needed — the function only
                     reads the max/min high/low and the final close.
        range_high:  High of the reference period (e.g. Monday's high).
        range_low:   Low  of the reference period (e.g. Monday's low).

    Returns dict with scalar values — NOT arrays (range state is a per-symbol classification):
        state          str   "above" | "below" | "deviate_high" | "deviate_low" | "inside"
        current_price  float Most recent close in obs window.
        period_high    float Highest high across the observation window.
        period_low     float Lowest low  across the observation window.
        range_high     float Reference high (echoed back).
        range_low      float Reference low  (echoed back).
        range_mid      float Midpoint: (range_high + range_low) / 2.
        range_size     float range_high - range_low.
        pct_position   float Where current price sits within range
                             (0.0 = exactly at low, 1.0 = exactly at high; can be <0 or >1).
        pct_from_high  float % distance of current price from range_high
                             (positive = above high, negative = below high).
        pct_from_low   float % distance of current price from range_low
                             (positive = above low, negative = below low).
    """
    if df is None or df.empty or range_high <= range_low:
        mid = (range_high + range_low) / 2.0 if range_high > range_low else float("nan")
        return {
            "state":         "inside",
            "current_price": float("nan"),
            "period_high":   float("nan"),
            "period_low":    float("nan"),
            "range_high":    range_high,
            "range_low":     range_low,
            "range_mid":     mid,
            "range_size":    max(range_high - range_low, 0.0),
            "pct_position":  float("nan"),
            "pct_from_high": float("nan"),
            "pct_from_low":  float("nan"),
        }

    highs  = np.asarray(df["high"],  dtype=float)
    lows   = np.asarray(df["low"],   dtype=float)
    closes = np.asarray(df["close"], dtype=float)

    period_high   = float(np.nanmax(highs))
    period_low    = float(np.nanmin(lows))
    current_price = float(closes[-1])

    range_size = float(range_high - range_low)
    range_mid  = float((range_high + range_low) / 2.0)

    # ── State classification ───────────────────────────────────────────────────
    # Priority order matters: current position is checked first,
    # then whether the window ever exceeded the boundary.
    if current_price > range_high:
        state = "above"           # currently above range high (breakout holding)
    elif current_price < range_low:
        state = "below"           # currently below range low (breakdown holding)
    elif period_high > range_high:
        state = "deviate_high"    # wicked above high, now back inside — failed breakout
    elif period_low < range_low:
        state = "deviate_low"     # wicked below low, now back inside — reclaim / failed breakdown
    else:
        state = "inside"          # Never broke outside range boundaries

    # ── Derived metrics ────────────────────────────────────────────────────────
    pct_position  = (current_price - range_low) / range_size
    pct_from_high = (current_price - range_high) / range_high * 100.0 if range_high > 0 else 0.0
    pct_from_low  = (current_price - range_low)  / range_low  * 100.0 if range_low  > 0 else 0.0

    return {
        "state":         state,
        "current_price": current_price,
        "period_high":   period_high,
        "period_low":    period_low,
        "range_high":    range_high,
        "range_low":     range_low,
        "range_mid":     range_mid,
        "range_size":    range_size,
        "pct_position":  pct_position,
        "pct_from_high": pct_from_high,
        "pct_from_low":  pct_from_low,
    }
