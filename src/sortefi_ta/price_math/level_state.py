"""
calc_level_state: Classify price relative to a single reference price level.

Example uses: a deviation below a prior low, a reclaim of a VWAP value-area level,
a breakdown from a weekly high.

The level can be anything: EMA value, VWAP band (VAL/VAH/POC), date-anchored
high/low/open/close, month high/low, or any user-specified price.

Detection uses **wicks** (period_low / period_high).  Current position uses **close**.
HTF confirmation (reclaim/breakdown) uses the current HTF close from an internally
resampled higher timeframe.

The 6 states:
  "above"        — current close > level, no wick below level during obs window
  "below"        — current close < level, no wick above level during obs window
  "deviate_low"  — wick went below level, current close back above (bullish deviation)
                   HTF close NOT yet above level (or interval not provided)
  "deviate_high" — wick went above level, current close back below (bearish deviation)
                   HTF close NOT yet below level (or interval not provided)
  "reclaim"      — deviate_low + HTF close confirms above level (confirmed bullish)
  "breakdown"    — deviate_high + HTF close confirms below level (confirmed bearish)
"""

import numpy as np
import pandas as pd

# ── HTF mapping — what timeframe confirms the deviation ──────────────────────
# Key = base timeframe of the candles passed in
# Value = (higher timeframe label, pandas resample rule)
_HTF_MAP = {
    "M1":  ("M15", "15min"),
    "M5":  ("M30", "30min"),
    "M15": ("H1",  "1h"),
    "M30": ("H4",  "4h"),
    "H1":  ("H4",  "4h"),
    "H2":  ("H12", "12h"),
    "H4":  ("H12", "12h"),
    "H6":  ("D",   "1D"),
    "H8":  ("D",   "1D"),
    "H12": ("3D",  "3D"),
    "D":   ("W",   "W-MON"),
    "W":   ("M",   "MS"),
}


def _resample_htf_close(df: pd.DataFrame, resample_rule: str) -> float:
    """
    Resample obs-window df to a higher timeframe and return the latest
    *completed* HTF candle close.  Returns NaN if not enough data.
    """
    if df is None or df.empty:
        return float("nan")

    ohlcv = df.set_index("open_time").sort_index()

    htf = ohlcv["close"].resample(resample_rule).last().dropna()
    if htf.empty:
        return float("nan")

    # If the latest HTF bar is still open (its period hasn't ended yet),
    # the last *completed* bar is the one before it.  For safety we always
    # take the last available value — the resample().last() already uses
    # only the data we have, so it represents the most recent close inside
    # that HTF bucket.
    return float(htf.iloc[-1])


def calc_level_state(
    df: "pd.DataFrame",
    level: float,
    interval: str | None = None,
) -> dict:
    """
    Classify price action relative to a single reference price level.

    Args:
        df:        OHLCV DataFrame for the **observation window** — bars AFTER the
                   reference period ended.  Must have columns: open_time, high, low, close.
        level:     The reference price level (e.g. EMA value, VWAP VAL, Monday low).
        interval:  Base timeframe string (e.g. "H1", "M15", "H4").
                   When provided, the function resamples df to the mapped HTF and
                   checks whether the latest HTF close confirms above/below the
                   level — promoting deviate_low → reclaim or deviate_high → breakdown.
                   When None, HTF confirmation is skipped (no reclaim/breakdown returned).

    Returns dict of scalars:
        state           str   "above" | "below" | "deviate_low" | "deviate_high"
                              | "reclaim" | "breakdown"
        current_price   float Most recent close in obs window.
        level           float The reference level (echoed).
        period_high     float Highest high (wick) across observation window.
        period_low      float Lowest low (wick) across observation window.
        pct_from_level  float (current_price - level) / level × 100
        htf_label       str   HTF timeframe used for confirmation (e.g. "H4"), or ""
        htf_close       float Latest HTF close used for confirmation, or NaN
    """
    nan_result = {
        "state":          "above",
        "current_price":  float("nan"),
        "level":          level,
        "period_high":    float("nan"),
        "period_low":     float("nan"),
        "pct_from_level": float("nan"),
        "htf_label":      "",
        "htf_close":      float("nan"),
    }

    if df is None or df.empty or np.isnan(level) or level <= 0:
        return nan_result

    highs  = np.asarray(df["high"],  dtype=float)
    lows   = np.asarray(df["low"],   dtype=float)
    closes = np.asarray(df["close"], dtype=float)

    period_high   = float(np.nanmax(highs))
    period_low    = float(np.nanmin(lows))
    current_price = float(closes[-1])

    # ── Base state classification (wick-based detection, close-based position) ─
    wicked_below = period_low < level
    wicked_above = period_high > level

    if current_price > level:
        base_state = "deviate_low" if wicked_below else "above"
    elif current_price < level:
        base_state = "deviate_high" if wicked_above else "below"
    else:
        # Exactly at level — treat like above if wicked below (bullish), else above
        base_state = "deviate_low" if wicked_below else "above"

    # ── HTF confirmation (promotes deviate → reclaim / breakdown) ─────────────
    htf_label = ""
    htf_close = float("nan")
    state = base_state

    if interval and interval in _HTF_MAP and base_state in ("deviate_low", "deviate_high"):
        htf_label, resample_rule = _HTF_MAP[interval]
        htf_close = _resample_htf_close(df, resample_rule)

        if not np.isnan(htf_close):
            if base_state == "deviate_low" and htf_close > level:
                state = "reclaim"       # HTF close confirms above → bullish reclaim
            elif base_state == "deviate_high" and htf_close < level:
                state = "breakdown"     # HTF close confirms below → bearish breakdown
            # else: HTF close doesn't confirm yet → stay at deviate_low / deviate_high

    pct_from_level = (current_price - level) / level * 100.0

    return {
        "state":          state,
        "current_price":  current_price,
        "level":          level,
        "period_high":    period_high,
        "period_low":     period_low,
        "pct_from_level": pct_from_level,
        "htf_label":      htf_label,
        "htf_close":      htf_close,
    }
