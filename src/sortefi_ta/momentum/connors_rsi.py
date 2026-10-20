import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_connors_rsi(
    df: pd.DataFrame,
    rsi_length: int = 3,
    streak_length: int = 2,
    roc_length: int = 100,
) -> dict[str, np.ndarray]:
    """
    ConnorsRSI — composite momentum indicator: avg(RSI, Streak RSI, Percent Rank).

    Components:
      1. RSI(close, rsi_length)
      2. RSI(updown_streak, streak_length) — streak = consecutive up/down count
      3. PercentRank(ROC(close, 1), roc_length)

    Args:
        df:            DataFrame with 'close' column.
        rsi_length:    RSI period on close (default 3).
        streak_length: RSI period on up/down streak (default 2).
        roc_length:    Lookback for percent rank of 1-bar ROC (default 100).

    Returns:
        crsi       — ConnorsRSI values 0–100 (ndarray)
        rsi        — RSI component (ndarray)
        streak_rsi — Streak RSI component (ndarray)
        pct_rank   — Percent Rank component (ndarray)
    """
    n = len(df)
    nan = _nan_array(n)
    if n < 2:
        return {"crsi": nan.copy(), "rsi": nan.copy(), "streak_rsi": nan.copy(), "pct_rank": nan.copy()}

    close = df["close"].to_numpy(dtype=float)

    # 1. Standard RSI on close
    rsi = _rsi(close, rsi_length)

    # 2. Up/Down streak → RSI on streak
    streak = _updown_streak(close)
    streak_rsi = _rsi(streak, streak_length)

    # 3. Percent rank of 1-bar ROC over roc_length bars
    roc1 = np.full(n, np.nan)
    for i in range(1, n):
        if close[i - 1] != 0 and np.isfinite(close[i]) and np.isfinite(close[i - 1]):
            roc1[i] = (close[i] - close[i - 1]) / close[i - 1] * 100.0

    pct_rank = np.full(n, np.nan)
    for i in range(roc_length, n):
        current = roc1[i]
        if not np.isfinite(current):
            continue
        window = roc1[i - roc_length + 1 : i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) == 0:
            continue
        pct_rank[i] = np.sum(valid < current) / len(valid) * 100.0

    # CRSI = average of 3 components
    crsi = np.full(n, np.nan)
    for i in range(n):
        r, s, p = rsi[i], streak_rsi[i], pct_rank[i]
        if np.isfinite(r) and np.isfinite(s) and np.isfinite(p):
            crsi[i] = (r + s + p) / 3.0

    return {"crsi": crsi, "rsi": rsi, "streak_rsi": streak_rsi, "pct_rank": pct_rank}


def _updown_streak(close: np.ndarray) -> np.ndarray:
    """Consecutive up/down streak count. Matches TV's updown() function."""
    n = len(close)
    ud = np.zeros(n)
    for i in range(1, n):
        if not np.isfinite(close[i]) or not np.isfinite(close[i - 1]):
            ud[i] = 0
        elif close[i] == close[i - 1]:
            ud[i] = 0
        elif close[i] > close[i - 1]:
            ud[i] = ud[i - 1] + 1 if ud[i - 1] > 0 else 1
        else:
            ud[i] = ud[i - 1] - 1 if ud[i - 1] < 0 else -1
    return ud


def _rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI on arbitrary data array."""
    n = len(data)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.nanmean(gains[:period])
    avg_loss = np.nanmean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return out
