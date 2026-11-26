import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def _wilder_sma_seeded(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's average (alpha = 1 / period) seeded like TradingView: the first value, on candle `period`,
    is the simple average of the first `period` changes (candles 1 … period; candle 0 has no change), then
    avg = prev + (x − prev) / period. Seeding from the first change alone (0.1.x) agrees with this on long
    histories but not near the warm-up edge, where a short history could read far from TradingView."""
    out = np.full(x.size, np.nan)
    if x.size <= period:
        return out
    seeded = np.full(x.size, np.nan)
    seeded[period] = np.nanmean(x[1:period + 1]) if np.isfinite(x[1:period + 1]).any() else np.nan
    seeded[period + 1:] = x[period + 1:]
    s = pd.Series(seeded).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(dtype=float)
    out[period:] = s[period:]
    return out


def calc_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> dict[str, np.ndarray]:
    """RSI (Wilder's smoothing, seeded with the simple average of the first `period` changes — TradingView's
    convention). Blank for the first `period` candles. Returns {"rsi": ndarray}."""
    n = len(df)
    period = int(period)
    if period < 1 or n < period + 1:
        return {"rsi": _nan_array(n)}
    close = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    move = np.diff(close, prepend=np.nan)
    gain = np.where(np.isnan(move), np.nan, np.where(move > 0, move, 0.0))
    loss = np.where(np.isnan(move), np.nan, np.where(move < 0, -move, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        up, down = _wilder_sma_seeded(gain, period), _wilder_sma_seeded(loss, period)
        rsi = 100.0 * up / (up + down)
    return {"rsi": rsi}
