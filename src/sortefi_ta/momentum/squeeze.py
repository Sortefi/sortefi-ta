import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def _rolling_line_end(values: pd.Series, period: int) -> np.ndarray:
    """End point of the least-squares line fitted to each rolling window of `period` values
    (windows with a missing value give NaN)."""
    t = np.arange(period, dtype=float)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()

    def _end(y: np.ndarray) -> float:
        slope = (t_c * (y - y.mean())).sum() / denom
        return y.mean() + slope * (period - 1 - t.mean())

    return values.rolling(period, min_periods=period).apply(_end, raw=True).to_numpy()


def calc_squeeze(
    df: pd.DataFrame,
    length: int = 20,
    mult: float = 2.0,
    length_kc: int = 20,
    mult_kc: float = 1.5,
) -> dict[str, np.ndarray]:
    """Squeeze (volatility compression) with a momentum line.

    The squeeze idea: when Bollinger Bands contract inside Keltner Channels, volatility is
    unusually low ("squeeze on"); when they expand back outside, it is released ("squeeze off").

      Bollinger Bands  = SMA(close, length) ± mult_kc × sample std-dev(close, length)
                         (band width uses `mult_kc`, as in the widely used community version
                         of this indicator; `mult` is accepted for compatibility)
      Keltner Channels = SMA(close, length_kc) ± mult_kc × SMA(true range, length_kc)
      squeeze_on       = both Bollinger bands inside the Keltner channel
      squeeze_off      = both Bollinger bands outside the Keltner channel
      squeeze_value    = momentum: end point of a length_kc-bar regression line through
                         close − mean(Donchian midline, SMA(close, length_kc))

    Returns {"squeeze_on", "squeeze_off", "squeeze_value"} (on/off as 1.0 / 0.0).
    """
    n = len(df)
    empty = {"squeeze_on": _nan_array(n), "squeeze_off": _nan_array(n), "squeeze_value": _nan_array(n)}
    if n < length + 1:
        return empty

    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)

    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                           axis=1).max(axis=1, skipna=False)
    true_range.iloc[0] = np.nan

    bb_mid = close.rolling(length, min_periods=length).mean()
    bb_dev = mult_kc * close.rolling(length, min_periods=length).std(ddof=1)
    kc_mid = close.rolling(length_kc, min_periods=length_kc).mean()
    kc_dev = mult_kc * true_range.rolling(length_kc, min_periods=length_kc).mean()

    bb_up, bb_lo = (bb_mid + bb_dev).to_numpy(), (bb_mid - bb_dev).to_numpy()
    kc_up, kc_lo = (kc_mid + kc_dev).to_numpy(), (kc_mid - kc_dev).to_numpy()
    with np.errstate(invalid="ignore"):
        on = (bb_lo > kc_lo) & (bb_up < kc_up)
        off = (bb_lo < kc_lo) & (bb_up > kc_up)

    donchian_mid = 0.5 * (high.rolling(length_kc, min_periods=length_kc).max()
                          + low.rolling(length_kc, min_periods=length_kc).min())
    momentum_base = close - (donchian_mid + kc_mid) / 2.0
    value = _rolling_line_end(momentum_base, length_kc)

    return {"squeeze_on": on.astype(float), "squeeze_off": off.astype(float), "squeeze_value": value}
