import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array

_MIN_LAG = 2


def _hurst_diffusion(x: np.ndarray, max_lag: int) -> float:
    """Hurst exponent of one window of log prices from how far the price diffuses with the lag:
    std(x[t+k] − x[t]) ∝ k^H, so H is the slope of log(std of lag-k differences) against log(k)
    over lags 2..max_lag. Unbiased for a random walk (H = 0.5), unlike the classic rescaled-range
    estimate, which reads ~0.7 on short windows. > 0.5 trending (persistent), < 0.5 mean-reverting."""
    lags = np.arange(_MIN_LAG, max_lag + 1)
    tau = np.array([np.std(x[lag:] - x[:-lag], ddof=1) for lag in lags])
    good = tau > 0
    if good.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[good]), np.log(tau[good]), 1)
    return float(min(max(slope, 0.0), 1.0))


def calc_hurst(df: pd.DataFrame, period: int = 100) -> dict[str, np.ndarray]:
    """Rolling Hurst exponent of the close over `period` candles (diffusion scaling of lagged
    differences). Returns {"hurst"}: ~0.5 random walk · > 0.5 trending · < 0.5 mean-reverting.
    Needs `period` closes per value; lags run 2..period/4."""
    n = len(df)
    out = _nan_array(n)
    if n < period or period < 16:
        return {"hurst": out}
    x = np.log(df["close"].astype(float).to_numpy())
    max_lag = max(_MIN_LAG + 2, period // 4)
    for i in range(period - 1, n):
        out[i] = _hurst_diffusion(x[i - period + 1: i + 1], max_lag)
    return {"hurst": out}
