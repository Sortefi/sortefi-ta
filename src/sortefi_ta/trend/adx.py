import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def _rma_tv(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's average (alpha = 1 / period) the way TradingView's `ta.rma` starts it: blank until `period` values
    in a row exist, the first value = their simple average, then avg = prev + (x − prev) / period."""
    out = np.full(x.size, np.nan)
    run, end = 0, -1
    for i, ok in enumerate(np.isfinite(x)):
        run = run + 1 if ok else 0
        if run >= period:
            end = i
            break
    if end < 0:
        return out
    seeded = np.full(x.size, np.nan)
    seeded[end] = x[end - period + 1: end + 1].mean()
    seeded[end + 1:] = x[end + 1:]
    s = pd.Series(seeded).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(dtype=float)
    out[end:] = s[end:]
    return out


def calc_adx(df: pd.DataFrame, period: int = 14) -> dict[str, np.ndarray]:
    """ADX with +DI / −DI, as TradingView's DMI draws them. Returns {"adx", "di_plus", "di_minus"}.

    Candle 0 has no move and no true range (there is no previous candle), so +DM / −DM / TR start on candle 1;
    each is Wilder-smoothed from the simple average of its first `period` values (+DI / −DI first on candle
    `period`); ADX = Wilder's average of DX = |+DI − −DI| ÷ (+DI + −DI), seeded the same way (first on candle
    2·period − 1). In 0.1.x the direction lines started from candle 0 — one candle early,
    so short histories read differently (the same seed gap fixed for RSI)."""
    n = len(df)
    period = int(period)
    empty = {"adx": _nan_array(n), "di_plus": _nan_array(n), "di_minus": _nan_array(n)}
    if period < 1 or n < period * 2:
        return empty
    h, l_, c = (pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float) for k in ("high", "low", "close"))
    up = np.r_[np.nan, np.diff(h)]
    dn = np.r_[np.nan, -np.diff(l_)]
    with np.errstate(invalid="ignore", divide="ignore"):
        pdm = np.where(np.isnan(up), np.nan, np.where((up > dn) & (up > 0), up, 0.0))
        mdm = np.where(np.isnan(dn), np.nan, np.where((dn > up) & (dn > 0), dn, 0.0))
        pc = np.r_[np.nan, c[:-1]]
        tr = np.where(np.isnan(pc), np.nan, np.fmax(h - l_, np.fmax(np.abs(h - pc), np.abs(l_ - pc))))
        trur = _rma_tv(tr, period)
        trur = np.where(trur == 0, np.nan, trur)              # x ÷ 0 is blank, then carried forward (fixnan)
        plus = pd.Series(100.0 * _rma_tv(pdm, period) / trur).ffill().to_numpy(dtype=float)
        minus = pd.Series(100.0 * _rma_tv(mdm, period) / trur).ffill().to_numpy(dtype=float)
        total = plus + minus
        dx = np.abs(plus - minus) / np.where(total == 0, 1.0, total)
    adx = 100.0 * _rma_tv(dx, period)
    return {"adx": adx, "di_plus": plus, "di_minus": minus}
