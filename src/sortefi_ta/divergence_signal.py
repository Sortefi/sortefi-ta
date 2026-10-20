"""divergence_signal() — one deterministic call for "is this divergence present?".

Divergence checks depend on pivot settings and on whether only the most recent bars
("active") or the whole history ("historical") is scanned. This helper fixes those
choices so results are consistent: you choose the indicator, direction and active or
historical, and the windowing is done the same way every time.

Wraps the 10 calc_*_divergence functions; defaults to method="swing" (ATR-significant
price swings, no bar-distance cap — suited to long-span, volatility-adaptive divergences).
"""

import pandas as pd

from sortefi_ta.momentum.rsi_divergence import calc_rsi_divergence
from sortefi_ta.momentum.macd_divergence import calc_macd_divergence
from sortefi_ta.momentum.stochastic_divergence import calc_stochastic_divergence
from sortefi_ta.momentum.cci_divergence import calc_cci_divergence
from sortefi_ta.momentum.mfi_divergence import calc_mfi_divergence
from sortefi_ta.momentum.ao_divergence import calc_ao_divergence
from sortefi_ta.momentum.kdj_divergence import calc_kdj_divergence
from sortefi_ta.volume.obv_divergence import calc_obv_divergence
from sortefi_ta.volume.cmf_divergence import calc_cmf_divergence
from sortefi_ta.volume.volume_divergence import calc_volume_divergence

_DIVERGENCE_FUNCS = {
    "rsi": calc_rsi_divergence,
    "macd": calc_macd_divergence,
    "stochastic": calc_stochastic_divergence,
    "stoch": calc_stochastic_divergence,
    "cci": calc_cci_divergence,
    "mfi": calc_mfi_divergence,
    "ao": calc_ao_divergence,
    "kdj": calc_kdj_divergence,
    "obv": calc_obv_divergence,
    "cmf": calc_cmf_divergence,
    "volume": calc_volume_divergence,
}

_BULL = {"bull", "bullish", "up", "long"}
_BEAR = {"bear", "bearish", "down", "short"}


def _clamp_to_window(df: pd.DataFrame, arrays: tuple, window_start) -> tuple:
    """Restrict full-length (one-per-bar) arrays to bars at/after `window_start`,
    excluding leading warm-up rows before it. Used by the historical scan so a divergence
    inside those rows is not counted.

    Does nothing when `window_start` is empty/None or `df` has no `open_time`.
    """
    if not window_start or "open_time" not in df.columns:
        return arrays
    ws = pd.Timestamp(window_start)
    if ws.tzinfo is None:
        ws = ws.tz_localize("UTC")
    ot = df["open_time"]
    if getattr(ot.dt, "tz", None) is None:
        ot = ot.dt.tz_localize("UTC")
    mask = (ot >= ws).to_numpy()
    return tuple(a[mask] for a in arrays)


def divergence_signal(
    df: pd.DataFrame,
    indicator: str = "rsi",
    direction: str = "both",
    *,
    when: str = "active",
    recent_bars: int = 10,
    method: str = "swing",
    window_start=None,
    **kwargs,
) -> bool:
    """True when the requested divergence is present in `df`.

    Args:
        df:          single-symbol OHLCV DataFrame at the target timeframe
                     (already resampled, sorted by open_time).
        indicator:   rsi | macd | stochastic | cci | mfi | ao | kdj | obv | cmf | volume.
        direction:   "bull", "bear", or "both".
        when:        "active"  → only the last `recent_bars` bars (DEFAULT — a divergence
                                 that resolved months ago is not tradeable today);
                     "historical" → anywhere in the requested window (pass `window_start`
                                 to exclude leading warm-up rows).
        recent_bars: size of the active tail (default 10 ≈ 2 trading weeks on daily;
                     scale up for finer TFs / down for coarser).
        window_start: start of the window you care about. Only
                     used when when="historical" — clamps the scan to bars at/after it so
                     warm-up rows aren't scanned. Empty/None ⇒ scan the whole array.
        method:      "swing" (ATR-significant price swings, no bar cap — use for
                     multi-month / dynamic; DEFAULT) or "tv" (fixed pivot window).
        **kwargs:    passed through to the underlying calc_*_divergence
                     (e.g. rsi_period=14, atr_mult=1.5, fractal_period=2).

    Returns:
        bool — whether the divergence of the requested direction exists in the
        chosen window. Use it directly as a per-symbol filter.
    """
    key = str(indicator).lower().strip().replace("_divergence", "").replace(" ", "")
    fn = _DIVERGENCE_FUNCS.get(key)
    if fn is None:
        raise ValueError(
            f"unknown divergence indicator: {indicator!r} "
            f"(expected one of {sorted(set(_DIVERGENCE_FUNCS))})"
        )

    div = fn(df, method=method, **kwargs)
    bull = div["bull_div"]
    bear = div["bear_div"]

    if str(when).lower().startswith("hist"):
        bull, bear = _clamp_to_window(df, (bull, bear), window_start)
    else:
        nb = max(1, int(recent_bars))
        bull = bull[-nb:]
        bear = bear[-nb:]

    has_bull = bool(bull.any())
    has_bear = bool(bear.any())

    d = str(direction).lower().strip()
    if d in _BULL:
        return has_bull
    if d in _BEAR:
        return has_bear
    return has_bull or has_bear


def divergence_strength(
    df: pd.DataFrame,
    indicator: str = "rsi",
    direction: str = "bear",
    *,
    when: str = "active",
    recent_bars: int = 10,
    method: str = "swing",
    window_start=None,
    **kwargs,
) -> float:
    """Strength of the requested divergence — use it to RANK symbols (top-N by conviction).

    Returns the indicator's counter-move magnitude at the STRONGEST qualifying divergence
    in the chosen window (for RSI: how many RSI points the oscillator diverged — bounded
    0-100 and directly comparable across symbols), or 0.0 when none is present. So
    `divergence_strength(...) > 0` is exactly equivalent to `divergence_signal(...)`, and a
    larger value means a wider, higher-conviction divergence.

    Typical use — collect (symbol, strength), sort descending, keep the top N:

        scored = []
        for sym, g in candles.groupby("symbol", sort=False):
            s = divergence_strength(g, "rsi", "bear", when="active", method="swing")
            if s > 0:
                scored.append((sym, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in scored[:25]]

    Args mirror divergence_signal(). Pass a concrete `direction` ("bull" or "bear") —
    ranking a mixed "both" score blends two different setups and is rarely intended.
    """
    key = str(indicator).lower().strip().replace("_divergence", "").replace(" ", "")
    fn = _DIVERGENCE_FUNCS.get(key)
    if fn is None:
        raise ValueError(
            f"unknown divergence indicator: {indicator!r} "
            f"(expected one of {sorted(set(_DIVERGENCE_FUNCS))})"
        )

    div = fn(df, method=method, **kwargs)
    bull = div["bull_strength"]
    bear = div["bear_strength"]

    if str(when).lower().startswith("hist"):
        bull, bear = _clamp_to_window(df, (bull, bear), window_start)
    else:
        nb = max(1, int(recent_bars))
        bull = bull[-nb:]
        bear = bear[-nb:]

    bull_max = float(bull.max()) if bull.size else 0.0
    bear_max = float(bear.max()) if bear.size else 0.0

    d = str(direction).lower().strip()
    if d in _BULL:
        return max(0.0, bull_max)
    if d in _BEAR:
        return max(0.0, bear_max)
    return max(0.0, bull_max, bear_max)
