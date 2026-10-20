"""Vectorized OHLC building blocks for candlestick pattern detection."""
import numpy as np
import pandas as pd


def ohlc_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLC as float64 arrays."""
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )


def body(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed body: positive = bullish, negative = bearish."""
    return c - o


def abs_body(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(c - o)


def candle_range(h: np.ndarray, l: np.ndarray) -> np.ndarray:
    return h - l


def body_ratio(o: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray) -> np.ndarray:
    """abs(body) / range, 0.0 where range == 0."""
    rng = candle_range(h, l)
    safe_rng = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, abs_body(o, c) / safe_rng, 0.0)


def upper_wick(o: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    return h - np.maximum(o, c)


def lower_wick(o: np.ndarray, c: np.ndarray, l: np.ndarray) -> np.ndarray:
    return np.minimum(o, c) - l


def upper_wick_ratio(o: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Upper wick as fraction of range, 0.0 where range == 0."""
    rng = candle_range(h, l)
    safe_rng = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, upper_wick(o, c, h) / safe_rng, 0.0)


def lower_wick_ratio(o: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Lower wick as fraction of range, 0.0 where range == 0."""
    rng = candle_range(h, l)
    safe_rng = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, lower_wick(o, c, l) / safe_rng, 0.0)


def is_bullish(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c > o


def is_bearish(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c < o


def body_top(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(o, c)


def body_bottom(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.minimum(o, c)


def body_center(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (o + c) / 2.0


def zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


def ones(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)
