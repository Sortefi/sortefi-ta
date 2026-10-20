import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array, ta


def calc_ppo(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close"
) -> dict[str, np.ndarray]:
    """Percentage Price Oscillator. Returns {"ppo", "ppo_signal", "ppo_histogram"}."""
    n = len(df)
    empty = {"ppo": _nan_array(n), "ppo_signal": _nan_array(n), "ppo_histogram": _nan_array(n)}
    if ta is None or n < slow + signal:
        return empty
    result = ta.ppo(df[col], fast=fast, slow=slow, signal=signal)
    if result is None or (hasattr(result, "empty") and result.empty):
        return empty
    if isinstance(result, pd.DataFrame):
        # columns returned: PPO, PPOh (histogram), PPOs (signal)
        cols = result.columns.tolist()
        return {
            "ppo":           result[cols[0]].to_numpy(dtype=float),
            "ppo_signal":    result[cols[2]].to_numpy(dtype=float) if len(cols) > 2 else _nan_array(n),
            "ppo_histogram": result[cols[1]].to_numpy(dtype=float) if len(cols) > 1 else _nan_array(n),
        }
    return {"ppo": result.to_numpy(dtype=float), "ppo_signal": _nan_array(n), "ppo_histogram": _nan_array(n)}
