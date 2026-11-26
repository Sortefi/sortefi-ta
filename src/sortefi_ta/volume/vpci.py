import numpy as np
import pandas as pd
from sortefi_ta._common import _nan_array


def calc_vpci(df: pd.DataFrame, short: int = 5, long: int = 20) -> dict[str, np.ndarray]:
    """Volume Price Confirmation Indicator (Buff Dormeier, 2007). Returns {"vpci", "vpci_smooth"}.

    VPCI = VPC × VPR × VM, where
      VPC (volume-price confirmation) = VWMA(long) − SMA(long)  — price, volume-weighted vs plain
      VPR (volume-price ratio)        = VWMA(short) ÷ SMA(short)
      VM  (volume multiplier)         = SMA(volume, short) ÷ SMA(volume, long)
    `vpci_smooth` is the SMA(short) of VPCI (the signal line traders draw on it).
    Positive and rising → volume confirms the up-move; negative → volume confirms the down-move;
    price up with VPCI falling → the move lacks volume.
    """
    n = len(df)
    if n < long + 1 or "volume" not in df:
        return {"vpci": _nan_array(n), "vpci_smooth": _nan_array(n)}
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    pv = close * vol

    def vwma(k: int) -> pd.Series:
        return pv.rolling(k).sum() / vol.rolling(k).sum().replace(0.0, np.nan)

    vpc = vwma(long) - close.rolling(long).mean()
    vpr = vwma(short) / close.rolling(short).mean()
    vm = vol.rolling(short).mean() / vol.rolling(long).mean().replace(0.0, np.nan)
    vpci = vpc * vpr * vm
    return {"vpci": vpci.to_numpy(dtype=float),
            "vpci_smooth": vpci.rolling(short).mean().to_numpy(dtype=float)}
