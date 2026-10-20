"""sortefi_ta._ta — the library's own implementations of the building-block calculations.

Each function takes pandas-ta-style arguments and returns a pandas Series or DataFrame with
pandas-ta-style column names. Numerically they reproduce the pandas-ta 0.4.67b0 native
calculations (MIT licensed).

Every function returns None when the input is shorter than the window it needs.
"""
from sortefi_ta._ta._base import (  # noqa: F401
    sma, ema, rma, wma, true_range, dev, mad, linreg, mom, roc, change,
)
from sortefi_ta._ta._momentum import *    # noqa: F401,F403
from sortefi_ta._ta._trend import *       # noqa: F401,F403
from sortefi_ta._ta._volvol import *      # noqa: F401,F403
from sortefi_ta._ta._more import *     # noqa: F401,F403
