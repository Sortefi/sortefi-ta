"""Small shared helpers for the options package."""
import numpy as np

_CALL = {"C", "CE", "CALL", "CALLS"}
_PUT = {"P", "PE", "PUT", "PUTS"}


def _type_code(v) -> float:
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else -1.0
    if isinstance(v, str):
        s = v.strip().upper()
        if s in _CALL:
            return 1.0
        if s in _PUT:
            return -1.0
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    if f == 1.0:
        return 1.0
    if f == -1.0:
        return -1.0
    return np.nan


_type_code_v = np.frompyfunc(_type_code, 1, 1)


def option_sign(opt_type) -> np.ndarray:
    """Map option types to +1 (call) / -1 (put) / NaN (unknown).

    Accepts 'C'/'P', 'CE'/'PE', 'call'/'put' (any case), True (call) / False (put),
    or +1 / -1, as a scalar or array-like.
    """
    if isinstance(opt_type, str) or np.isscalar(opt_type):
        return np.asarray(_type_code(opt_type), dtype=float)
    arr = np.asarray(opt_type, dtype=object)
    if arr.size == 0:
        return np.asarray(arr, dtype=float)
    return np.asarray(_type_code_v(arr), dtype=float)


def as_float(*args):
    """Convert each argument to a float ndarray (non-numeric -> NaN) and broadcast them."""
    out = []
    for a in args:
        try:
            out.append(np.asarray(a, dtype=float))
        except (TypeError, ValueError):
            arr = np.asarray(a, dtype=object)
            out.append(np.asarray(_to_float_v(arr), dtype=float))
    return np.broadcast_arrays(*out)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


_to_float_v = np.frompyfunc(_to_float, 1, 1)


def unwrap(x):
    """Return a Python float for 0-d results, else the array."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x
