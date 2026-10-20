"""Regenerate tests/snapshot.npz — ONLY when a change in numbers is intended and reviewed."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sortefi_ta as T          # noqa: E402
from _data import make_df, EXTRA_ARGS   # noqa: E402

def flatten():
    out = {}
    for seed in (1, 2):
        df = make_df(seed=seed, drift=0.002 * (seed - 1))
        for name, fn in sorted(T.INDICATOR_FUNCTIONS.items()):
            res = fn(df.copy(), **EXTRA_ARGS.get(name, {}))
            for k, v in (res.items() if isinstance(res, dict) else [("_", res)]):
                arr = np.asarray(v)
                if arr.dtype.kind in "fiub":
                    out[f"{seed}|{name}|{k}"] = arr.astype(float)
    return out

if __name__ == "__main__":
    np.savez_compressed(Path(__file__).with_name("snapshot.npz"), **flatten())
    print("snapshot written")
