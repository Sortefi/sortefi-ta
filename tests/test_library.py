import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sortefi_ta as T                     # noqa: E402
from _data import make_df, EXTRA_ARGS      # noqa: E402
from make_snapshot import flatten          # noqa: E402

NAMES = sorted(T.INDICATOR_FUNCTIONS)


@pytest.mark.parametrize("name", NAMES)
def test_runs_and_returns_one_value_per_row(name):
    df = make_df()
    res = T.INDICATOR_FUNCTIONS[name](df, **EXTRA_ARGS.get(name, {}))
    if isinstance(res, dict):
        for v in res.values():
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1:
                assert v.size == len(df)


@pytest.mark.parametrize("name", NAMES)
def test_short_input_does_not_crash(name):
    T.INDICATOR_FUNCTIONS[name](make_df(n=5), **EXTRA_ARGS.get(name, {}))


def test_matches_reviewed_snapshot():
    want = np.load(Path(__file__).with_name("snapshot.npz"))
    got = flatten()
    assert set(got) == set(want.files)
    for k in want.files:
        np.testing.assert_allclose(got[k], want[k], rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=k)


def test_hand_checked_values():
    df = pd.DataFrame({"open": np.arange(1, 11.0), "high": np.arange(1, 11.0) + 1,
                       "low": np.arange(1, 11.0) - 1, "close": np.arange(1, 11.0),
                       "volume": np.ones(10)})
    sma = T.calc_sma(df, period=3)["sma"]
    assert np.isnan(sma[1]) and sma[2] == pytest.approx(2.0) and sma[-1] == pytest.approx(9.0)
    # a steady +1 rise has RSI 100 (no losses)
    assert T.calc_rsi(df, period=3)["rsi"][-1] == pytest.approx(100.0)


def test_no_optional_libraries_needed():
    import sortefi_ta._ta as core
    src = "".join(p.read_text() for p in Path(core.__file__).parent.glob("*.py"))
    for lib in ("pandas_ta", "talib", "numba"):
        assert f"import {lib}" not in src


def test_alma_offset_moves_the_average():
    df = make_df()
    a = T.calc_alma(df, offset_pct=0.85)["alma"][-1]
    b = T.calc_alma(df, offset_pct=0.5)["alma"][-1]
    assert np.isfinite(a) and np.isfinite(b) and a != b


@pytest.mark.parametrize("k,d", [(1, 3), (3, 1), (1, 1)])
def test_stochrsi_without_smoothing(k, d):
    res = T.calc_stochrsi(make_df(), k=k, d=d)
    assert np.isfinite(res["stochrsi_k"][-1]) and np.isfinite(res["stochrsi_d"][-1])
    if k == 1 and d == 1:
        np.testing.assert_allclose(res["stochrsi_k"], res["stochrsi_d"], equal_nan=True)
