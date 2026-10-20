# sortefi-ta

[![PyPI](https://img.shields.io/pypi/v/sortefi-ta)](https://pypi.org/project/sortefi-ta/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/sortefi-ta/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-sortefi.com-8E5E47)](https://www.sortefi.com/docs/sortefi-ta)

Technical indicators, candlestick patterns, chart patterns and price maths for **pandas**.
Pure numpy + pandas: `pip install sortefi-ta` and go, no C library to compile.

Built and used in production by [Sortefi](https://sortefi.com), where you can describe a
stock, crypto or forex screen in plain English and get the matching symbols.

Step-by-step install and usage guide: **[sortefi.com/docs/sortefi-ta](https://www.sortefi.com/docs/sortefi-ta)**

## Install

```bash
pip install sortefi-ta
```

Python 3.10+, numpy and pandas only.

## Quick start

```python
import pandas as pd
import sortefi_ta as ta

df = pd.read_csv("candles.csv")          # columns: open, high, low, close, volume (oldest first)

rsi = ta.calc_rsi(df, period=14)["rsi"]          # numpy array, one value per row
macd = ta.calc_macd(df)                           # {"macd", "signal", "histogram"}
pats = ta.calc_candlestick_patterns(df, ["hammer", "bullish_engulfing"])

print(rsi[-1])                    # latest RSI
print(pats["hammer"][-1] == 1.0)  # did the latest candle form a hammer?
```

Every function takes a DataFrame and returns a dict of numpy arrays aligned with the rows,
where `[-1]` is the latest candle. Warm-up rows are `NaN`. `ta.INDICATOR_FUNCTIONS` lists them all.

## What's inside

| Group | Count | Functions |
|---|---|---|
| Momentum | 29 | ao, ao_divergence, bop, cci, cci_divergence, cmo, connors_rsi, coppock, fisher, kdj, kdj_divergence, laguerre_rsi, macd, macd_divergence, mfi, mfi_divergence, mom, ppo, roc, rsi, rsi_divergence, squeeze, stochastic, stochastic_divergence, stochrsi, trix, tsi, uo, williams_r |
| Trend & moving averages | 24 | adx, alligator, alma, aroon, chop, dema, dpo, ema, hma, ichimoku, kama, linreg, psar, sma, smma, ssl_channel, stage, supertrend, t3, tema, vortex, vwma, wma, zlema |
| Volatility | 18 | accbands, adr, atr, bbands, donchian, gk_volatility, hv, keltner, massi, natr, parkinson_vol, realized_vol, rvi, true_range, ui, vol_percentile, vol_ratio, yz_volatility |
| Volume | 15 | ad, adosc, anchored_vwap, cmf, cmf_divergence, efi, eom, kvo, nvi, obv, obv_divergence, pvt, rvol, volume_divergence, vwap |
| Price maths | 14 | avg_daily_range, body_size, change, cumulative_return, distance_from_high, distance_from_low, gap, level_state, midpoint, range, range_pct, range_state, relative_position, wick_ratio |
| Statistics | 7 | entropy, kurtosis, mad, skew, stdev, variance, zscore |
| Support & resistance | 6 | equal_levels, fibonacci_levels, fractals, fvg, pivot_points, sfp |
| Chart patterns | 6 | cup_handle, first_pullback, htf, nr7, three_weeks_tight, vcp |
| Divergence helpers | 2 | divergence_signal, divergence_strength |
| Candlestick patterns | 59 | advance_block, bearish_abandoned_baby, bearish_engulfing, bearish_harami, bearish_harami_cross, bearish_hikkake, bearish_kicking, bearish_meeting_lines, bearish_separating_lines, bearish_three_line_strike, bearish_tri_star, belt_hold, bullish_abandoned_baby, bullish_engulfing, bullish_harami, bullish_harami_cross, bullish_hikkake, bullish_kicking, bullish_meeting_lines, bullish_separating_lines, bullish_three_line_strike, bullish_tri_star, dark_cloud_cover, deliberation, doji, dragonfly_doji, evening_doji_star, evening_star, falling_three_methods, falling_window, gravestone_doji, hammer, high_wave, homing_pigeon, identical_three_crows, in_neck, inverted_hammer, ladder_bottom, long_legged_doji, marubozu, matching_low, morning_doji_star, morning_star, on_neck, piercing, rising_three_methods, rising_window, spinning_top, stick_sandwich, three_black_crows, three_inside_down, three_inside_up, three_outside_down, three_outside_up, three_white_soldiers, thrusting, tweezer_bottom, tweezer_top, upside_gap_two_crows |

## Notes

- **Candlestick patterns are shape-only.** A hammer is a hammer whatever came before it;
  combine with a trend tool (moving-average slope, ADX, `calc_stage`) when you need
  "hammer after a downtrend".
- **Every threshold is in the source.** Each pattern lives in its own short file with its
  rule written out, e.g. `candlestick/hammer.py`.
- **Numbers follow common charting conventions** (EMA seeded with an SMA, Wilder smoothing
  for RSI/ATR/ADX, population standard deviation for Bollinger Bands).

## Tests

```bash
pip install -e ".[test]" && pytest
```

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
