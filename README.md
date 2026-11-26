# sortefi-ta

[![PyPI](https://img.shields.io/pypi/v/sortefi-ta)](https://pypi.org/project/sortefi-ta/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/sortefi-ta/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-sortefi.com-8E5E47)](https://www.sortefi.com/docs/sortefi-ta)

Technical indicators, candlestick patterns, chart patterns, price maths, options maths and
performance statistics for **pandas**. Pure numpy + pandas: `pip install sortefi-ta` and go, no C library to compile.

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

## What changed in 2.0.0

2.0.0 changes some outputs, so read this before upgrading from 0.1.x:

| Function | Change |
|---|---|
| `calc_rsi` | Wilder's averages now start from the simple average of the first `period` changes, as TradingView does. Long histories are unchanged; values near the start of a short history move. |
| `calc_stochrsi` | Uses that RSI, and `period` is now the stochastic window (0.1.x silently used `rsi_period`). Same result when both are equal and the history is long. |
| `calc_adx` | +DI / −DI / ADX seeded like TradingView's DMI: +DI / −DI first on candle `period`, ADX on candle `2·period − 1`. |
| `calc_vwap`, `calc_anchored_vwap` | The VWAP line and its bands use (high + low + close) / 3, TradingView's default source (was (high + low) / 2). Volume-profile levels are unchanged. |
| `calc_macd`, `calc_ppo`, `calc_ao` | On a perfectly flat price, values are exactly 0 instead of tiny floating-point residues. No other output changes. |

New: `calc_renko` + `renko_bricks`, `calc_vpci`, `calc_hurst`, the `flat_open` / `flat_close`
candlestick patterns, the `sortefi_ta.options` module, and `sortefi_ta.performance`.
Full list in [CHANGELOG.md](CHANGELOG.md).

## What's inside

| Group | Count | Functions |
|---|---|---|
| Momentum | 29 | ao, ao_divergence, bop, cci, cci_divergence, cmo, connors_rsi, coppock, fisher, kdj, kdj_divergence, laguerre_rsi, macd, macd_divergence, mfi, mfi_divergence, mom, ppo, roc, rsi, rsi_divergence, squeeze, stochastic, stochastic_divergence, stochrsi, trix, tsi, uo, williams_r |
| Trend & moving averages | 25 | adx, alligator, alma, aroon, chop, dema, dpo, ema, hma, ichimoku, kama, linreg, psar, renko, sma, smma, ssl_channel, stage, supertrend, t3, tema, vortex, vwma, wma, zlema |
| Volatility | 18 | accbands, adr, atr, bbands, donchian, gk_volatility, hv, keltner, massi, natr, parkinson_vol, realized_vol, rvi, true_range, ui, vol_percentile, vol_ratio, yz_volatility |
| Volume | 16 | ad, adosc, anchored_vwap, cmf, cmf_divergence, efi, eom, kvo, nvi, obv, obv_divergence, pvt, rvol, volume_divergence, vpci, vwap |
| Price maths | 14 | avg_daily_range, body_size, change, cumulative_return, distance_from_high, distance_from_low, gap, level_state, midpoint, range, range_pct, range_state, relative_position, wick_ratio |
| Statistics | 8 | entropy, hurst, kurtosis, mad, skew, stdev, variance, zscore |
| Support & resistance | 6 | equal_levels, fibonacci_levels, fractals, fvg, pivot_points, sfp |
| Chart patterns | 6 | cup_handle, first_pullback, htf, nr7, three_weeks_tight, vcp |
| Divergence helpers | 2 | divergence_signal, divergence_strength |

Renko: `ta.calc_renko(df, box=2.5)` gives the brick state on every candle (direction, bricks made,
latest brick level, bricks in a row); leave `box` out for an ATR(14) box. `ta.renko_bricks(df, box)`
lists every brick. Traditional construction, anchored at the first close.
| Candlestick patterns | 61 | advance_block, bearish_abandoned_baby, bearish_engulfing, bearish_harami, bearish_harami_cross, bearish_hikkake, bearish_kicking, bearish_meeting_lines, bearish_separating_lines, bearish_three_line_strike, bearish_tri_star, belt_hold, bullish_abandoned_baby, bullish_engulfing, bullish_harami, bullish_harami_cross, bullish_hikkake, bullish_kicking, bullish_meeting_lines, bullish_separating_lines, bullish_three_line_strike, bullish_tri_star, dark_cloud_cover, deliberation, doji, dragonfly_doji, evening_doji_star, evening_star, falling_three_methods, falling_window, flat_close, flat_open, gravestone_doji, hammer, high_wave, homing_pigeon, identical_three_crows, in_neck, inverted_hammer, ladder_bottom, long_legged_doji, marubozu, matching_low, morning_doji_star, morning_star, on_neck, piercing, rising_three_methods, rising_window, spinning_top, stick_sandwich, three_black_crows, three_inside_down, three_inside_up, three_outside_down, three_outside_up, three_white_soldiers, thrusting, tweezer_bottom, tweezer_top, upside_gap_two_crows |

## Options

`sortefi_ta.options` prices European options, computes greeks and implied volatility, and
summarises option chains and multi-leg strategies. It takes arrays and returns arrays
(scalars, numpy arrays and pandas Series all broadcast). Bad inputs give `NaN`, never an exception.

```python
from sortefi_ta import options as opt

# Black-76 on a forward / future (on spot: opt.bsm_greeks(spot, strike, t, vol, r, q, opt_type))
g = opt.black76_greeks(forward=24150, strike=[24000, 24500], t=7/365, vol=0.13, opt_type=["PE", "CE"])
g["delta"], g["theta"]                         # one value per contract

iv = opt.implied_vol_black76(price=182.5, forward=24150, strike=24000, t=7/365, opt_type="P")

# chain DataFrame columns: expiry, strike, opt_type, price, oi, volume, [iv], forward, t
opt.pcr(chain, by="oi"); opt.max_pain(chain); opt.atm_iv(chain); opt.skew_25d(chain)

legs = [dict(kind="C", strike=100, premium=5, qty=1), dict(kind="C", strike=110, premium=2, qty=-1)]
opt.breakevens(legs), opt.max_profit(legs), opt.max_loss(legs)   # [103.], 7.0, 3.0
opt.probability_of_profit(legs, forward=100, vol=0.25, t=30/365)
```

| Quantity | Convention |
|---|---|
| vol | decimal per year (0.20 = 20 %) |
| t | years (calendar days / 365) |
| r, q | continuously compounded annual decimals |
| delta, gamma | per 1 unit of the underlying (forward for Black-76, spot for BSM) |
| vega | price change per 1 vol point (dV/dσ ÷ 100) |
| theta | price change per calendar day (dV/dt ÷ 365) |
| rho | price change per 1 % rate move (dV/dr ÷ 100); Black-76 holds the forward fixed |
| opt_type | `C`/`P`, `CE`/`PE`, `call`/`put`, `True`/`False` or ±1 |

| Group | Functions |
|---|---|
| Pricing | black76_price, black76_greeks, bsm_price, bsm_greeks, norm_cdf, norm_pdf |
| Implied vol | implied_vol_black76, implied_vol_bsm, implied_vol_nse (BSM with r = 10 %, q = 0, the convention behind NSE's published IV) |
| Chain | pcr, max_pain, atm_iv, term_structure, constant_maturity_iv, iv_rank, iv_percentile, skew_25d, expected_move, straddle_move, buildup, gamma_exposure, parity_forward |
| Strategies | Leg, payoff_at_expiry, breakevens, max_profit, max_loss, probability_of_profit, value_now |

How the numbers are checked:

- **Normal CDF without scipy.** It uses W. J. Cody's rational approximations of erfc
  (relative error about 1e-16), vectorised in numpy.
- **Implied vol.** It uses the normalisation and branch split from Jäckel's "Let's Be Rational"
  (not the paper's code), solved with a bracketed Newton method. On a 5,000-point grid
  (strike/forward 0.5–2, 1 day to 2 years, vol 5–300 %) it agrees with `lets_be_rational`
  to about 1e-10 in vol wherever the price pins the vol down. It solves 40,000 contracts in about 0.1 s.
- **Prices and greeks** match QuantLib's analytic engine to about 1e-15.
- **Deribit.** Black-76 reproduces Deribit's delta, gamma and vega when given r = 0, the
  expiry's `underlying_price` as the forward, and `mark_iv`. Deribit's delta is the plain
  delta, not premium-adjusted, even for coin-settled options. Its theta is
  `max(theta × min(1, days left), −price)`. Checked on a frozen sample of 296 options.
- **NSE.** `implied_vol_nse(last_price, spot, strike, days / 365, type)` reproduces the IV on
  NSE's option chain to its 2-decimal rounding.

`straddle_move` returns call + put. Under a lognormal model that is roughly the mean
absolute move to expiry. The "0.85 × straddle ≈ one standard deviation" figure that traders
quote is only a rule of thumb.

## Performance statistics

`sortefi_ta.performance` turns a return series into one number: Sharpe, Sortino, Omega,
information ratio, value at risk and expected shortfall, beta, alpha and more.

```python
from sortefi_ta import performance as pf

r = pf.simple_returns(df["close"])
pf.sharpe_ratio(r, risk_free=0.05, periods_per_year=252)   # risk-free is an ANNUAL rate
pf.value_at_risk(r, level=0.05, method="normal")           # "hist" (default) or "normal"
pf.beta(r, pf.simple_returns(benchmark["close"]))
```

| Group | Functions |
|---|---|
| Returns | simple_returns, log_returns, cumulative_return, total_return, annualized_return, annualized_volatility, cagr |
| Ratios | sharpe_ratio, sortino_ratio, omega_ratio, information_ratio, tail_ratio |
| Downside risk | downside_deviation, value_at_risk, conditional_value_at_risk |
| Distribution | win_rate, skewness, kurtosis |
| Benchmark | beta, alpha, correlation |

Sample standard deviation (ddof = 1); annualise with `periods_per_year` (252 daily stocks,
365 daily crypto, 52 weekly, 12 monthly). Empty, too-short or zero-variance input returns `None`.

## Notes

- **Candlestick patterns are shape-only.** A hammer is a hammer whatever came before it;
  combine with a trend tool (moving-average slope, ADX, `calc_stage`) when you need
  "hammer after a downtrend".
- **Every threshold is in the source.** Each pattern lives in its own short file with its
  rule written out, e.g. `candlestick/hammer.py`.
- **Numbers follow common charting conventions** (EMA seeded with an SMA; Wilder smoothing
  for RSI and ADX seeded with a simple average, as TradingView does; population standard
  deviation for Bollinger Bands).

## Tests

```bash
pip install -e ".[test]" && pytest
```

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
