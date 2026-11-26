# Changelog

## 2.0.0

Some outputs change (see "Changed"), hence the major version.

### Changed
- **`calc_rsi`** — Wilder's averages are seeded with the simple average of the first `period`
  changes, as TradingView does (0.1.x seeded from the first change alone). Long histories agree
  to rounding; values near the start of a short history move toward TradingView's.
- **`calc_stochrsi`** — built on that RSI. `period` is now the stochastic window, as documented;
  0.1.x used `rsi_period` for it even when `period` was given.
- **`calc_adx`** — +DI / −DI / ADX seeded like TradingView's DMI (+DI / −DI first on candle
  `period`, ADX on candle `2·period − 1`); 0.1.x started the direction lines one candle early.
- **`calc_vwap`, `calc_anchored_vwap`** — the VWAP line and its standard-deviation bands use
  (high + low + close) / 3, TradingView's default source (was (high + low) / 2). POC / VAL / VAH
  binning is unchanged.
- **`calc_macd`, `calc_ppo`, `calc_ao`** — on a perfectly flat price the lines are exactly 0
  instead of ±1e-14 floating-point residue, so a `> 0` check can't flip on a halted series. No
  non-flat output changes.

### Added
- **Renko** — `calc_renko(df, box=None, atr_period=14)` (per-candle direction, bricks made, latest
  brick level, bricks in a row) and `renko_bricks(...)` (every brick). Traditional construction,
  anchored at the first close; `box=None` uses ATR(14) on the last candle.
- **`calc_vpci`** — Volume Price Confirmation Indicator (Dormeier).
- **`calc_hurst`** — rolling Hurst exponent.
- **Candlestick patterns `flat_open` and `flat_close`** — a candle that opens (or closes) at its
  high or low, within 0.3 % of price (adjustable via the module's `FLAT_TOL`).
- **`sortefi_ta.options`** — Black-76 and Black-Scholes-Merton prices and greeks, implied
  volatility (including NSE's published convention), chain analytics (put/call ratio, max pain,
  ATM IV, term structure, constant-maturity IV, IV rank / percentile, 25-delta skew, expected
  move, straddle move, open-interest build-up, gamma exposure, put-call-parity forward) and
  multi-leg strategy payoffs, breakevens and probability of profit. numpy only — no scipy.
- **`sortefi_ta.performance`** — returns, annualised return and volatility, CAGR, Sharpe,
  Sortino, Omega, information and tail ratios, downside deviation, value at risk and expected
  shortfall (historical or normal), win rate, skewness, kurtosis, beta, alpha, correlation.

## 0.1.0

First release: 122 indicator functions and 59 candlestick patterns.
