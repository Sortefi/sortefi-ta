"""
sortefi_ta — technical indicators, candlestick patterns, chart patterns and price maths
for OHLCV data in pandas. Open source (MIT) by Sortefi — https://sortefi.com

Every function takes a DataFrame with open / high / low / close / volume columns (oldest row
first) and returns a dict of numpy arrays, one value per row, where [-1] is the latest candle.

    import sortefi_ta as ta
    rsi = ta.calc_rsi(df, period=14)["rsi"]

INDICATOR_FUNCTIONS maps every public function name to the function.
"""

# -- Momentum --
from sortefi_ta.momentum import (
    calc_rsi,
    calc_macd,
    calc_macd_divergence,
    calc_stochastic_divergence,
    calc_cci_divergence,
    calc_mfi_divergence,
    calc_ao_divergence,
    calc_kdj_divergence,
    calc_stochastic,
    calc_williams_r,
    calc_cci,
    calc_stochrsi,
    calc_roc,
    calc_mfi,
    calc_cmo,
    calc_ao,
    calc_tsi,
    calc_uo,
    calc_bop,
    calc_fisher,
    calc_kdj,
    calc_ppo,
    calc_mom,
    calc_trix,
    calc_squeeze,
    calc_rsi_divergence,
    calc_laguerre_rsi,
    calc_coppock,
    calc_connors_rsi,
)

# -- Trend --
from sortefi_ta.trend import (
    calc_sma,
    calc_stage,
    calc_ema,
    calc_smma,
    calc_zlema,
    calc_adx,
    calc_ichimoku,
    calc_supertrend,
    calc_psar,
    calc_aroon,
    calc_dema,
    calc_tema,
    calc_wma,
    calc_hma,
    calc_kama,
    calc_vwma,
    calc_t3,
    calc_linreg,
    calc_chop,
    calc_dpo,
    calc_vortex,
    calc_alma,
    calc_alligator,
    calc_ssl_channel,
)

# -- Volatility --
from sortefi_ta.volatility import (
    calc_bbands,
    calc_atr,
    calc_keltner,
    calc_donchian,
    calc_true_range,
    calc_natr,
    calc_adr,
    calc_massi,
    calc_accbands,
    calc_ui,
    calc_rvi,
    calc_hv,
    calc_parkinson_vol,
    calc_gk_volatility,
    calc_yz_volatility,
    calc_realized_vol,
    calc_vol_percentile,
    calc_vol_ratio,
)

# -- Volume --
from sortefi_ta.volume import (
    calc_vwap,
    calc_anchored_vwap,
    calc_obv,
    calc_obv_divergence,
    calc_cmf_divergence,
    calc_volume_divergence,
    calc_ad,
    calc_cmf,
    calc_adosc,
    calc_efi,
    calc_eom,
    calc_kvo,
    calc_pvt,
    calc_nvi,
    calc_rvol,
)

# -- Support/Resistance --
from sortefi_ta.support_resistance import (
    calc_pivot_points,
    calc_fibonacci_levels,
    calc_fvg,
    calc_fractals,
    calc_sfp,
    calc_equal_levels,
)

# -- Price Math --
from sortefi_ta.price_math import (
    calc_midpoint,
    calc_body_size,
    calc_range,
    calc_range_pct,
    calc_change,
    calc_gap,
    calc_distance_from_high,
    calc_distance_from_low,
    calc_relative_position,
    calc_cumulative_return,
    calc_avg_daily_range,
    calc_wick_ratio,
    calc_range_state,
    calc_level_state,
)

# -- Candlestick Patterns --
from sortefi_ta.candlestick import calc_candlestick_patterns

# -- Chart Patterns (multi-bar swing setups) --
from sortefi_ta.chart_patterns import (
    calc_nr7,
    calc_first_pullback,
    calc_three_weeks_tight,
    calc_htf,
    calc_cup_handle,
    calc_vcp,
)

# -- Statistics --
from sortefi_ta.statistics import (
    calc_zscore,
    calc_stdev,
    calc_variance,
    calc_skew,
    calc_kurtosis,
    calc_entropy,
    calc_mad,
)

# High-level divergence helpers (deterministic active/historical slicing + ranking)
from sortefi_ta.divergence_signal import divergence_signal, divergence_strength


# Every public function by name
INDICATOR_FUNCTIONS: dict[str, callable] = {
    # Divergence convenience helpers (wrap the calc_*_divergence functions)
    "divergence_signal": divergence_signal,
    "divergence_strength": divergence_strength,
    # Momentum
    "calc_rsi": calc_rsi,
    "calc_macd": calc_macd,
    "calc_stochastic": calc_stochastic,
    "calc_williams_r": calc_williams_r,
    "calc_cci": calc_cci,
    "calc_stochrsi": calc_stochrsi,
    "calc_roc": calc_roc,
    "calc_mfi": calc_mfi,
    "calc_cmo": calc_cmo,
    "calc_ao": calc_ao,
    "calc_tsi": calc_tsi,
    "calc_uo": calc_uo,
    "calc_bop": calc_bop,
    "calc_fisher": calc_fisher,
    "calc_kdj": calc_kdj,
    "calc_ppo": calc_ppo,
    "calc_mom": calc_mom,
    "calc_trix": calc_trix,
    "calc_squeeze": calc_squeeze,
    "calc_laguerre_rsi": calc_laguerre_rsi,
    "calc_coppock": calc_coppock,
    "calc_connors_rsi": calc_connors_rsi,
    "calc_rsi_divergence": calc_rsi_divergence,
    "calc_macd_divergence": calc_macd_divergence,
    "calc_stochastic_divergence": calc_stochastic_divergence,
    "calc_cci_divergence": calc_cci_divergence,
    "calc_mfi_divergence": calc_mfi_divergence,
    "calc_ao_divergence": calc_ao_divergence,
    "calc_kdj_divergence": calc_kdj_divergence,
    # Trend
    "calc_sma": calc_sma,
    "calc_stage": calc_stage,
    "calc_ema": calc_ema,
    "calc_smma": calc_smma,
    "calc_zlema": calc_zlema,
    "calc_adx": calc_adx,
    "calc_ichimoku": calc_ichimoku,
    "calc_supertrend": calc_supertrend,
    "calc_psar": calc_psar,
    "calc_aroon": calc_aroon,
    "calc_dema": calc_dema,
    "calc_tema": calc_tema,
    "calc_wma": calc_wma,
    "calc_hma": calc_hma,
    "calc_kama": calc_kama,
    "calc_vwma": calc_vwma,
    "calc_t3": calc_t3,
    "calc_linreg": calc_linreg,
    "calc_chop": calc_chop,
    "calc_dpo": calc_dpo,
    "calc_vortex": calc_vortex,
    "calc_alma": calc_alma,
    "calc_alligator": calc_alligator,
    "calc_ssl_channel": calc_ssl_channel,
    # Volatility
    "calc_bbands": calc_bbands,
    "calc_atr": calc_atr,
    "calc_keltner": calc_keltner,
    "calc_donchian": calc_donchian,
    "calc_true_range": calc_true_range,
    "calc_natr": calc_natr,
    "calc_adr": calc_adr,
    "calc_massi": calc_massi,
    "calc_accbands": calc_accbands,
    "calc_ui": calc_ui,
    "calc_rvi": calc_rvi,
    "calc_hv": calc_hv,
    "calc_parkinson_vol": calc_parkinson_vol,
    "calc_gk_volatility": calc_gk_volatility,
    "calc_yz_volatility": calc_yz_volatility,
    "calc_realized_vol": calc_realized_vol,
    "calc_vol_percentile": calc_vol_percentile,
    "calc_vol_ratio": calc_vol_ratio,
    # Volume
    "calc_vwap": calc_vwap,
    "calc_anchored_vwap": calc_anchored_vwap,
    "calc_obv": calc_obv,
    "calc_obv_divergence": calc_obv_divergence,
    "calc_cmf_divergence": calc_cmf_divergence,
    "calc_volume_divergence": calc_volume_divergence,
    "calc_ad": calc_ad,
    "calc_cmf": calc_cmf,
    "calc_adosc": calc_adosc,
    "calc_efi": calc_efi,
    "calc_eom": calc_eom,
    "calc_kvo": calc_kvo,
    "calc_pvt": calc_pvt,
    "calc_nvi": calc_nvi,
    "calc_rvol": calc_rvol,
    # Support/Resistance
    "calc_pivot_points": calc_pivot_points,
    "calc_fibonacci_levels": calc_fibonacci_levels,
    "calc_fvg": calc_fvg,
    "calc_fractals": calc_fractals,
    "calc_sfp": calc_sfp,
    "calc_equal_levels": calc_equal_levels,
    # Price Math
    "calc_midpoint": calc_midpoint,
    "calc_body_size": calc_body_size,
    "calc_range": calc_range,
    "calc_range_pct": calc_range_pct,
    "calc_change": calc_change,
    "calc_gap": calc_gap,
    "calc_distance_from_high": calc_distance_from_high,
    "calc_distance_from_low": calc_distance_from_low,
    "calc_relative_position": calc_relative_position,
    "calc_cumulative_return": calc_cumulative_return,
    "calc_avg_daily_range": calc_avg_daily_range,
    "calc_wick_ratio": calc_wick_ratio,
    "calc_range_state": calc_range_state,
    "calc_level_state": calc_level_state,
    # Candlestick Patterns
    "calc_candlestick_patterns": calc_candlestick_patterns,
    # Chart Patterns (multi-bar swing setups)
    "calc_nr7": calc_nr7,
    "calc_first_pullback": calc_first_pullback,
    "calc_three_weeks_tight": calc_three_weeks_tight,
    "calc_htf": calc_htf,
    "calc_cup_handle": calc_cup_handle,
    "calc_vcp": calc_vcp,
    # Statistics
    "calc_zscore": calc_zscore,
    "calc_stdev": calc_stdev,
    "calc_variance": calc_variance,
    "calc_skew": calc_skew,
    "calc_kurtosis": calc_kurtosis,
    "calc_entropy": calc_entropy,
    "calc_mad": calc_mad,
}
