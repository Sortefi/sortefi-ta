"""Options maths: Black-76 / BSM pricing and greeks, implied volatility, chain analytics
and multi-leg strategy payoffs. numpy + pandas only.

    from sortefi_ta import options as opt
    g = opt.black76_greeks(forward=100, strike=105, t=30/365, vol=0.25, opt_type="C")
    iv = opt.implied_vol_black76(price=1.2, forward=100, strike=105, t=30/365)

Units: vol decimal, t years, rates continuous decimals; vega per vol point, theta per
calendar day, rho per 1 % rate move. Bad inputs give NaN, never an exception.
"""
from ._normal import norm_cdf, norm_pdf
from .pricing import black76_price, black76_greeks, bsm_price, bsm_greeks
from .iv import implied_vol_black76, implied_vol_bsm, implied_vol_nse
from .chain import (
    pcr,
    max_pain,
    atm_iv,
    term_structure,
    constant_maturity_iv,
    iv_rank,
    iv_percentile,
    skew_25d,
    expected_move,
    straddle_move,
    buildup,
    gamma_exposure,
    parity_forward,
)
from .strategy import (
    Leg,
    payoff_at_expiry,
    breakevens,
    max_profit,
    max_loss,
    probability_of_profit,
    value_now,
)

__all__ = [
    "norm_cdf", "norm_pdf",
    "black76_price", "black76_greeks", "bsm_price", "bsm_greeks",
    "implied_vol_black76", "implied_vol_bsm", "implied_vol_nse",
    "pcr", "max_pain", "atm_iv", "term_structure", "constant_maturity_iv",
    "iv_rank", "iv_percentile", "skew_25d", "expected_move", "straddle_move", "buildup",
    "gamma_exposure", "parity_forward",
    "Leg", "payoff_at_expiry", "breakevens", "max_profit", "max_loss",
    "probability_of_profit", "value_now",
]
