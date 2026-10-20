"""Candlestick pattern detection — single entry point.

Usage:
    result = calc_candlestick_patterns(df, pattern="bullish_engulfing")
    result = calc_candlestick_patterns(df, pattern=["doji", "hammer"])
"""
import importlib
import numpy as np
import pandas as pd
from . import _helpers as h

# Pattern name → module name mapping (lazy-loaded)
_PATTERN_MODULES: dict[str, str] = {
    # Single candle (10)
    "doji":                    "doji",
    "long_legged_doji":        "long_legged_doji",
    "dragonfly_doji":          "dragonfly_doji",
    "gravestone_doji":         "gravestone_doji",
    "hammer":                  "hammer",
    "inverted_hammer":         "inverted_hammer",
    "marubozu":                "marubozu",
    "spinning_top":            "spinning_top",
    "belt_hold":               "belt_hold",
    "high_wave":               "high_wave",
    # Two candle (15)
    "bullish_engulfing":       "bullish_engulfing",
    "bearish_engulfing":       "bearish_engulfing",
    "bullish_harami":          "bullish_harami",
    "bearish_harami":          "bearish_harami",
    "bullish_harami_cross":    "bullish_harami_cross",
    "bearish_harami_cross":    "bearish_harami_cross",
    "dark_cloud_cover":        "dark_cloud_cover",
    "piercing":                "piercing",
    "tweezer_top":             "tweezer_top",
    "tweezer_bottom":          "tweezer_bottom",
    "bullish_kicking":         "bullish_kicking",
    "bearish_kicking":         "bearish_kicking",
    "rising_window":           "rising_window",
    "falling_window":          "falling_window",
    "bullish_meeting_lines":   "bullish_meeting_lines",
    "bearish_meeting_lines":   "bearish_meeting_lines",
    "bullish_separating_lines":"bullish_separating_lines",
    "bearish_separating_lines":"bearish_separating_lines",
    "on_neck":                 "on_neck",
    "in_neck":                 "in_neck",
    "thrusting":               "thrusting",
    "homing_pigeon":           "homing_pigeon",
    "matching_low":            "matching_low",
    # Three candle (13)
    "morning_star":            "morning_star",
    "evening_star":            "evening_star",
    "morning_doji_star":       "morning_doji_star",
    "evening_doji_star":       "evening_doji_star",
    "three_white_soldiers":    "three_white_soldiers",
    "three_black_crows":       "three_black_crows",
    "identical_three_crows":   "identical_three_crows",
    "three_inside_up":         "three_inside_up",
    "three_inside_down":       "three_inside_down",
    "three_outside_up":        "three_outside_up",
    "three_outside_down":      "three_outside_down",
    "bullish_abandoned_baby":  "bullish_abandoned_baby",
    "bearish_abandoned_baby":  "bearish_abandoned_baby",
    "bullish_tri_star":        "bullish_tri_star",
    "bearish_tri_star":        "bearish_tri_star",
    "advance_block":           "advance_block",
    "deliberation":            "deliberation",
    "stick_sandwich":          "stick_sandwich",
    "upside_gap_two_crows":    "upside_gap_two_crows",
    # Multi candle — 4-5 bars (7)
    "bullish_three_line_strike":"bullish_three_line_strike",
    "bearish_three_line_strike":"bearish_three_line_strike",
    "rising_three_methods":    "rising_three_methods",
    "falling_three_methods":   "falling_three_methods",
    "ladder_bottom":           "ladder_bottom",
    "bullish_hikkake":         "bullish_hikkake",
    "bearish_hikkake":         "bearish_hikkake",
}

# Aliases — common alternative names map to canonical pattern names
_ALIASES: dict[str, str] = {
    "shooting_star":     "inverted_hammer",
    "hanging_man":       "hammer",
    "piercing_line":     "piercing",
    "engulfing":         "bullish_engulfing",  # default to bullish when unspecified
    "harami":            "bullish_harami",
    "harami_cross":      "bullish_harami_cross",
    "kicking":           "bullish_kicking",
    "meeting_lines":     "bullish_meeting_lines",
    "separating_lines":  "bullish_separating_lines",
    "tweezers":          "tweezer_top",
    "tri_star":          "bullish_tri_star",
    "abandoned_baby":    "bullish_abandoned_baby",
    "three_line_strike": "bullish_three_line_strike",
    "hikkake":           "bullish_hikkake",
    "window":            "rising_window",
}

# Cache loaded detect functions
_DETECT_CACHE: dict[str, callable] = {}


def _get_detect(pattern: str):
    """Lazy-load and cache the detect function for a pattern."""
    if pattern in _DETECT_CACHE:
        return _DETECT_CACHE[pattern]
    module_name = _PATTERN_MODULES[pattern]
    mod = importlib.import_module(f".{module_name}", package=__name__)
    _DETECT_CACHE[pattern] = mod.detect
    return mod.detect


def calc_candlestick_patterns(
    df: pd.DataFrame,
    pattern: str | list[str] = "doji",
) -> dict[str, np.ndarray]:
    """Detect one or more candlestick patterns.

    Args:
        df: OHLCV DataFrame with open/high/low/close columns.
        pattern: Pattern name or list of names. See _PATTERN_MODULES for all valid names.

    Returns:
        Dict mapping pattern name → float array (1.0 = detected, 0.0 = not).
    """
    if isinstance(pattern, str):
        pattern = [pattern]

    o, hi, lo, c = h.ohlc_arrays(df)
    result = {}
    for p in pattern:
        name = _ALIASES.get(p, p)
        if name not in _PATTERN_MODULES:
            raise ValueError(
                f"Unknown candlestick pattern: '{p}'. "
                f"Valid patterns: {sorted(_PATTERN_MODULES.keys())}"
            )
        detect_fn = _get_detect(name)
        result[p] = detect_fn(o, hi, lo, c)
    return result


# All valid pattern names
VALID_PATTERNS = sorted(_PATTERN_MODULES.keys())
VALID_ALIASES = dict(_ALIASES)

__all__ = ["calc_candlestick_patterns", "VALID_PATTERNS", "VALID_ALIASES"]
