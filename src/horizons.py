"""Construct h-day-ahead average targets for multi-step volatility forecasting."""
from __future__ import annotations

import pandas as pd


def build_h_step_target(rv: pd.Series, h: int) -> pd.Series:
    """Return the h-step-ahead mean target; the last h-1 rows are NaN and the caller must drop them."""
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    if h == 1:
        return rv.copy()
    # Row d holds mean(RV_d, ..., RV_{d+h-1}); the last h-1 rows are NaN.
    return rv.rolling(h).mean().shift(-(h - 1))
