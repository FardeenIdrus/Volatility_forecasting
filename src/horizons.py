"""src/horizons.py: multi-step-ahead target helper.

Used by Stages 6, 7, 8 to produce h-day-ahead targets consistent with paper §4's
RV_{t+h|t+1} convention plus our shift-1 feature alignment in merge_master.py.

Under that alignment, the row at date d has features representing info through
end of day d-1 and target RV_d. So:
  - h=1:  target = RV at row d = RV_d.
  - h=22: target = mean(RV_d, RV_{d+1}, ..., RV_{d+21}) per paper §4.
The last h-1 rows have NaN target and must be dropped before fitting.
"""
from __future__ import annotations

import pandas as pd


def build_h_step_target(rv: pd.Series, h: int) -> pd.Series:
    """h-step-ahead average target aligned to our feature-shift convention.

    For h=1 returns RV unchanged. For h>1 returns RV.rolling(h).mean().shift(-(h-1))
    so the row at date d contains mean(RV_d, ..., RV_{d+h-1}).
    Last h-1 rows are NaN; caller must dropna(subset=["y_target"]) before fitting.
    """
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    if h == 1:
        return rv.copy()
    return rv.rolling(h).mean().shift(-(h - 1))
