"""src/build_predictors.py: Stage 3. Stock-specific predictors (M1W, Δlog($VOL), EA).

For each ticker in TICKERS:
  1. Loads the corresponding RV trading-day index from data/interim/rv_<TICKER>.csv.
  2. Re-reads the raw 1-min OHLCV file (via build_rv.load_raw + filter_session) to
     compute per-day close (last bar's close) and total volume.
  3. Computes:
       - M1W_t  = sum_{i=1..5} r_{t-i}  where r_t = log(c_t) - log(c_{t-1})
         (5-trading-day cumulative log return ending at t-1)
       - d_log_dvol_t = log(c_t * v_t) - log(c_{t-1} * v_{t-1})
  4. Fetches earnings-announcement dates from yfinance (limit=100, gives us ~25
     years of history per ticker, well beyond our 2016-2024 window).
       - EA_t = 1 if date t is an EA date (as recorded by yfinance), else 0.
       - We mark EA on the announcement date itself, not the next-day reaction date.
         This is a literal reading of the paper's "EA on the day of forecast" and
         matches what the paper presumably does. To be disclosed in report §4.
  5. Writes data/interim/covariates_<TICKER>.csv with columns: date, M1W, d_log_dvol, EA.

NaN expectations:
  - M1W: NaN for the first ~6 trading days of the sample (need 5 prior log returns,
    and the first return is itself NaN). Cleared by Stage 4's RV^M 22-day lag drop.
  - d_log_dvol: NaN for the first trading day. Same.
  - EA: never NaN (always 0 or 1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from build_rv import load_raw, filter_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_predictors")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

TICKERS = ["AAPL", "JPM", "AMZN"]
M1W_WINDOW = 5
YF_EARNINGS_LIMIT = 100  # Yahoo caps at 100; gives us ~25 years per ticker, plenty


def daily_close_and_volume(ticker: str) -> pd.DataFrame:
    """From the raw 1-min file, aggregate to per-day (close, total volume).

    Uses every trading day for which we have any session data, including
    half-days that Stage 1 dropped from RV. The close on a half-day is the
    close of its last 1-min bar (typically 13:00); the volume is the sum of
    its session 1-min volumes. Including half-days gives us a denser daily
    series for computing daily log returns and dollar volume. We then re-filter
    to the RV trading-day index when saving, so the final covariates align
    one-to-one with RV.
    """
    raw = filter_session(load_raw(ticker))
    daily = raw.groupby(raw.index.normalize()).agg(
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index.name = "date"
    return daily


def compute_m1w_and_dvol(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute M1W and Δlog($VOL) from a per-day (close, volume) DataFrame."""
    log_close = np.log(daily["close"])
    log_ret = log_close.diff()                          # r_t = log c_t - log c_{t-1}
    m1w = log_ret.rolling(M1W_WINDOW).sum().shift(1)    # sum_{i=1..5} r_{t-i}

    log_dvol = np.log(daily["close"] * daily["volume"])
    d_log_dvol = log_dvol.diff()                        # log $VOL_t - log $VOL_{t-1}

    out = pd.DataFrame({"M1W": m1w, "d_log_dvol": d_log_dvol})
    out.index.name = "date"
    return out


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """Fetch unique earnings-announcement dates from yfinance. Returns a
    DatetimeIndex of date-only Timestamps (time stripped, deduped, sorted)."""
    log.info("  fetching earnings dates from yfinance for %s", ticker)
    ed = yf.Ticker(ticker).get_earnings_dates(limit=YF_EARNINGS_LIMIT)
    if ed is None or ed.empty:
        log.warning("  yfinance returned no earnings dates for %s", ticker)
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(sorted(set(ed.index.date)))
    log.info("  got %d unique EA dates spanning %s to %s",
             len(dates), dates.min().date(), dates.max().date())
    return dates


def build_for_ticker(ticker: str) -> pd.DataFrame:
    """Build the predictors DataFrame for one ticker, aligned to its RV trading-day index."""
    log.info("processing %s", ticker)
    rv_df = pd.read_csv(INTERIM_DIR / f"rv_{ticker}.csv",
                        parse_dates=["date"], index_col="date")
    rv_dates = rv_df.index
    log.info("  RV trading-day index: %d days, %s to %s",
             len(rv_dates), rv_dates.min().date(), rv_dates.max().date())

    daily = daily_close_and_volume(ticker)
    log.info("  daily aggregates from raw: %d days (includes half-days)", len(daily))

    df = compute_m1w_and_dvol(daily)
    df = df.reindex(rv_dates)  # align to RV trading-day index

    ea_dates = fetch_earnings_dates(ticker)
    df["EA"] = rv_dates.isin(ea_dates).astype(np.int64)

    ea_in_window = int(df["EA"].sum())
    log.info("  EA dates in our trading window: %d", ea_in_window)
    return df


def main() -> None:
    """Build covariates_<TICKER>.csv for each ticker in TICKERS."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        df = build_for_ticker(ticker)
        out = INTERIM_DIR / f"covariates_{ticker}.csv"
        df.to_csv(out)
        log.info("  wrote %s  rows=%d  nan_counts=%s",
                 out, len(df), df.isna().sum().to_dict())


if __name__ == "__main__":
    main()
