"""Stock-specific predictors: 1-week momentum, dollar-volume change, and an earnings dummy."""

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
    """Aggregate the raw 1-min file to per-day (close, total volume), including half-days."""
    # Half-days are kept here for a denser daily log-return series; build_for_ticker
    # re-aligns the result to the RV trading-day index.
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
    log_ret = log_close.diff()
    m1w = log_ret.rolling(M1W_WINDOW).sum().shift(1)    # shift(1): sum of r_{t-5..t-1}

    log_dvol = np.log(daily["close"] * daily["volume"])
    d_log_dvol = log_dvol.diff()

    out = pd.DataFrame({"M1W": m1w, "d_log_dvol": d_log_dvol})
    out.index.name = "date"
    return out


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """Fetch unique earnings-announcement dates from yfinance as a sorted DatetimeIndex."""
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
    df = df.reindex(rv_dates)

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
