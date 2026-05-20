"""Merge realized variance, lagged HAR terms, stock predictors, and macro covariates into per-stock master DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_master")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

TICKERS = ["AAPL", "JPM", "AMZN"]
HAR_WEEKLY = 5
HAR_MONTHLY = 22


def load_rv_with_lags(ticker: str) -> pd.DataFrame:
    """Load rv_<ticker>.csv and return RV plus its 1-day, 5-day, 22-day lagged means."""
    rv = pd.read_csv(INTERIM_DIR / f"rv_{ticker}.csv",
                     parse_dates=["date"], index_col="date")
    out = pd.DataFrame(index=rv.index)
    out["RV"] = rv["RV"]
    out["RVD"] = rv["RV"].shift(1)
    out["RVW"] = rv["RV"].rolling(HAR_WEEKLY).mean().shift(1)
    out["RVM"] = rv["RV"].rolling(HAR_MONTHLY).mean().shift(1)
    out["RQ_lag"] = rv["RQ"].shift(1)
    return out


def load_stock_predictors(ticker: str) -> pd.DataFrame:
    """Load covariates_<ticker>.csv and apply the per-feature predictive-lag policy."""
    cov = pd.read_csv(INTERIM_DIR / f"covariates_{ticker}.csv",
                      parse_dates=["date"], index_col="date")
    out = pd.DataFrame(index=cov.index)
    out["M1W"] = cov["M1W"]                          # already at correct lag
    out["d_log_dvol"] = cov["d_log_dvol"].shift(1)   # shift for predictive validity
    out["EA"] = cov["EA"]                            # forward-looking; no shift
    return out


def align_macro(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align the external macro series to the trading-day index and shift one day for predictive validity."""
    vix = pd.read_csv(EXTERNAL_DIR / "vix.csv",
                      parse_dates=["date"], index_col="date")["value"]
    vix_aligned = vix.reindex(trading_dates)

    us3m = pd.read_csv(EXTERNAL_DIR / "us3m.csv",
                       parse_dates=["date"], index_col="date")
    # Policy: ffill rate first, then diff (avoids diff-after-NaN propagation)
    rate_aligned = us3m["rate"].reindex(trading_dates).ffill()
    d_rate = rate_aligned.diff()

    hsi = pd.read_csv(EXTERNAL_DIR / "hsi.csv",
                      parse_dates=["date"], index_col="date")
    # Policy: missing US-trading-day-but-HK-holiday rows -> log_ret_sq = 0
    hsi_sq = hsi["log_ret_sq"].reindex(trading_dates).fillna(0)

    ads = pd.read_csv(EXTERNAL_DIR / "ads.csv",
                      parse_dates=["date"], index_col="date")["value"]
    ads_aligned = ads.reindex(trading_dates)

    epu = pd.read_csv(EXTERNAL_DIR / "epu.csv",
                      parse_dates=["date"], index_col="date")["value"]
    epu_aligned = epu.reindex(trading_dates)

    out = pd.DataFrame({
        "VIX":    vix_aligned.shift(1),
        "d_US3M": d_rate.shift(1),
        "HSI":    hsi_sq.shift(1),
        "ADS":    ads_aligned.shift(1),
        "EPU":    epu_aligned.shift(1),
    }, index=trading_dates)
    return out


def build_master(ticker: str) -> pd.DataFrame:
    """Assemble the per-ticker master DataFrame and drop leading RVM-NaN rows."""
    log.info("building master for %s", ticker)
    rv_df = load_rv_with_lags(ticker)
    cov_df = load_stock_predictors(ticker)
    macro_df = align_macro(rv_df.index)

    df = rv_df.join(cov_df).join(macro_df)
    log.info("  pre-drop:  rows=%d  any-NaN rows=%d",
             len(df), int(df.isna().any(axis=1).sum()))

    df = df.dropna(subset=["RVM"])
    log.info("  post-drop: rows=%d  any-NaN rows=%d  per-col-NaN=%s",
             len(df), int(df.isna().any(axis=1).sum()),
             df.isna().sum().to_dict())
    return df


def main() -> None:
    """Build and save master_<TICKER>.csv for each ticker in TICKERS."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        df = build_master(ticker)
        out = FINAL_DIR / f"master_{ticker}.csv"
        df.to_csv(out)
        log.info("  wrote %s  rows=%d  cols=%d",
                 out.name, len(df), len(df.columns))


if __name__ == "__main__":
    main()
