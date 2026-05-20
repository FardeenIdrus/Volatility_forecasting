"""Download the five daily macro covariates (VIX, US3M, HSI, ADS, EPU) to data/external/."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_macro")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

START_DATE = pd.Timestamp("2015-12-01")  # ~1 month buffer before RV starts 2016-01-04, for first-diff
END_DATE = pd.Timestamp("2024-12-31")

FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id={sid}&cosd={cosd}&coed={coed}")
ADS_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/"
           "surveys-and-data/ads/ads_index_most_current_vintage.xlsx")
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"


def _http_get(url: str) -> bytes:
    """Download a URL using requests with its default User-Agent. Returns raw bytes."""
    log.info("  GET %s", url[:100] + ("..." if len(url) > 100 else ""))
    # FRED rejects browser-like User-Agents; the requests-library default works.
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def _save(df: pd.DataFrame, name: str) -> Path:
    """Write a DataFrame to data/external/<name>.csv and log row/date/NaN stats."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_DIR / f"{name}.csv"
    df.to_csv(path)
    log.info("  wrote %s  rows=%d  dates=%s..%s  nan_counts=%s",
             path.name, len(df), df.index.min().date(), df.index.max().date(),
             df.isna().sum().to_dict())
    return path


def _fred_series(series_id: str, col_name: str) -> pd.Series:
    """Fetch a daily FRED series via direct CSV. Returns Series indexed by date."""
    url = FRED_URL.format(sid=series_id,
                          cosd=START_DATE.strftime("%Y-%m-%d"),
                          coed=END_DATE.strftime("%Y-%m-%d"))
    raw = _http_get(url)
    df = pd.read_csv(io.BytesIO(raw), na_values=".")
    df["date"] = pd.to_datetime(df["observation_date"])
    s = (df.set_index("date")[series_id]
           .astype(np.float64)
           .sort_index()
           .rename(col_name))
    return s.loc[START_DATE:END_DATE]


def fetch_vix() -> None:
    """Download VIXCLS from FRED and save to data/external/vix.csv."""
    log.info("fetching VIX (FRED VIXCLS)")
    s = _fred_series("VIXCLS", "value")
    _save(s.to_frame(), "vix")


def fetch_us3m() -> None:
    """Download DTB3 from FRED, first-difference it, save to data/external/us3m.csv."""
    log.info("fetching US3M (FRED DTB3)")
    rate = _fred_series("DTB3", "rate")
    df = pd.DataFrame({"rate": rate, "d_rate": rate.diff()})
    _save(df, "us3m")


def fetch_hsi() -> None:
    """Download Hang Seng index from Yahoo, compute log return and its square."""
    log.info("fetching HSI (Yahoo ^HSI)")
    raw = yf.download(
        "^HSI",
        start=START_DATE.strftime("%Y-%m-%d"),
        end=(END_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False, auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame for ^HSI")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].astype(np.float64).sort_index()
    close.index.name = "date"
    log_ret = np.log(close).diff()
    df = pd.DataFrame({
        "close": close,
        "log_ret": log_ret,
        "log_ret_sq": log_ret ** 2,
    })
    _save(df.loc[START_DATE:END_DATE], "hsi")


def fetch_ads() -> None:
    """Download the Philly Fed ADS Business Conditions Index xlsx, save the index column."""
    log.info("fetching ADS (Philly Fed xlsx)")
    raw = _http_get(ADS_URL)
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    # Columns observed in 2026-05 vintage: Date (YYYY:MM:DD), ADS_Index, RECBARS
    df["date"] = pd.to_datetime(df["Date"], format="%Y:%m:%d")
    s = (df.set_index("date")["ADS_Index"]
           .astype(np.float64)
           .sort_index()
           .rename("value"))
    _save(s.loc[START_DATE:END_DATE].to_frame(), "ads")


def fetch_epu() -> None:
    """Download the daily US EPU index from policyuncertainty.com, save the index column."""
    log.info("fetching EPU (policyuncertainty.com)")
    raw = _http_get(EPU_URL)
    df = pd.read_csv(io.BytesIO(raw))
    # Columns observed in 2026-05 vintage: day, month, year, daily_policy_index
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    s = (df.set_index("date")["daily_policy_index"]
           .astype(np.float64)
           .sort_index()
           .rename("value"))
    _save(s.loc[START_DATE:END_DATE].to_frame(), "epu")


def main() -> None:
    """Fetch all five macro series in order and log a 'done' line at the end."""
    log.info("Stage 2: fetching macro covariates (target window %s to %s)",
             START_DATE.date(), END_DATE.date())
    fetch_vix()
    fetch_us3m()
    fetch_hsi()
    fetch_ads()
    fetch_epu()
    log.info("done")


if __name__ == "__main__":
    main()
