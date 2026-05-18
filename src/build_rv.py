"""src/build_rv.py: Stage 1. Daily realized variance and quarticity from 1-min OHLCV bars.

For each ticker in TICKERS this script:
  1. Loads data/raw/<TICKER>.txt and parses MM/DD/YYYY + HH:MM into a DatetimeIndex.
  2. Restricts each day to the regular NYSE session 09:30 to 15:59 inclusive.
  3. Applies the bar-count + forward-fill policy:
       - drops days with fewer than MIN_BARS_PER_DAY bars (genuine half-days / halts),
       - reindexes the remaining days to the full 390-minute grid and forward-fills
         missing 1-min bars via previous-tick interpolation (BNHLS 2009),
       - back-fills only if the leading 09:30 bar itself is missing.
  4. Constructs 78 intraday 5-min log returns per day (see _five_min_returns_from_closes).
  5. Computes RV_t = sum of squared 5-min returns.
  6. Computes RQ_t = (n/3) * sum of (5-min return)^4 with n = 78.
  7. Writes data/interim/rv_<TICKER>.csv with columns: date, RV, RQ.

Output is in raw, dimensionless variance units. Any display rescaling (annualised
percent-vol = sqrt(RV * 252) * 100) happens downstream in plot/report code, never here.

Side-effects (per ticker):
  - Appends dropped-day rows (ticker, date, n_bars) to results/logs/dropped_days.csv.
  - Appends filled-minute rows (ticker, date, n_filled) to results/logs/filled_minutes.csv.
  - Replaces any pre-existing rows for the same ticker in those log files, so re-runs
    overwrite cleanly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_rv")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
DROPPED_DAYS_PATH = LOGS_DIR / "dropped_days.csv"
FILLED_MINUTES_PATH = LOGS_DIR / "filled_minutes.csv"

TICKERS = ["AAPL", "JPM", "AMZN"]
SESSION_START = pd.Timestamp("09:30").time()
SESSION_END = pd.Timestamp("15:59").time()
EXPECTED_BARS_PER_DAY = 390
MIN_BARS_PER_DAY = 350  # keep days with >= 350 bars; rest dropped as half-days/halts
N_5MIN_RETURNS = 78
BARS_PER_5MIN = 5


def load_raw(ticker: str) -> pd.DataFrame:
    """Read a 1-min OHLCV file and return a DataFrame indexed by minute timestamp."""
    path = RAW_DIR / f"{ticker}.txt"
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"date": str, "time": str, "open": np.float64, "high": np.float64,
               "low": np.float64, "close": np.float64, "volume": np.int64},
    )
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["date", "time"]).set_index(ts).sort_index()
    df.index.name = "timestamp"
    return df


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars whose time-of-day lies within [09:30, 15:59]."""
    mask = (df.index.time >= SESSION_START) & (df.index.time <= SESSION_END)
    return df.loc[mask]


def _five_min_returns_from_closes(closes: np.ndarray) -> np.ndarray:
    """Given a full 390-element close vector (no NaN), return 78 5-min log returns.

    Sampling indices = [0, 4, 9, 14, ..., 389], giving 79 prices. 78 consecutive log
    differences are returned. First return spans 09:30:59 -> 09:34:59 (a 4-minute
    return; anchored on the 09:30 bar's close).
    """
    assert closes.shape == (EXPECTED_BARS_PER_DAY,), f"expected {EXPECTED_BARS_PER_DAY} closes"
    sample_idx = np.concatenate((
        [0],
        np.arange(BARS_PER_5MIN - 1, EXPECTED_BARS_PER_DAY, BARS_PER_5MIN),
    ))
    assert len(sample_idx) == N_5MIN_RETURNS + 1
    return np.diff(np.log(closes[sample_idx]))


def compute_day_rv_rq(day_df: pd.DataFrame, day_normalized: pd.Timestamp) -> tuple[float, float, int]:
    """Process one trading day: reindex to full 390-minute grid, forward-fill any
    missing 1-min bars, and compute (RV, RQ, n_filled_minutes).

    Forward-fill (previous-tick interpolation) is the standard TAQ-cleaning rule
    from Barndorff-Nielsen, Hansen, Lunde & Shephard (2009), which the paper §2
    explicitly cites as its cleaning reference. Filled minutes contribute zero
    1-min returns, which is the no-trade ≡ no-price-change assumption.
    """
    expected_index = pd.date_range(
        start=day_normalized + pd.Timedelta(hours=9, minutes=30),
        end=day_normalized + pd.Timedelta(hours=15, minutes=59),
        freq="1min",
    )
    close = day_df["close"].reindex(expected_index)
    n_filled = int(close.isna().sum())
    close = close.ffill()
    if pd.isna(close.iloc[0]):
        close = close.bfill()
    assert close.notna().all(), f"unfillable NaNs on {day_normalized.date()}"
    rets = _five_min_returns_from_closes(close.to_numpy())
    rv = float(np.sum(rets ** 2))
    rq = float((N_5MIN_RETURNS / 3.0) * np.sum(rets ** 4))
    return rv, rq, n_filled


def reset_ticker_in_log(ticker: str, log_path: Path) -> None:
    """Remove any existing rows for `ticker` from `log_path`, so re-runs overwrite."""
    if not log_path.exists():
        return
    existing = pd.read_csv(log_path)
    remaining = existing.loc[existing["ticker"] != ticker]
    if remaining.empty:
        log_path.unlink()
    else:
        remaining.to_csv(log_path, index=False)


def append_log(ticker: str, df: pd.DataFrame, log_path: Path) -> None:
    """Append per-ticker rows to a log file with a leading ticker column."""
    if df.empty:
        return
    out = df.copy()
    out.insert(0, "ticker", ticker)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    header = not log_path.exists()
    out.to_csv(log_path, mode="a", header=header, index=False)


def build_rv_for_ticker(ticker: str, min_bars: int = MIN_BARS_PER_DAY) -> pd.DataFrame:
    """Build the daily RV/RQ DataFrame for one ticker, end to end."""
    log.info("processing %s", ticker)
    raw = load_raw(ticker)
    log.info("  loaded %d 1-min bars spanning %s to %s",
             len(raw), raw.index.min().date(), raw.index.max().date())
    raw = filter_session(raw)

    counts = raw.groupby(raw.index.normalize()).size()
    kept_dates = counts[counts >= min_bars].index
    dropped = counts[counts < min_bars].rename("n_bars").reset_index()
    dropped.columns = ["date", "n_bars"]

    log.info("  kept %d days (>= %d bars), dropped %d days (< %d bars)",
             len(kept_dates), min_bars, len(dropped), min_bars)
    if not dropped.empty:
        sample = dropped.head(10).to_dict(orient="records")
        log.info("  first dropped days: %s", sample)

    reset_ticker_in_log(ticker, DROPPED_DAYS_PATH)
    reset_ticker_in_log(ticker, FILLED_MINUTES_PATH)
    append_log(ticker, dropped, DROPPED_DAYS_PATH)

    raw = raw.loc[raw.index.normalize().isin(kept_dates)]

    records: list[dict] = []
    fill_records: list[dict] = []
    for day, day_df in raw.groupby(raw.index.normalize()):
        rv, rq, n_filled = compute_day_rv_rq(day_df, day)
        records.append({"date": day.date(), "RV": rv, "RQ": rq})
        if n_filled > 0:
            fill_records.append({"date": day.date(), "n_filled": n_filled})

    out = pd.DataFrame(records).set_index("date")
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    if fill_records:
        fr = pd.DataFrame(fill_records)
        log.info("  %d/%d kept days needed forward-fill (median %d min, max %d min)",
                 len(fr), len(out), int(fr["n_filled"].median()), int(fr["n_filled"].max()))
        append_log(ticker, fr, FILLED_MINUTES_PATH)
    else:
        log.info("  no forward-fill needed (all kept days had full 390 bars)")

    return out


def main(argv: list[str] | None = None) -> None:
    """Build daily RV/RQ for the tickers passed on the CLI (default: all three)."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tickers = (argv if argv is not None else sys.argv[1:]) or TICKERS
    for ticker in tickers:
        rv_df = build_rv_for_ticker(ticker)
        out_path = INTERIM_DIR / f"rv_{ticker}.csv"
        rv_df.to_csv(out_path)
        log.info("  wrote %s (%d rows)", out_path, len(rv_df))
        rv = rv_df["RV"]
        ann_vol_pct = np.sqrt(rv * 252) * 100
        log.info("  RV  raw   mean=%.3e median=%.3e max=%.3e",
                 rv.mean(), rv.median(), rv.max())
        log.info("  Ann-vol %% mean=%.2f median=%.2f max=%.2f",
                 ann_vol_pct.mean(), ann_vol_pct.median(), ann_vol_pct.max())


if __name__ == "__main__":
    main()
