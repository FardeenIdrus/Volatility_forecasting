"""Chronological 70/10/20 train/validation/test split with training-set standardisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("split")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "data" / "final"

TICKERS = ["AAPL", "JPM", "AMZN"]
TARGET_COL = "RV"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# Test fraction is implicitly 1 - TRAIN_FRAC - VAL_FRAC = 0.20


@dataclass
class StockSplit:
    """One stock's chronological train/validation/test split and training-set scaler."""
    ticker: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_mean: pd.Series
    train_std: pd.Series
    X_train_std: pd.DataFrame
    X_val_std: pd.DataFrame
    X_test_std: pd.DataFrame


def split_stock(ticker: str,
                train_frac: float = TRAIN_FRAC,
                val_frac: float = VAL_FRAC) -> StockSplit:
    """Load master_<ticker>.csv, split 70/10/20 chronologically, and standardise."""
    path = FINAL_DIR / f"master_{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    # Training-set scaler. ddof=0 matches sklearn StandardScaler convention.
    train_mean = X_train.mean()
    train_std = X_train.std(ddof=0)

    if (train_std == 0).any():
        zero_cols = train_std[train_std == 0].index.tolist()
        raise ValueError(
            f"{ticker}: zero std in training set for columns {zero_cols}"
        )

    X_train_std = (X_train - train_mean) / train_std
    X_val_std = (X_val - train_mean) / train_std
    X_test_std = (X_test - train_mean) / train_std

    log.info("split %s  train=%d (%s..%s)  val=%d (%s..%s)  test=%d (%s..%s)",
             ticker,
             len(X_train), X_train.index.min().date(), X_train.index.max().date(),
             len(X_val), X_val.index.min().date(), X_val.index.max().date(),
             len(X_test), X_test.index.min().date(), X_test.index.max().date())

    return StockSplit(
        ticker=ticker,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        train_mean=train_mean, train_std=train_std,
        X_train_std=X_train_std,
        X_val_std=X_val_std,
        X_test_std=X_test_std,
    )


def split_all_stocks() -> dict[str, StockSplit]:
    """Return a {ticker: StockSplit} dict for all tickers in TICKERS."""
    return {tkr: split_stock(tkr) for tkr in TICKERS}


def _print_diagnostic(sp: StockSplit) -> None:
    """Print the per-stock split diagnostic block."""
    print("=" * 96)
    print(f"  {sp.ticker}")
    print("=" * 96)
    print(f"  train: {len(sp.X_train):4d} rows   "
          f"({sp.X_train.index.min().date()}  to  {sp.X_train.index.max().date()})")
    print(f"  val:   {len(sp.X_val):4d} rows   "
          f"({sp.X_val.index.min().date()}  to  {sp.X_val.index.max().date()})")
    print(f"  test:  {len(sp.X_test):4d} rows   "
          f"({sp.X_test.index.min().date()}  to  {sp.X_test.index.max().date()})")
    print()
    print(f"  training-set scaler (mean, std; ddof=0):")
    scaler_df = pd.DataFrame({"train_mean": sp.train_mean, "train_std": sp.train_std})
    print(scaler_df.to_string())
    print()
    print(f"  X_train_std head (first 3 rows):")
    print(sp.X_train_std.head(3).to_string())
    print()


def main() -> None:
    """Diagnostic-only entry point. Runs split for all 3 stocks and prints summary."""
    splits = split_all_stocks()
    print()
    for sp in splits.values():
        _print_diagnostic(sp)


if __name__ == "__main__":
    main()
