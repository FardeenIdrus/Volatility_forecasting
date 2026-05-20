"""Exhibit 2: relative MSE vs HAR split by realized-RV decile (mirrors paper Figure 5)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rv_decile_split")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
TABLES_REGIME = PROJECT_ROOT / "results" / "tables" / "regime"

TICKERS = ["AAPL", "JPM", "AMZN"]
HORIZONS = [1, 5, 22]
INFO_SET = "M_ALL"
BASELINE = "HAR"
REPORT_MODELS = ["HAR-X", "LogHAR", "ElasticNet", "RF", "NN_2_e10"]
ALL_MODELS = [BASELINE] + REPORT_MODELS
N_DECILES = 10


def load_cell(ticker: str, h: int) -> pd.DataFrame:
    """Load the realized target and per-model test predictions for one (ticker, M_ALL, h) cell."""
    actual = None
    preds = {}
    for model in ALL_MODELS:
        path = PREDICTIONS_DIR / f"h{h}" / f"{ticker}_{INFO_SET}_{model}_h{h}.csv"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        preds[model] = df["predicted"].rename(model)
        if actual is None:
            actual = df["actual"].rename("actual")
        else:
            assert actual.equals(df["actual"].rename("actual")), \
                f"realized target mismatch: {ticker} h{h} {model}"
    return pd.concat([actual, *preds.values()], axis=1)


def run_horizon(h: int) -> pd.DataFrame:
    """Build the long-format realized-RV-decile relative-MSE table for one horizon."""
    rows = []
    for ticker in TICKERS:
        cell = load_cell(ticker, h)
        actual = cell["actual"]
        # Rank-based qcut so the 10 deciles are exactly equal-count regardless of ties.
        decile = pd.qcut(actual.rank(method="first"), N_DECILES,
                         labels=range(1, N_DECILES + 1))
        for d in range(1, N_DECILES + 1):
            sub = cell.loc[decile == d]
            a = sub["actual"]
            mse = {m: float(((a - sub[m]) ** 2).mean()) for m in ALL_MODELS}
            mse_har = mse[BASELINE]
            for model in REPORT_MODELS:
                rows.append({
                    "stock": ticker,
                    "decile": d,
                    "rv_lo": float(a.min()),
                    "rv_hi": float(a.max()),
                    "model": model,
                    "n_days": int(len(sub)),
                    "mse": mse[model],
                    "rel_mse_to_HAR": mse[model] / mse_har,
                })
    return pd.DataFrame(rows)


def print_horizon(h: int, df: pd.DataFrame) -> None:
    """Print a decile x model pivot of relative MSE for one horizon."""
    print("\n" + "=" * 78)
    print(f"  h={h}  —  relative MSE vs HAR by realized-RV decile (M_ALL)")
    print("=" * 78)
    for ticker in TICKERS:
        sub = df[df["stock"] == ticker]
        pivot = sub.pivot(index="decile", columns="model",
                          values="rel_mse_to_HAR")[REPORT_MODELS]
        nday = sub.groupby("decile")["n_days"].first()
        print(f"\n  {ticker}  (decile 1 = lowest realized RV, 10 = highest;  "
              f"n/decile {nday.min()}-{nday.max()})")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))


def loghar_summary(tables: dict[int, pd.DataFrame]) -> None:
    """Print, per horizon and stock, the deciles where LogHAR beats HAR."""
    print("\n" + "=" * 78)
    print("  LogHAR rel-MSE vs HAR by realized-RV decile (< 1 = LogHAR beats HAR)")
    print("=" * 78)
    for h in HORIZONS:
        print(f"\n  h={h}:")
        df = tables[h]
        for ticker in TICKERS:
            sub = (df[(df["stock"] == ticker) & (df["model"] == "LogHAR")]
                   .set_index("decile")["rel_mse_to_HAR"])
            beats = [int(d) for d in sub.index if sub[d] < 1.0]
            print(f"    {ticker:5s}: beats HAR in deciles {beats};  "
                  f"D1={sub[1]:.3f}  D10={sub[10]:.3f}  "
                  f"min={sub.min():.3f}@D{int(sub.idxmin())}")


def main() -> None:
    """Compute the realized-RV-decile relative-MSE split for all horizons; write one CSV each."""
    TABLES_REGIME.mkdir(parents=True, exist_ok=True)
    tables = {}
    for h in HORIZONS:
        df = run_horizon(h)
        out = TABLES_REGIME / f"rv_decile_split_h{h}.csv"
        df.to_csv(out, index=False)
        log.info("wrote %s  rows=%d", out.name, len(df))
        tables[h] = df
    for h in HORIZONS:
        print_horizon(h, tables[h])
    loghar_summary(tables)


if __name__ == "__main__":
    main()
