"""src/inference/regime_split.py: Stage 11. VIX-tercile regime split of test-set MSE.

For each horizon h in {1, 5, 22} and each stock on M_ALL, the test set is split
into three VIX regimes — low / mid / high volatility — by tercile of the VIX
predictor already in the master DataFrame. Within each regime we compute the
relative MSE vs HAR for the 5 representative models (HAR-X, LogHAR, ElasticNet,
RF, NN_2_e10).

Regime definition: pd.qcut on the *rank* of the VIX column (rank-based so the
three bins are exactly equal-count regardless of VIX ties), restricted to each
cell's test dates. VIX in the master is the lagged (shift-1) VIX predictor; it is
market-wide, so the regime dates are essentially common across the three stocks.

Output: results/tables/regime_split_h{1,5,22}.csv — long format
        (ticker, regime, model, n_days, vix_lo, vix_hi, mse, rel_mse_to_HAR).

Framing: this is *context* for why our 2023-24 test-period numbers differ from
the paper's 2014-17 test period — NOT a deliberate regime-extension claim.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regime_split")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TICKERS = ["AAPL", "JPM", "AMZN"]
HORIZONS = [1, 5, 22]
INFO_SET = "M_ALL"
BASELINE = "HAR"
REPORT_MODELS = ["HAR-X", "LogHAR", "ElasticNet", "RF", "NN_2_e10"]
ALL_MODELS = [BASELINE] + REPORT_MODELS
REGIME_LABELS = ["low", "mid", "high"]


def load_loss(ticker: str, model: str, h: int) -> pd.Series:
    """Per-day squared-error loss Series for one (ticker, M_ALL, model, h) cell."""
    path = PREDICTIONS_DIR / f"{ticker}_{INFO_SET}_{model}_h{h}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return ((df["actual"] - df["predicted"]) ** 2).rename(model)


def assign_regime(ticker: str, test_dates: pd.DatetimeIndex) -> pd.Series:
    """VIX-tercile regime label for each test date. Rank-based qcut → equal bins."""
    master = pd.read_csv(FINAL_DIR / f"master_{ticker}.csv",
                         parse_dates=["date"], index_col="date")
    vix = master["VIX"].reindex(test_dates)
    if vix.isna().any():
        log.warning("  %s: %d test dates have NaN VIX — excluded from all regimes",
                    ticker, int(vix.isna().sum()))
    regime = pd.qcut(vix.rank(method="first"), 3, labels=REGIME_LABELS)
    return vix, regime


def run_horizon(h: int) -> pd.DataFrame:
    """Compute the regime-split relative-MSE table for one horizon."""
    rows = []
    for ticker in TICKERS:
        losses = pd.concat([load_loss(ticker, m, h) for m in ALL_MODELS], axis=1)
        vix, regime = assign_regime(ticker, losses.index)
        for reg in REGIME_LABELS:
            mask = (regime == reg)
            sub = losses.loc[mask.to_numpy()]
            mse = sub.mean()
            mse_har = mse[BASELINE]
            vix_reg = vix.loc[mask.to_numpy()]
            for model in REPORT_MODELS:
                rows.append({
                    "ticker": ticker,
                    "regime": reg,
                    "model": model,
                    "n_days": int(mask.sum()),
                    "vix_lo": round(float(vix_reg.min()), 2),
                    "vix_hi": round(float(vix_reg.max()), 2),
                    "mse": float(mse[model]),
                    "rel_mse_to_HAR": float(mse[model] / mse_har),
                })
    return pd.DataFrame(rows)


def print_horizon(h: int, df: pd.DataFrame) -> None:
    """Print a readable regime × model pivot of relative MSE for one horizon."""
    print("\n" + "=" * 78)
    print(f"  h={h}  —  relative MSE vs HAR by VIX regime (M_ALL)")
    print("=" * 78)
    for ticker in TICKERS:
        sub = df[df["ticker"] == ticker]
        pivot = sub.pivot(index="regime", columns="model",
                          values="rel_mse_to_HAR").reindex(REGIME_LABELS)[REPORT_MODELS]
        nday = sub.groupby("regime")["n_days"].first().reindex(REGIME_LABELS)
        vlo = sub.groupby("regime")["vix_lo"].first().reindex(REGIME_LABELS)
        vhi = sub.groupby("regime")["vix_hi"].first().reindex(REGIME_LABELS)
        print(f"\n  {ticker}  (regime VIX ranges: "
              f"low {vlo['low']:.1f}-{vhi['low']:.1f}, "
              f"mid {vlo['mid']:.1f}-{vhi['mid']:.1f}, "
              f"high {vlo['high']:.1f}-{vhi['high']:.1f};  "
              f"n/regime: {nday['low']}/{nday['mid']}/{nday['high']})")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))


def flip_analysis(tables: dict[int, pd.DataFrame]) -> None:
    """Print, per horizon, which models cross rel-MSE 1.0 across regimes."""
    print("\n" + "=" * 78)
    print("  Cross-regime flip analysis (rel MSE < 1 = beats HAR, > 1 = loses)")
    print("=" * 78)
    for h, df in tables.items():
        print(f"\n  h={h}:")
        for model in REPORT_MODELS:
            sub = df[df["model"] == model]
            rmin, rmax = sub["rel_mse_to_HAR"].min(), sub["rel_mse_to_HAR"].max()
            n_beat = int((sub["rel_mse_to_HAR"] < 1.0).sum())
            n_cells = len(sub)
            if rmax < 1.0:
                verdict = "beats HAR in ALL 9 cells"
            elif rmin > 1.0:
                verdict = "loses to HAR in ALL 9 cells"
            else:
                verdict = f"FLIPS — beats HAR in {n_beat}/{n_cells} cells"
            print(f"    {model:11s}: rel MSE {rmin:.3f}–{rmax:.3f}   {verdict}")


def main() -> None:
    """Run Stage 11 for all three horizons; write one CSV each; print diagnostics."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for h in HORIZONS:
        df = run_horizon(h)
        out = TABLES_DIR / f"regime_split_h{h}.csv"
        df.to_csv(out, index=False)
        log.info("wrote %s  rows=%d", out.name, len(df))
        tables[h] = df
    for h in HORIZONS:
        print_horizon(h, tables[h])
    flip_analysis(tables)


if __name__ == "__main__":
    main()
