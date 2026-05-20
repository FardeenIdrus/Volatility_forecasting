"""src/inference/run_stage8.py: Stage 8. Statistical inference on 1-day predictions.

For each (stock × info-set) combination:
  1. Relative MSE vs HAR baseline.
  2. Pairwise Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample
     correction. One-sided alternative H1: MSE_i > MSE_j (model i is worse).
  3. Model Confidence Set at 90% confidence using arch.bootstrap.MCS, stationary
     bootstrap, 10,000 replications, squared-error loss.

Reads results/predictions/<ticker>_<infoset>_<model>_h1.csv.
Writes:
  results/tables/summary.csv          long: ticker, info_set, model, n_test, mse, rel_mse_to_HAR
  results/tables/mcs_inclusion.csv    long: ticker, info_set, model, in_mcs, mcs_pvalue
  results/tables/mse_<infoset>.csv          wide pivot (rows=model, cols=ticker)
  results/tables/relative_mse_<infoset>.csv  same but relative-to-HAR
  results/tables/dm_<ticker>_<infoset>_stat.csv     pairwise DM stat matrix
  results/tables/dm_<ticker>_<infoset>_pvalue.csv   pairwise one-sided p-value matrix
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stage8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TICKERS = ["AAPL", "JPM", "AMZN"]
INFOSETS = ["M_HAR", "M_ALL"]
HORIZON = 1  # overridden via CLI: `python src/inference/run_stage8.py 22`
# DM Newey-West HAC bandwidth = HORIZON - 1; HLN small-sample multiplier uses HORIZON.

# Model ordering for tables: HAR family first, then ML
MODEL_ORDER = {
    "M_HAR": ["HAR", "LogHAR", "HARQ", "Lasso", "ElasticNet", "RF", "GB", "NN_2_e10"],
    "M_ALL": ["HAR", "HAR-X", "LogHAR", "HARQ", "Lasso", "ElasticNet", "RF", "GB", "NN_2_e10"],
}

BASELINE_MODEL = "HAR"
MCS_SIZE = 0.10       # 1 - 0.90 confidence
MCS_REPS = 10000
MCS_SEED = 42


def load_predictions(ticker: str, info_set: str, model: str) -> pd.DataFrame | None:
    """Load one prediction CSV; return None if the file doesn't exist."""
    path = PREDICTIONS_DIR / f"{ticker}_{info_set}_{model}_h{HORIZON}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def squared_loss(df: pd.DataFrame) -> pd.Series:
    """Per-day squared error loss from a (actual, predicted) DataFrame."""
    return (df["actual"] - df["predicted"]) ** 2


def collect_losses(ticker: str, info_set: str) -> pd.DataFrame:
    """Return a T x M DataFrame of per-day losses for all models available on disk."""
    losses = {}
    for model in MODEL_ORDER[info_set]:
        df = load_predictions(ticker, info_set, model)
        if df is None:
            log.warning("  missing: %s × %s × %s", ticker, info_set, model)
            continue
        losses[model] = squared_loss(df)
    return pd.DataFrame(losses)


def dm_test_hln(loss_i: pd.Series, loss_j: pd.Series, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    H0: E[loss_i - loss_j] = 0
    H1: E[loss_i - loss_j] > 0   (one-sided; model i has higher loss = worse)

    Newey-West HAC variance with Bartlett kernel, bandwidth h-1.
    HLN multiplier: sqrt((T + 1 - 2h + h(h-1)/T) / T).
    p-value under t_{T-1}.
    """
    d = (loss_i - loss_j).dropna().to_numpy()
    T = len(d)
    if T == 0:
        return float("nan"), float("nan")
    d_bar = float(d.mean())
    centered = d - d_bar
    gamma_0 = float(np.sum(centered ** 2) / T)
    nw_var = gamma_0
    for lag in range(1, h):
        gamma_l = float(np.sum(centered[lag:] * centered[:-lag]) / T)
        nw_var += 2 * (1 - lag / h) * gamma_l
    if nw_var <= 0:
        return float("nan"), float("nan")
    dm_unadj = d_bar / np.sqrt(nw_var / T)
    hln_mult = float(np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm_stat = dm_unadj * hln_mult
    p_value = 1.0 - float(stats.t.cdf(dm_stat, df=T - 1))
    return float(dm_stat), float(p_value)


def compute_pairwise_dm(losses: pd.DataFrame, h: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise DM stat and one-sided p-value matrices.

    Cell [i, j] is the test of H1: model_i has higher MSE than model_j.
    Diagonal cells are set to (0, 1) by convention.
    """
    models = list(losses.columns)
    stat = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    pval = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for i in models:
        for j in models:
            if i == j:
                stat.loc[i, j] = 0.0
                pval.loc[i, j] = 1.0
                continue
            s, p = dm_test_hln(losses[i], losses[j], h=h)
            stat.loc[i, j] = s
            pval.loc[i, j] = p
    return stat, pval


def run_mcs(losses: pd.DataFrame) -> pd.DataFrame:
    """Run MCS at 90% confidence; return DataFrame [model, in_mcs, mcs_pvalue]."""
    mcs = MCS(
        losses.to_numpy(),
        size=MCS_SIZE,
        reps=MCS_REPS,
        bootstrap="stationary",
        seed=MCS_SEED,
    )
    mcs.compute()
    included_idx = set(mcs.included)
    pvals = mcs.pvalues
    rows = []
    for idx, model in enumerate(losses.columns):
        try:
            p = float(pvals.loc[idx, "Pvalue"])
        except Exception:
            p = float("nan")
        rows.append({"model": model, "in_mcs": idx in included_idx, "mcs_pvalue": p})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Run Stage 8 inference: relative MSE, pairwise DM, and MCS for all cells.

    CLI:
      python src/inference/run_stage8.py        # h=1 (default)
      python src/inference/run_stage8.py 22     # h=22 (paper §4 monthly horizon)
    """
    global HORIZON
    args = argv if argv is not None else sys.argv[1:]
    if args:
        HORIZON = int(args[0])
    log.info("HORIZON h=%d", HORIZON)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    mcs_long_rows: list[pd.DataFrame] = []

    for ticker in TICKERS:
        for info_set in INFOSETS:
            log.info("== %s × %s ==", ticker, info_set)
            losses = collect_losses(ticker, info_set)
            if losses.empty:
                log.warning("  no predictions for %s × %s, skipping", ticker, info_set)
                continue
            log.info("  models found: %s", list(losses.columns))
            log.info("  n_test obs: %d", len(losses))

            # Relative MSE
            mse_per_model = losses.mean()
            mse_har = mse_per_model[BASELINE_MODEL]
            for model in losses.columns:
                summary_rows.append({
                    "ticker": ticker,
                    "info_set": info_set,
                    "model": model,
                    "n_test": len(losses),
                    "test_mse": float(mse_per_model[model]),
                    "rel_mse_to_HAR": float(mse_per_model[model] / mse_har),
                })

            # Pairwise DM
            log.info("  computing pairwise DM (%d models, h=%d)", len(losses.columns), HORIZON)
            stat, pval = compute_pairwise_dm(losses, h=HORIZON)
            stat.to_csv(TABLES_DIR / f"dm_{ticker}_{info_set}_h{HORIZON}_stat.csv")
            pval.to_csv(TABLES_DIR / f"dm_{ticker}_{info_set}_h{HORIZON}_pvalue.csv")

            # MCS
            log.info("  computing MCS (%d reps, seed=%d)", MCS_REPS, MCS_SEED)
            mcs_df = run_mcs(losses)
            mcs_df.insert(0, "ticker", ticker)
            mcs_df.insert(1, "info_set", info_set)
            mcs_long_rows.append(mcs_df)

    # Save aggregated tables
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / f"summary_h{HORIZON}.csv", index=False)

    mcs_combined = pd.concat(mcs_long_rows, ignore_index=True)
    mcs_combined.to_csv(TABLES_DIR / f"mcs_inclusion_h{HORIZON}.csv", index=False)

    # Wide-format pivots, one per info_set
    for info_set in INFOSETS:
        sub = summary[summary["info_set"] == info_set]
        order = [m for m in MODEL_ORDER[info_set] if m in sub["model"].unique()]
        pivot_mse = sub.pivot(index="model", columns="ticker", values="test_mse").reindex(order)
        pivot_rel = sub.pivot(index="model", columns="ticker", values="rel_mse_to_HAR").reindex(order)
        pivot_mse.to_csv(TABLES_DIR / f"mse_{info_set}_h{HORIZON}.csv")
        pivot_rel.to_csv(TABLES_DIR / f"relative_mse_{info_set}_h{HORIZON}.csv")

    # Console summary
    print("\n" + "=" * 80)
    print("  Relative MSE vs HAR (per stock × info-set)")
    print("=" * 80)
    for info_set in INFOSETS:
        sub = summary[summary["info_set"] == info_set]
        order = [m for m in MODEL_ORDER[info_set] if m in sub["model"].unique()]
        pivot = sub.pivot(index="model", columns="ticker", values="rel_mse_to_HAR").reindex(order)
        print(f"\n  {info_set}:")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("  MCS inclusion at 90% confidence (Y = in set, N = excluded)")
    print("=" * 80)
    for info_set in INFOSETS:
        sub = mcs_combined[mcs_combined["info_set"] == info_set]
        order = [m for m in MODEL_ORDER[info_set] if m in sub["model"].unique()]
        pivot = sub.pivot(index="model", columns="ticker", values="in_mcs").reindex(order)
        pivot_str = pivot.replace({True: "Y", False: "N", np.nan: "-"})
        print(f"\n  {info_set}:")
        print(pivot_str.to_string())


if __name__ == "__main__":
    main()
