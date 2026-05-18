"""src/models/har_family.py: Stage 6. HAR family fits with rolling-window forecasts.

For each (ticker, info_set, model_name) at the 1-day horizon:
  - Build features per spec (see ModelSpec). LogHAR log-transforms RV-lags and
    additionally VIX in M_ALL; HARQ adds the RVD * sqrt(RQ_lag) interaction.
  - Initial fit on the first 1776-day window (train+val). For LogHAR this fit
    also fixes the Jensen-bias-correction variance. Paper footnote 11:
    "var(f̂(Z_t)) is the variance of the residuals in the training and validation
    set", i.e. computed ONCE and reused across all rolling steps.
  - Rolling OLS forecast through the test set:
      window size = 1776 days; refit at each step using the most recent observed
      1776 days, predict the next day.
  - LogHAR: predict in log space; bias-correct via exp(pred + 0.5 * var_fixed).
  - Negative-clipping for all models (per paper §1.6): if pred < 0, replace with
    min(in-sample RV in current rolling window).
  - HARQ additionally applies the BPQ-2016 upper-clip: if pred > max(in-sample
    RV), replace with min(in-sample RV).
  - Save predictions to results/predictions/<ticker>_<infoset>_<model>_h1.csv.

This script does not standardise X (HAR family uses raw features, per paper §1.3
which standardises only ML models). Coefficients are in raw-feature units.

HAR-X is omitted from M_HAR (it equals HAR there per paper Tables 2/6).
HAR is included in both M_HAR and M_ALL. It's the same model (the benchmark
whose MSE is 1.000 in the paper's tables for each info-set).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split import split_all_stocks, TICKERS, FINAL_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("har_family")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

INFOSETS = ["M_HAR", "M_ALL"]
HAR_LAGS = ["RVD", "RVW", "RVM"]
M_ALL_EXTRAS = ["M1W", "d_log_dvol", "EA", "VIX", "d_US3M", "HSI", "ADS", "EPU"]
WINDOW_SIZE = 1776  # train (1554) + val (222)


@dataclass
class ModelSpec:
    name: str
    info_set: str
    feature_cols: list[str]
    log_cols: list[str] = field(default_factory=list)
    log_target: bool = False
    has_harq_interaction: bool = False
    apply_harq_upper_clip: bool = False  # HARQ-only: also clip if pred > max(in-sample RV)


def get_specs(info_set: str) -> list[ModelSpec]:
    """Return the list of HAR-family ModelSpecs for the given info_set."""
    if info_set == "M_HAR":
        return [
            ModelSpec("HAR",    info_set, HAR_LAGS),
            ModelSpec("LogHAR", info_set, HAR_LAGS,
                      log_cols=HAR_LAGS, log_target=True),
            ModelSpec("HARQ",   info_set, HAR_LAGS,
                      has_harq_interaction=True, apply_harq_upper_clip=True),
        ]
    if info_set == "M_ALL":
        all_feats = HAR_LAGS + M_ALL_EXTRAS
        return [
            ModelSpec("HAR",    info_set, HAR_LAGS),
            ModelSpec("HAR-X",  info_set, all_feats),
            ModelSpec("LogHAR", info_set, all_feats,
                      log_cols=HAR_LAGS + ["VIX"], log_target=True),
            ModelSpec("HARQ",   info_set, all_feats,
                      has_harq_interaction=True, apply_harq_upper_clip=True),
        ]
    raise ValueError(f"unknown info_set {info_set}")


def build_features(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Construct feature DataFrame for a model: apply log-transforms, append HARQ
    interaction if needed. The HARQ interaction is RVD * sqrt(RQ_lag)."""
    out = pd.DataFrame(index=df.index)
    for col in spec.feature_cols:
        if col in spec.log_cols:
            out[f"log_{col}"] = np.log(df[col])
        else:
            out[col] = df[col]
    if spec.has_harq_interaction:
        out["RVD_x_sqrtRQ"] = df["RVD"] * np.sqrt(df["RQ_lag"])
    return out


def fit_ols(X: pd.DataFrame, y: pd.Series):
    """Fit OLS with explicit constant column. Returns statsmodels result."""
    X_c = X.copy()
    X_c.insert(0, "const", 1.0)
    return sm.OLS(y, X_c).fit()


def predict_ols(ols, X_row: pd.DataFrame) -> float:
    """Predict a single (or multi-row) X using a fitted OLS, with explicit constant."""
    X_c = X_row.copy()
    X_c.insert(0, "const", 1.0)
    return float(ols.predict(X_c).iloc[0])


def initial_fit(master: pd.DataFrame, spec: ModelSpec,
                test_start_idx: int, window_size: int):
    """Fit on the first rolling window (= train+val). Used for the coefficient
    diagnostic and, for LogHAR, to fix the Jensen-bias-correction variance."""
    fit_df = master.iloc[test_start_idx - window_size:test_start_idx]
    X = build_features(fit_df, spec)
    y = np.log(fit_df["RV"]) if spec.log_target else fit_df["RV"]
    return fit_ols(X, y)


def rolling_forecast(master: pd.DataFrame, spec: ModelSpec,
                     test_start_idx: int, window_size: int,
                     fixed_resid_var: float | None
                     ) -> tuple[pd.Series, int, int]:
    """One-day-ahead rolling-window forecast across the test segment.

    fixed_resid_var: if not None, used in the LogHAR Jensen bias correction
                     (computed once on the initial fit). Required when log_target.

    Returns (predictions, n_neg_clip, n_max_clip):
      - predictions: Series indexed by test dates
      - n_neg_clip:  count of times pred < 0 was replaced with min(in-sample RV)
                     (applies to all models per paper §1.6)
      - n_max_clip:  count of times pred > max(in-sample RV) was replaced
                     (HARQ only; the BPQ-2016 upper clip)
    """
    if spec.log_target and fixed_resid_var is None:
        raise ValueError("LogHAR requires a fixed_resid_var from initial_fit")

    n = len(master)
    out_dates = master.index[test_start_idx:]
    preds = pd.Series(index=out_dates, dtype=np.float64, name=spec.name)
    n_neg = 0
    n_max = 0

    for i, target_pos in enumerate(range(test_start_idx, n)):
        fit_start = target_pos - window_size
        fit_df = master.iloc[fit_start:target_pos]
        target_row = master.iloc[[target_pos]]

        X_fit = build_features(fit_df, spec)
        X_pred = build_features(target_row, spec)
        y_fit = np.log(fit_df["RV"]) if spec.log_target else fit_df["RV"]

        ols = fit_ols(X_fit, y_fit)
        pred = predict_ols(ols, X_pred)

        if spec.log_target:
            pred = float(np.exp(pred + 0.5 * fixed_resid_var))

        # Negative-clip for all models (paper §1.6)
        if pred < 0:
            pred = float(fit_df["RV"].min())
            n_neg += 1

        # HARQ-only upper clip (BPQ 2016)
        if spec.apply_harq_upper_clip and pred > float(fit_df["RV"].max()):
            pred = float(fit_df["RV"].min())
            n_max += 1

        preds.iloc[i] = pred

    return preds, n_neg, n_max


def run_one(ticker: str, info_set: str, spec: ModelSpec, master: pd.DataFrame,
            test_start_idx: int) -> dict:
    """Fit one HAR-family (ticker, info_set, spec) combination and return its result dict."""
    log.info("  fitting %s × %s × %s", ticker, info_set, spec.name)
    ols0 = initial_fit(master, spec, test_start_idx, WINDOW_SIZE)
    fixed_var = float(ols0.resid.var()) if spec.log_target else None
    preds, n_neg, n_max = rolling_forecast(master, spec,
                                            test_start_idx, WINDOW_SIZE,
                                            fixed_resid_var=fixed_var)
    actual = master["RV"].iloc[test_start_idx:]
    mse = float(((preds - actual) ** 2).mean())

    out_path = PREDICTIONS_DIR / f"{ticker}_{info_set}_{spec.name}_h1.csv"
    pd.DataFrame({"actual": actual, "predicted": preds}).to_csv(out_path)

    return {
        "ticker": ticker,
        "info_set": info_set,
        "model": spec.name,
        "test_mse": mse,
        "mean_predicted": float(preds.mean()),
        "mean_actual": float(actual.mean()),
        "n_test": len(actual),
        "n_neg_clip": n_neg,
        "n_max_clip": n_max if spec.apply_harq_upper_clip else None,
        "initial_params": ols0.params.to_dict(),
        "initial_pvalues": ols0.pvalues.to_dict(),
        "initial_rsquared": float(ols0.rsquared),
        "initial_resid_var": fixed_var,  # only set for LogHAR
    }


def main() -> None:
    """Run all HAR-family fits across 3 stocks and 2 info-sets, then print diagnostics."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    splits = split_all_stocks()

    all_results = []
    for ticker in TICKERS:
        sp = splits[ticker]
        master = pd.read_csv(FINAL_DIR / f"master_{ticker}.csv",
                             parse_dates=["date"], index_col="date").sort_index()
        test_start_idx = len(sp.X_train) + len(sp.X_val)
        assert test_start_idx == WINDOW_SIZE, \
            f"expected test_start_idx={WINDOW_SIZE}, got {test_start_idx}"
        log.info("== %s ==  master rows=%d  test_start_idx=%d  n_test=%d",
                 ticker, len(master), test_start_idx, len(master) - test_start_idx)
        for info_set in INFOSETS:
            for spec in get_specs(info_set):
                all_results.append(run_one(ticker, info_set, spec, master, test_start_idx))

    # Print diagnostic table
    for r in all_results:
        print()
        print("=" * 96)
        print(f"  {r['ticker']}  ×  {r['info_set']}  ×  {r['model']}")
        print("=" * 96)
        coef_df = pd.DataFrame({
            "coef":   pd.Series(r["initial_params"]),
            "pvalue": pd.Series(r["initial_pvalues"]),
        })
        print(f"  Initial fit (window = first 1776 days = train+val):")
        print(coef_df.to_string(float_format=lambda x: f"{x: .4e}"))
        print(f"  R² = {r['initial_rsquared']:.4f}")
        if r["initial_resid_var"] is not None:
            print(f"  LogHAR Jensen variance (fixed, on train+val residuals): "
                  f"{r['initial_resid_var']:.4e}")
        print(f"  Test MSE          : {r['test_mse']:.4e}")
        print(f"  Mean predicted RV : {r['mean_predicted']:.4e}   "
              f"Mean actual RV: {r['mean_actual']:.4e}")
        print(f"  Negative-clips    : {r['n_neg_clip']} of {r['n_test']} test days "
              f"({100*r['n_neg_clip']/r['n_test']:.1f}%)")
        if r["n_max_clip"] is not None:
            print(f"  HARQ upper-clips  : {r['n_max_clip']} of {r['n_test']} test days "
                  f"({100*r['n_max_clip']/r['n_test']:.1f}%)")
        else:
            print(f"  HARQ upper-clips  : N/A (not HARQ)")


if __name__ == "__main__":
    main()
