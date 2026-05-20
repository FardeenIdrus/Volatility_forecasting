"""Exhibit 1: ACF of in-sample fitted RV series for JPM (mirrors paper Figure 8).

One-shot train+val fit (not the rolling-window scheme used in the main results):
HAR and RF are fit once on train+val and predict in-sample; NN_2^10 retrains its
logged top-10 seeds and predicts in-sample. ACF is computed to lag 50. This script
computes the fitted series, the ACFs, and a sanity report, and persists the ACF
data; it does NOT plot (plotting is a separate, gated step).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import acf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "models"))
from split import split_stock  # noqa: E402
from horizons import build_h_step_target  # noqa: E402
from har_family import build_features, fit_ols, ModelSpec, HAR_LAGS  # noqa: E402
from ml_models import fit_one_nn, M_ALL_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("acf_persistence")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data" / "final"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
TABLES_APPENDIX = PROJECT_ROOT / "results" / "tables" / "appendix"
FIGURES_APPENDIX = PROJECT_ROOT / "results" / "figures" / "appendix"

TICKER = "JPM"
HORIZONS = [1, 22]
MODELS = ["HAR", "RF", "NN_2_e10"]
N_TRAIN = 1554
N_TRAINVAL = 1776          # train (1554) + validation (222)
MAX_LAG = 50
NN_SEED_LOG = {1: LOGS_DIR / "stage7_h1_nn.log", 22: LOGS_DIR / "stage7_h22.log"}
RF_PARAMS = dict(n_estimators=500, max_features=1 / 3, min_samples_leaf=5,
                 random_state=42, n_jobs=-1)


def top10_seeds(log_path: Path, ticker: str) -> list[int]:
    """Parse the top-10 NN seeds for one ticker's M_ALL block from a Stage 7 log."""
    m = re.search(rf"^\s*{ticker} . M_ALL:\s*$.*?^\s*top-10 seeds\s*:\s*\[([^\]]+)\]",
                  log_path.read_text(), re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError(f"top-10 seeds not found for {ticker} in {log_path}")
    return [int(s) for s in m.group(1).split(",")]


def har_fitted(master: pd.DataFrame, y_target: pd.Series) -> tuple[np.ndarray, int]:
    """In-sample HAR fitted RV from a one-shot OLS fit on train+val."""
    fit_df = master.iloc[:N_TRAINVAL]
    y = y_target.iloc[:N_TRAINVAL]
    X = build_features(fit_df, ModelSpec("HAR", "M_ALL", HAR_LAGS))
    ols = fit_ols(X, y)
    X_c = X.copy()
    X_c.insert(0, "const", 1.0)
    fitted = ols.predict(X_c).to_numpy()
    n_neg = int((fitted < 0).sum())
    return np.where(fitted < 0, float(y.min()), fitted), n_neg


def rf_fitted(master: pd.DataFrame, y_target: pd.Series) -> tuple[np.ndarray, int]:
    """In-sample RF fitted RV from a one-shot fit on train+val (M_ALL features)."""
    X = master.iloc[:N_TRAINVAL][M_ALL_FEATURES].to_numpy(dtype=np.float64)
    y = y_target.iloc[:N_TRAINVAL].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = np.where(X.std(axis=0, ddof=0) == 0, 1.0, X.std(axis=0, ddof=0))
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit((X - mean) / std, y)
    fitted = rf.predict((X - mean) / std)
    n_neg = int((fitted < 0).sum())
    return np.where(fitted < 0, float(y.min()), fitted), n_neg


def nn_fitted(sp, y_target: pd.Series, seeds: list[int]) -> tuple[np.ndarray, int]:
    """In-sample NN_2^10 fitted RV: retrain the logged top-10 seeds, predict on train+val."""
    X_tr = sp.X_train_std[M_ALL_FEATURES].to_numpy(dtype=np.float64)
    X_va = sp.X_val_std[M_ALL_FEATURES].to_numpy(dtype=np.float64)
    y_tr_raw = y_target.iloc[:N_TRAIN].to_numpy(dtype=np.float64)
    y_va_raw = y_target.iloc[N_TRAIN:N_TRAINVAL].to_numpy(dtype=np.float64)
    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_tr = (y_tr_raw - y_mean) / y_std
    y_va = (y_va_raw - y_mean) / y_std
    X_in = np.vstack([X_tr, X_va])
    preds = []
    for i, seed in enumerate(seeds):
        model, _, _ = fit_one_nn(X_tr, y_tr, X_va, y_va, seed)
        preds.append(model.predict(X_in, verbose=0).flatten())
        log.info("    NN seed %d (%d/%d) trained", seed, i + 1, len(seeds))
    fitted = np.column_stack(preds).mean(axis=1) * y_std + y_mean
    n_neg = int((fitted < 0).sum())
    return np.where(fitted < 0, float(y_tr_raw.min()), fitted), n_neg


def sanity_report(results: dict) -> None:
    """Print the lag-0 / NaN / negative-clip sanity check for all six fitted series."""
    print("\n" + "=" * 96)
    print("  EXHIBIT 1 SANITY CHECK  —  in-sample fitted RV, JPM M_ALL  (ACF to lag 50)")
    print("=" * 96)
    print(f"  {'cell':14s} {'n_obs':>6} {'nan':>4} {'clip':>5} "
          f"{'fit_min':>11} {'fit_mean':>11} {'fit_max':>11} "
          f"{'acf0':>6} {'acf1':>6} {'acf5':>6} {'acf10':>6} {'acf25':>6} {'acf50':>6}")
    print("  " + "-" * 94)
    issues = []
    for h in HORIZONS:
        for model in MODELS:
            f, n_neg, a = results[(h, model)]
            n_nan = int(np.isnan(f).sum()) + int(np.isnan(a).sum())
            print(f"  h{h:<2d}/{model:9s} {len(f):6d} {n_nan:4d} {n_neg:5d} "
                  f"{f.min():11.3e} {f.mean():11.3e} {f.max():11.3e} "
                  f"{a[0]:6.3f} {a[1]:6.3f} {a[5]:6.3f} {a[10]:6.3f} "
                  f"{a[25]:6.3f} {a[50]:6.3f}")
            if n_nan:
                issues.append(f"h{h}/{model}: {n_nan} NaN value(s)")
            if abs(a[0] - 1.0) > 1e-9:
                issues.append(f"h{h}/{model}: acf[0]={a[0]:.6f}, expected 1.0")
            if n_neg > 18:
                issues.append(f"h{h}/{model}: {n_neg} negative-clips (>1% of {N_TRAINVAL})")
    print("  " + "-" * 94)
    if issues:
        print("  VERDICT: ISSUES FOUND")
        for it in issues:
            print(f"    - {it}")
    else:
        print("  VERDICT: PASS  —  no NaN; acf[0]=1.000 for all 6 series; "
              "negative-clipping negligible.")


def make_figure() -> None:
    """Render the 2-panel figA3 ACF figure from the persisted ACF data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
    })
    colour = {"HAR": "#0072B2", "RF": "#E69F00", "NN_2_e10": "#009E73"}
    label = {"HAR": "HAR", "RF": "RF", "NN_2_e10": "NN$_2^{10}$"}

    df = pd.read_csv(TABLES_APPENDIX / "figA3_acf_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (tag, h, htext) in zip(
            axes, [("A", 1, "$h$ = 1  (1-day)"), ("B", 22, "$h$ = 22  (1-month)")]):
        for model in MODELS:
            ax.plot(df["lag"], df[f"h{h}_{model}"], color=colour[model],
                    lw=1.6, label=label[model], zorder=3)
        ax.set_title(f"Panel {tag}:  {htext}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag (trading days)")
        ax.set_xlim(0, MAX_LAG)
        ax.set_ylim(0, 1.02)
        ax.tick_params(length=0)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Autocorrelation of fitted RV")
    fig.tight_layout()
    FIGURES_APPENDIX.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_APPENDIX / f"figA3_acf_persistence.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figA3_acf_persistence.{png,pdf}")


def main(argv: list[str] | None = None) -> None:
    """Compute the fitted series, ACFs and sanity report; pass 'plot' to render the figure."""
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "plot":
        make_figure()
        return
    TABLES_APPENDIX.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(FINAL_DIR / f"master_{TICKER}.csv",
                         parse_dates=["date"], index_col="date").sort_index()
    sp = split_stock(TICKER)
    results = {}
    for h in HORIZONS:
        y_target = build_h_step_target(master["RV"], h)
        seeds = top10_seeds(NN_SEED_LOG[h], TICKER)
        log.info("h=%d  NN top-10 seeds: %s", h, seeds)
        fitted = {
            "HAR": har_fitted(master, y_target),
            "RF": rf_fitted(master, y_target),
            "NN_2_e10": nn_fitted(sp, y_target, seeds),
        }
        for model, (f, n_neg) in fitted.items():
            results[(h, model)] = (f, n_neg, acf(f, nlags=MAX_LAG, fft=False))
            log.info("h=%d  %s fitted (n_neg_clip=%d)", h, model, n_neg)

    sanity_report(results)

    out = pd.DataFrame({"lag": range(MAX_LAG + 1)})
    for h in HORIZONS:
        for model in MODELS:
            out[f"h{h}_{model}"] = results[(h, model)][2]
    path = TABLES_APPENDIX / "figA3_acf_data.csv"
    out.to_csv(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
