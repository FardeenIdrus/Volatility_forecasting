"""Accumulated Local Effects variable importance for the random forest and neural-net models."""
from __future__ import annotations

import logging
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))
from split import split_all_stocks, TICKERS  # noqa: E402
from ml_models import fit_one_nn, M_ALL_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ale")
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
# Top-10 NN seeds are parsed from the most recent h=1 NN run log written by
# ml_models.py (after the NN y-standardisation fix); see stage7_h1_nn.log.
NN_SEED_LOG = PROJECT_ROOT / "results" / "logs" / "stage7_h1_nn.log"
ALE_MODELS = ("RF", "NN_2_e10")

N_BINS = 100
NN_TOP = 10
RF_PARAMS = dict(
    n_estimators=500, max_features=1 / 3, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)


# ---------- ALE computation (paper Eq. 28-31) ----------

def compute_ale(
    predict_fn, X: np.ndarray, feature_idx: int, n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centred ALE curve and raw variable-importance score for one feature (Apley & Zhu 2020, Eq. 28-31)."""
    z = X[:, feature_idx]
    T = len(z)
    edges = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
    K = len(edges) - 1
    if K < 2:
        log.warning("    feature idx %d collapses to %d bins, skipping", feature_idx, K)
        return edges, np.zeros(max(K + 1, 1)), 0.0

    bin_idx = np.clip(np.searchsorted(edges, z, side="right"), 1, K)

    X_lower = X.copy()
    X_upper = X.copy()
    X_lower[:, feature_idx] = edges[bin_idx - 1]
    X_upper[:, feature_idx] = edges[bin_idx]

    # Batch all bins into a single predict call: lower- and upper-edge copies stacked.
    preds = predict_fn(np.vstack([X_lower, X_upper]))
    diff = preds[T:] - preds[:T]

    local_effects = np.zeros(K)
    for k in range(1, K + 1):
        mask = (bin_idx == k)
        if mask.any():
            local_effects[k - 1] = float(diff[mask].mean())

    ale_at_edges = np.concatenate(([0.0], np.cumsum(local_effects)))
    ale_at_obs = ale_at_edges[bin_idx]
    centring = float(ale_at_obs.mean())
    ale_centred = ale_at_edges - centring
    vi = float((ale_at_obs - centring).std())
    return edges, ale_centred, vi


def compute_all_features(
    predict_fn, X: np.ndarray, feature_names: list[str],
) -> tuple[dict, np.ndarray]:
    """Run compute_ale across all features. Returns (results dict, normalised VI vec)."""
    results: dict = {}
    vi_raw = np.zeros(len(feature_names))
    for j, name in enumerate(feature_names):
        log.info("    ALE feature %2d/%d  %s", j + 1, len(feature_names), name)
        edges, ale_centred, vi = compute_ale(predict_fn, X, j)
        results[name] = (edges, ale_centred, vi)
        vi_raw[j] = vi
    total = vi_raw.sum()
    vi_norm = vi_raw / total if total > 0 else vi_raw
    return results, vi_norm


# ---------- NN re-creation ----------

def parse_top_seeds_from_log(log_path: Path) -> dict[str, list[int]]:
    """Extract top-10 NN seeds per ticker from the NN ensemble diagnostic log."""
    text = log_path.read_text()
    pattern = re.compile(
        r"^\s*(\w+) × M_ALL:\s*$.*?^\s*top-10 seeds\s*:\s*\[([^\]]+)\]",
        re.MULTILINE | re.DOTALL,
    )
    out: dict[str, list[int]] = {}
    for m in pattern.finditer(text):
        ticker = m.group(1)
        seeds = [int(s.strip()) for s in m.group(2).split(",")]
        out[ticker] = seeds
        log.info("  parsed %d top-%d NN seeds for %s × M_ALL: %s",
                 len(seeds), NN_TOP, ticker, seeds)
    return out


def retrain_nn_ensemble(X_train, y_train, X_val, y_val, seeds: list[int]):
    """Retrain only the top-K NN seeds. Returns list of fitted Keras models."""
    nets = []
    for seed in seeds:
        log.info("    NN seed=%d retraining", seed)
        model, _val_mse, _n_epochs = fit_one_nn(X_train, y_train, X_val, y_val, seed)
        nets.append(model)
    return nets


# ---------- plotting ----------

def plot_ale_curves(ale_results, vi_norm, feature_names, ticker, model_name, save_path):
    n = len(feature_names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows), squeeze=False)
    for j, name in enumerate(feature_names):
        ax = axes[j // ncols, j % ncols]
        edges, ale_centred, _ = ale_results[name]
        ax.plot(edges, ale_centred, lw=1.2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{name}  (VI={vi_norm[j]:.3f})", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("z (standardised)", fontsize=8)
        ax.set_ylabel("centred ALE", fontsize=8)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(f"ALE — {ticker} × {model_name} (M_ALL, {N_BINS} quantile bins)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_vi_bar(vi_norm, feature_names, ticker, model_name, save_path):
    order = np.argsort(vi_norm)
    names = [feature_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(feature_names))))
    ax.barh(names, vi_norm[order])
    ax.set_xlabel("Variable importance (normalised, sums to 1)")
    ax.set_title(f"VI — {ticker} × {model_name} (M_ALL)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_tables(ale_results, vi_norm, feature_names, ticker, model_name):
    rows = []
    for name in feature_names:
        edges, ale_centred, _ = ale_results[name]
        for i, (e, a) in enumerate(zip(edges, ale_centred)):
            rows.append({"feature": name, "edge_idx": i,
                         "z_std": float(e), "ale_centred": float(a)})
    pd.DataFrame(rows).to_csv(
        TABLES_DIR / f"ale_{ticker}_{model_name}.csv", index=False,
    )
    pd.DataFrame([{"feature": n, "vi_norm": float(v)}
                  for n, v in zip(feature_names, vi_norm)]).to_csv(
        TABLES_DIR / f"vi_{ticker}_{model_name}.csv", index=False,
    )


# ---------- driver ----------

def run_for_stock(sp, top_seeds: list[int], ticker: str,
                  models: tuple[str, ...] = ALE_MODELS) -> None:
    log.info("== %s ==", ticker)
    cols = M_ALL_FEATURES
    X_train_std = sp.X_train_std[cols].to_numpy()
    y_train_arr = sp.y_train.to_numpy()
    X_val_std = sp.X_val_std[cols].to_numpy()
    y_val_arr = sp.y_val.to_numpy()

    if "RF" in models:
        log.info("  RF retraining (n_train=%d, J=%d)", *X_train_std.shape)
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train_std, y_train_arr)
        rf_results, rf_vi = compute_all_features(rf.predict, X_train_std, cols)
        plot_ale_curves(rf_results, rf_vi, cols, ticker, "RF",
                        FIGURES_DIR / f"ale_{ticker}_RF.png")
        plot_vi_bar(rf_vi, cols, ticker, "RF",
                    FIGURES_DIR / f"vi_{ticker}_RF.png")
        save_tables(rf_results, rf_vi, cols, ticker, "RF")
        log.info("  RF VI sorted: %s",
                 sorted(zip(cols, rf_vi.tolist()), key=lambda x: -x[1]))

    if "NN_2_e10" in models:
        # Standardise y exactly as ml_models.py does, so the retrained seeds reproduce
        # the original ensemble (raw-y training would give different, ill-conditioned
        # nets). nn_predict inverts back to raw RV units. ALE is differences of
        # predictions, so the +mean cancels; VI normalisation also cancels the
        # *std factor, but we invert anyway to keep ALE curves in raw RV units.
        y_mean = float(y_train_arr.mean())
        y_std = float(y_train_arr.std())  # ddof=0
        y_train_t = (y_train_arr - y_mean) / y_std
        y_val_t = (y_val_arr - y_mean) / y_std
        log.info("  NN retraining top-%d seeds %s (y-standardised)",
                 len(top_seeds), top_seeds)
        nets = retrain_nn_ensemble(X_train_std, y_train_t, X_val_std, y_val_t,
                                   top_seeds)

        def nn_predict(X, _nets=nets, _m=y_mean, _s=y_std):
            ps = np.column_stack([m.predict(X, verbose=0).flatten() for m in _nets])
            return ps.mean(axis=1) * _s + _m

        nn_results, nn_vi = compute_all_features(nn_predict, X_train_std, cols)
        plot_ale_curves(nn_results, nn_vi, cols, ticker, "NN_2_e10",
                        FIGURES_DIR / f"ale_{ticker}_NN_2_e10.png")
        plot_vi_bar(nn_vi, cols, ticker, "NN_2_e10",
                    FIGURES_DIR / f"vi_{ticker}_NN_2_e10.png")
        save_tables(nn_results, nn_vi, cols, ticker, "NN_2_e10")
        log.info("  NN VI sorted: %s",
                 sorted(zip(cols, nn_vi.tolist()), key=lambda x: -x[1]))


def main(argv: list[str] | None = None) -> None:
    """Compute ALE variable importance for the requested tickers and models."""
    # CLI: python src/inference/ale.py [TICKER ...] [MODEL ...]  (args mix freely;
    #   defaults to all 3 stocks and both models).
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42); random.seed(42); tf.random.set_seed(42)

    args = argv if argv is not None else sys.argv[1:]
    bad = [a for a in args if a not in TICKERS and a not in ALE_MODELS]
    if bad:
        raise ValueError(f"unknown args: {bad}; valid tickers={TICKERS}, "
                         f"models={list(ALE_MODELS)}")
    requested_tickers = [a for a in args if a in TICKERS] or list(TICKERS)
    requested_models = tuple(a for a in args if a in ALE_MODELS) or ALE_MODELS
    log.info("tickers=%s  models=%s", requested_tickers, list(requested_models))

    top_seeds = parse_top_seeds_from_log(NN_SEED_LOG)
    if "NN_2_e10" in requested_models:
        missing = [t for t in requested_tickers
                   if len(top_seeds.get(t, [])) != NN_TOP]
        if missing:
            raise RuntimeError(f"could not parse {NN_TOP} top NN seeds for: {missing}")

    splits = split_all_stocks()
    for ticker in requested_tickers:
        run_for_stock(splits[ticker], top_seeds.get(ticker, []), ticker,
                      requested_models)


if __name__ == "__main__":
    main()
