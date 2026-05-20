"""Machine-learning realized-variance forecasters: rolling-window Lasso, ElasticNet,
random forest and gradient boosting, plus a fixed-window neural-net ensemble."""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split import split_all_stocks, TICKERS, FINAL_DIR  # noqa: E402
from horizons import build_h_step_target  # noqa: E402

# Global seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ml_models")
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

HAR_FEATURES = ["RVD", "RVW", "RVM"]
M_ALL_EXTRAS = ["M1W", "d_log_dvol", "EA", "VIX", "d_US3M", "HSI", "ADS", "EPU"]
M_ALL_FEATURES = HAR_FEATURES + M_ALL_EXTRAS

INFOSETS = ["M_HAR", "M_ALL"]
ML_MODELS = ["Lasso", "ElasticNet", "RF", "GB", "NN_2_e10"]

# Rolling-window sizes per paper §2:
#   HAR family + BG + RF:       "we merge the training and validation set"     -> 1776
#   Lasso, ElasticNet, GB:      "rolling scheme without concatenation"          -> 1554
ROLLING_WINDOW_LARGE = 1554 + 222  # 1776: train + val
ROLLING_WINDOW_SMALL = 1554        # train only

NN_TOTAL = 100
NN_TOP = 10
NN_MAX_EPOCHS = 500
NN_PATIENCE = 100
NN_BATCH_SIZE = 32
NN_LR = 0.001
NN_DROPOUT = 0.2
NN_LRELU_ALPHA = 0.01


def feature_cols(info_set: str) -> list[str]:
    """Return the feature columns to use for the given info_set."""
    return HAR_FEATURES if info_set == "M_HAR" else M_ALL_FEATURES


# ---------- rolling ML refits ----------

def rolling_ml_forecast(
    master: pd.DataFrame,
    feature_cols_: list[str],
    test_start_idx: int,
    window_size: int,
    model_factory,
    h: int,
) -> tuple[pd.Series, list[dict], int]:
    """One-step-ahead rolling-window forecast for a single ML model."""
    n = len(master)
    out_dates = master.index[test_start_idx:]
    preds = pd.Series(index=out_dates, dtype=np.float64)
    diag: list[dict] = []
    n_neg = 0
    t0 = time.time()

    for i, target_pos in enumerate(range(test_start_idx, n)):
        # Lag the window by h-1 rows so every training label is realised (no look-ahead).
        fit_end = target_pos - (h - 1)
        fit_start = max(0, fit_end - window_size)
        fit_slice = master.iloc[fit_start:fit_end]
        target_row = master.iloc[[target_pos]]

        X_fit = fit_slice[feature_cols_].to_numpy()
        y_fit = fit_slice["y_target"].to_numpy()
        X_pred = target_row[feature_cols_].to_numpy()

        mean = X_fit.mean(axis=0)
        std = X_fit.std(axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        X_fit_std = (X_fit - mean) / std
        X_pred_std = (X_pred - mean) / std

        model = model_factory(X_fit_std, y_fit)
        pred = float(model.predict(X_pred_std)[0])

        step_info: dict = {}
        if hasattr(model, "alpha_"):
            step_info["alpha"] = float(model.alpha_)
        if hasattr(model, "l1_ratio_"):
            step_info["l1_ratio"] = float(model.l1_ratio_)
        if step_info:
            diag.append(step_info)

        if pred < 0:
            pred = float(fit_slice["y_target"].min())
            n_neg += 1

        preds.iloc[i] = pred

        if (i + 1) % 50 == 0:
            log.info("    rolling step %d/%d  elapsed=%.1fs",
                     i + 1, len(out_dates), time.time() - t0)

    return preds, diag, n_neg


# ---------- NN ensemble ----------

def make_nn(n_features: int) -> tf.keras.Model:
    """Construct a fresh NN_2 model with the project's standard architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(4, kernel_initializer="glorot_normal"),
        tf.keras.layers.LeakyReLU(negative_slope=NN_LRELU_ALPHA),
        tf.keras.layers.Dropout(NN_DROPOUT),
        tf.keras.layers.Dense(2, kernel_initializer="glorot_normal"),
        tf.keras.layers.LeakyReLU(negative_slope=NN_LRELU_ALPHA),
        tf.keras.layers.Dropout(NN_DROPOUT),
        tf.keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_normal"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=NN_LR), loss="mse")
    return model


def fit_one_nn(X_train, y_train, X_val, y_val, seed: int):
    """Train a single NN with a given seed. Returns (model, val_mse, n_epochs)."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_nn(X_train.shape[1])
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True
    )
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NN_MAX_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )
    val_pred = model.predict(X_val, verbose=0).flatten()
    val_mse = float(((val_pred - y_val) ** 2).mean())
    n_epochs = len(hist.history["loss"])
    return model, val_mse, n_epochs


def fit_nn_ensemble(X_train, y_train, X_val, y_val):
    """Train NN_TOTAL nets with seeds 0..NN_TOTAL-1, return all of them + diagnostics."""
    val_mses, epoch_counts, nets = [], [], []
    t0 = time.time()
    for seed in range(NN_TOTAL):
        m, vmse, neps = fit_one_nn(X_train, y_train, X_val, y_val, seed)
        val_mses.append(vmse)
        epoch_counts.append(neps)
        nets.append(m)
        if (seed + 1) % 25 == 0:
            log.info("    NN progress %d/%d  elapsed=%.1fs",
                     seed + 1, NN_TOTAL, time.time() - t0)
    val_mses = np.array(val_mses)
    epoch_counts = np.array(epoch_counts)
    top_idx = np.argsort(val_mses)[:NN_TOP]
    return nets, top_idx, val_mses, epoch_counts


def predict_nn_ensemble(nets, top_idx, X) -> np.ndarray:
    """Mean prediction over the selected top-K nets in the ensemble."""
    preds = np.column_stack([nets[i].predict(X, verbose=0).flatten() for i in top_idx])
    return preds.mean(axis=1)


# ---------- driver ----------

def run_one(ticker: str, info_set: str, model_name: str, sp, master: pd.DataFrame,
            h: int) -> dict:
    """Fit one (ticker, info_set, model) cell, save test predictions, return diagnostics."""
    log.info("  %s × %s × %s (h=%d)", ticker, info_set, model_name, h)
    cols = feature_cols(info_set)
    test_start_idx = len(sp.X_train) + len(sp.X_val)
    assert test_start_idx == ROLLING_WINDOW_LARGE, \
        f"expected test_start_idx={ROLLING_WINDOW_LARGE}, got {test_start_idx}"

    t_start = time.time()
    nn_diag = None
    n_neg = 0
    preds: pd.Series

    if model_name == "Lasso":
        def factory(X, y):
            m = LassoCV(
                alphas=np.logspace(-5, 2, 1000),
                cv=TimeSeriesSplit(n_splits=5),
                random_state=42, max_iter=20000,
            )
            m.fit(X, y)
            return m
        preds, rolling_diag, n_neg = rolling_ml_forecast(
            master, cols, test_start_idx, ROLLING_WINDOW_SMALL, factory, h
        )
        alphas = [d["alpha"] for d in rolling_diag]
        chosen = {
            "alpha_median": float(np.median(alphas)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "window": ROLLING_WINDOW_SMALL,
        }

    elif model_name == "ElasticNet":
        def factory(X, y):
            m = ElasticNetCV(
                l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                cv=TimeSeriesSplit(n_splits=5),
                random_state=42, max_iter=20000,
            )
            m.fit(X, y)
            return m
        preds, rolling_diag, n_neg = rolling_ml_forecast(
            master, cols, test_start_idx, ROLLING_WINDOW_SMALL, factory, h
        )
        alphas = [d["alpha"] for d in rolling_diag]
        l1s = [d["l1_ratio"] for d in rolling_diag]
        chosen = {
            "alpha_median": float(np.median(alphas)),
            "l1_ratio_median": float(np.median(l1s)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "window": ROLLING_WINDOW_SMALL,
        }

    elif model_name == "RF":
        n_features = len(cols)
        rf_max_features = None if n_features <= 3 else 1 / 3
        def factory(X, y):
            m = RandomForestRegressor(
                n_estimators=500, max_features=rf_max_features, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            m.fit(X, y)
            return m
        preds, _, n_neg = rolling_ml_forecast(
            master, cols, test_start_idx, ROLLING_WINDOW_LARGE, factory, h
        )
        chosen = {
            "n_estimators": 500, "max_features": rf_max_features,
            "min_samples_leaf": 5, "window": ROLLING_WINDOW_LARGE,
        }

    elif model_name == "GB":
        # Tune learning_rate ONCE on initial train/val split, freeze for all rolling
        # refits. Matches paper Table A.6 grid {0.01, 0.1}.
        gb_val_mse_by_lr: dict[float, float] = {}
        X_train_arr = sp.X_train_std[cols].to_numpy()
        y_train_arr = sp.y_train.to_numpy()
        X_val_arr = sp.X_val_std[cols].to_numpy()
        y_val_arr = sp.y_val.to_numpy()
        for lr in (0.01, 0.1):
            cand = GradientBoostingRegressor(
                n_estimators=500, learning_rate=lr, max_depth=2, random_state=42
            )
            cand.fit(X_train_arr, y_train_arr)
            val_pred = cand.predict(X_val_arr)
            gb_val_mse_by_lr[lr] = float(((val_pred - y_val_arr) ** 2).mean())
        best_lr = min(gb_val_mse_by_lr, key=gb_val_mse_by_lr.get)
        log.info("    GB lr=%s selected (val_mse_by_lr=%s)", best_lr, gb_val_mse_by_lr)

        def factory(X, y, _lr=best_lr):
            m = GradientBoostingRegressor(
                n_estimators=500, learning_rate=_lr, max_depth=2, random_state=42
            )
            m.fit(X, y)
            return m
        preds, _, n_neg = rolling_ml_forecast(
            master, cols, test_start_idx, ROLLING_WINDOW_SMALL, factory, h
        )
        chosen = {
            "n_estimators": 500, "learning_rate": best_lr, "max_depth": 2,
            "val_mse_by_lr": {str(k): v for k, v in gb_val_mse_by_lr.items()},
            "window": ROLLING_WINDOW_SMALL,
        }

    elif model_name == "NN_2_e10":
        # Fixed-window per paper §2: "weights are only found once in the initial
        # validation sample and not rolled forward... outside our budget" to roll.
        X_train = sp.X_train_std[cols].to_numpy()
        y_train_raw = sp.y_train.to_numpy()
        X_val = sp.X_val_std[cols].to_numpy()
        y_val_raw = sp.y_val.to_numpy()
        X_test = sp.X_test_std[cols].to_numpy()

        # Standardise the target for NN training only. RV-scale targets (~1e-4)
        # are badly conditioned for a linear-head NN under Adam(lr=1e-3);
        # standardising y puts the optimisation in its natural regime. Applied
        # at both horizons for consistency. Predictions invert to raw RV units
        # before negative-clipping/output. Linear/tree models do not need this.
        y_mean = float(y_train_raw.mean())
        y_std = float(y_train_raw.std())  # ddof=0
        y_train = (y_train_raw - y_mean) / y_std
        y_val = (y_val_raw - y_mean) / y_std

        nets, top_idx, val_mses, epoch_counts = fit_nn_ensemble(
            X_train, y_train, X_val, y_val
        )
        preds_arr = predict_nn_ensemble(nets, top_idx, X_test) * y_std + y_mean
        n_neg = int((preds_arr < 0).sum())
        if n_neg > 0:
            preds_arr = np.where(preds_arr < 0, float(sp.y_train.min()), preds_arr)
        preds = pd.Series(preds_arr, index=sp.y_test.index)

        # val_mses are in standardised units; rescale to raw RV units (MSE ~ std^2)
        # so the NN diagnostic stays comparable. Top-10 ranking is unaffected.
        val_mses = val_mses * (y_std ** 2)

        nn_diag = {
            "top10_val_mse_mean": float(val_mses[top_idx].mean()),
            "all100_val_mse_mean": float(val_mses.mean()),
            "all100_val_mse_std": float(val_mses.std()),
            "best_val_mse": float(val_mses.min()),
            "worst_val_mse": float(val_mses.max()),
            "avg_epochs": float(epoch_counts.mean()),
            "top_seeds": top_idx.tolist(),
        }
        chosen = {"n_total": NN_TOTAL, "n_top": NN_TOP}

    else:
        raise ValueError(f"unknown model_name {model_name}")

    elapsed = time.time() - t_start

    y_test = sp.y_test
    preds_aligned = preds.reindex(y_test.index)
    mse = float(((preds_aligned - y_test) ** 2).mean())
    out_path = PREDICTIONS_DIR / f"{ticker}_{info_set}_{model_name}_h{h}.csv"
    pd.DataFrame({
        "actual": y_test,
        "predicted": preds_aligned,
    }).to_csv(out_path)

    return {
        "ticker": ticker, "info_set": info_set, "model": model_name, "horizon": h,
        "test_mse": mse, "mean_predicted": float(preds_aligned.mean()),
        "mean_actual": float(y_test.mean()),
        "n_neg_clip": n_neg, "elapsed_seconds": elapsed,
        "hyperparams": chosen, "nn_diag": nn_diag,
    }


def print_diagnostic(results: list[dict]) -> None:
    """Print the diagnostic summary: MSE table, hyperparams, clips, NN stats."""
    df = pd.DataFrame([{
        "ticker": r["ticker"], "info_set": r["info_set"], "model": r["model"],
        "test_mse": r["test_mse"], "n_neg_clip": r["n_neg_clip"],
        "secs": r["elapsed_seconds"],
    } for r in results])
    pivot = df.pivot_table(index=["ticker", "model"], columns="info_set",
                          values="test_mse")
    print("\n" + "=" * 80)
    print("  Test MSE (per stock × info-set × model)")
    print("=" * 80)
    print(pivot.to_string(float_format=lambda x: f"{x:.4e}"))
    print()

    print("=" * 80)
    print("  Hyperparameter selections")
    print("=" * 80)
    for r in results:
        if r["model"] in ("Lasso", "ElasticNet"):
            print(f"  {r['ticker']:>5} × {r['info_set']:<6} × {r['model']:<10}: "
                  f"{r['hyperparams']}")
    print()

    print("=" * 80)
    print("  Negative-clip counts (test set)")
    print("=" * 80)
    clip_df = df[df["n_neg_clip"] > 0]
    if clip_df.empty:
        print("  No negative-clips across any ML model.")
    else:
        print(clip_df[["ticker", "info_set", "model", "n_neg_clip"]].to_string(index=False))
    print()

    print("=" * 80)
    print("  NN ensemble diagnostics")
    print("=" * 80)
    for r in results:
        if r["nn_diag"] is None:
            continue
        d = r["nn_diag"]
        print(f"  {r['ticker']} × {r['info_set']}:")
        print(f"    top-10 val MSE mean : {d['top10_val_mse_mean']:.4e}")
        print(f"    all-100 val MSE mean: {d['all100_val_mse_mean']:.4e}    "
              f"std: {d['all100_val_mse_std']:.4e}")
        print(f"    best  val MSE       : {d['best_val_mse']:.4e}    "
              f"worst: {d['worst_val_mse']:.4e}")
        print(f"    avg epochs (early-stopped): {d['avg_epochs']:.1f}")
        print(f"    training time       : {r['elapsed_seconds']:.1f}s")
        print(f"    top-10 seeds        : {d['top_seeds']}")


def main(argv: list[str] | None = None):
    """Run the five ML models across three stocks and two info-sets, then print diagnostics."""
    # CLI: python src/models/ml_models.py [h] [model1,model2,...]
    #   `... 22` runs h=22 for all models; `... 1 NN_2_e10` runs h=1 for the NN only.
    args = argv if argv is not None else sys.argv[1:]
    h = int(args[0]) if args else 1
    model_filter = args[1].split(",") if len(args) > 1 else ML_MODELS
    log.info("HORIZON h=%d  models=%s", h, model_filter)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    splits = split_all_stocks()
    results = []
    for ticker in TICKERS:
        sp = splits[ticker]
        master = pd.read_csv(
            FINAL_DIR / f"master_{ticker}.csv",
            parse_dates=["date"], index_col="date",
        ).sort_index()
        master["y_target"] = build_h_step_target(master["RV"], h)
        n_before = len(master)
        master = master.dropna(subset=["y_target"])
        # Override sp.y_* with h-step targets aligned to existing split indices.
        # For h>1 the test set is truncated by h-1 rows (NaN target tail dropped).
        sp.y_train = master["y_target"].iloc[:len(sp.X_train)]
        sp.y_val = master["y_target"].iloc[len(sp.X_train):len(sp.X_train) + len(sp.X_val)]
        test_dates_kept = master.index[len(sp.X_train) + len(sp.X_val):]
        sp.y_test = master["y_target"].loc[test_dates_kept]
        sp.X_test = sp.X_test.loc[test_dates_kept]
        sp.X_test_std = sp.X_test_std.loc[test_dates_kept]
        log.info("== %s ==  master rows=%d (dropped %d for h=%d tail)  "
                 "test_start_idx=%d  n_test=%d",
                 ticker, len(master), n_before - len(master), h,
                 len(sp.X_train) + len(sp.X_val), len(test_dates_kept))
        for info_set in INFOSETS:
            for model_name in ML_MODELS:
                if model_name not in model_filter:
                    continue
                results.append(run_one(ticker, info_set, model_name, sp, master, h))
    print_diagnostic(results)


if __name__ == "__main__":
    main()
