"""src/models/ml_models.py: Stage 7. Five ML models with fixed-window training.

Per Stage 7 design:
  - All models fit ONCE on training data (no rolling). Validation set is used for
    early stopping (NN) or CV tuning (Lasso/EN; internal TimeSeriesSplit on train).
  - Use STANDARDISED features (sp.X_*_std from split.py).
  - Lasso: LassoCV(cv=TimeSeriesSplit(5), random_state=42).
  - ElasticNet: ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99], cv=TimeSeriesSplit(5)).
  - RF: RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42).
  - GB: GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=2,
        random_state=42). Depth 2 matches paper Table A.6 tuning grid {1, 2}.
  - NN_2_e10: 100 nets with seeds 0..99; architecture:
        Dense(4)->LeakyReLU(0.01)->Dropout(0.2)
       ->Dense(2)->LeakyReLU(0.01)->Dropout(0.2)
       ->Dense(1, linear)
      Adam(lr=0.001), batch=32, max 500 epochs, EarlyStopping(patience=20,
      restore_best_weights=True). Final prediction = mean over the top 10 nets
      (by validation MSE on the 222-row validation set).

Apply negative-clip (replace pred < 0 with min(in-sample RV) where in-sample = y_train)
for all ML models for consistency with Stage 6.

Saves predictions to results/predictions/<ticker>_<infoset>_<model>_h1.csv.
"""

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
from split import split_all_stocks, TICKERS  # noqa: E402

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

NN_TOTAL = 100
NN_TOP = 10
NN_MAX_EPOCHS = 500
NN_PATIENCE = 20
NN_BATCH_SIZE = 32
NN_LR = 0.001
NN_DROPOUT = 0.2
NN_LRELU_ALPHA = 0.01


def feature_cols(info_set: str) -> list[str]:
    """Return the feature columns to use for the given info_set."""
    return HAR_FEATURES if info_set == "M_HAR" else M_ALL_FEATURES


# ---------- NN ensemble ----------

def make_nn(n_features: int) -> tf.keras.Model:
    """Construct a fresh NN_2 model with the project's standard architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(4),
        tf.keras.layers.LeakyReLU(negative_slope=NN_LRELU_ALPHA),
        tf.keras.layers.Dropout(NN_DROPOUT),
        tf.keras.layers.Dense(2),
        tf.keras.layers.LeakyReLU(negative_slope=NN_LRELU_ALPHA),
        tf.keras.layers.Dropout(NN_DROPOUT),
        tf.keras.layers.Dense(1, activation="linear"),
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

def run_one(ticker: str, info_set: str, model_name: str, sp) -> dict:
    """Fit one (ticker, info_set, model) combination, save predictions, return result dict."""
    log.info("  %s × %s × %s", ticker, info_set, model_name)
    cols = feature_cols(info_set)
    X_train = sp.X_train_std[cols].to_numpy()
    y_train = sp.y_train.to_numpy()
    X_val = sp.X_val_std[cols].to_numpy()
    y_val = sp.y_val.to_numpy()
    X_test = sp.X_test_std[cols].to_numpy()
    y_test = sp.y_test.to_numpy()

    nn_diag = None
    t_start = time.time()

    if model_name == "Lasso":
        m = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42, max_iter=20000)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        chosen = {"alpha": float(m.alpha_)}
    elif model_name == "ElasticNet":
        m = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            cv=TimeSeriesSplit(n_splits=5),
            random_state=42, max_iter=20000,
        )
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        chosen = {"alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_)}
    elif model_name == "RF":
        m = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        chosen = {"n_estimators": 500, "max_features": "sqrt"}
    elif model_name == "GB":
        m = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42
        )
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        chosen = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 2}
    elif model_name == "NN_2_e10":
        nets, top_idx, val_mses, epoch_counts = fit_nn_ensemble(
            X_train, y_train, X_val, y_val
        )
        preds = predict_nn_ensemble(nets, top_idx, X_test)
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

    # Negative clip (per Stage 7 instruction)
    n_neg = int((preds < 0).sum())
    if n_neg > 0:
        preds = np.where(preds < 0, float(sp.y_train.min()), preds)

    mse = float(((preds - y_test) ** 2).mean())
    out_path = PREDICTIONS_DIR / f"{ticker}_{info_set}_{model_name}_h1.csv"
    pd.DataFrame({
        "actual": sp.y_test,
        "predicted": pd.Series(preds, index=sp.y_test.index),
    }).to_csv(out_path)

    return {
        "ticker": ticker, "info_set": info_set, "model": model_name,
        "test_mse": mse, "mean_predicted": float(preds.mean()),
        "mean_actual": float(y_test.mean()),
        "n_neg_clip": n_neg, "elapsed_seconds": elapsed,
        "hyperparams": chosen, "nn_diag": nn_diag,
    }


def print_diagnostic(results: list[dict]) -> None:
    """Print the Stage 7 diagnostic summary: MSE table, hyperparams, clips, NN stats."""
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


def main():
    """Run all 5 ML models across 3 stocks and 2 info-sets, then print diagnostics."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    splits = split_all_stocks()
    results = []
    for ticker in TICKERS:
        sp = splits[ticker]
        log.info("== %s ==", ticker)
        for info_set in INFOSETS:
            for model_name in ML_MODELS:
                results.append(run_one(ticker, info_set, model_name, sp))
    print_diagnostic(results)


if __name__ == "__main__":
    main()
