# Realized Volatility Forecasting: HAR Models vs Machine Learning

## Overview

This project replicates the methodology of Christensen, Siggaard & Veliyev
(2023) to test whether machine learning models outperform the HAR family at
forecasting the realized variance of large-cap US equities. It covers three
stocks (Apple, JPMorgan, Amazon) over 2016-2024 using 1-minute price data, and
evaluates nine forecasting models across two predictor sets and three horizons
(1 day, 1 week, 1 month). Forecasts are compared with the Diebold-Mariano test
and the Model Confidence Set, and predictor influence is quantified with
Accumulated Local Effects.

## Key Findings

- **The log-HAR model is the most reliable forecaster.** LogHAR beats the HAR
  baseline for Apple and JPMorgan at every horizon, cutting JPMorgan's one-month
  MSE by 47% (relative MSE 0.53), and is retained in the Model Confidence Set in
  11 of 12 main-horizon cells. It is weaker only on Amazon, the one stock outside
  the original study's sample.
- **Tree ensembles collapse at long horizons.** Random forest and gradient
  boosting are competitive at the one-day horizon (relative MSE near 1.0) but
  deteriorate sharply at one month: gradient boosting on the full predictor set
  reaches 4.0x, 3.4x, and 11.9x the HAR baseline MSE for Apple, Amazon, and
  JPMorgan.
- **The "gains grow with horizon" result does not replicate.** The original
  paper finds the ML advantage widens as the horizon lengthens. Here every
  flexible model except LogHAR gets worse from one day to one month; the
  neural-net ensemble for JPMorgan moves from 0.84 to 1.51 relative MSE.
- **Extra predictors help linear models, not trees.** Adding eight macro and
  firm-level covariates improves LogHAR and Lasso but destabilises trees. A
  VIX-tercile split shows the tree deterioration concentrates in the calm and
  stressed extremes: random forest reaches 6.9x HAR in JPMorgan's low-volatility
  regime at the one-month horizon.

## Methodology

Daily realized variance and quarticity are built from 78 five-minute returns per
session, with a previous-tick interpolation rule for missing minutes. Two
predictor sets are used: the three HAR lags alone (M_HAR), and those lags plus
eight covariates (momentum, dollar volume, an earnings dummy, VIX, the 3-month
T-bill rate, Hang Seng return, the ADS business-conditions index, and economic
policy uncertainty). HAR-family, linear, and tree models use rolling-window
refits; the neural-net ensemble uses a fixed window and averages the best 10 of
100 random seeds. At multi-step horizons the training window is lagged by h-1
days to remove look-ahead. Inference uses the Diebold-Mariano test with the
Harvey-Leybourne-Newbold small-sample correction and a 10,000-replication
stationary-bootstrap Model Confidence Set.

## Repository Structure

```
src/
  build_rv.py              realized variance and quarticity from 1-min bars
  fetch_macro.py           macro covariate downloads
  build_predictors.py      stock-specific predictors
  merge_master.py          per-stock merged model inputs
  split.py                 chronological train/validation/test split
  horizons.py              multi-step target construction
  models/har_family.py     HAR, HAR-X, LogHAR, HARQ
  models/ml_models.py      Lasso, ElasticNet, random forest, GB, neural net
  inference/statistical_inference.py  relative MSE, DM test, MCS
  inference/ale.py         Accumulated Local Effects variable importance
  inference/regime_split.py  VIX-regime MSE breakdown
  make_figures.py          report figures and tables
data/                      raw, interim, external, and final model inputs
results/                   predictions, tables, and figures
```

## Reproduction

Place 1-minute OHLCV files in `data/raw/` as `<TICKER>.txt`, install
dependencies with `pip install -r requirements.txt`, then run the pipeline:

```
python src/build_rv.py
python src/fetch_macro.py
python src/build_predictors.py
python src/merge_master.py
python src/models/har_family.py
python src/models/ml_models.py
python src/inference/statistical_inference.py
python src/inference/ale.py
python src/inference/regime_split.py
python src/make_figures.py
```

Multi-step horizons take a horizon argument, for example
`python src/models/ml_models.py 22`. All randomness is seeded and intermediate
artifacts are written to `data/` and `results/`, so each stage can be re-run
independently.

## References

Christensen, K., Siggaard, M., & Veliyev, B. (2023). A Machine Learning Approach
to Volatility Forecasting. *Journal of Financial Econometrics*, 21(5), 1680-1727.

Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
*Journal of Financial Econometrics*, 7(2), 174-196.

Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the Errors: A
Simple Approach for Improved Volatility Forecasting. *Journal of Econometrics*,
192(1), 1-18.

Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal
of Business & Economic Statistics*, 13(3), 253-263.
