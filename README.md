# Volatility Forecasting Replication

Replication of Christensen, Siggaard and Veliyev (2023, *Journal of Financial
Econometrics*), "A Machine Learning Approach to Volatility Forecasting", applied
to AAPL, JPM, and AMZN over 2016 to 2024.

The deliverable is a 3-page report plus appendix. This repository contains the
data pipeline, model fits, and statistical inference code.

## Models

Nine models, two information sets, three stocks, one-day forecast horizon.

| Family             | Models                              |
|--------------------|-------------------------------------|
| HAR family         | HAR, HAR-X, LogHAR, HARQ            |
| Regularised linear | Lasso, Elastic Net                  |
| Trees              | Random Forest, Gradient Boosting    |
| Neural net         | NN_2 with 10-net ensemble of 100    |

Plus a naive random-walk benchmark. M_HAR contains daily, weekly, and monthly
RV lags. M_ALL adds 8 covariates: 1-week momentum, log dollar volume change,
earnings-announcement dummy, VIX, Hang Seng squared log return, ADS business
conditions index, 3-month T-bill change, and the EPU index.

## Directory layout

```
data/
  raw/        1-min OHLCV files (gitignored)
  interim/    daily RV, RQ, stock-specific covariates
  external/   daily macro series (VIX, HSI, US3M, ADS, EPU)
  final/      per-stock master DataFrames
results/
  predictions/  one CSV per (stock, infoset, model, horizon) (gitignored)
  figures/      final figures for the report
  logs/         dropped-days log, fill log (gitignored)
src/
  build_rv.py            Stage 1: daily RV and RQ from 1-min bars
  fetch_macro.py         Stage 2: macro series from FRED, Philly Fed, EPU
  build_predictors.py    Stage 3: stock-specific predictors
  merge_master.py        Stage 4: assemble per-stock master DF
  split.py               Stage 5: train/val/test split, standardisation
  models/
    har_family.py        Stage 6: HAR, HAR-X, LogHAR, HARQ
    ml_models.py         Stage 7: Lasso, EN, RF, GB, NN_2
```

## Reproduce

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/build_rv.py
python src/fetch_macro.py
python src/build_predictors.py
python src/merge_master.py
python src/models/har_family.py
python src/models/ml_models.py    # approximately 4 hours
```

Each stage reads from `data/` and writes to `data/` or `results/`. Stages can
be re-run independently once their inputs exist. Stages 8 to 13 (Diebold-Mariano
tests, Model Confidence Set, Accumulated Local Effects, figures, report writing)
are pending.

## Known deviations from the paper

- Three stocks (AAPL, JPM, AMZN) rather than 29 DJIA constituents.
- Sample window is 2016 to 2024 rather than 2001 to 2017.
- IV is dropped from M_ALL. The paper uses firm-specific OptionMetrics implied
  volatility; we have no equivalent source. VIX is kept separately.
- Test set is roughly half the size of the paper's, which reduces statistical
  power for the DM test and Model Confidence Set.
- AMZN was added to the DJIA in 2024 and is not in the paper's sample. It is
  included here because we have the data.
