# Realised Volatility Forecasting: HAR Models vs Machine Learning

## Overview

This project replicates the methodology of Christensen, Siggaard & Veliyev
(2023, *Journal of Financial Econometrics*) to test whether off-the-shelf
machine-learning models can outperform the HAR family — the long-standing
benchmark for realised-variance prediction. Accurate volatility forecasts
underpin risk management, derivatives pricing, and portfolio allocation, so
whether flexible ML methods add value over a parsimonious linear benchmark is a
question of direct practical interest.

The study covers three large-cap US stocks — Apple, JPMorgan, and Amazon — over
2016–2024, built from 1-minute price data. Nine forecasting models are evaluated
across two predictor sets and three horizons (1 day, 1 week, 1 month), with
formal predictive-accuracy testing and a model-agnostic variable-importance
analysis. The replication is methodological rather than numerical: the data
window and cross-section differ from the original study, so the aim is a
faithful re-implementation and an honest comparison of findings.

## Key Findings

- **LogHAR is the most reliable forecaster.** The log-transformed HAR beats the
  HAR benchmark for Apple and JPMorgan at every horizon — cutting JPMorgan's
  one-month MSE by 47% (relative MSE 0.53) — and survives in the Model
  Confidence Set in 11 of 12 main-horizon cells. A single nonlinearity, the log
  transform, does most of the work, not ML flexibility.
- **Tree ensembles collapse at long horizons.** Random forest and gradient
  boosting are competitive at one day (relative MSE near 1.0) but deteriorate
  sharply at one month: gradient boosting on the full predictor set reaches
  4.0×, 3.4×, and 11.9× the HAR benchmark MSE for Apple, Amazon, and JPMorgan.
- **The paper's central "gains grow with the horizon" result does not
  replicate.** In the original study the ML advantage widens with the forecast
  horizon. Here every flexible model except LogHAR gets *worse* from one day to
  one month; the neural-net ensemble for JPMorgan moves from 0.84 to 1.51
  relative MSE.
- **The extra predictors help linear models, not trees.** Adding eight macro
  and firm-level covariates improves LogHAR and Lasso but destabilises the tree
  ensembles. A VIX-regime split shows the tree deterioration concentrates in the
  calm and stressed extremes — random forest reaches 6.9× the HAR benchmark in
  JPMorgan's low-volatility regime at one month.

## Methodology

**Data.** Daily realised variance and realised quarticity are constructed from
78 five-minute returns per trading session, with a previous-tick interpolation
rule for missing intraday minutes. Each stock's series is split chronologically
into 70% training, 10% validation, and 20% test, with all predictors
standardised on training-window statistics only.

**Models.** Nine models span four families: the HAR family (HAR, HAR-X, LogHAR,
HARQ), regularised linear models (Lasso, Elastic Net), tree ensembles (random
forest, gradient boosting), and a neural-net ensemble (NN₂¹⁰ — ten networks
averaged from the best of 100 seeds). Two predictor sets are compared: the three
HAR lags alone (M_HAR), and those lags plus eight covariates (M_ALL) — one-week
momentum, dollar-volume change, an earnings-announcement dummy, VIX, the 3-month
T-bill rate, the Hang Seng return, the ADS business-conditions index, and
economic policy uncertainty. The HAR family and the regularised-linear and tree
models are refitted on rolling windows; the neural-net ensemble uses a fixed
window. At multi-step horizons the training window is lagged by h−1 days to
remove look-ahead.

**Inference.** Forecast accuracy is compared with the Diebold-Mariano test
(strengthened with the Harvey-Leybourne-Newbold small-sample correction) and the
Model Confidence Set (10,000-replication stationary bootstrap, 90% confidence).
Predictor influence is quantified with hand-rolled Accumulated Local Effects.
Robustness exhibits split test-set accuracy by VIX regime and by
realised-variance decile, and compare fitted-series persistence via the
autocorrelation function.

## Results

Relative MSE versus the HAR benchmark on the full predictor set (M_ALL); values
below 1.0 beat HAR.

| Model | AAPL 1-day | JPM 1-day | AMZN 1-day | AAPL 1-month | JPM 1-month | AMZN 1-month |
|---|---|---|---|---|---|---|
| LogHAR            | 0.90 | 0.67 | 0.92 | 0.71 | 0.53  | 1.15 |
| Random forest     | 0.98 | 0.80 | 1.08 | 1.36 | 3.80  | 2.62 |
| Gradient boosting | 1.10 | 0.75 | 1.12 | 4.00 | 11.89 | 3.41 |
| NN₂¹⁰             | 0.97 | 0.84 | 0.97 | 0.99 | 1.51  | 1.00 |

At the one-day horizon the flexible models are broadly competitive with HAR, in
line with the original study. At one month the picture diverges sharply: LogHAR
pulls clear of the field, while the tree ensembles diverge from the benchmark
rather than beating it. Two supporting exhibits localise this — an
RV-decile split shows LogHAR's edge is broad-based at one day but retreats from
the high-volatility deciles at one month, and an ACF comparison shows the
random-forest and neural-net fitted series carry markedly higher persistence
than HAR at the monthly horizon.

## Repository Structure

```
src/
  build_rv.py                  realised variance and quarticity from 1-min bars
  fetch_macro.py               macro covariate downloads
  build_predictors.py          stock-specific predictors
  merge_master.py              per-stock merged model inputs
  split.py                     chronological train/validation/test split
  horizons.py                  multi-step target construction
  models/har_family.py         HAR, HAR-X, LogHAR, HARQ
  models/ml_models.py          Lasso, Elastic Net, random forest, GB, neural net
  inference/statistical_inference.py  relative MSE, DM test, MCS
  inference/ale.py             Accumulated Local Effects variable importance
  inference/regime_split.py    VIX-regime MSE breakdown
  inference/rv_decile_split.py realised-variance-decile MSE breakdown
  inference/acf_persistence.py fitted-series autocorrelation analysis
  make_figures.py              figures and summary tables
data/                          raw, interim, external, and final model inputs
results/                       predictions, tables, and figures
```

## Reproduction

Place 1-minute OHLCV files in `data/raw/` as `<TICKER>.txt`, install
dependencies with `pip install -r requirements.txt`, then run the pipeline in
order:

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
python src/inference/rv_decile_split.py
python src/inference/acf_persistence.py
python src/make_figures.py
```

Multi-step horizons take a horizon argument, for example
`python src/models/ml_models.py 22`. All randomness is seeded and every stage
writes intermediate artefacts to `data/` and `results/`, so stages can be re-run
independently.

## References

Christensen, K., Siggaard, M., & Veliyev, B. (2023). A Machine Learning Approach
to Volatility Forecasting. *Journal of Financial Econometrics*, 21(5), 1680–1727.

Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
*Journal of Financial Econometrics*, 7(2), 174–196.

Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the Errors: A
Simple Approach for Improved Volatility Forecasting. *Journal of Econometrics*,
192(1), 1–18.

Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal
of Business & Economic Statistics*, 13(3), 253–263.

Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set.
*Econometrica*, 79(2), 453–497.

Apley, D. W., & Zhu, J. (2020). Visualizing the Effects of Predictor Variables
in Black Box Supervised Learning Models. *Journal of the Royal Statistical
Society: Series B*, 82(4), 1059–1086.
