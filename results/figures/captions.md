# Report figure & table captions

Drop into the LaTeX `\caption{}` slots; edit freely.

## Figure 1 — Relative MSE vs HAR (main)

Out-of-sample MSE of each model relative to the HAR benchmark (ratio < 1 = beats
HAR), for AAPL, JPM, AMZN. Within each stock panel, bars are grouped by the five
representative models; colour = information set (M_HAR vs M_ALL), hatch =
horizon (plain h=1, hatched h=22). Dashed line = HAR baseline. Stars: one-sided
Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction) that the
model beats HAR — * p<0.10, ** p<0.05, *** p<0.01. Bars above 2.0 are capped and
value-labelled. HAR-X has only M_ALL bars; in M_HAR it reduces to HAR. Our
2023-24 test period is shorter and calmer than the paper's 2001-17 sample, so
magnitudes are not directly comparable — read for qualitative pattern.

## Table 1 — Relative MSE with DM significance (main)

Out-of-sample MSE relative to HAR, five models x three stocks x two information
sets x two horizons (h = 1 day, h = 22 days). Stars: one-sided HLN-corrected DM
that the model beats HAR — * p<0.10, ** p<0.05, *** p<0.01. "$-$" = HAR-X is
identical to HAR in M_HAR (paper Tables 2/6).

## Figure 2 — ALE-based variable importance (main)

Variable importance from Accumulated Local Effects (Apley & Zhu 2020), computed
hand-rolled per the paper's Eq. 28-31 with 100 quantile bins. Rows = stock,
columns = model (RF, NN$_2^{10}$), M_ALL, 1-day horizon. Each panel: the 11
M_ALL predictors, horizontal bars sorted by VI (normalised to sum to 1). NN
values are from the post-y-standardisation ensemble re-run. EA collapses to
VI = 0 (a binary indicator degenerates under quantile binning — a known ALE
limitation the paper also notes).

## Figure A1 — Realized variance time series (appendix)

Daily realized variance for each stock, in annualised standard-deviation units
(sqrt(RV x 252) x 100). Dashed lines mark the chronological 70/10/20
train/validation/test boundaries. The 2020 COVID spike and the 2022 episode are
visible; the test period (from March 2023) is comparatively calm.

## Figure A2 — Relative MSE vs HAR, all three horizons (appendix)

Relative MSE vs HAR as a 3x3 small-multiples grid: rows = stocks (AAPL, JPM,
AMZN), columns = forecast horizon (1-day, 1-week, 1-month). Within each panel,
two bars per model give the M_HAR and M_ALL information sets (colour); HAR-X has
only an M_ALL bar. Dashed line = HAR baseline. DM stars and the 2.0 bar cap as
in Figure 1.

## Table A1 — Full relative-MSE table (appendix)

Relative MSE vs HAR for all nine models x three stocks x two information sets x
three horizons. Stars and "$-$" as in Table 1. HAR is the benchmark (1.000).

## Table A2 — Model Confidence Set inclusion (appendix)

MCS membership at 90% confidence (stationary bootstrap, 10,000 replications) per
(stock, info-set, horizon). Y = retained, N = eliminated. $\dagger$ marks cells
where the MCS is unreliable: at h=5 and h=22 the heavy-tailed RF/GB loss series
inflate the stationary-bootstrap variances and block elimination. $\ddagger$
marks a borderline cell. In flagged cells rely on the relative-MSE and DM
evidence instead.

## Table A3 — Regime-split relative MSE (appendix)

Relative MSE vs HAR (M_ALL) with the test set split into VIX terciles
(low/mid/high volatility); VIX ranges in the column headers. Context for why our
test-period magnitudes differ from the paper's — the deterioration of the
extended linear models concentrates in the high-VIX regime — not a deliberate
regime-extension claim.
