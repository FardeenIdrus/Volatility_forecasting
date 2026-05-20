# Verification Pass — Implementation vs Christensen, Siggaard & Veliyev (2023, JFE)

Full-pass paranoid verification of our code against `rv1.pdf`. Per-section: paper reference,
code reference, and a verdict — ✓ matches / ⚠ deviates with reason (document) / ✗ deviates
without reason (must fix).

**Note on the brief's section/equation numbers**: the request used section labels like
"§1.7", "§1.10", "§1.12" and equation ranges that do not match the paper. The paper's actual
structure is §1.1 Setting, §1.2 HAR family, §1.3 Regularization, §1.4 Tree-Based Regression,
§1.5 Neural Network, §1.6 Forecast Comparison, §2 Data, §3 Results (§3.2 = ALE), §4 Longer-Run
Forecasting. Equations run 1–31. Citations below use the paper's *actual* numbering.

**Headline**: no ✗ findings (no unreasoned deviations). Several ⚠ deviations, all with a
defensible reason, listed for §12 documentation / §4 disclosure in the consolidated table at
the end. The single most consequential is the Lasso/EN hyperparameter-tuning method (the
request's premise that "the paper uses 5-fold CV" is incorrect — see Section 3).

---

## Section 1 — RV construction (`build_rv.py`)

Paper: §1.1 eq 2 (RV, p. 1683), eq 11 (RQ, p. 1684); §2 (p. 1692) "we construct a 5-min
log-return series and compute ... daily realized variance RV_t with n=78".

| Item | Paper | Code | Verdict |
|---|---|---|---|
| 5-min returns from 1-min bars, n=78 | eq 2, §2 "n=78" | `N_5MIN_RETURNS = 78` ([build_rv.py:50](../../src/build_rv.py#L50)); `_five_min_returns_from_closes` builds 79 sample points → 78 log-diffs ([build_rv.py:76-89](../../src/build_rv.py#L76-L89)) | ✓ |
| RV formula | eq 2: RV = Σ\|Δ^n X\|² | `rv = np.sum(rets ** 2)` ([build_rv.py:113](../../src/build_rv.py#L113)) | ✓ |
| RQ formula | eq 11: RQ = (n/3) Σ\|Δ^n X\|⁴ | `rq = (N_5MIN_RETURNS/3.0) * np.sum(rets**4)` ([build_rv.py:114](../../src/build_rv.py#L114)) | ✓ |
| No overnight return | eq 2 is intraday by construction (Σ over j=1..n intraday returns) | First 5-min return = `log(c[4]) - log(c[0])` = 09:34 close vs 09:30 close — within-day; no prev-close-to-open term ([build_rv.py:84-89](../../src/build_rv.py#L84-L89)) | ✓ |
| Intraday-only first return | Paper does not state the first-interval convention explicitly | `sample_idx = [0] + [4,9,...,389]` → first 5-min return spans **4 minutes** (09:30→09:34), the other 77 span 5 minutes | ⚠ |
| Display units | Table 1 reports annualised values (RVD mean ≈ 20.67); paper gives no formula | `sqrt(RV*252)*100` for the log line only, not persisted ([build_rv.py:202](../../src/build_rv.py#L202)) | ✓ |

**⚠ First-interval is 4-min, not 5-min.** 390 one-minute bars (09:30–15:59) yield 78 five-minute
intervals only if the first interval is anchored on the 09:30 bar's close (09:30→09:34 = 4 min).
The paper's idealised grid would be 09:30, 09:35, …, 16:00 — but NYSE 1-min TAQ data has no
16:00 bar, so the paper would face the same issue. Defensible, resolved in CLAUDE.md §7.1.
**Action**: already documented; mention in §2/§4 as a minor RV-construction detail.

**Related data deviation (cross-reference, not strictly "RV construction")**: days with
350–389 bars are kept and missing intra-session minutes are forward-filled (previous-tick);
days <350 bars dropped ([build_rv.py:49,108-110](../../src/build_rv.py#L49)). Paper §2 cites
BNHLS (2009) cleaning but states no bar-count rule. Already in CLAUDE.md §7.1/§12. ⚠ — disclose.

---

## Section 2 — HAR family (`har_family.py`)

Paper: §1.2 eq 5 (HAR), eq 6 (LogHAR), eq 10 (HARQ), eq 11 (RQ), pp. 1683-1684; HAR-X §1.2
p. 1684; LogHAR transform §2 p. 1693; insanity filter / negative-clip §1.6 p. 1691; HAR
rolling window §2 p. 1692.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| HAR | eq 5: RV = β₀+β₁RVD+β₂RVW+β₃RVM | `ModelSpec("HAR", …, HAR_LAGS)`, OLS ([har_family.py:70,103-107](../../src/models/har_family.py#L70)) | ✓ |
| LogHAR (M_HAR) | eq 6: log RV on log RV-lags | `log_cols=HAR_LAGS, log_target=True` ([har_family.py:71-72](../../src/models/har_family.py#L71)) | ✓ |
| LogHAR (M_ALL) log-transform RV+VIX only | §2: "in the LogHAR we also log-transform VIX and IV" | `log_cols=HAR_LAGS + ["VIX"]` — RV-lags + VIX logged, other extras in levels ([har_family.py:82](../../src/models/har_family.py#L82)) | ✓ (IV omitted, see §12) |
| HARQ | eq 10: (β₁+β₁Q·RQ^½)·RVD term | `out["RVD_x_sqrtRQ"] = df["RVD"]*sqrt(df["RQ_lag"])` added alongside plain RVD/RVW/RVM ([har_family.py:98-99](../../src/models/har_family.py#L98-L99)) — algebraically expands eq 10 | ✓ |
| HAR-X covariate list | §1.2: HAR-X = HAR + all extra covariates linearly | `ModelSpec("HAR-X", …, HAR_LAGS + M_ALL_EXTRAS)` ([har_family.py:80](../../src/models/har_family.py#L80)); omitted from M_HAR (≡ HAR) | ✓ |
| Negative-clip (all models) | §1.6: "If a model predicts volatility to be negative, we replace the forecast with the minimum in-sample realized variance" | `if pred < 0: pred = min(fit_df["y_target"])` — applied to every spec ([har_family.py:179-181](../../src/models/har_family.py#L179-L181)) | ✓ |
| HARQ insanity filter (upper bound) | §1.6: "We also adopt the insanity filter from Bollerslev, Patton, and Quaedvlieg (2016) for the HARQ" — **rule not reproduced in rv1** | HARQ-only: `pred > max(fit_df["y_target"]) → min` ([har_family.py:183-186](../../src/models/har_family.py#L183-L186)) | ⚠ |
| HAR rolling window | §2: HAR family "merge the training and validation set ... rolling window forecast" | `rolling_forecast`, `WINDOW_SIZE = 1776` = train(1554)+val(222) ([har_family.py:52,132-190](../../src/models/har_family.py#L52)) | ✓ |

**⚠ HARQ upper-clip.** The paper cites the BPQ (2016) insanity filter but does **not** reproduce
the rule. We cannot verify our exact implementation against `rv1.pdf` alone. Our rule replaces
HARQ forecasts that are negative *or* exceed the in-window max target with the in-window
minimum target. The negative-clip half is verbatim paper §1.6; the upper-clip half is our
documented interpretation (CLAUDE.md §6.9). **Uncertain — flagged honestly.**

**Note** — threshold uses `y_target` (= daily RV at h=1, h-day mean RV at h>1). At h=1 this is
exactly paper §1.6's "in-sample RV"; at h>1 it is the in-window minimum/maximum of the h-day
mean target — a consistent extension.

---

## Section 3 — Regularised linear: Lasso, Elastic Net (`ml_models.py`)

Paper: §1.3 eq 15 (ridge), eq 16 (lasso), eq 17 (EN), pp. 1686-1687; standardisation §2 p. 1693;
hyperparameter tuning Appendix A.4 p. 1720 + footnote 22; rolling scheme §2 p. 1692; grids
Table A.6 p. 1721.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| Predictor standardisation, training-set stats | §2: "standardize the input data with the sample mean and sample variance from the training set" | Per-rolling-window mean/std, ddof=0 ([ml_models.py:148-152](../../src/models/ml_models.py#L148-L152)) | ⚠ |
| α (λ) tuning method | **Appendix A.4 footnote 22: "Standard k-fold cross-validation has not been conducted, as it violates the time-series structure in our data."** Paper tunes on the single validation set | `LassoCV(cv=TimeSeriesSplit(5))`, `ElasticNetCV(cv=TimeSeriesSplit(5))` ([ml_models.py:268-271,288-292](../../src/models/ml_models.py#L268-L271)) | ⚠ |
| λ grid | Table A.6: λ ∈ [10⁻⁵,10²], 1000-point partition | `alphas=np.logspace(-5, 2, 1000)` ([ml_models.py:269](../../src/models/ml_models.py#L269)) | ✓ |
| EN α (l1_ratio) grid | Table A.6: α ∈ [0,1], 10-point partition | `l1_ratio=[0.1,0.2,…,0.9,0.99]` — 10 points ([ml_models.py:289](../../src/models/ml_models.py#L289)) | ⚠ |
| Rolling scheme, train-only window | §2: for RR/LA/EN/GB "a rolling scheme without concatenation of the training and validation set" | `ROLLING_WINDOW_SMALL = 1554` (train only) ([ml_models.py:87](../../src/models/ml_models.py#L87)) | ✓ |

**⚠ MOST CONSEQUENTIAL FINDING — tuning method.** The request's premise ("paper uses 5-fold
over validation set") is **incorrect**. Appendix A.4 footnote 22 states the paper did **not**
use k-fold CV at all; it tunes by fitting on the training set across a grid and selecting the
hyperparameter with the best *single validation-set* MSE. We use `LassoCV`/`ElasticNetCV` with
`cv=TimeSeriesSplit(5)` — 5-fold expanding-window CV inside each rolling window. **Reason it is
⚠ not ✗**: `TimeSeriesSplit` preserves temporal order, so it does *not* commit the look-ahead
sin the paper's footnote 22 warns against; and under our 445-step rolling refit, re-evaluating
against a fixed calendar-time validation block at every step is ill-posed (the block drifts
arbitrarily far from later windows). Internal time-series CV on each rolling window is a
defensible adaptation — but it **is** a deviation. **Action: document in CLAUDE.md §12, surface
in §4.**

**⚠ Per-window standardisation.** Paper §2 says "from the training set". Under rolling refits
the "training set" is the current rolling window; we standardise per-window. Spirit-consistent;
note in §2.

**⚠ EN grid endpoint.** `0.99` substituted for `1.0` (`ElasticNetCV` rejects pure-L1). Also
note: paper's α and sklearn's `l1_ratio` are inversely defined (paper footnote 2: EN nests LA
at α=0, RR at α=1; sklearn `l1_ratio=1` = pure L1). Immaterial for model selection — the CV
searches the same convex space — but the diagnostic logs `l1_ratio` in sklearn's convention.

---

## Section 4 — Tree methods: RF, GB (`ml_models.py`)

Paper: §1.4 eq 19-22, pp. 1687-1689; hyperparameters Table A.6 p. 1721.

| Item | Paper (Table A.6) | Code | Verdict |
|---|---|---|---|
| RF n_estimators | "Trees = 500" (fixed) | `n_estimators=500` ([ml_models.py:312](../../src/models/ml_models.py#L312)) | ✓ |
| RF min node size | "Min. node size = 5" | `min_samples_leaf=5` ([ml_models.py:313](../../src/models/ml_models.py#L313)) | ✓ |
| RF feature split | "Feature split = J/3" | `max_features=1/3`; `None` when J≤3 (M_HAR) ([ml_models.py:310-313](../../src/models/ml_models.py#L310-L313)) | ⚠ |
| RF rolling window | §2: RF "also adopted" the train+val merge | `ROLLING_WINDOW_LARGE = 1776` ([ml_models.py:86](../../src/models/ml_models.py#L86)) | ✓ |
| GB n_estimators | "Trees ∈ {50,100,…,500}" (**tuned**) | `n_estimators=500` **fixed** ([ml_models.py:336](../../src/models/ml_models.py#L336)) | ⚠ |
| GB depth | "Depth ∈ {1,2}" (**tuned**) | `max_depth=2` **fixed** ([ml_models.py:336](../../src/models/ml_models.py#L336)) | ⚠ |
| GB learning rate | "Learning rate ∈ {0.01,0.1}" (tuned) | tuned over {0.01,0.1} by validation MSE ([ml_models.py:334-341](../../src/models/ml_models.py#L334-L341)) | ✓ |

**⚠ RF feature split.** sklearn float `max_features=1/3` selects `max(1, int(1/3·J))` features
per split. M_ALL J=11 → 3 features; the paper's J/3 with its J=12 → 4. Slight difference from
both rounding and our IV omission (J=11 vs 12). M_HAR J=3 → J/3=1 is degenerate (depth-1 stump);
we fall back to `max_features=None`. Documented (CLAUDE.md §7.3); disclose in §4.

**⚠ GB partial tuning.** Table A.6 marks GB's n_estimators, depth **and** learning rate as
tuned (asterisks). We tune only `learning_rate`, fixing `n_estimators=500` and `max_depth=2`
(both at the top of the paper's grids). Reason: keeps the rolling-refit tractable and depth=2
is within the paper's {1,2}. But it **is** a partial deviation — the paper tunes 3 GB
hyperparameters, we tune 1. **Action: document in §12, surface in §4.**

---

## Section 5 — Neural network NN_2^10 (`ml_models.py`)

Paper: §1.5 eq 23-25, pp. 1689-1691; Appendix A.5 pp. 1721-1722; Table A.6 p. 1721.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| Architecture input→4→2→1 | §1.5: "NN2 is two-layered with 4 and 2 neurons" | `Dense(4) → Dense(2) → Dense(1, linear)` ([ml_models.py:182-191](../../src/models/ml_models.py#L182-L191)) | ✓ |
| Activation L-ReLU, c=0.01 | eq 25, footnote 8: "c=0.01" | `LeakyReLU(negative_slope=0.01)` ([ml_models.py:185,188,96](../../src/models/ml_models.py#L185)) | ✓ |
| Optimiser Adam, lr=0.001 | §1.5: "predetermine the learning rate at 0.001"; Table A.6 "Adam = default" | `Adam(learning_rate=0.001)` ([ml_models.py:192,94](../../src/models/ml_models.py#L192)) | ✓ |
| Epochs = 500 | Table A.6: "Epochs = 500" | `NN_MAX_EPOCHS = 500` ([ml_models.py:91](../../src/models/ml_models.py#L91)) | ✓ |
| Early-stopping patience = 100 | Appendix A.5: "we set the patience to 100" | `NN_PATIENCE = 100` ([ml_models.py:92](../../src/models/ml_models.py#L92)) | ✓ |
| Glorot normal initialiser | Table A.6: "Initializer: Glorot normal" | `kernel_initializer="glorot_normal"` on all Dense layers ([ml_models.py:184,187,190](../../src/models/ml_models.py#L184)) | ✓ |
| Ensemble: 100 nets → top-10 by val MSE → mean | §1.5 + A.5: "100 independent neural networks ... an ensemble of the ten best" | `NN_TOTAL=100, NN_TOP=10`; `top_idx = argsort(val_mses)[:10]`; mean over top-10 ([ml_models.py:89-90,234,238-241](../../src/models/ml_models.py#L89-L90)) | ✓ |
| Fixed window (no rolling) | §2: "fixed window estimation for the NNs" | NN branch uses `sp.X_*_std` once, no rolling ([ml_models.py, run_one NN_2_e10 branch](../../src/models/ml_models.py)) | ✓ |
| Batch size | Not specified in paper | `NN_BATCH_SIZE = 32` ([ml_models.py:93](../../src/models/ml_models.py#L93)) | ✓ (paper silent) |
| Dropout rate | Table A.6: "Drop-out = 0.8"; A.5: "drop-out rate is set to 0.8 following Goodfellow, Bengio, and Courville (2016)" | `Dropout(0.2)` (Keras) ([ml_models.py:186,189,95](../../src/models/ml_models.py#L186)) | ✓ (see investigation) |
| Target standardisation | Paper standardises **predictors** (§2); **silent on the target** | NN target y-standardised on training set, predicted in std space, inverted before negative-clip ([ml_models.py, run_one NN_2_e10 branch](../../src/models/ml_models.py)) | ⚠ |

**Dropout convention — investigation (as requested).** Keras `Dropout(rate)` interprets `rate`
as the **drop** probability — `Dropout(0.2)` drops 20%, keeps 80%. The paper says "drop-out rate
is set to 0.8". Determination:

1. The paper cites **Goodfellow, Bengio & Courville (2016)**, *Deep Learning*. That book
   parameterises dropout by the probability of **including** (keeping) a unit, and its standard
   recommendation is **0.8 for input units**, 0.5 for hidden units — i.e. 0.8 is a *keep*
   probability.
2. **Srivastava et al. (2014)** (the original dropout paper, also cited in A.5) likewise uses
   `p` = probability of **retaining** a unit.
3. The network is tiny — hidden layers of 4 and 2 neurons. Dropping 80% would retain ≈0.8 and
   ≈0.4 neurons on average — destructive and nonsensical. Keeping 80% retains ≈3.2 and ≈1.6.

**Conclusion**: the paper's "drop-out rate 0.8" is a **keep probability** (Goodfellow/Srivastava
convention), equivalent to a 0.2 *drop* probability. Keras `Dropout(0.2)` therefore **matches**
the paper's intent. The paper's wording is loose (the phrase "drop-out rate" more often means
drop-probability), so this is an interpretation — but a strongly justified one. ✓ with caveat;
record the reasoning so §4/appendix can defend it if challenged.

**⚠ Target y-standardisation.** Paper §2 standardises predictors and is **silent on the target**.
We standardise the NN target on the training-set mean/std (ddof=0), train and predict in
standardised space, then invert to raw RV units before the negative-clip. This is a
numerical-conditioning fix (raw RV ≈ 1e-4 is ill-conditioned for a linear-head NN under
Adam(1e-3); it caused the JPM×M_ALL h=22 71× blow-up). Already in CLAUDE.md §7.3 and §12.
Applied at both horizons. **Action: keep documented; surface in §4 as an implementation
measure not present in the paper.**

---

## Section 6 — Diebold-Mariano test (`run_stage8.py`)

Paper: §1.6 p. 1691 — "we compute a pairwise Diebold and Mariano (1995) test ... with a
one-sided alternative"; Tables 2-7 notes — "H0: MSE_i = MSE_j against a one-sided alternative
H1: MSE_i > MSE_j". **The paper gives no DM equations** (the request's "eq 22-25" is wrong —
eq 22 is the GB update). **The paper does not cite Harvey-Leybourne-Newbold** — HLN (1997) is
absent from the reference list.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| One-sided alternative H1: MSE_i > MSE_j | Tables 2-7 notes | `dm_test_hln`: H1 E[loss_i−loss_j]>0, model i worse; one-sided p = 1−t.cdf ([run_stage8.py:80-107](../../src/inference/run_stage8.py#L80-L107)) | ✓ |
| HLN small-sample correction | **Not in the paper** | `hln_mult = sqrt((T+1-2h+h(h-1)/T)/T)` ([run_stage8.py:104](../../src/inference/run_stage8.py#L104)) | ⚠ |
| NW HAC variance, bandwidth h−1 | Paper does not specify the HAC lag | `for lag in range(1, h)` Bartlett weights `(1-lag/h)`; h=1 → sample variance ([run_stage8.py:97-100](../../src/inference/run_stage8.py#L97-L100)) | ✓ (standard) |
| Reference distribution | Not specified | `stats.t.cdf(dm_stat, df=T-1)` ([run_stage8.py:106](../../src/inference/run_stage8.py#L106)) | ⚠ (pairs with HLN) |

**⚠ HLN correction is our addition.** The paper specifies only "Diebold and Mariano (1995)" —
which uses an asymptotic N(0,1) reference. We apply the Harvey-Leybourne-Newbold (1997)
small-sample correction and a t_{T−1} reference. The HLN formula and t-reference are correctly
implemented, but they are **not the paper's stated method**. Reason: our test set (~445 days)
is smaller than the paper's (847); HLN is the standard small-sample DM correction and is more
conservative. Defensible methodological improvement, but a deviation. Already in CLAUDE.md
§4.6/§7.4. **Action: surface in §2/§4 ("we strengthen the DM test with the HLN correction").**

---

## Section 7 — Model Confidence Set (`run_stage8.py`)

Paper: §1.6 p. 1691 — "a Model Confidence Set (MCS) of Hansen, Lunde, and Nason (2011)";
Figure 4 caption — "overall confidence level is set to 90%". **No MCS equations in the paper**
(the request's "eq 26-27" is wrong — eq 26 is the OOS loss).

| Item | Paper | Code | Verdict |
|---|---|---|---|
| 90% confidence | Figure 4: "confidence level is set to 90%" | `MCS_SIZE = 0.10` ([run_stage8.py:50](../../src/inference/run_stage8.py#L50)) | ✓ |
| MCS procedure | "Hansen, Lunde, and Nason (2011)" | `arch.bootstrap.MCS` ([run_stage8.py:133-140](../../src/inference/run_stage8.py#L133-L140)) | ✓ |
| Bootstrap type | **Not specified** by the paper | `bootstrap="stationary"` ([run_stage8.py:137](../../src/inference/run_stage8.py#L137)) | ⚠ |
| Replications | **Not specified** | `MCS_REPS = 10000` ([run_stage8.py:51](../../src/inference/run_stage8.py#L51)) | ⚠ |
| Elimination statistic | Not specified (HLN 2011 define T_R and T_max) | `arch` default (T_R / range statistic) — `method` not passed ([run_stage8.py:133-140](../../src/inference/run_stage8.py#L133-L140)) | ⚠ |
| Block length | Not specified | `arch` auto-selects (Politis-White optimal) — `block_size` not passed | ✓ (per CLAUDE.md §7.5) |
| Loss | §1.6 eq 26: squared error | `squared_loss` ([run_stage8.py:63-65](../../src/inference/run_stage8.py#L63-L65)) | ✓ |

**⚠ MCS configuration choices.** The paper specifies only "HLN (2011) MCS at 90%". Bootstrap
type (stationary), replications (10,000), and the elimination statistic (arch's default) are
**our choices** — the paper is silent. All standard and documented in CLAUDE.md §7.5. **Action:
note in §2 that the bootstrap configuration is our specification.**

**Known issue (not a paper deviation)**: at h=22 and h=5 the MCS is corrupted in some cells by
RF/GB heavy-tailed losses inflating bootstrap variances (JPM M_ALL + AAPL M_HAR at h=22; AAPL
M_HAR + AAPL M_ALL at h=5). Already recorded in PROJECT_NOTES §10; report leans on relative
MSE + DM in those cells.

---

## Section 8 — ALE / variable importance (`ale.py`)

Paper: §3.2 eq 27 (ALE integral), eq 28 (uncentered estimator), eq 29 (centered), eq 30
(I(Z_j)), eq 31 (VI), pp. 1700-1702; footnote 21 (p. 1701) — "we partition Z_j into 100
subintervals containing equally many observations".

| Item | Paper | Code | Verdict |
|---|---|---|---|
| 100 quantile bins | Footnote 21 | `N_BINS = 100`; `np.quantile(z, linspace(0,1,101))` ([ale.py:52](../../src/inference/ale.py#L52)) | ✓ |
| Uncentered ALE | eq 28: per-bin mean local effect, then cumsum | per-bin `mean(f(z_k)-f(z_{k-1}))`, `np.cumsum` ([ale.py compute_ale](../../src/inference/ale.py)) | ✓ |
| Centering | eq 29: subtract (1/T₀)Σ uncentered-ALE(z_jt) | subtract mean of uncentered ALE at obs; obs evaluated at the bin's **upper edge** | ⚠ |
| VI = std of centered ALE | eq 30: I(Z_j) = sqrt((1/(T₀−1))Σ[f^ALE(z_jt)]²) | `vi = (ale_at_obs - centring).std()` — numpy default **ddof=0** | ⚠ (immaterial) |
| VI normalised to sum 1 | eq 31: VI(Z_j) = I(Z_j)/ΣI(Z_j) | `vi_norm = vi_raw / vi_raw.sum()` ([ale.py compute_all_features](../../src/inference/ale.py)) | ✓ |

**⚠ Centering at the upper bin edge.** Eq 29 centers by the uncentered ALE evaluated "at z_jt".
The uncentered ALE is a step function on bin edges; we evaluate each observation at its bin's
upper edge. Paper eq 28-29 do not pin the within-bin convention. Curve shape identical; VI
robust. Minor — note only.

**⚠ ddof in I(Z_j) — paranoid catch, turns out immaterial.** Eq 30 divides by (T₀−1) (ddof=1);
our `.std()` uses numpy ddof=0. **But** VI is normalised (eq 31), and the factor
sqrt(T₀/(T₀−1)) is identical for every predictor (same T₀), so it **cancels exactly** in the
normalisation. Reported VI is bit-identical either way. ✓ in effect — no action needed,
mention only if a reviewer asks.

---

## Section 9 — Horizons (`horizons.py`, `har_family.py`, `ml_models.py`)

Paper: §4 p. 1703 — "We replace the dependent variable ... to next-week average realized
variance (RV_{t+5|t+1}) and next-month average realized variance (RV_{t+22|t+1})";
RV_{t-1|t-h} averaging definition eq 5 / eq 8.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| h=1 target = RV_{t+1} | §4 | `build_h_step_target(rv,1)` returns RV unchanged ([horizons.py](../../src/horizons.py)) | ✓ |
| h=5 target = mean(RV_{t+1..t+5}) | §4: RV_{t+5\|t+1} | `rv.rolling(5).mean().shift(-4)` ([horizons.py](../../src/horizons.py)) | ✓ |
| h=22 target = mean(RV_{t+1..t+22}) | §4: RV_{t+22\|t+1} | `rv.rolling(22).mean().shift(-21)` ([horizons.py](../../src/horizons.py)) | ✓ |
| Predictor lag structure consistent across horizons | §4: same information set Z_t, only the dependent variable changes | Features unchanged across horizons; only `y_target` changes ([har_family.py:245](../../src/models/har_family.py#L245), [ml_models.py:497](../../src/models/ml_models.py#L497)) | ✓ |
| No look-ahead in rolling training labels | Implied (real-time forecasting) | Training window lagged by h−1 so every label is realised at the origin ([har_family.py:124-126,162-164](../../src/models/har_family.py#L124-L126); [ml_models.py:139-141](../../src/models/ml_models.py#L139-L141)) | ✓ |

**Indexing note.** Under our shift-1 feature convention (row d has features through end of
day d−1 and target RV_d), `rolling(h).mean().shift(-(h-1))` puts mean(RV_d,…,RV_{d+h-1}) on
row d, which equals the paper's RV_{t+h|t+1}. Confirmed correct; the `shift(-(h-1))` choice
was deliberately verified (vs `shift(-h)`). ✓ — matches paper §4. The look-ahead-lag fix is a
**bug fix** (the pre-fix code contaminated the last h−1 training labels at h>1) — already in
PROJECT_NOTES §10 / CLAUDE.md.

---

## Section 10 — Train/validation/test split (`split.py`)

Paper: §2 p. 1692 — "a training set of 70% ... a validation set of 10% ... a test set of 20%";
standardisation §2 p. 1693.

| Item | Paper | Code | Verdict |
|---|---|---|---|
| Chronological 70/10/20, no shuffle | §2 | `TRAIN_FRAC=0.70, VAL_FRAC=0.10`; `df.sort_index()` then `iloc` slices ([split.py:32-34,65,70-72](../../src/split.py#L32-L34)) | ✓ |
| Standardisation uses training-set stats only | §2: "sample mean and sample variance from the training set" | `train_mean = X_train.mean()`, `train_std = X_train.std(ddof=0)`; applied to val/test ([split.py:82-83,91-93](../../src/split.py#L82-L93)) | ✓ |
| ddof for std | Paper says "sample variance" | `ddof=0` (matches sklearn `StandardScaler`) ([split.py:83](../../src/split.py#L83)) | ✓ |

**Note** — our split sizes are 1554/222/445 (h=1; test shorter at h>1), vs the paper's
2964/424/847. Same 70/10/20 *procedure*; the absolute sizes differ because our sample is
~2,221 days vs the paper's 4,257. Data-driven, already in CLAUDE.md §4.2/§12.

**Note** — HAR family and trees are fit on **raw** (unstandardised) features; only RR/LA/EN/NN
need standardisation per paper §2 ("for RR, LA, EN, and the NN"). Our rolling ML code
standardises RF/GB inputs too, which is harmless (trees are invariant to monotone per-feature
rescaling) and immaterial. ✓.

---

## Consolidated deviations — for CLAUDE.md §12 / report §4

No ✗ (unreasoned) findings. ⚠ deviations, each with a reason:

| # | Deviation | Type | Status |
|---|---|---|---|
| 1 | Lasso/EN tuned via `TimeSeriesSplit(5)` CV, not the paper's single-validation-set tuning (paper footnote 22: no k-fold CV) | Methodological | **Most consequential. Document §12, surface §4.** |
| 2 | GB: only `learning_rate` tuned; `n_estimators` and `depth` fixed (paper Table A.6 tunes all three) | Methodological | Document §12, surface §4 |
| 3 | DM test uses the HLN small-sample correction + t-reference; paper specifies plain DM (1995) | Methodological (improvement) | Document §12, mention §2/§4 |
| 4 | NN target y-standardised (paper standardises predictors only, silent on target) | Implementation (numerical) | Already CLAUDE.md §7.3/§12; surface §4 |
| 5 | HARQ upper insanity-clip rule is our interpretation (paper cites BPQ 2016 without the formula) | Implementation (uncertain) | Already CLAUDE.md §6.9; flag uncertainty |
| 6 | MCS bootstrap type / reps / elimination statistic are our choices (paper silent) | Implementation | Already CLAUDE.md §7.5; note §2 |
| 7 | RF `max_features=1/3` → 3 features at J=11 (vs paper J/3 with J=12 → 4); M_HAR fallback to `None` | Implementation (data-driven J) | Already CLAUDE.md §7.3; note §4 |
| 8 | First 5-min return spans 4 minutes (09:30→09:34), not 5 | Implementation (data-forced) | Already CLAUDE.md §7.1; minor note |
| 9 | Per-window standardisation under rolling refits (paper says "training set") | Implementation | Note §2 |
| 10 | EN l1_ratio grid endpoint 0.99 (not 1.0); paper α and sklearn l1_ratio inversely defined | Implementation (cosmetic) | Note only |
| 11 | RV forward-fill + 350-bar rule (paper cites BNHLS cleaning, no bar-count rule) | Data-driven | Already CLAUDE.md §7.1/§12 |

**Non-issues investigated and cleared**: ALE VI ddof=0 vs eq 30's ddof=1 (cancels in
normalisation); RF/GB input standardisation (immaterial for trees); batch_size=32 (paper
silent, standard default).

## Overall verdict

The implementation is **methodologically faithful** to `rv1.pdf`. Every model equation
(HAR, LogHAR, HARQ, eq 5/6/10/11; RV/RQ eq 2/11; NN eq 23-25; ALE eq 27-31) is correctly
implemented. The negative-clip, fixed-window NN, rolling HAR, ensemble selection, 100-bin ALE,
70/10/20 split, training-set standardisation, and Glorot-normal / patience-100 / Adam-0.001 NN
settings all match. The dropout convention question resolves to a match (Keras `Dropout(0.2)`
= the paper's keep-probability 0.8).

The 11 ⚠ deviations are all reasoned: data constraints (3 stocks, IV omitted, sample period,
bar-count rule), deliberate scope or robustness choices (HLN correction, CV tuning), or
numerical-correctness measures (NN y-standardisation). The two that most need honest
prominence in §4 are **#1 (CV tuning method)** and **#2 (partial GB tuning)** — both are real
methodological departures, not just unspecified details. The verification surfaced no errors
requiring a fix before report-writing.
