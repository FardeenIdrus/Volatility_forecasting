"""Generate the report figures and tables from the results CSVs."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_figures")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

TICKERS = ["AAPL", "JPM", "AMZN"]
INFOSETS = ["M_HAR", "M_ALL"]
HORIZONS_MAIN = [1, 22]
HORIZONS_ALL = [1, 5, 22]

MODELS5 = ["HAR-X", "LogHAR", "ElasticNet", "RF", "NN_2_e10"]
MODELS9 = ["HAR", "HAR-X", "LogHAR", "HARQ", "Lasso", "ElasticNet", "RF", "GB", "NN_2_e10"]
DISPLAY = {"HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "HARQ": "HARQ",
           "Lasso": "Lasso", "ElasticNet": "EN", "RF": "RF", "GB": "GB",
           "NN_2_e10": "NN$_2^{10}$"}
DISPLAY_TXT = {"HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "HARQ": "HARQ",
               "Lasso": "Lasso", "ElasticNet": "EN", "RF": "RF", "GB": "GB",
               "NN_2_e10": "NN2_10"}
FEATURE_DISPLAY = {"d_log_dvol": "dVOL", "d_US3M": "dUS3M"}

COL_MHAR = "#0072B2"   # blue  (colourblind-safe)
COL_MALL = "#E69F00"   # orange
COL_VI = "#0072B2"
Y_CAP = 2.0

# MCS cells corrupted by RF/GB heavy-tailed losses at the longer horizons.
MCS_CORRUPT = {(22, "JPM", "M_ALL"), (22, "AAPL", "M_HAR"),
               (5, "AAPL", "M_HAR"), (5, "AAPL", "M_ALL")}
MCS_BORDERLINE = {(5, "AMZN", "M_HAR")}

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


# ---------- data access ----------

def load_summary(h: int) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"summary_h{h}.csv")


def rel_mse(summary: pd.DataFrame, ticker: str, infoset: str, model: str):
    row = summary[(summary.ticker == ticker) & (summary.info_set == infoset)
                  & (summary.model == model)]
    return None if row.empty else float(row["rel_mse_to_HAR"].iloc[0])


def dm_p_beats_har(ticker: str, infoset: str, h: int, model: str):
    """One-sided DM p-value that `model` beats HAR (cell [HAR, model])."""
    pv = pd.read_csv(TABLES_DIR / f"dm_{ticker}_{infoset}_h{h}_pvalue.csv", index_col=0)
    if "HAR" not in pv.index or model not in pv.columns:
        return None
    return float(pv.loc["HAR", model])


def stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------- Figure 1 / A2: relative-MSE bar charts ----------

def _relmse_panels(horizons: list[int], hatch_map: dict, save_stem: str,
                    fig_w: float) -> None:
    """Grouped relative-MSE bar chart, one panel per stock, for the given horizons."""
    summaries = {h: load_summary(h) for h in horizons}
    combos = [(h, iset) for h in horizons for iset in INFOSETS]
    n_combo = len(combos)
    bar_w = 0.74 / n_combo

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4.0), sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        for mi, model in enumerate(MODELS5):
            present = [(h, iset, rel_mse(summaries[h], ticker, iset, model))
                       for (h, iset) in combos]
            present = [(h, iset, v) for (h, iset, v) in present if v is not None]
            k = len(present)
            # symmetric offsets centred on the model tick
            slots = np.linspace(-(k - 1) / 2, (k - 1) / 2, k)
            for (h, iset, val), slot in zip(present, slots):
                x = mi + slot * bar_w
                colour = COL_MHAR if iset == "M_HAR" else COL_MALL
                drawn = min(val, Y_CAP)
                ax.bar(x, drawn, bar_w, color=colour, hatch=hatch_map[h],
                       edgecolor="white", linewidth=0.5, zorder=3)
                if val > Y_CAP:
                    ax.annotate(f"{val:.1f}", (x, Y_CAP), textcoords="offset points",
                                xytext=(0, 1), ha="center", va="bottom",
                                fontsize=6, color="0.25")
                s = stars(dm_p_beats_har(ticker, iset, h, model))
                if s:
                    ax.text(x, drawn + 0.04, s, ha="center", va="bottom",
                            fontsize=7, color="0.1")
        ax.axhline(1.0, ls="--", color="0.35", lw=0.9, zorder=2)
        ax.set_xticks(range(len(MODELS5)))
        ax.set_xticklabels([DISPLAY[m] for m in MODELS5])
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylim(0, Y_CAP + 0.18)
        ax.set_xlim(-0.55, len(MODELS5) - 0.45)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Relative MSE vs HAR  (<1 = beats HAR)")

    handles = [mpatches.Patch(facecolor=COL_MHAR, label="M_HAR"),
               mpatches.Patch(facecolor=COL_MALL, label="M_ALL")]
    handles += [mpatches.Patch(facecolor="0.75", hatch=hatch_map[h],
                               label=f"h={h}") for h in horizons]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
    fig.text(0.5, -0.03,
             "DM stars (one-sided, HLN-corrected, model beats HAR): "
             "* p<0.10  ** p<0.05  *** p<0.01.  Bars above 2.0 are capped and value-labelled.",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{save_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", save_stem)


def figure1() -> None:
    _relmse_panels(HORIZONS_MAIN, {1: "", 22: "////"}, "fig1_relative_mse", 12.0)


def figureA2() -> None:
    """Relative-MSE small multiples: rows = stocks, columns = horizons."""
    summaries = {h: load_summary(h) for h in HORIZONS_ALL}
    hlabel = {1: "$h$ = 1  (1-day)", 5: "$h$ = 5  (1-week)",
              22: "$h$ = 22  (1-month)"}
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), sharey=True)
    bar_w = 0.30          # drawn bar width
    group_spread = 0.44   # centre-to-centre span of the 2 bars; wider than bar_w
                          # so each bar-centred star clears its neighbour
    for ri, ticker in enumerate(TICKERS):
        for ci, h in enumerate(HORIZONS_ALL):
            ax = axes[ri, ci]
            summ = summaries[h]
            for mi, model in enumerate(MODELS5):
                present = [(iset, rel_mse(summ, ticker, iset, model))
                           for iset in INFOSETS]
                present = [(iset, v) for (iset, v) in present if v is not None]
                k = len(present)
                slots = np.linspace(-(k - 1) / 2, (k - 1) / 2, k)
                for (iset, val), slot in zip(present, slots):
                    x = mi + slot * group_spread  # bar centre (== star centre)
                    colour = COL_MHAR if iset == "M_HAR" else COL_MALL
                    drawn = min(val, Y_CAP)
                    ax.bar(x, drawn, bar_w, color=colour, edgecolor="white",
                           linewidth=0.5, zorder=3)
                    if val > Y_CAP:  # value label at this bar's own x
                        ax.annotate(f"{val:.1f}", (x, Y_CAP),
                                    textcoords="offset points", xytext=(0, 1),
                                    ha="center", va="bottom", fontsize=6.5,
                                    color="0.25")
                    s = stars(dm_p_beats_har(ticker, iset, h, model))
                    if s:  # star centred on this bar's own x
                        ax.text(x, drawn + 0.05, s, ha="center", va="bottom",
                                fontsize=6.5, color="0.1")
            ax.axhline(1.0, ls="--", color="0.35", lw=0.9, zorder=2)
            ax.set_xticks(range(len(MODELS5)))
            ax.set_xticklabels([DISPLAY[m] for m in MODELS5], fontsize=8)
            ax.set_ylim(0, Y_CAP + 0.22)
            ax.set_xlim(-0.55, len(MODELS5) - 0.45)
            ax.tick_params(axis="x", length=0)
            if ri == 0:
                ax.set_title(hlabel[h], fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{ticker}\nRel. MSE vs HAR", fontsize=9,
                              fontweight="bold")
    handles = [mpatches.Patch(facecolor=COL_MHAR, label="M_HAR"),
               mpatches.Patch(facecolor=COL_MALL, label="M_ALL")]
    fig.legend(handles=handles, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=9)
    fig.text(0.5, -0.02, "DM stars (one-sided, HLN-corrected, model beats HAR): "
             "* p<0.10  ** p<0.05  *** p<0.01.  Bars above 2.0 capped, value-labelled.",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"figA2_relative_mse_3h.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figA2_relative_mse_3h.{png,pdf}")


# ---------- Table 1 / A1: relative-MSE tables ----------

def _cell(summary, ticker, infoset, h, model) -> str:
    val = rel_mse(summary, ticker, infoset, model)
    if val is None:
        return "--"
    if model == "HAR":
        return f"{val:.3f}"
    return f"{val:.3f}{stars(dm_p_beats_har(ticker, infoset, h, model))}"


def _relmse_table(models: list[int], horizons: list[int], stem: str,
                  caption_id: str, sideways: bool = False) -> pd.DataFrame:
    summaries = {h: load_summary(h) for h in horizons}
    col_keys = [(t, i, h) for t in TICKERS for i in INFOSETS for h in horizons]
    df = pd.DataFrame(index=[DISPLAY_TXT[m] for m in models],
                      columns=[f"{t}_{i}_h{h}" for (t, i, h) in col_keys])
    for m in models:
        for (t, i, h) in col_keys:
            df.loc[DISPLAY_TXT[m], f"{t}_{i}_h{h}"] = _cell(summaries[h], t, i, h, m)
    df.index.name = "model"
    df.to_csv(TABLES_DIR / f"{stem}.csv")

    nh = len(horizons)
    colspec = "l " + " ".join(["r" * nh] * (3 * len(INFOSETS)))
    hcols = " & ".join([" & ".join(f"$h$={h}" for h in horizons)] * 6)
    lines = [
        f"% {caption_id}: relative MSE vs HAR with DM significance stars.",
        r"% Stars: * p<0.10 ** p<0.05 *** p<0.01 (one-sided HLN-corrected DM, beats HAR).",
        r"% '$-$' = HAR-X equals HAR in M_HAR.",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{{nh * 2}}}{{c}}{{{t}}}" for t in TICKERS)
        + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * nh * k}-{1 + 2 * nh * (k + 1)}}}"
                for k in range(3)),
        "Model & " + " & ".join(rf"\multicolumn{{{nh}}}{{c}}{{{i.replace('_', chr(92) + '_')}}}"
                                 for _ in TICKERS for i in INFOSETS) + r" \\",
        " & " + hcols + r" \\",
        r"\midrule",
    ]
    for m in models:
        cells = [_cell(summaries[h], t, i, h, m).replace("--", r"$-$")
                 for (t, i, h) in col_keys]
        lines.append(DISPLAY[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if sideways:
        lines = ([r"% Requires \usepackage{rotating} in the LaTeX preamble.",
                  r"\begin{sidewaystable}",
                  r"  \centering",
                  r"  \footnotesize",
                  r"  % \caption{...}  \label{...}  -- caption in results/figures/captions.md"]
                 + lines + [r"\end{sidewaystable}"])
    (TABLES_DIR / f"{stem}.tex").write_text("\n".join(lines) + "\n")
    log.info("wrote %s.{csv,tex}", stem)
    return df


def table1() -> pd.DataFrame:
    return _relmse_table(MODELS5, HORIZONS_MAIN, "table1_main", "Table 1")


def tableA1() -> pd.DataFrame:
    return _relmse_table(MODELS9, HORIZONS_ALL, "tableA1_full_relmse", "Table A1",
                         sideways=True)


# ---------- Figure 2: ALE-based variable importance ----------

def figure2() -> None:
    """ALE-based VI horizontal bars: rows = stock, cols = {RF, NN}, M_ALL, h=1."""
    models = ["RF", "NN_2_e10"]
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.5))
    for ri, ticker in enumerate(TICKERS):
        for ci, model in enumerate(models):
            ax = axes[ri, ci]
            vi = pd.read_csv(TABLES_DIR / f"vi_{ticker}_{model}.csv")
            vi = vi.sort_values("vi_norm")  # ascending → largest at top of barh
            labels = [FEATURE_DISPLAY.get(f, f) for f in vi["feature"]]
            ax.barh(labels, vi["vi_norm"], color=COL_VI, edgecolor="white",
                    linewidth=0.5, zorder=3)
            ax.set_title(f"{ticker}  —  {DISPLAY[model]}", fontsize=10,
                         fontweight="bold")
            ax.set_xlim(0, 0.42)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            if ri == 2:
                ax.set_xlabel("Variable importance (normalised, sums to 1)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig2_vi.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig2_vi.{png,pdf}")


# ---------- Figure A1: RV time series ----------

def figureA1() -> None:
    """RV time series (annualised vol %) per stock with train/val/test boundary marks."""
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=False)
    for ax, ticker in zip(axes, TICKERS):
        master = pd.read_csv(FINAL_DIR / f"master_{ticker}.csv",
                             parse_dates=["date"], index_col="date").sort_index()
        ann_vol = np.sqrt(master["RV"] * 252) * 100
        ax.plot(ann_vol.index, ann_vol.values, color=COL_MHAR, lw=0.7, zorder=3)
        train_end = master.index[1554]
        val_end = master.index[1776]
        for d, lab in [(train_end, "train | val"), (val_end, "val | test")]:
            ax.axvline(d, color="0.35", ls="--", lw=1.0, zorder=2)
            ax.text(d, ax.get_ylim()[1] * 0.96, f" {lab}", fontsize=7,
                    color="0.35", va="top", ha="left")
        ax.set_title(ticker, fontsize=11, fontweight="bold")
        ax.set_ylabel("Annualised vol (%)")
        ax.margins(x=0.01)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"figA1_rv_series.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figA1_rv_series.{png,pdf}")


# ---------- Table A2: MCS inclusion ----------

def tableA2() -> None:
    """MCS inclusion (Y/N) per (stock, info-set, horizon); corrupted cells flagged."""
    long_rows = []
    for h in HORIZONS_ALL:
        mcs = pd.read_csv(TABLES_DIR / f"mcs_inclusion_h{h}.csv")
        for _, r in mcs.iterrows():
            key = (h, r["ticker"], r["info_set"])
            flag = ("corrupt" if key in MCS_CORRUPT
                    else "borderline" if key in MCS_BORDERLINE else "ok")
            long_rows.append({"horizon": h, "ticker": r["ticker"],
                              "info_set": r["info_set"], "model": r["model"],
                              "in_mcs": bool(r["in_mcs"]),
                              "mcs_pvalue": float(r["mcs_pvalue"]),
                              "mcs_reliability": flag})
    long = pd.DataFrame(long_rows)
    long.to_csv(TABLES_DIR / "tableA2_mcs.csv", index=False)

    lines = [r"% Table A2: MCS inclusion at 90%. Y = retained, N = eliminated.",
             r"% dagger = MCS unreliable (RF/GB heavy-tailed losses inflate the",
             r"% stationary-bootstrap variance); ddagger = borderline. See footnote."]
    for h in HORIZONS_ALL:
        sub = long[long.horizon == h]
        cols = [(t, i) for t in TICKERS for i in INFOSETS]
        lines += [rf"\begin{{tabular}}{{l cccccc}}", r"\toprule",
                  rf"\multicolumn{{7}}{{l}}{{\textbf{{Horizon $h={h}$}}}} \\",
                  r"\midrule",
                  "Model & " + " & ".join(
                      f"{t} {i.replace('_', chr(92) + '_')}"
                      + ("$^\\dagger$" if (h, t, i) in MCS_CORRUPT else
                         "$^\\ddagger$" if (h, t, i) in MCS_BORDERLINE else "")
                      for (t, i) in cols) + r" \\",
                  r"\midrule"]
        for m in MODELS9:
            cells = []
            for (t, i) in cols:
                row = sub[(sub.ticker == t) & (sub.info_set == i) & (sub.model == m)]
                cells.append("Y" if (not row.empty and row["in_mcs"].iloc[0])
                             else ("N" if not row.empty else "$-$"))
            lines.append(DISPLAY[m] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{1em}", ""]
    lines += [r"% $^\dagger$ MCS unreliable in this cell (heavy-tailed RF/GB losses).",
              r"% $^\ddagger$ borderline; treat with caution."]
    (TABLES_DIR / "tableA2_mcs.tex").write_text("\n".join(lines) + "\n")
    log.info("wrote tableA2_mcs.{csv,tex}")


# ---------- Table A3: regime split (LaTeX of the regime-split CSVs) ----------

def tableA3() -> None:
    """LaTeX of the regime-split CSVs; VIX tercile ranges in column headers."""
    lines = [r"% Table A3: regime-split relative MSE vs HAR (M_ALL).",
             r"% Test set split into VIX terciles. Rel MSE < 1 = beats HAR."]
    for h in HORIZONS_ALL:
        df = pd.read_csv(TABLES_DIR / f"regime_split_h{h}.csv")
        # representative VIX ranges (market-wide VIX → ~common across stocks)
        rng = {}
        for reg in ["low", "mid", "high"]:
            sub = df[df.regime == reg]
            rng[reg] = (sub["vix_lo"].min(), sub["vix_hi"].max())
        lines += [r"\begin{tabular}{ll rrr}", r"\toprule",
                  rf"\multicolumn{{5}}{{l}}{{\textbf{{Horizon $h={h}$}}}} \\",
                  r"\midrule",
                  "Stock & Model & "
                  + rf"low VIX & mid VIX & high VIX \\",
                  " & & "
                  + rf"({rng['low'][0]:.0f}--{rng['low'][1]:.0f}) & "
                  + rf"({rng['mid'][0]:.0f}--{rng['mid'][1]:.0f}) & "
                  + rf"({rng['high'][0]:.0f}--{rng['high'][1]:.0f}) \\",
                  r"\midrule"]
        for ti, ticker in enumerate(TICKERS):
            for model in MODELS5:
                sub = df[(df.ticker == ticker) & (df.model == model)]
                vals = {r["regime"]: r["rel_mse_to_HAR"] for _, r in sub.iterrows()}
                cells = " & ".join(f"{vals[reg]:.3f}"
                                   for reg in ["low", "mid", "high"])
                stock_lab = ticker if model == MODELS5[0] else ""
                lines.append(f"{stock_lab} & {DISPLAY[model]} & {cells} " + r"\\")
            if ti < 2:
                lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{1em}", ""]
    (TABLES_DIR / "tableA3_regime.tex").write_text("\n".join(lines) + "\n")
    log.info("wrote tableA3_regime.tex")


# ---------- captions ----------

def write_captions() -> None:
    text = r"""# Report figure & table captions

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
"""
    (FIGURES_DIR / "captions.md").write_text(text)
    log.info("wrote captions.md")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    figure1()
    t1 = table1()
    figure2()
    figureA1()
    figureA2()
    tA1 = tableA1()
    tableA2()
    tableA3()
    write_captions()
    print("\n" + "=" * 78)
    print("  Table 1 — relative MSE vs HAR (stars: * 10% ** 5% *** 1%, beats HAR)")
    print("=" * 78)
    print(t1.to_string())
    print("\n  Table A1 written:", len(tA1), "model rows x", len(tA1.columns), "cells")


if __name__ == "__main__":
    main()
