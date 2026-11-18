"""Build the LaTeX source of the supplementary tables added in the revision.

Purpose:  Turn the revision statistics into the supplementary tables that the
          manuscript refers to (FOOOF fit quality, peak-detection counts,
          Cohort x Condition band-power model, estimates and variance
          components of the retained mixed models, sensitivity models, gait
          parameters), so that no table is typed by hand.
Inputs:   outputs/stats/revision_2026-10/fit_quality_<date>.csv
          outputs/stats/revision_2026-10/fooof_detection_rates_<date>.csv
          outputs/stats/revision_2026-10/psd_mixed_anova_<date>.csv
          outputs/stats/revision_2026-10/lmm_submitted_fixef_<date>.csv
          outputs/stats/revision_2026-10/lmm_submitted_varcorr_<date>.csv
          outputs/stats/revision_2026-10/lmm_refit_anova_all_models_<date>.csv
          outputs/stats/revision_2026-10/lmm_random_slopes_anova_<date>.csv
          outputs/stats/revision_2026-10/gait_kinematics_<date>.csv
          outputs/stats/revision_2026-10/gait_kinematics_mocap_<mocap-date>.csv
Outputs:  outputs/tables/revision_2026-11/supplementary_tables_<date>.tex
          (+ .meta.json); the file is pasted into the Overleaf
          supplementary.tex between the markers
          "% BEGIN generated revision tables" / "% END generated revision tables"
Author:   Lena
Date:     2026-11-17
Seed:     config.yaml:repro.seed (42) via _common.setup_run (formatting only).

Usage:
  python code/analysis/revision/make_supplementary_tables.py --date 2026-11-16 --first-number 15
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PROJECT_ROOT, atomic_write, load_config, setup_run, write_sidecar  # noqa: E402

STATS = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "revision_2026-11"
BANDS = ["Theta", "Alpha", "Low Beta", "High Beta", "Gamma"]
BAND_TEX = {"Theta": "Theta", "Alpha": "Alpha", "Low Beta": "Low-beta", "High Beta": "High-beta", "Gamma": "Gamma"}
REGION_TEX = {"frontal": "Prefrontal", "motor": "Motor", "STN": "STN"}
COHORT_TEX = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}
BLOCK_TEX = {"baseline": "Rest", "pre8walk": "Walking"}


def fp(p: float) -> str:
    return "$<$ 0.001" if p < 0.001 else f"{p:.3f}"


def table(caption: str, label: str, colspec: str, header: str, rows: list[str], size: str = "\\small",
          note: str | None = None) -> str:
    body = "\n".join(rows)
    note_tex = f"\n\\par\\vspace{{2pt}}{{\\footnotesize {note}}}" if note else ""
    return (f"\\begin{{table*}}[ht]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}{note_tex}\n\\end{{table*}}\n")


def fit_quality(date: str) -> str:
    q = pd.read_csv(STATS / f"fit_quality_{date}.csv")
    q = q[(q.source == "unwarped") & (q.spectrum_variant == "A")]
    same = q.groupby(["cohort", "block", "modality"])[["r2_mean", "error_mean"]].nunique().max().max() == 1
    q = q[q.min_peak_height == 0.15]
    rows = []
    for mod, sig in (("eeg", "EEG (two ROIs)"), ("dbs", "STN-LFP")):
        for coh in ("AgeMatched_NFB", "Patient_NFB"):
            for blk in ("baseline", "pre8walk"):
                s = q[(q.modality == mod) & (q.cohort == coh) & (q.block == blk)]
                if s.empty:
                    continue
                r = s.iloc[0]
                rows.append(f"{sig} & {COHORT_TEX[coh]} & {BLOCK_TEX[blk]} & {int(r.n)} & "
                            f"{r.r2_mean:.3f} $\\pm$ {r.r2_sd:.3f} & {r.r2_min:.3f} & "
                            f"{r.error_mean:.3f} $\\pm$ {r.error_sd:.3f} \\\\")
    sens = ("The fit-quality values were identical for minimum peak heights of 0.05, 0.09 and 0.15 log units."
            if same else "Fit quality for minimum peak heights of 0.05 and 0.09 is given in the analysis outputs.")
    cap = ("\\textbf{Quality of the spectral parameterisation by cohort and condition.} Goodness of fit of the "
           "FOOOF model (4--45~Hz, fixed aperiodic mode, minimum peak height 0.15 log units) to the spectra "
           "computed without time warping. $n$, number of fitted spectra (participants $\\times$ ROIs); $R^2$ and "
           "mean absolute error (MAE, log$_{10}$ power) as mean $\\pm$ SD across spectra. " + sens)
    return table(cap, "tabR:fit_quality", "@{}lllllll@{}",
                 "\\textbf{Signal} & \\textbf{Cohort} & \\textbf{Condition} & \\textbf{$n$} & \\textbf{$R^2$} & "
                 "\\textbf{Min.\\ $R^2$} & \\textbf{MAE}", rows)


# Peak tables use one beta band (12-30 Hz): fitted beta peaks cluster around the
# 20 Hz limit between the sub-bands (fooof_peak_locations.py), so that a single
# peak near 20 Hz leaves one sub-band empty. A beta peak is present when the
# low- or the high-beta sub-band has one; its power and centre frequency are
# those of the higher of the two (= the highest fitted peak with CF in 12-30 Hz).
PEAK_BANDS = [("Alpha", "Alpha (8--12 Hz)"), ("Beta", "Beta (12--30 Hz)")]


def add_beta_peak(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    lo, hi = f["Low Beta_peak_present"].astype(bool), f["High Beta_peak_present"].astype(bool)
    f["Beta_peak_present"] = lo | hi
    use_hi = hi & (~lo | (f["High Beta_pw"].fillna(-1) > f["Low Beta_pw"].fillna(-1)))
    for k in ("cf", "pw"):
        f[f"Beta_{k}"] = f[f"Low Beta_{k}"].where(~use_hi, f[f"High Beta_{k}"])
    return f


def detection(date: str) -> str:
    f = pd.read_csv(STATS / f"fooof_unwarped_{date}.csv")
    f = add_beta_peak(f[(f.spectrum_variant == "A") & ((f.modality == "dbs") | (f.side == "Both"))])
    p = f[f.min_peak_height == 0.15]
    tests = pd.read_csv(STATS / f"fooof_peak_presence_tests_{date}.csv")  # fooof_peak_presence_tests.py
    rows = []
    regions = (("frontal", (p.modality == "eeg") & (p.region == "frontal")),
               ("motor", (p.modality == "eeg") & (p.region == "motor")), ("STN", p.modality == "dbs"))
    for reg, sel in regions:
        d = p[sel]
        for i, (band, name) in enumerate(PEAK_BANDS):
            cells = []
            for coh in ("AgeMatched_NFB", "Patient_NFB"):
                for blk in ("baseline", "pre8walk"):
                    s = d[(d.cohort == coh) & (d.block == blk)]
                    cells.append("--" if s.empty else f"{int(s[f'{band}_peak_present'].sum())}/{len(s)}")
            for blk in ("baseline", "pre8walk"):
                t = tests[(tests.region == reg) & (tests.band == band) & (tests.block == blk)]
                if t.empty:
                    cells.append("--")
                else:
                    x = t.iloc[0]
                    star = "*" if x.fisher_p_fdr <= 0.05 else ""
                    cells.append(f"{fp(x.fisher_p)} ({fp(x.fisher_p_fdr)}){star}")
            first = f"\\multirow{{2}}{{*}}{{\\textbf{{{REGION_TEX[reg]}}}}}" if i == 0 else ""
            rows.append(f"{first} & {name} & " + " & ".join(cells) + " \\\\")
        if reg != "STN":
            rows.append("\\midrule")
    tot = {h: (int(g[[f"{b}_peak_present" for b, _ in PEAK_BANDS]].to_numpy().sum()), 2 * len(g))
           for h, g in f.groupby("min_peak_height")}
    sens = "; ".join(f"{h:g}: {a}/{b}" for h, (a, b) in tot.items())
    cap = ("\\textbf{Number of participants with a detected spectral peak.} Counts of participants with a FOOOF "
           "peak in the alpha band and in the beta band, by region, cohort and condition (spectra computed without "
           "time warping; minimum peak height 0.15 log units). The beta band was treated as one band (12--30~Hz) "
           "for the peak summaries, because fitted beta peaks clustered around the 20~Hz limit between the low- "
           "and high-beta sub-bands. Peak power and centre frequency are summarised for these participants only. "
           "PD vs.\\ HC: Fisher's exact test on the proportion of participants with a peak; uncorrected $p$ and, in "
           "parentheses, $p$ after Benjamini--Hochberg correction across the eight cortical tests (two regions "
           "$\\times$ two bands $\\times$ two conditions); $^{*}p_{\\mathrm{FDR}} \\le 0.05$. "
           "Sensitivity to the minimum peak height: detected alpha and beta peaks across all regions, cohorts and "
           f"conditions were {sens} (minimum peak height: detected/possible).")
    return table(cap, "tabR:peak_detection", "@{}llccccll@{}",
                 "\\textbf{Region} & \\textbf{Band} & \\textbf{HC rest} & \\textbf{HC walking} & "
                 "\\textbf{PD rest} & \\textbf{PD walking} & \\textbf{PD vs.\\ HC, rest} & "
                 "\\textbf{PD vs.\\ HC, walking}", rows, size="\\footnotesize")


def psd_model(date: str) -> str:
    a = pd.read_csv(STATS / f"psd_mixed_anova_{date}.csv")
    a = a[a.analysis == "primary_A_adj"]
    rows = []
    for reg in ("frontal", "motor", "STN"):
        for i, band in enumerate(BANDS):
            cells = []
            for term in ("Cohort", "Condition", "Cohort:Condition"):
                s = a[(a.region == reg) & (a.band == band) & (a.term == term)]
                if s.empty:
                    cells.append("--")
                else:
                    x = s.iloc[0]
                    cells.append(f"$F_{{{int(x.df1)},{x.df2:.0f}}}$ = {x.F:.2f}, {fp(x.p)} ({fp(x.p_fdr_bands)})")
            first = f"\\multirow{{5}}{{*}}{{\\textbf{{{REGION_TEX[reg]}}}}}" if i == 0 else ""
            rows.append(f"{first} & {BAND_TEX[band]} & " + " & ".join(cells) + " \\\\")
        if reg != "STN":
            rows.append("\\midrule")
    cap = ("\\textbf{Cohort $\\times$ Condition mixed model on band power above the aperiodic fit.} Type III tests "
           "(Satterthwaite) of the model \\texttt{band power $\\sim$ Cohort * Condition + (1|Subject)} per region "
           "and band (STN, PD only: \\texttt{Condition + (1|Subject)}). Band power is the mean of the spectrum "
           "above the aperiodic fit within the band (log$_{10}$ units) and does not require a detected peak. "
           "Cells give $F$, the raw $p$ and, in parentheses, the $p$ after Benjamini--Hochberg correction across "
           "the five bands within region and term.")
    return table(cap, "tabR:psd_model", "@{}lllll@{}",
                 "\\textbf{Region} & \\textbf{Band} & \\textbf{Cohort} & \\textbf{Condition (rest vs.\\ walking)} & "
                 "\\textbf{Cohort $\\times$ Condition}", rows, size="\\footnotesize")


TERM_TEX = {
    "(Intercept)": "Intercept", "CohortPD": "Cohort (PD)", "Areamotor": "Region (motor)",
    "GaitPhaseIDS": "GaitPhase (IDS)", "GaitPhaseLLS": "GaitPhase (LLS)", "GaitPhaseRLS": "GaitPhase (RLS)",
    "CohortPD:Areamotor": "Cohort (PD) $\\times$ Region (motor)",
    "CohortPD:GaitPhaseIDS": "Cohort (PD) $\\times$ IDS", "CohortPD:GaitPhaseLLS": "Cohort (PD) $\\times$ LLS",
    "CohortPD:GaitPhaseRLS": "Cohort (PD) $\\times$ RLS",
    "HemisphereRight": "Hemisphere (right)",
    "HemisphereRight:GaitPhaseIDS": "Hemisphere (right) $\\times$ IDS",
    "HemisphereRight:GaitPhaseLLS": "Hemisphere (right) $\\times$ LLS",
    "HemisphereRight:GaitPhaseRLS": "Hemisphere (right) $\\times$ RLS",
}
VAR_TEX = {"SubjectID": "Participant intercept", "GaitCycle:SubjectID": "Gait-cycle intercept", "Residual": "Residual"}


def estimates(date: str, region: str) -> str:
    f = pd.read_csv(STATS / f"lmm_submitted_fixef_{date}.csv")
    v = pd.read_csv(STATS / f"lmm_submitted_varcorr_{date}.csv")
    f, v = f[f.region == region], v[v.region == region]
    rows = []
    for term in f[f.band == "Theta"].term:
        cells = []
        for band in BANDS:
            x = f[(f.band == band) & (f.term == term)].iloc[0]
            cells.append(f"\\makecell[l]{{{x.estimate:.2f} \\\\ {{[}}{x.lower:.2f}, {x.upper:.2f}{{]}}}}")
        rows.append(f"{TERM_TEX[term]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{@{}l}{\\textit{Variance components (dB$^2$)}} \\\\")
    for grp in ("SubjectID", "GaitCycle:SubjectID", "Residual"):
        s = v[v.grp == grp]
        if s.empty:
            continue
        rows.append(f"{VAR_TEX[grp]} & " + " & ".join(f"{s[s.band == b].iloc[0].vcov:.2f}" for b in BANDS) + " \\\\")
    if region == "cortical":
        cap = ("\\textbf{Fixed-effect estimates and variance components of the retained cortical model "
               "(baseline normalisation).} Estimates (dB) with 95\\% Wald confidence intervals from "
               "\\texttt{gERD/ERS $\\sim$ Cohort * Region + Cohort * GaitPhase + (1|Subject) + (1|GaitCycle:Subject)} "
               "(treatment coding; reference levels HC, prefrontal, FDS).")
        label = "tabR:cortical_estimates"
    else:
        cap = ("\\textbf{Fixed-effect estimates and variance components of the retained STN model "
               "(baseline normalisation).} Estimates (dB) with 95\\% Wald confidence intervals from "
               "\\texttt{gERD/ERS $\\sim$ Hemisphere * GaitPhase + (1|Subject)} "
               "(treatment coding; reference levels left STN, FDS).")
        label = "tabR:stn_estimates"
    return table(cap, label, "@{}llllll@{}",
                 "\\textbf{Term} & " + " & ".join(f"\\textbf{{{BAND_TEX[b]}}}" for b in BANDS), rows,
                 size="\\footnotesize")


def sensitivity(date: str, region: str) -> str:
    a = pd.read_csv(STATS / f"lmm_refit_anova_all_models_{date}.csv")
    a = a[a.region == region]
    if region == "cortical":
        terms = ["Cohort", "Area", "GaitPhase", "Cohort:Area", "Cohort:GaitPhase", "Area:GaitPhase",
                 "Cohort:Area:GaitPhase"]
    else:
        terms = ["Hemisphere", "GaitPhase", "Hemisphere:GaitPhase"]

    def cell(s: pd.DataFrame) -> str:
        if s.empty:
            return "--"
        x = s.iloc[0]
        return f"$F_{{{int(x.NumDF)},{x.DenDF:.0f}}}$ = {x.F:.2f}, {fp(x.p)}"

    rows = []
    for band in BANDS:
        for i, term in enumerate(terms):
            s = a[(a.band == band) & (a.term == term)]
            first = f"\\multirow{{{len(terms)}}}{{*}}{{\\textbf{{{BAND_TEX[band]}}}}}" if i == 0 else ""
            name = term.replace("Area", "Region").replace(":", " $\\times$ ")
            rows.append(f"{first} & {name} & {cell(s[s.model == 'submitted'])} & "
                        f"{cell(s[s.model == 'random_slopes'])} & {cell(s[s.model == 'aggregated_median'])} \\\\")
        if band != BANDS[-1]:
            rows.append("\\midrule")
    rs = pd.read_csv(STATS / f"lmm_random_slopes_anova_{date}.csv")
    rs = rs[rs.region == region].drop_duplicates("band")
    n_int_only = int((rs.structure.str.replace(" ", "") ==
                      ("(1|SubjectID)+(1|GaitCycle:SubjectID)" if region == "cortical" else "(1|SubjectID)")).sum())
    if region == "cortical":
        cap = ("\\textbf{Sensitivity of the cortical model effects to the random-effects structure (baseline "
               "normalisation).} Type III tests (Satterthwaite): $F$ and raw $p$. Retained model: participant and "
               "gait-cycle random intercepts, gait cycles as observations. Random slopes: fixed part "
               "Cohort $\\times$ Region $\\times$ GaitPhase with by-participant random slopes for Region, GaitPhase "
               "and their interaction (uncorrelated) plus the gait-cycle intercept; the first structure of a "
               "pre-defined sequence that converged without a singular fit was used (gamma: correlated slopes for "
               "Region and GaitPhase without their interaction). Participant level: the same fixed part on one "
               "median value per participant, region and gait phase, with a participant random intercept. "
               "--, term not in the model.")
        label = "tabR:cortical_sensitivity"
    else:
        cap = ("\\textbf{Sensitivity of the STN model effects to the random-effects structure (baseline "
               "normalisation).} As the cortical sensitivity table, for \\texttt{Hemisphere * GaitPhase}. Random "
               "slopes: by-participant random slopes for GaitPhase where the fit converged without a singular fit "
               f"(in {n_int_only} of 5 bands only the participant intercept was estimable, so that the model "
               "equals the retained model fitted by maximum likelihood). Participant level: one median value per "
               "participant and gait phase.")
        label = "tabR:stn_sensitivity"
    return table(cap, label, "@{}lllll@{}",
                 "\\textbf{Band} & \\textbf{Term} & \\textbf{Retained model} & \\textbf{Random slopes} & "
                 "\\textbf{Participant level}", rows, size="\\footnotesize")


def gait(date: str, mocap_date: str) -> str:
    m = pd.read_csv(STATS / f"gait_kinematics_mocap_{mocap_date}.csv").set_index("parameter")
    g = pd.read_csv(STATS / f"gait_kinematics_{date}.csv")
    g = g[g.cycle_set == "all"].set_index("parameter")
    spec = [
        ("walking_speed_cm_s", "Walking speed (cm/s)", 1), ("stride_length_cm", "Stride length (cm)", 1),
        ("step_length_cm", "Step length (cm)", 1), ("step_width_cm", "Step width (cm)", 1),
        ("cadence_steps_min", "Cadence (steps/min)", 1), ("gait_cycle_s", "Stride time (s)", 3),
        ("stance_s", "Stance time (s)", 3), ("swing_s", "Swing time (s)", 3),
        ("double_support_s", "Double-support time (s)", 3),
        ("step_length_asymmetry", "Step-length asymmetry (index)", 1),
        ("step_time_asymmetry", "Step-time asymmetry (index)", 1),
    ]
    rows = []
    for key, name, d in spec:
        x = m.loc[key]
        rows.append(f"{name} & {x.pd_mean:.{d}f} $\\pm$ {x.pd_sd:.{d}f} & {x.hc_mean:.{d}f} $\\pm$ {x.hc_sd:.{d}f} & "
                    f"{fp(x.welch_p)} & {x.hedges_g:.2f} \\\\")
    x = g.loc["Stride time, within-subject CV"]
    rows.append(f"Stride-time variability (CV, \\%)\\textsuperscript{{a}} & {x.PD_mean:.2f} $\\pm$ {x.PD_sd:.2f} & "
                f"{x.HC_mean:.2f} $\\pm$ {x.HC_sd:.2f} & {fp(x.welch_p)} & {x.hedges_g:.2f} \\\\")
    cap = ("\\textbf{Spatiotemporal gait parameters during overground walking.} Mean $\\pm$ SD across participants "
           "(PD $n = 10$, HC $n = 10$) from the motion-capture gait summary of the straight-walking segments; "
           "$p$ from Welch's $t$-test, $g$ Hedges' $g$ (PD $-$ HC). \\textsuperscript{a}Coefficient of variation "
           "of stride time across the segmented gait cycles of each participant, from the gait events used for "
           "the neural analyses.")
    return table(cap, "tabR:gait", "@{}lllll@{}",
                 "\\textbf{Parameter} & \\textbf{PD} & \\textbf{HC} & \\textbf{$p$} & \\textbf{$g$}", rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="date tag of the statistics, e.g. 2026-11-16")
    ap.add_argument("--mocap-date", default="2026-10-21")
    ap.add_argument("--first-number", type=int, default=15,
                    help="number of the first generated table (for the comment lines only)")
    args = ap.parse_args()
    meta = setup_run("make_supplementary_tables", load_config())
    print(json.dumps({"seed": meta["seed"], "pandas": importlib.metadata.version("pandas")}))

    blocks = [
        ("FOOOF fit quality", fit_quality(args.date)),
        ("Peak detection", detection(args.date)),
        ("Cohort x Condition band-power model", psd_model(args.date)),
        ("Retained cortical model: estimates and variances", estimates(args.date, "cortical")),
        ("Retained STN model: estimates and variances", estimates(args.date, "STN")),
        ("Cortical sensitivity models", sensitivity(args.date, "cortical")),
        ("STN sensitivity models", sensitivity(args.date, "STN")),
        ("Gait parameters", gait(args.date, args.mocap_date)),
    ]
    parts = ["% BEGIN generated revision tables (code/analysis/revision/make_supplementary_tables.py, "
             f"statistics of {args.date}); do not edit by hand"]
    for i, (name, tex) in enumerate(blocks):
        parts.append(f"% --- Table S{args.first_number + i}: {name}\n{tex}")
    parts.append("% END generated revision tables")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"supplementary_tables_{args.date}.tex"
    with atomic_write(out) as tmp:
        Path(tmp).write_text("\n".join(parts) + "\n")
    meta.update({"author": "Lena", "date": args.date, "mocap_date": args.mocap_date, "output": str(out),
                 "tables": {f"S{args.first_number + i}": n for i, (n, _) in enumerate(blocks)}})
    write_sidecar(out, meta)
    print(f"wrote {out}")
    for i, (n, _) in enumerate(blocks):
        print(f"  S{args.first_number + i}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Tabulars of the existing supplementary tables that carry the retained
# cortical model (post-hoc contrasts, baseline normalisation; Type III table
# and post-hoc contrasts, average-gait-cycle normalisation). They are parsed
# from the text output of the submitted R script
# (code/analysis/R/stats_lmm_cortical_complex_final.R -> outputs/stats/
# lmm_cortical_<date>.txt), so that the values are the ones the retained
# script itself prints (Kenward-Roger df, HC - PD, uncorrected p).
# Use: python make_supplementary_tables.py --date <d> --legacy-tabulars
# ---------------------------------------------------------------------------
import re  # noqa: E402

LEVEL_TEX = {"frontal": "Prefrontal Cortex", "motor": "Motor Cortex"}
BAND_TEX_CAP = {"Theta": "Theta", "Alpha": "Alpha", "Low Beta": "Low-Beta", "High Beta": "High-Beta",
                "Gamma": "Gamma"}


def _stars(p: float) -> str:
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "°" if p <= 0.1 else ""


def parse_lmm_txt(path: Path) -> dict:
    """{datatype: {band: {"anova": {term: (df1, df2, F, p)}, "posthoc": {family: {level: (df, t, p)}}}}}"""
    out: dict = {}
    dt = band = fam = level = None
    for line in path.read_text().splitlines():
        m = re.match(r"--- Analyzing Frequency Band: (.+) for (\w+) data ---", line)
        if m:
            band, dt = m.group(1), m.group(2)
            out.setdefault(dt, {})[band] = {"anova": {}, "posthoc": {"Area": {}, "GaitPhase": {}}}
            continue
        if band is None:
            continue
        m = re.match(r"^(Cohort:GaitPhase|Cohort:Area|Cohort|Area|GaitPhase)\s+[\d.eE+-]+\s+[\d.eE+-]+\s+(\d+)\s+"
                     r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+(<\s*)?([\d.eE+-]+)", line)
        if m and m.group(1) not in out[dt][band]["anova"]:
            p = float(m.group(6))
            out[dt][band]["anova"][m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)),
                                                  0.0 if m.group(5) else p)
            continue
        m = re.match(r"^(Area|GaitPhase) = (\w+):", line)
        if m:
            fam, level = m.group(1), m.group(2)
            continue
        m = re.match(r"^\s*HC - PD\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(<)?\s*([\d.]+)", line)
        if m and fam:
            out[dt][band]["posthoc"][fam][level] = (float(m.group(3)), float(m.group(4)),
                                                    0.0 if m.group(5) else float(m.group(6)))
    return out


def _p4(p: float) -> str:
    return "$<$0.0001" + _stars(p) if p < 0.0001 else f"{p:.4f}{_stars(p)}"


def posthoc_tabular(res: dict, datatype: str) -> str:
    rows = ["\\begin{tabular}{@{}ll rrr@{}}", "\\toprule",
            "\\textbf{Frequency Band} & \\textbf{Interaction Level} & \\textbf{df} & \\textbf{$t$-ratio} & "
            "\\textbf{$p$-value} \\\\", "\\midrule"]
    for bi, band in enumerate(BANDS):
        ph = res[datatype][band]["posthoc"]
        n = (1 + len(ph["Area"]) if ph["Area"] else 1) + 1 + len(ph["GaitPhase"])
        first = f"\\multirow{{{n}}}{{*}}{{\\textbf{{{BAND_TEX_CAP[band]}}}}}"
        if ph["Area"]:
            rows.append(f"{first} & \\textit{{Cohort $\\times$ Region}} & & & \\\\")
            for lv in ("frontal", "motor"):
                df, t, p = ph["Area"][lv]
                rows.append(f"& \\quad {LEVEL_TEX[lv]} & {df:.1f} & {t:.3f} & {_p4(p)} \\\\")
        else:
            rows.append(f"{first} & \\textit{{Cohort $\\times$ Region}} & "
                        "\\multicolumn{3}{l}{Not performed (ANOVA $\\textit{p} > 0.05$)} \\\\")
        rows.append("\\cmidrule(l){2-5}")
        if ph["GaitPhase"]:
            rows.append("& \\textit{Cohort $\\times$ GaitPhase} & & & \\\\")
            for lv in ("FDS", "IDS", "LLS", "RLS"):
                df, t, p = ph["GaitPhase"][lv]
                rows.append(f"& \\quad {lv} & {df:.1f} & {t:.3f} & {_p4(p)} \\\\")
        else:
            rows.append("& \\textit{Cohort $\\times$ GaitPhase} & "
                        "\\multicolumn{3}{l}{Not performed (ANOVA $\\textit{p} > 0.05$)} \\\\")
        rows.append("\\midrule" if bi < len(BANDS) - 1 else "\\bottomrule")
    rows.append("\\multicolumn{5}{l}{\\textit{Note:} ° indicates $0.05 < p \\le 0.1$; * indicates $p \\le 0.05$; "
                "** indicates $p \\le 0.01$; *** indicates $p \\le 0.001$.}")
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def anova_tabular(res: dict, datatype: str) -> str:
    names = [("Cohort", "Cohort"), ("Area", "Region"), ("GaitPhase", "GaitPhase"),
             ("Cohort:Area", "\\makecell[c]{Cohort \\\\ $\\times$ \\\\ Region}"),
             ("Cohort:GaitPhase", "\\makecell[c]{Cohort \\\\ $\\times$ \\\\ GaitPhase}")]
    rows = ["\\begin{tabular}{@{}l*{5}{l}@{}}", "\\toprule",
            "\\textbf{Variable} & \\textbf{Theta} & \\textbf{Alpha} & \\textbf{Low-Beta} & \\textbf{High-Beta} & "
            "\\textbf{Gamma} \\\\", "\\midrule", "\\addlinespace"]
    for term, name in names:
        cells = []
        for band in BANDS:
            df1, df2, F, p = res[datatype][band]["anova"][term]
            df2s = f"{df2:.0f}" if df2 > 100 else f"{df2:.1f}"
            ps = "p $<$ 0.001***" if p < 0.001 else f"p = {p:.3f}{_stars(p)}"
            cells.append(f"\\makecell[l]{{df = {df1}, {df2s} \\\\ F = {F:.2f} \\\\ {ps}}}")
        rows.append(f"{name} &\n" + " &\n".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows += ["\\bottomrule",
             "\\multicolumn{6}{l}{\\textit{Note:} ° indicates $0.05 < p \\le 0.1$; * indicates $p \\le 0.05$; "
             "** indicates $p \\le 0.01$; *** indicates $p \\le 0.001$.}", "\\end{tabular}"]
    return "\n".join(rows)


def legacy_tabulars(date: str) -> dict:
    res = parse_lmm_txt(PROJECT_ROOT / "outputs" / "stats" / f"lmm_cortical_{date}.txt")
    return {"tab5:basenorm_posthoc": posthoc_tabular(res, "basenorm"),
            "tab10:avggait_model_effects": anova_tabular(res, "avggait"),
            "tab11:avggait_posthoc": posthoc_tabular(res, "avggait")}


# ---------------------------------------------------------------------------
# Spectral section: main-text table of the band-power contrasts and two more
# supplementary tables (peak descriptives for detected peaks; aperiodic model).
# Use: python make_supplementary_tables.py --date <d> --spectral
# ---------------------------------------------------------------------------
def band_power_contrasts(date: str) -> str:
    """Main-text table: PD vs HC within condition and rest vs walking within cohort."""
    c = pd.read_csv(STATS / f"psd_mixed_anova_contrasts_{date}.csv")
    c = c[(c.analysis == "primary_A_adj") & (c.modality == "eeg")]

    def cell(s: pd.DataFrame, flip: bool) -> str:
        x = s.iloc[0]
        est, lo, hi = (-x.estimate, -x.upper, -x.lower) if flip else (x.estimate, x.lower, x.upper)
        st = "*" if x.p_fdr <= 0.05 else ""
        return f"\\makecell[l]{{{est:.2f} {{[}}{lo:.2f}, {hi:.2f}{{]}} \\\\ $p$ = {fp(x.p)} \\\\ $p_{{\\mathrm{{FDR}}}}$ = {fp(x.p_fdr)}{st}}}"

    rows = []
    for reg in ("frontal", "motor"):
        for i, band in enumerate(BANDS):
            s = c[(c.region == reg) & (c.band == band)]
            a = s[s.family == "PD_vs_HC_within_condition"]
            b = s[s.family == "rest_vs_walking_within_cohort"]
            cells = [cell(a[a.by == "rest"], True), cell(a[a.by == "walking"], True),
                     cell(b[b.by == "PD"], False), cell(b[b.by == "HC"], False)]
            first = f"\\multirow{{5}}{{*}}{{\\textbf{{{REGION_TEX[reg]}}}}}" if i == 0 else ""
            rows.append(f"{first} & {BAND_TEX[band]} & " + " & ".join(cells) + " \\\\")
            if i < 4:
                rows.append("\\addlinespace")
        if reg == "frontal":
            rows.append("\\midrule")
    return "\n".join(rows)


def peak_descriptives(date: str) -> str:
    f = pd.read_csv(STATS / f"fooof_unwarped_{date}.csv")
    f = add_beta_peak(f[(f.spectrum_variant == "A") & (f.min_peak_height == 0.15) & ((f.modality == "dbs") | (f.side == "Both"))])
    rows = []
    for reg, sel in (("frontal", (f.modality == "eeg") & (f.region == "frontal")),
                     ("motor", (f.modality == "eeg") & (f.region == "motor")), ("STN", f.modality == "dbs")):
        d = f[sel]
        for i, (band, band_name) in enumerate(PEAK_BANDS):
            cells = []
            for coh in ("AgeMatched_NFB", "Patient_NFB"):
                for blk in ("baseline", "pre8walk"):
                    s = d[(d.cohort == coh) & (d.block == blk)]
                    if s.empty:
                        cells.append("--")
                        continue
                    k = s[s[f"{band}_peak_present"].astype(bool)]
                    if len(k) == 0:
                        cells.append("\\makecell[l]{$n$ = 0}")
                    else:
                        sd_pw = f" $\\pm$ {k[f'{band}_pw'].std(ddof=1):.2f}" if len(k) > 1 else ""
                        sd_cf = f" $\\pm$ {k[f'{band}_cf'].std(ddof=1):.1f}" if len(k) > 1 else ""
                        cells.append(f"\\makecell[l]{{$n$ = {len(k)} \\\\ {k[f'{band}_pw'].mean():.2f}{sd_pw} \\\\ "
                                     f"{k[f'{band}_cf'].mean():.1f}{sd_cf} Hz}}")
            first = f"\\multirow{{2}}{{*}}{{\\textbf{{{REGION_TEX[reg]}}}}}" if i == 0 else ""
            rows.append(f"{first} & {band_name} & " + " & ".join(cells) + " \\\\")
            if i < 1:
                rows.append("\\addlinespace")
        if reg != "STN":
            rows.append("\\midrule")
    cap = ("\\textbf{Peak power and centre frequency of detected spectral peaks.} Descriptive summary for the "
           "participants with a detected FOOOF peak in the alpha band and in the beta band (12--30~Hz, treated as one band; spectra computed without time warping; minimum "
           "peak height 0.15 log units): number of participants with a peak ($n$), peak power above the aperiodic "
           "fit (log$_{10}$ units; mean $\\pm$ SD) and centre frequency (mean $\\pm$ SD). Observations without a "
           "detected peak are omitted, so that these values are not compared statistically between cohorts or "
           "conditions.")
    return table(cap, "tabR:peak_descriptives", "@{}llllll@{}",
                 "\\textbf{Region} & \\textbf{Band} & \\textbf{HC rest} & \\textbf{HC walking} & \\textbf{PD rest} & "
                 "\\textbf{PD walking}", rows, size="\\footnotesize")


def aperiodic(date: str) -> str:
    a = pd.read_csv(STATS / f"aperiodic_mixed_anova_{date}.csv")
    d = pd.read_csv(STATS / f"aperiodic_mixed_anova_desc_{date}.csv")
    rows = []
    for reg in ("frontal", "motor", "STN"):
        for i, par in enumerate(("exponent", "offset")):
            means = []
            for coh in ("HC", "PD"):
                for cond in ("rest", "walking"):
                    s = d[(d.region == reg) & (d.parameter == par) & (d.Cohort == coh) & (d.Condition == cond)]
                    means.append("--" if s.empty else f"{s.iloc[0]['mean']:.2f} $\\pm$ {s.iloc[0]['sd']:.2f}")
            tests = []
            for term in ("Cohort", "Condition", "Cohort:Condition"):
                s = a[(a.region == reg) & (a.parameter == par) & (a.term == term)]
                tests.append("--" if s.empty else f"{s.iloc[0].F:.2f}, {fp(s.iloc[0].p)}")
            first = f"\\multirow{{2}}{{*}}{{\\textbf{{{REGION_TEX[reg]}}}}}" if i == 0 else ""
            rows.append(f"{first} & {par.capitalize()} & " + " & ".join(means + tests) + " \\\\")
        if reg != "STN":
            rows.append("\\midrule")
    cap = ("\\textbf{Aperiodic spectral parameters do not differ between cohorts or between rest and walking.} "
           "FOOOF aperiodic exponent and offset (log$_{10}$ units; mean $\\pm$ SD) from the spectra computed without "
           "time warping, and Type III tests ($F$, $p$) of \\texttt{parameter $\\sim$ Cohort * Condition + "
           "(1|Subject)} per cortical region ($F_{1,18}$) and of \\texttt{parameter $\\sim$ Condition + "
           "(1|Subject)} for the STN ($F_{1,9}$; PD only).")
    return table(cap, "tabR:aperiodic", "@{}lllllllll@{}",
                 "\\textbf{Region} & \\textbf{Parameter} & \\textbf{HC rest} & \\textbf{HC walking} & \\textbf{PD rest} "
                 "& \\textbf{PD walking} & \\textbf{Cohort} & \\textbf{Condition} & \\textbf{Cohort $\\times$ Cond.}",
                 rows, size="\\footnotesize")


def equivalence(date: str) -> str:
    """Equivalence tests of the phase-specific cohort differences (null_strength.R, section 1)."""
    a = pd.read_csv(STATS / f"null_strength_{date}.csv")
    t = a[(a.block == "tost_Cohort_x_GaitPhase") & (a.summary == "median")]
    rows = []
    for bi, band in enumerate(BANDS):
        for i, ph in enumerate(("IDS", "LLS", "FDS", "RLS")):
            x = t[(t.band == band) & (t.level == ph)].iloc[0]
            first = f"\\multirow{{4}}{{*}}{{\\textbf{{{BAND_TEX[band]}}}}}" if i == 0 else ""
            y1 = "yes" if bool(x.equiv_1p0) else "no"
            y5 = "yes" if bool(x.equiv_0p5) else "no"
            rows.append(f"{first} & {ph} & {x.estimate:.2f} {{[}}{x.ci90_lo:.2f}, {x.ci90_hi:.2f}{{]}} & "
                        f"{fp(x.tost_p_1p0)} & {y1} & {fp(x.tost_p_0p5)} & {y5} \\\\")
        if bi < len(BANDS) - 1:
            rows.append("\\midrule")
    cap = ("\\textbf{Equivalence tests of the phase-specific cohort differences in cortical gait-phase modulation.} "
           "Per participant and gait phase, the median baseline-normalised gERD/ERS was averaged across the two "
           "regions and centred on the participant's mean over the four gait phases, which removes the overall "
           "cohort offset. Difference, PD $-$ HC of the centred value (dB) with its 90\\% confidence interval "
           "(Welch); $p_{\\mathrm{TOST}}$, two one-sided tests against bounds of $\\pm$1.0 and $\\pm$0.5~dB; "
           "equivalent, 90\\% confidence interval within the bounds. The bounds correspond to about 40\\% and 20\\% "
           "of the gamma cohort offset and were not derived from an independent benchmark.")
    return table(cap, "tabR:equivalence", "@{}lllllll@{}",
                 "\\textbf{Band} & \\textbf{Phase} & \\textbf{Difference {[}90\\% CI{]}} & "
                 "\\textbf{$p_{\\mathrm{TOST}}$ ($\\pm$1.0)} & \\textbf{Equivalent} & "
                 "\\textbf{$p_{\\mathrm{TOST}}$ ($\\pm$0.5)} & \\textbf{Equivalent}", rows, size="\\footnotesize")


# ---------------------------------------------------------------------------
# Post-hoc contrasts of the retained models with the family-wise correction
# (lmm_submitted_posthoc_families.R). Replaces the tabulars parsed from the
# submitted script's text output (legacy_tabulars) for the post-hoc tables.
# ---------------------------------------------------------------------------
def posthoc_family_table(date: str, suffix: str, region: str, label: str, caption: str,
                         phase_map: dict | None = None) -> str:
    c = pd.read_csv(STATS / f"lmm_submitted_posthoc_{date}{suffix}.csv")
    c = c[c.region == region]
    phase_map = phase_map or {}

    def row(x, name: str) -> str:
        st = "*" if x.p_fdr_family <= 0.05 else ""
        return (f"& \\quad {name} & {x.estimate:.2f} {{[}}{x['lower.CL']:.2f}, {x['upper.CL']:.2f}{{]}} & "
                f"{x['t.ratio']:.2f} ({x.df:.1f}) & {fp(x.p_raw)} & {fp(x.p_fdr_family)}{st} \\\\")

    rows = []
    for bi, band in enumerate(BANDS):
        b = c[c.band == band]
        blocks = []
        if region == "cortical":
            a = b[b.family == "Cohort:Area"]
            blocks.append(("\\textit{Cohort $\\times$ Region}",
                           [(a[a.level == lv].iloc[0], LEVEL_TEX[lv]) for lv in ("frontal", "motor")]))
            g = b[b.family == "Cohort:GaitPhase"]
            blocks.append(("\\textit{Cohort $\\times$ GaitPhase}",
                           [(g[g.level == lv].iloc[0], lv) for lv in ("IDS", "LLS", "FDS", "RLS")]))
        else:
            g = b[b.family == "Hemisphere:GaitPhase"]
            blocks.append(("\\textit{Hemisphere $\\times$ GaitPhase}",
                           [(g[g.level == lv].iloc[0], phase_map.get(lv, lv)) for lv in ("IDS", "LLS", "FDS", "RLS")]))
        n = sum(1 + len(items) for _, items in blocks)
        for k, (title, items) in enumerate(blocks):
            first = f"\\multirow{{{n}}}{{*}}{{\\textbf{{{BAND_TEX_CAP[band]}}}}}" if k == 0 else ""
            if k:
                rows.append("\\cmidrule(l){2-6}")
            rows.append(f"{first} & {title} & & & & \\\\")
            rows += [row(x, name) for x, name in items]
        if bi < len(BANDS) - 1:
            rows.append("\\midrule")
    header = ("\\textbf{Frequency Band} & \\textbf{Level} & \\textbf{Estimate {[}95\\% CI{]} (dB)} & "
              "\\textbf{$t$ (df)} & \\textbf{$p$} & \\textbf{$p_{\\mathrm{FDR}}$}")
    return table(caption, label, "@{}llllll@{}", header, rows, size="\\footnotesize")


FAMILY_NOTE_C = ("$p$, uncorrected; $p_{\\mathrm{FDR}}$, Benjamini--Hochberg correction across the ten per-region "
                 "contrasts (five bands $\\times$ two regions) and, separately, across the 20 per-gait-phase contrasts "
                 "(five bands $\\times$ four phases). $^{*}p_{\\mathrm{FDR}} \\le 0.05$.")
FAMILY_NOTE_S = ("$p$, uncorrected; $p_{\\mathrm{FDR}}$, Benjamini--Hochberg correction across the 20 contrasts (five "
                 "bands $\\times$ four gait phases). $^{*}p_{\\mathrm{FDR}} \\le 0.05$.")
EMM = "(estimated marginal means, Satterthwaite degrees of freedom)"


def posthoc_family_tables(date: str) -> dict:
    ic = {"LLS": "CLS", "RLS": "ILS"}
    return {
        "tab5:basenorm_posthoc": posthoc_family_table(
            date, "", "cortical", "tab5:basenorm_posthoc",
            "\\textbf{Post-hoc contrasts of the cortical baseline-normalised model: after correction, the PD-vs-HC "
            "difference is significant in prefrontal gamma.} PD $-$ HC contrasts per region and per gait phase from "
            f"the LMM in Table~2 of the main text {EMM}. " + FAMILY_NOTE_C),
        "tab11:avggait_posthoc": posthoc_family_table(
            date, "_avggait", "cortical", "tab11:avggait_posthoc",
            "\\textbf{Post-hoc contrasts of the cortical average-gait-cycle-normalised model: PD-vs-HC differences "
            "during the double-support phases (IDS, FDS) in theta, alpha and low-beta survive correction.} PD $-$ HC "
            f"contrasts per region and per gait phase from the LMM in Table~\\ref{{tab10:avggait_model_effects}} {EMM}. "
            + FAMILY_NOTE_C),
        "tab7:stn_basenorm_posthoc": posthoc_family_table(
            date, "", "STN", "tab7:stn_basenorm_posthoc",
            "\\textbf{Post-hoc contrasts of the STN baseline-normalised model: no left-vs-right contrast survives "
            "correction in any gait phase or band.} Left $-$ Right contrasts per gait phase from the LMM in Table~3 "
            f"of the main text {EMM}. " + FAMILY_NOTE_S),
        "tab13:stn_avggait_posthoc": posthoc_family_table(
            date, "_avggait", "STN", "tab13:stn_avggait_posthoc",
            "\\textbf{Post-hoc contrasts of the STN average-gait-cycle-normalised model: left-vs-right contrasts in "
            "theta and gamma are significant only before correction.} Left $-$ Right contrasts per gait phase from "
            f"the LMM in Table~\\ref{{tab12:avggait_posthoc}} {EMM}. " + FAMILY_NOTE_S),
        "tab15:stn_ipsicontra_basenorm_posthoc": posthoc_family_table(
            date, "_ipsicontra", "STN", "tab15:stn_ipsicontra_basenorm_posthoc",
            "\\textbf{Post-hoc contrasts of the ipsi/contra-aligned STN model (baseline-normalised): no left-vs-right "
            "contrast survives correction in any gait phase or band.} Left $-$ Right contrasts per gait phase from "
            f"the LMM in Table~\\ref{{tab14:stn_ipsicontra_basenorm_LMM}} {EMM}. " + FAMILY_NOTE_S, ic),
        "tab17:stn_ipsicontra_avggait_posthoc": posthoc_family_table(
            date, "_avggait_ipsicontra", "STN", "tab17:stn_ipsicontra_avggait_posthoc",
            "\\textbf{Post-hoc contrasts of the ipsi/contra-aligned STN model (average-gait-cycle-normalised): "
            "left-vs-right contrasts in theta (FDS) and gamma (IDS) are significant only before correction.} "
            "Left $-$ Right contrasts per gait phase from the LMM in "
            f"Table~\\ref{{tab16:stn_ipsicontra_avggait_LMM}} {EMM}. " + FAMILY_NOTE_S, ic),
    }
