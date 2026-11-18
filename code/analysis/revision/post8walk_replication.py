"""Cross-block replication of the revision results: pre8walk (manuscript) vs post8walk.

Purpose:  Compare, subject by subject, the two walking blocks of the same
          session: (a) the walking-vs-rest offset (cycle mean of the
          baseline-normalised gERD, dB) per area and band, with emphasis on
          high beta and gamma; (b) the similarity of each subject's
          within-cycle modulation pattern between blocks (4-phase avggait
          pattern and the full time-resolved band time course); (c) a
          side-by-side table of the cohort-level RM-ANOVA terms (GaitPhase,
          Cohort, Cohort x GaitPhase) and of the cluster tests in both
          blocks. Also records which subjects are present per modality and
          block.
Inputs:   data/derivatives/gerd/<subj>_{pre8walk,post8walk}-{eeg,dbs}-{basenorm,avggait}-gerd.csv
              (compute_gerd_table.py; per cycle x area x hemisphere x phase)
          data/derivatives/tfr_gavg/{pre8walk,post8walk}-{eeg,dbs}-tfr-gavg[-avggait].nc
              (aggregate_tfr_gavg.py; per-subject regional TFRs, per_subject)
          outputs/stats/revision_2026-10/rm_anova_classic_2026-10-14.csv          (pre8walk)
          outputs/stats/revision_2026-10/rm_anova_post8walk_2026-10-14.csv        (post8walk)
          outputs/stats/revision_2026-10/rm_anova_classic_contrasts_2026-10-14.csv
          outputs/stats/revision_2026-10/rm_anova_post8walk_contrasts_2026-10-14.csv
          outputs/stats/revision_2026-10/cluster_perm_2026-10-14.csv (pre8walk re-run; falls back to cluster_perm_2026-10-13.csv)
          outputs/stats/revision_2026-10/cluster_perm_post8walk_2026-10-14.csv    (post8walk)
          code/analysis/config.yaml (freq_bands, cohorts, repro.seed)
Outputs:  outputs/stats/revision_2026-10/post8walk_subjects_2026-10-14.csv
          outputs/stats/revision_2026-10/cross_block_offsets_per_subject_2026-10-14.csv
          outputs/stats/revision_2026-10/cross_block_offsets_2026-10-14.csv        (summary per area x band)
          outputs/stats/revision_2026-10/cross_block_pattern_similarity_per_subject_2026-10-14.csv
          outputs/stats/revision_2026-10/cross_block_pattern_similarity_2026-10-14.csv
          outputs/stats/revision_2026-10/cross_block_cohort_table_2026-10-14.csv
          outputs/stats/revision_2026-10/cross_block_cluster_table_2026-10-14.csv
          outputs/stats/revision_2026-10/post8walk_replication_2026-10-14.txt
          outputs/figures/revision_2026-10/cross_block_offsets.png
          outputs/figures/revision_2026-10/cross_block_pattern_similarity.png
Author:   Lena
Date:     2026-10-14
Seed:     42 (numpy; the analysis is deterministic, the seed is set for consistency)

Description:
  Offset: for every subject, block, area (motor, frontal: Hemisphere "Both";
  STN for PD) and band, the mean across kept cycles and the four gait phases
  of the per-cycle basenorm medians (dB vs rest). Across subjects (both
  cohorts pooled, and per cohort) the two blocks are compared with Pearson
  r, Spearman rho, the paired difference post - pre (mean, 95 % CI, paired
  t, Wilcoxon signed-rank) and the between-cohort difference per block.
  Pattern similarity: (i) Pearson r across the four phase medians (IDS,
  LLS, FDS, RLS) of the avggait table, (ii) Pearson r across the 1100
  (EEG) / 275 (STN) warped time samples of the band-averaged avggait
  per-subject TFR. Cohort comparison of the Fisher-z similarities with a
  Welch t-test. All tests are two-sided; no correction for multiple
  comparisons is applied (descriptive replication summary).
  HC scaling: the HC post8walk proc fifs store microvolts as volts (1e6 x
  amplitude) while the baseline block is in volts, so HC post8walk basenorm
  values carry a constant +120 dB (10*log10(1e12)); HC_CORRECTION_DB is
  subtracted from HC post8walk basenorm values (gERD tables) before use.
  avggait values are unaffected.

Usage:
  python code/analysis/revision/post8walk_replication.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

DATE = "2026-10-14"
BLOCKS = ("pre8walk", "post8walk")
PHASES = ("IDS", "LLS", "FDS", "RLS")
BANDS = ("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
AREAS = ("motor", "frontal", "STN")
OUT = PROJECT_ROOT / "outputs/stats/revision_2026-10"
FIG = PROJECT_ROOT / "outputs/figures/revision_2026-10"
HC_CORRECTION_DB = 120.0   # HC post8walk basenorm: microvolt-as-volt export, see docstring
HC_SUBJECTS: set[str] = set()


def _read_gerd(gerd_dir: Path, subj: str, block: str, modality: str, norm: str) -> pd.DataFrame | None:
    p = gerd_dir / f"{subj}_{block}-{modality}-{norm}-gerd.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if modality == "eeg":
        d = d[d["Hemisphere"] == "Both"].copy()
    if block == "post8walk" and norm == "basenorm" and modality == "eeg" and subj in HC_SUBJECTS:
        cols = [f"ERD_ERS_{b}" for b in BANDS]
        d[cols] = d[cols] - HC_CORRECTION_DB
    return d


def _fisher(r: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return np.arctanh(r)


def _paired_summary(pre: np.ndarray, post: np.ndarray) -> dict:
    ok = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[ok], post[ok]
    n = int(ok.sum())
    out = dict(n=n)
    if n < 3:
        return out
    d = post - pre
    r, p_r = stats.pearsonr(pre, post)
    rho, p_rho = stats.spearmanr(pre, post)
    t, p_t = stats.ttest_rel(post, pre)
    try:
        w, p_w = stats.wilcoxon(d)
    except ValueError:
        w, p_w = np.nan, np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    out.update(mean_pre=pre.mean(), mean_post=post.mean(), mean_diff=d.mean(),
               ci_lo=d.mean() - ci, ci_hi=d.mean() + ci, sd_diff=d.std(ddof=1),
               pearson_r=r, pearson_p=p_r, spearman_rho=rho, spearman_p=p_rho,
               paired_t=t, paired_t_p=p_t, wilcoxon_W=w, wilcoxon_p=p_w)
    return out


def main() -> int:
    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("post8walk_replication", config)
    seed = int(config["repro"]["seed"])
    np.random.seed(seed)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    gerd_dir = paths.derivatives_root / "gerd"
    cohorts = config["cohorts"]
    group_of = {s: ("PD" if c == "Patient_NFB" else "HC") for c, ss in cohorts.items() for s in ss}
    subjects = [s for c in cohorts for s in cohorts[c]]
    HC_SUBJECTS.update(cohorts["AgeMatched_NFB"])
    lines = [f"post8walk replication summary, {DATE}, seed {seed}",
             f"HC post8walk basenorm values corrected by -{HC_CORRECTION_DB:g} dB (microvolt-as-volt export)", ""]

    # ---- 1. subject inventory per block x modality ----
    inv_rows = []
    for s in subjects:
        row = dict(SubjectID=s, Group=group_of[s])
        for b in BLOCKS:
            for m in ("eeg", "dbs"):
                d = _read_gerd(gerd_dir, s, b, m, "basenorm")
                row[f"{b}_{m}_cycles"] = int(d["Epoch"].nunique()) if d is not None else 0
        inv_rows.append(row)
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(OUT / f"post8walk_subjects_{DATE}.csv", index=False)
    lines.append("Subjects with per-cycle gERD tables (kept cycles):")
    for b in BLOCKS:
        for m in ("eeg", "dbs"):
            col = f"{b}_{m}_cycles"
            present = inv[inv[col] > 0]
            lines.append(f"  {b:9s} {m}: n={len(present)} (PD {int((present.Group == 'PD').sum())}, "
                         f"HC {int((present.Group == 'HC').sum())}); cycles median {present[col].median():.0f} "
                         f"[{present[col].min()}-{present[col].max()}]; missing: "
                         f"{', '.join(inv[inv[col] == 0].SubjectID) or 'none'}")
    lines.append("")

    # ---- 2. per-subject offsets (basenorm cycle mean) and 4-phase avggait patterns ----
    off_rows, pat4 = [], {}
    for s in subjects:
        for b in BLOCKS:
            for m, areas in (("eeg", ("motor", "frontal")), ("dbs", ("STN",))):
                d_bn = _read_gerd(gerd_dir, s, b, m, "basenorm")
                d_ag = _read_gerd(gerd_dir, s, b, m, "avggait")
                if d_bn is None:
                    continue
                for a in areas:
                    da = d_bn[d_bn["Area"] == a]
                    if da.empty:
                        continue
                    row = dict(SubjectID=s, Group=group_of[s], block=b, Area=a, n_cycles=int(da["Epoch"].nunique()))
                    for band in BANDS:
                        row[band] = float(da[f"ERD_ERS_{band}"].mean())          # mean over cycles and phases
                    off_rows.append(row)
                    if d_ag is not None:
                        dg = d_ag[d_ag["Area"] == a]
                        for band in BANDS:
                            pat4[(s, b, a, band)] = np.array(
                                [dg.loc[dg["GaitPhase"] == ph, f"ERD_ERS_{band}"].median() for ph in PHASES])
    off = pd.DataFrame(off_rows)
    wide = off.pivot_table(index=["SubjectID", "Group", "Area"], columns="block", values=list(BANDS))
    wide.columns = [f"{band}_{b}" for band, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / f"cross_block_offsets_per_subject_{DATE}.csv", index=False)

    summ_rows = []
    lines.append("Walking-vs-rest offset (cycle mean of basenorm gERD, dB): post8walk vs pre8walk per subject")
    for a in AREAS:
        for band in BANDS:
            for grp in ("all", "PD", "HC"):
                w = wide[wide.Area == a] if grp == "all" else wide[(wide.Area == a) & (wide.Group == grp)]
                if w.empty or f"{band}_pre8walk" not in w:
                    continue
                res = _paired_summary(w[f"{band}_pre8walk"].values, w[f"{band}_post8walk"].values)
                summ_rows.append(dict(Area=a, band=band, group=grp, **res))
                if grp == "all" and res.get("n", 0) >= 3 and band in ("High_Beta", "Gamma"):
                    lines.append(f"  {a:8s} {band:9s} n={res['n']:2d}  pre {res['mean_pre']:+.2f}  post {res['mean_post']:+.2f}  "
                                 f"diff {res['mean_diff']:+.2f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}] "
                                 f"paired t p={res['paired_t_p']:.3f}  r={res['pearson_r']:.2f} (p={res['pearson_p']:.3f}) "
                                 f"rho={res['spearman_rho']:.2f}")
            # PD - HC offset per block (Welch t), cortical only
            if a != "STN":
                for b in BLOCKS:
                    w = wide[wide.Area == a]
                    pd_v = w.loc[w.Group == "PD", f"{band}_{b}"].dropna().values
                    hc_v = w.loc[w.Group == "HC", f"{band}_{b}"].dropna().values
                    t, p = stats.ttest_ind(pd_v, hc_v, equal_var=False)
                    summ_rows.append(dict(Area=a, band=band, group=f"PD_minus_HC_{b}", n=len(pd_v) + len(hc_v),
                                          mean_pre=np.nan, mean_post=np.nan, mean_diff=pd_v.mean() - hc_v.mean(),
                                          paired_t=t, paired_t_p=p))
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT / f"cross_block_offsets_{DATE}.csv", index=False)
    lines.append("")
    lines.append("PD minus HC offset per block (Welch t on subject cycle means, dB):")
    for a in ("motor", "frontal"):
        for band in ("High_Beta", "Gamma"):
            parts = []
            for b in BLOCKS:
                r = summ[(summ.Area == a) & (summ.band == band) & (summ.group == f"PD_minus_HC_{b}")].iloc[0]
                parts.append(f"{b} {r.mean_diff:+.2f} dB (p={r.paired_t_p:.3f})")
            lines.append(f"  {a:8s} {band:9s} " + "   ".join(parts))
    lines.append("")

    # ---- 3. pattern similarity between blocks ----
    sim_rows = []
    tc = {}   # band-averaged avggait time courses from the per-subject regional TFRs
    freq_bands = {k.replace(" ", "_"): tuple(v) for k, v in config["freq_bands"].items()}
    for b in BLOCKS:
        for m in ("eeg", "dbs"):
            p = paths.tfr_gavg_dir / f"{b}-{m}-tfr-gavg-avggait.nc"
            if not p.exists():
                continue
            ds = xr.open_dataset(p)
            for s in subjects:
                if s not in ds.subject.values:
                    continue
                if m == "eeg":
                    for a in ("motor", "frontal"):
                        arr = ds.per_subject.sel(subject=s, region=a, side="Both")
                        for band in BANDS:
                            lo, hi = freq_bands[band]
                            v = arr.sel(freq=slice(lo, hi)).mean("freq").values
                            if np.isfinite(v).any():
                                tc[(s, b, a, band)] = v
                else:
                    arr = ds.per_subject.sel(subject=s, region="STN").max("side", skipna=True)
                    for band in BANDS:
                        lo, hi = freq_bands[band]
                        v = arr.sel(freq=slice(lo, hi)).mean("freq").values
                        if np.isfinite(v).any():
                            tc[(s, b, "STN", band)] = v
    for s in subjects:
        for a in AREAS:
            for band in BANDS:
                k_pre, k_post = (s, "pre8walk", a, band), (s, "post8walk", a, band)
                row = dict(SubjectID=s, Group=group_of[s], Area=a, band=band)
                if k_pre in pat4 and k_post in pat4:
                    x, y = pat4[k_pre], pat4[k_post]
                    row["r_4phase"] = float(np.corrcoef(x, y)[0, 1]) if np.isfinite(x).all() and np.isfinite(y).all() else np.nan
                    row["depth_pre"] = float(np.ptp(x)); row["depth_post"] = float(np.ptp(y))
                if k_pre in tc and k_post in tc:
                    x, y = tc[k_pre], tc[k_post]
                    ok = np.isfinite(x) & np.isfinite(y)
                    row["r_timecourse"] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else np.nan
                if "r_4phase" in row or "r_timecourse" in row:
                    sim_rows.append(row)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT / f"cross_block_pattern_similarity_per_subject_{DATE}.csv", index=False)

    sim_summ = []
    lines.append("Within-cycle modulation pattern similarity between blocks (avggait), per area x band:")
    lines.append("  r_4phase = Pearson r over the 4 phase medians; r_tc = Pearson r over the warped time course (band mean)")
    for a in AREAS:
        for band in BANDS:
            d = sim[(sim.Area == a) & (sim.band == band)]
            if d.empty:
                continue
            row = dict(Area=a, band=band, n=len(d))
            for col, tag in (("r_4phase", "4phase"), ("r_timecourse", "tc")):
                if col not in d or d[col].isna().all():
                    continue
                v = d[col].dropna().values
                z = _fisher(v)
                t1, p1 = stats.ttest_1samp(z, 0)
                row[f"median_r_{tag}"] = float(np.median(v)); row[f"mean_z_{tag}"] = float(z.mean())
                row[f"frac_pos_{tag}"] = float((v > 0).mean()); row[f"p_z_vs0_{tag}"] = p1
                for grp in ("PD", "HC"):
                    vg = d.loc[d.Group == grp, col].dropna().values
                    row[f"median_r_{tag}_{grp}"] = float(np.median(vg)) if len(vg) else np.nan
                zp = _fisher(d.loc[d.Group == "PD", col].dropna().values)
                zh = _fisher(d.loc[d.Group == "HC", col].dropna().values)
                if len(zp) > 2 and len(zh) > 2:
                    row[f"p_PDvsHC_{tag}"] = stats.ttest_ind(zp, zh, equal_var=False).pvalue
            sim_summ.append(row)
            lines.append(f"  {a:8s} {band:9s} n={row['n']:2d}  r_4phase median {row.get('median_r_4phase', np.nan):+.2f} "
                         f"(PD {row.get('median_r_4phase_PD', np.nan):+.2f}, HC {row.get('median_r_4phase_HC', np.nan):+.2f}; "
                         f"{row.get('frac_pos_4phase', np.nan)*100:.0f} % > 0)   "
                         f"r_tc median {row.get('median_r_tc', np.nan):+.2f} "
                         f"(PD {row.get('median_r_tc_PD', np.nan):+.2f}, HC {row.get('median_r_tc_HC', np.nan):+.2f}; "
                         f"{row.get('frac_pos_tc', np.nan)*100:.0f} % > 0; z vs 0 p={row.get('p_z_vs0_tc', np.nan):.3g}; "
                         f"PD vs HC p={row.get('p_PDvsHC_tc', np.nan):.2f})")
    pd.DataFrame(sim_summ).to_csv(OUT / f"cross_block_pattern_similarity_{DATE}.csv", index=False)
    lines.append("")

    # cohort-level pattern similarity (median across subjects per cohort, time course)
    lines.append("Cohort-level (median across subjects) avggait time-course correlation pre vs post, per area x band:")
    for a in AREAS:
        for band in BANDS:
            parts = []
            for grp in ("PD", "HC"):
                if a == "STN" and grp == "HC":
                    continue
                pre = [tc[k] for k in tc if k[1] == "pre8walk" and k[2] == a and k[3] == band and group_of[k[0]] == grp]
                post = [tc[k] for k in tc if k[1] == "post8walk" and k[2] == a and k[3] == band and group_of[k[0]] == grp]
                if len(pre) < 2 or len(post) < 2:
                    continue
                x, y = np.nanmedian(np.array(pre), axis=0), np.nanmedian(np.array(post), axis=0)
                ok = np.isfinite(x) & np.isfinite(y)
                parts.append(f"{grp} r={np.corrcoef(x[ok], y[ok])[0, 1]:+.2f} (n={len(pre)}/{len(post)})")
            if parts:
                lines.append(f"  {a:8s} {band:9s} " + "   ".join(parts))
    lines.append("")

    # ---- 4. side-by-side cohort-level tables ----
    rm_pre = pd.read_csv(OUT / "rm_anova_classic_2026-10-14.csv")
    rm_post = pd.read_csv(OUT / f"rm_anova_post8walk_{DATE}.csv")
    terms = ["GaitPhase", "Cohort", "Cohort:GaitPhase", "Cohort:Area", "Area", "Area:GaitPhase", "Cohort:Area:GaitPhase",
             "Hemisphere", "Hemisphere:GaitPhase"]
    tab_rows = []
    for region in ("cortical", "STN"):
        for band in BANDS:
            for term in terms:
                a = rm_pre[(rm_pre.region == region) & (rm_pre.summary == "median") & (rm_pre.band == band) & (rm_pre.term == term)]
                b = rm_post[(rm_post.region == region) & (rm_post.summary == "median") & (rm_post.band == band) & (rm_post.term == term)]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                tab_rows.append(dict(region=region, band=band, term=term,
                                     F_pre=a.F, df1=a.df1, df2_pre=a.df2, p_gg_pre=a.p_gg, p_bh_pre=a.p_gg_bh_bands, eta2p_pre=a.eta2p,
                                     F_post=b.F, df2_post=b.df2, p_gg_post=b.p_gg, p_bh_post=b.p_gg_bh_bands, eta2p_post=b.eta2p,
                                     sig_pre=a.p_gg < 0.05, sig_post=b.p_gg < 0.05))
    tab = pd.DataFrame(tab_rows)
    tab.to_csv(OUT / f"cross_block_cohort_table_{DATE}.csv", index=False)
    lines.append("RM-ANOVA (subject medians, GG-corrected p) side by side, pre8walk | post8walk:")
    lines.append(f"  {'region':8s} {'band':9s} {'term':22s} {'F_pre':>7s} {'p_pre':>8s} {'eta_pre':>7s} | {'F_post':>7s} {'p_post':>8s} {'eta_post':>8s}")
    for _, r in tab.iterrows():
        if r.term in ("GaitPhase", "Cohort", "Cohort:GaitPhase", "Hemisphere:GaitPhase"):
            lines.append(f"  {r.region:8s} {r.band:9s} {r.term:22s} {r.F_pre:7.2f} {r.p_gg_pre:8.4f} {r.eta2p_pre:7.3f} | "
                         f"{r.F_post:7.2f} {r.p_gg_post:8.4f} {r.eta2p_post:8.3f}")
    lines.append("")

    # per-phase PD - HC contrasts side by side (high beta, gamma)
    c_pre = pd.read_csv(OUT / "rm_anova_classic_contrasts_2026-10-14.csv")
    c_post = pd.read_csv(OUT / f"rm_anova_post8walk_contrasts_{DATE}.csv")
    lines.append("PD minus HC per gait phase (Welch t on subject means, dB), pre8walk | post8walk:")
    for band in ("High_Beta", "Gamma"):
        for ph in PHASES:
            a = c_pre[(c_pre.region == "cortical") & (c_pre.summary == "median") & (c_pre.family == "PD_vs_HC_by_phase") & (c_pre.band == band) & (c_pre.level == ph)].iloc[0]
            b = c_post[(c_post.region == "cortical") & (c_post.summary == "median") & (c_post.family == "PD_vs_HC_by_phase") & (c_post.band == band) & (c_post.level == ph)].iloc[0]
            lines.append(f"  {band:9s} {ph}  {a.diff_PD_minus_HC:+.2f} [{a.ci_lo:+.2f}, {a.ci_hi:+.2f}] p={a.p:.3f} | "
                         f"{b.diff_PD_minus_HC:+.2f} [{b.ci_lo:+.2f}, {b.ci_hi:+.2f}] p={b.p:.3f}")
    lines.append("")

    # pre8walk reference: the 2026-10-14 re-run of cluster_perm_revision.py (on the
    # pre8walk grand averages regenerated 2026-10-14) when present, else the 2026-10-13 run.
    pre_cluster = OUT / "cluster_perm_2026-10-14.csv"
    if not pre_cluster.exists():
        pre_cluster = OUT / "cluster_perm_2026-10-13.csv"
    cl_pre = pd.read_csv(pre_cluster)
    cl_post = pd.read_csv(OUT / f"cluster_perm_post8walk_{DATE}.csv")
    cl = cl_pre.merge(cl_post.drop(columns=["block"]), on=["test", "normalisation", "region", "side"], suffixes=("_pre", "_post"))
    cl.insert(0, "pre8walk_source", pre_cluster.name)
    cl.to_csv(OUT / f"cross_block_cluster_table_{DATE}.csv", index=False)
    lines.append(f"Cluster tests side by side (n significant clusters, min cluster p), pre8walk ({pre_cluster.name}) | post8walk:")
    for _, r in cl.iterrows():
        lines.append(f"  {r.test:28s} {r.region:8s} {r.side:8s} n={r.n_group1_pre}/{r.n_group2_pre} sig={r.n_significant_pre} "
                     f"minp={r.min_cluster_p_pre:.4f} | n={r.n_group1_post}/{r.n_group2_post} sig={r.n_significant_post} "
                     f"minp={r.min_cluster_p_post:.4f}  {r.significant_clusters_post if isinstance(r.significant_clusters_post, str) else ''}")

    # ---- 5. figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for i, band in enumerate(("High_Beta", "Gamma")):
        for j, a in enumerate(AREAS):
            ax = axes[i, j]
            w = wide[wide.Area == a]
            for grp, col in (("PD", "tab:red"), ("HC", "tab:blue")):
                g = w[w.Group == grp]
                ax.scatter(g[f"{band}_pre8walk"], g[f"{band}_post8walk"], c=col, label=grp, s=28, alpha=0.8)
            allv = np.concatenate([w[f"{band}_pre8walk"].dropna(), w[f"{band}_post8walk"].dropna()])
            if allv.size:
                lo, hi = allv.min() - 0.5, allv.max() + 0.5
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            res = summ[(summ.Area == a) & (summ.band == band) & (summ.group == "all")]
            if not res.empty and res.iloc[0].get("n", 0) >= 3:
                r0 = res.iloc[0]
                ax.set_title(f"{a} {band.replace('_', ' ')}: r={r0.pearson_r:.2f}, diff {r0.mean_diff:+.2f} dB (p={r0.paired_t_p:.2f})", fontsize=8)
            ax.set_xlabel("pre8walk offset (dB vs rest)"); ax.set_ylabel("post8walk offset (dB vs rest)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Walking-vs-rest offset per subject: first vs second walking block", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cross_block_offsets.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for j, a in enumerate(AREAS):
        ax = axes[j]
        d = sim[sim.Area == a]
        for bi, band in enumerate(BANDS):
            for grp, col, dx in (("PD", "tab:red", -0.15), ("HC", "tab:blue", 0.15)):
                v = d[(d.band == band) & (d.Group == grp)]["r_timecourse"].dropna().values
                if v.size:
                    ax.scatter(np.full(v.size, bi + dx) + np.random.uniform(-0.05, 0.05, v.size), v, c=col, s=16, alpha=0.7,
                               label=grp if bi == 0 else None)
                    ax.hlines(np.median(v), bi + dx - 0.12, bi + dx + 0.12, colors=col, lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([b.replace("_", " ") for b in BANDS], fontsize=8)
        ax.set_title(a, fontsize=9)
        if j == 0:
            ax.set_ylabel("r (avggait time course, pre vs post)"); ax.legend(fontsize=8)
    fig.suptitle("Within-cycle modulation pattern similarity between walking blocks (per subject)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cross_block_pattern_similarity.png", dpi=150); plt.close(fig)

    txt = OUT / f"post8walk_replication_{DATE}.txt"
    txt.write_text("\n".join(lines) + "\n")
    write_sidecar(txt, {**meta, "n_subjects_offsets": int(wide.SubjectID.nunique()),
                        "n_subjects_similarity": int(sim.SubjectID.nunique()) if not sim.empty else 0})
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
