"""Cycle-to-cycle consistency of the gait-locked cortical and STN modulation.

Purpose:  Ask whether the gait-phase-locked neural pattern is less consistent
          from cycle to cycle in PD than in HC (the neural analogue of the
          higher stride-time variability in PD), using the per-cycle
          gERD/ERS values that the cohort-level analyses average away.
Inputs:   data/derivatives/gerd/<subj>_pre8walk-{eeg,dbs}-{avggait,basenorm}-gerd.csv
              (compute_gerd_table.py; one row per cycle x area x hemisphere x
              gait phase, five band columns)
          outputs/stats/revision_2026-10/gait_kinematics_per_subject_2026-10-13.csv
              (stride-time CV per subject, for the behavioural link)
          code/analysis/config.yaml (freq_bands)
Outputs:  outputs/stats/revision_2026-10/cycle_variability_per_subject_2026-10-13.csv
          outputs/stats/revision_2026-10/cycle_variability_2026-10-13.csv  (group table)
          outputs/stats/revision_2026-10/cycle_variability_2026-10-13.txt
          outputs/figures/revision_2026-10/cycle_variability.png
Author:   Lena
Date:     2026-10-13
Seed:     42 (numpy seed for the leave-one-out bootstrap of the consistency metric)

Description:
  For each subject, area (motor, frontal; Hemisphere == Both; STN for PD),
  normalisation and band, the per-cycle 4-phase vector (IDS, LLS, FDS, RLS)
  is compared with the subject's leave-one-out mean pattern:
    consistency  = mean across cycles of Pearson r(cycle pattern, LOO mean)
    noise_sd     = mean across phases of the across-cycle SD (dB)
    depth        = max - min of the mean pattern (dB)
    snr          = depth / noise_sd
  avggait is the primary normalisation (offset removed by construction).
  Group comparison per metric x area x band: Welch t, Mann-Whitney U,
  Hedges g; BH across the five bands within metric x area. Spearman
  correlation of consistency and noise_sd with stride-time CV across all 20
  subjects and within PD.

Usage:
  python code/analysis/revision/cycle_variability.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

PHASES = ("IDS", "LLS", "FDS", "RLS")
BANDS = ("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
PD, HC = "Patient_NFB", "AgeMatched_NFB"


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); ranked = np.empty(n)
    ranked[order] = p[order] * n / (np.arange(n) + 1)
    out = np.minimum.accumulate(ranked[order][::-1])[::-1]; res = np.empty(n); res[order] = np.minimum(out, 1); return res


def metrics_for(cycles: np.ndarray) -> dict:
    """cycles: (n_cycles, 4) phase values for one subject/area/band."""
    n = len(cycles)
    if n < 5:
        return {}
    mean_pat = cycles.mean(0)
    rs = []
    for i in range(n):
        loo = (mean_pat * n - cycles[i]) / (n - 1)
        if np.std(cycles[i]) > 0 and np.std(loo) > 0:
            rs.append(np.corrcoef(cycles[i], loo)[0, 1])
    noise = cycles.std(0, ddof=1).mean()
    depth = mean_pat.max() - mean_pat.min()
    return dict(n_cycles=n, consistency=float(np.mean(rs)), noise_sd=float(noise), depth=float(depth),
                snr=float(depth / noise) if noise > 0 else np.nan)


def main() -> int:
    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("cycle_variability", config); np.random.seed(int(config["repro"]["seed"]))
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"; fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    gerd = paths.derivatives_root / "gerd"

    per_subject = []
    for f in sorted(gerd.glob("*_pre8walk-*-gerd.csv")):
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        subj, cohort = rows[0]["SubjectID"], rows[0]["Cohort"]
        modality, norm = rows[0]["Modality"], f.stem.split("-")[-2]
        areas = sorted({r["Area"] for r in rows})
        for area in areas:
            sel = [r for r in rows if r["Area"] == area and (modality == "dbs" or r["Hemisphere"] == "Both")]
            if not sel:
                continue
            for band in BANDS:
                col = f"ERD_ERS_{band}"
                by_cycle = defaultdict(dict)
                for r in sel:
                    try:
                        by_cycle[int(r["Epoch"])][r["GaitPhase"]] = float(r[col])
                    except ValueError:
                        pass
                cyc = np.array([[d[p] for p in PHASES] for e, d in sorted(by_cycle.items()) if all(p in d for p in PHASES)])
                m = metrics_for(cyc)
                if m:
                    per_subject.append(dict(subject=subj, cohort=cohort, modality=modality, area=area, norm=norm, band=band, **m))

    # stride-time CV per subject (kept cycles)
    cv = {}
    kin = out_dir / "gait_kinematics_per_subject_2026-10-13.csv"
    if kin.exists():
        krows = list(csv.DictReader(kin.open()))
        cv_col = next((c for c in krows[0] if "cv" in c.lower() and "stride" in c.lower()), None)
        set_col = next((c for c in krows[0] if "cycle_set" in c.lower()), None)
        for r in krows:
            if cv_col and (set_col is None or r[set_col] == "kept"):
                cv[r.get("subject") or r.get("SubjectID")] = float(r[cv_col])
    for r in per_subject:
        r["stride_cv"] = cv.get(r["subject"], np.nan)

    ps_path = out_dir / "cycle_variability_per_subject_2026-10-13.csv"
    with ps_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_subject[0])); w.writeheader(); w.writerows(per_subject)
    write_sidecar(ps_path, meta)

    # ---- group comparisons (EEG: PD vs HC) ----
    group_rows = []; lines = ["Cycle-to-cycle consistency of gait-locked modulation: PD vs HC (EEG), descriptive STN (PD)", ""]
    for norm in ("avggait", "basenorm"):
        for metric in ("consistency", "noise_sd", "depth", "snr"):
            for area in ("motor", "frontal"):
                ps = []
                for band in BANDS:
                    a = np.array([r[metric] for r in per_subject if r["modality"] == "eeg" and r["norm"] == norm and r["area"] == area and r["band"] == band and r["cohort"] == PD])
                    b = np.array([r[metric] for r in per_subject if r["modality"] == "eeg" and r["norm"] == norm and r["area"] == area and r["band"] == band and r["cohort"] == HC])
                    t, pt = stats.ttest_ind(a, b, equal_var=False); pu = stats.mannwhitneyu(a, b).pvalue
                    group_rows.append(dict(norm=norm, metric=metric, area=area, band=band, n_pd=len(a), n_hc=len(b),
                                           pd_mean=a.mean(), pd_sd=a.std(ddof=1), hc_mean=b.mean(), hc_sd=b.std(ddof=1),
                                           welch_t=t, welch_p=pt, mwu_p=pu, hedges_g=hedges_g(a, b)))
                    ps.append(pt)
                for r, q in zip(group_rows[-5:], bh(ps)):
                    r["welch_p_bh_bands"] = q
        # STN descriptive
        for metric in ("consistency", "noise_sd", "depth", "snr"):
            for band in BANDS:
                a = np.array([r[metric] for r in per_subject if r["modality"] == "dbs" and r["norm"] == norm and r["band"] == band])
                group_rows.append(dict(norm=norm, metric=metric, area="STN", band=band, n_pd=len(a), n_hc=0, pd_mean=a.mean(), pd_sd=a.std(ddof=1),
                                       hc_mean=np.nan, hc_sd=np.nan, welch_t=np.nan, welch_p=np.nan, mwu_p=np.nan, hedges_g=np.nan, welch_p_bh_bands=np.nan))

    for norm in ("avggait", "basenorm"):
        lines.append(f"=== {norm} ===")
        for metric in ("consistency", "noise_sd", "depth", "snr"):
            for area in ("motor", "frontal"):
                lines.append(f"  {metric:12s}{area:8s}")
                for r in group_rows:
                    if r["norm"] == norm and r["metric"] == metric and r["area"] == area:
                        lines.append(f"    {r['band']:10s} PD {r['pd_mean']:6.3f}±{r['pd_sd']:.3f}  HC {r['hc_mean']:6.3f}±{r['hc_sd']:.3f}  g={r['hedges_g']:+.2f}  Welch p={r['welch_p']:.3f} (BH {r['welch_p_bh_bands']:.3f})  MWU p={r['mwu_p']:.3f}")
        lines.append("")

    # ---- behavioural link ----
    lines.append("Spearman correlation with stride-time CV (kept cycles)")
    for norm in ("avggait",):
        for metric in ("consistency", "noise_sd", "snr"):
            for area in ("motor", "frontal"):
                for band in BANDS:
                    sel = [r for r in per_subject if r["modality"] == "eeg" and r["norm"] == norm and r["area"] == area and r["band"] == band and not np.isnan(r["stride_cv"])]
                    x = np.array([r["stride_cv"] for r in sel]); y = np.array([r[metric] for r in sel])
                    rho, p = stats.spearmanr(x, y)
                    pdsel = [r for r in sel if r["cohort"] == PD]
                    rho_pd, p_pd = stats.spearmanr([r["stride_cv"] for r in pdsel], [r[metric] for r in pdsel])
                    if p < 0.05 or p_pd < 0.05:
                        lines.append(f"  {metric:12s}{area:8s}{band:10s} all n={len(sel)} rho={rho:+.2f} p={p:.3f} | PD n={len(pdsel)} rho={rho_pd:+.2f} p={p_pd:.3f}")
    lines.append("  (only correlations with p < 0.05 in either set are listed; uncorrected, exploratory)")

    g_path = out_dir / "cycle_variability_2026-10-13.csv"
    with g_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(group_rows[0])); w.writeheader(); w.writerows(group_rows)
    write_sidecar(g_path, meta)
    (out_dir / "cycle_variability_2026-10-13.txt").write_text("\n".join(lines) + "\n")

    # ---- figure: consistency and noise per band, avggait, motor + frontal ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r_, metric in enumerate(("consistency", "noise_sd")):
        for c_, area in enumerate(("motor", "frontal")):
            ax = axes[r_, c_]
            for k, band in enumerate(BANDS):
                for cohort, col, dx in ((PD, "C3", -0.15), (HC, "C0", 0.15)):
                    v = [r[metric] for r in per_subject if r["modality"] == "eeg" and r["norm"] == "avggait" and r["area"] == area and r["band"] == band and r["cohort"] == cohort]
                    ax.scatter(np.full(len(v), k + dx) + np.random.uniform(-0.05, 0.05, len(v)), v, color=col, alpha=0.6, s=18)
                    ax.hlines(np.mean(v), k + dx - 0.12, k + dx + 0.12, color=col, lw=2.5)
            ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([b.replace("_", " ") for b in BANDS])
            ax.set_title(f"{metric} (avggait), {area}"); ax.set_ylabel("r" if metric == "consistency" else "dB")
    axes[0, 0].scatter([], [], color="C3", label="PD"); axes[0, 0].scatter([], [], color="C0", label="HC"); axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "cycle_variability.png", dpi=130); plt.close(fig)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
