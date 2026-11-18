"""Direct PD vs HC comparison of cortical spectra at seated rest and during walking.

Purpose:  Locate the baseline-normalised high-beta/gamma offset: compare
          absolute (log) power between cohorts separately at rest and during
          walking, per band and across the full 4-45 Hz spectrum (1-D cluster
          permutation), and compare the aperiodic parameters at rest.
Inputs:   data/derivatives/psd_unwarped/<subj>_{baseline,pre8walk}-eeg-psd.csv
              (psd_unwarped.py; spectrum variant A, ROI-mean Welch PSD)
          outputs/stats/revision_2026-10/fooof_unwarped_2026-10-13.csv
              (fooof_refit.py; aperiodic offset/exponent, variant A, mph 0.15)
          code/analysis/config.yaml (freq_bands, repro.seed)
Outputs:  outputs/stats/revision_2026-10/rest_spectra_bands_2026-10-13.csv
          outputs/stats/revision_2026-10/rest_spectra_clusters_2026-10-13.csv
          outputs/stats/revision_2026-10/rest_spectra_2026-10-13.txt
          outputs/figures/revision_2026-10/rest_spectra_<region>.png
Author:   Lena
Date:     2026-10-13
Seed:     42 (config repro.seed; MNE permutation seed)

Usage:
  python code/analysis/revision/rest_spectra_comparison.py
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

PD, HC = "Patient_NFB", "AgeMatched_NFB"
BLOCKS = {"baseline": "rest", "pre8walk": "walking"}


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def main() -> int:
    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("rest_spectra_comparison", config); seed = int(config["repro"]["seed"])
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"; fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    bands = {k: tuple(v) for k, v in config["freq_bands"].items()}

    # ---- load spectra: spec[(subj, block, region)] = (freqs, log10 power) ----
    spec = {}
    for f in sorted((paths.derivatives_root / "psd_unwarped").glob("*-eeg-psd.csv")):
        rows = [r for r in csv.DictReader(f.open()) if r["spectrum_variant"] == "A" and r["side"] == "Both"]
        if not rows:
            continue
        subj, block = rows[0]["subject"], rows[0]["block"]
        for region in ("frontal", "motor"):
            rr = [r for r in rows if r["region"] == region]
            if rr:
                spec[(subj, block, region)] = (np.array([float(r["freq"]) for r in rr]),
                                               np.log10(np.array([float(r["power"]) for r in rr])))
    subjects = sorted({k[0] for k in spec})
    cohort = {s: (PD if s.startswith("P4-2") else HC) for s in subjects}

    lines = [f"PD vs HC cortical spectra, absolute log10 power (variant A), seed {seed}", ""]
    band_rows, cl_rows = [], []
    import mne

    for region in ("frontal", "motor"):
        for block, cond in BLOCKS.items():
            fr = None; X = {PD: [], HC: []}
            for s in subjects:
                if (s, block, region) in spec:
                    fr, p = spec[(s, block, region)]; X[cohort[s]].append(p)
            Xpd, Xhc = np.array(X[PD]), np.array(X[HC])
            # per band: 10*log10 mean power in band (dB), Welch t, MWU, Hedges g
            for band, (lo, hi) in bands.items():
                m = (fr >= lo) & (fr < hi)
                a = 10 * np.log10(np.mean(10 ** Xpd[:, m], axis=1)); b = 10 * np.log10(np.mean(10 ** Xhc[:, m], axis=1))
                t, pt = stats.ttest_ind(a, b, equal_var=False); u, pu = stats.mannwhitneyu(a, b)
                band_rows.append(dict(region=region, condition=cond, band=band, n_pd=len(a), n_hc=len(b),
                                      pd_mean_db=a.mean(), pd_sd=a.std(ddof=1), hc_mean_db=b.mean(), hc_sd=b.std(ddof=1),
                                      diff_db=a.mean() - b.mean(), welch_t=t, welch_p=pt, mwu_p=pu, hedges_g=hedges_g(a, b)))
            # 1-D cluster permutation across frequency on log10 power
            T, clusters, cp, _ = mne.stats.permutation_cluster_test([Xpd, Xhc], n_permutations=5000, tail=0,
                                                                    out_type="mask", seed=seed, verbose="WARNING")
            diff = Xpd.mean(0) - Xhc.mean(0)
            for c, p in zip(clusters, cp):
                cl_rows.append(dict(region=region, condition=cond, f_lo=fr[c].min(), f_hi=fr[c].max(),
                                    mean_diff_db=10 * diff[c].mean(), cluster_p=p, significant=p < 0.05))
            sig = [r for r in cl_rows if r["region"] == region and r["condition"] == cond and r["significant"]]
            lines.append(f"{region:8s}{cond:8s} n={len(Xpd)}/{len(Xhc)}  significant clusters: " +
                         ("; ".join(f"{r['f_lo']:.0f}-{r['f_hi']:.0f} Hz ({r['mean_diff_db']:+.1f} dB, p={r['cluster_p']:.3f})" for r in sig) or "none") +
                         (f"  (min cluster p = {min(cp):.3f})" if len(cp) else ""))
            # figure data
            if block == "baseline":
                fig_rest = (fr, Xpd, Xhc, clusters, cp)
            else:
                fig_walk = (fr, Xpd, Xhc, clusters, cp)

        # ---- figure: group mean ± SEM spectra, rest and walking ----
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (fr, Xpd, Xhc, clusters, cp), title in zip(axes[:2], (fig_rest, fig_walk), ("seated rest", "walking")):
            for X, color, lab in ((Xpd, "C3", "PD"), (Xhc, "C0", "HC")):
                mu, se = 10 * X.mean(0), 10 * X.std(0, ddof=1) / np.sqrt(len(X))
                ax.plot(fr, mu, color=color, label=f"{lab} (n={len(X)})"); ax.fill_between(fr, mu - se, mu + se, color=color, alpha=0.25)
            for c, p in zip(clusters, cp):
                if p < 0.05:
                    ax.axvspan(fr[c].min(), fr[c].max(), color="grey", alpha=0.25)
            ax.set_title(f"{region}, {title}"); ax.set_xlabel("Hz"); ax.set_ylabel("10 log10 power (dB)"); ax.legend()
        ax = axes[2]
        for (fr, Xpd, Xhc, _, _), ls, lab in ((fig_rest, "--", "rest"), (fig_walk, "-", "walking")):
            ax.plot(fr, 10 * (Xpd.mean(0) - Xhc.mean(0)), color="k", ls=ls, label=f"PD - HC, {lab}")
        ax.axhline(0, color="grey", lw=0.5); ax.set_title(f"{region}, cohort difference"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"rest_spectra_{region}.png", dpi=130); plt.close(fig)

    # ---- aperiodic parameters at rest and walking (variant A, mph 0.15) ----
    ap = defaultdict(lambda: {PD: [], HC: []})
    for r in csv.DictReader((out_dir / "fooof_unwarped_2026-10-13.csv").open()):
        if r["modality"] == "eeg" and r["spectrum_variant"] == "A" and abs(float(r["min_peak_height"]) - 0.15) < 1e-9:
            for par in ("aperiodic_offset", "aperiodic_exponent"):
                ap[(r["region"], BLOCKS[r["block"]], par)][r["cohort"]].append(float(r[par]))
    lines.append(""); lines.append("Aperiodic parameters (FOOOF, variant A): PD vs HC")
    for key in sorted(ap):
        a, b = np.array(ap[key][PD]), np.array(ap[key][HC])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        lines.append(f"  {key[0]:8s}{key[1]:8s}{key[2]:19s} PD {a.mean():6.2f}±{a.std(ddof=1):.2f}  HC {b.mean():6.2f}±{b.std(ddof=1):.2f}  diff {a.mean()-b.mean():+.2f}  Welch p={p:.3f}  g={hedges_g(a,b):+.2f}")
        band_rows.append(dict(region=key[0], condition=key[1], band=key[2], n_pd=len(a), n_hc=len(b), pd_mean_db=a.mean(), pd_sd=a.std(ddof=1),
                              hc_mean_db=b.mean(), hc_sd=b.std(ddof=1), diff_db=a.mean() - b.mean(), welch_t=t, welch_p=p,
                              mwu_p=stats.mannwhitneyu(a, b).pvalue, hedges_g=hedges_g(a, b)))

    lines.append(""); lines.append("Band power (10 log10 mean power in band, dB): PD vs HC")
    for r in band_rows:
        if r["band"] in bands:
            lines.append(f"  {r['region']:8s}{r['condition']:8s}{r['band']:10s} PD {r['pd_mean_db']:7.1f}±{r['pd_sd']:.1f}  HC {r['hc_mean_db']:7.1f}±{r['hc_sd']:.1f}  diff {r['diff_db']:+.2f} dB  Welch p={r['welch_p']:.3f}  MWU p={r['mwu_p']:.3f}  g={r['hedges_g']:+.2f}")

    for name, rows in (("rest_spectra_bands_2026-10-13.csv", band_rows), ("rest_spectra_clusters_2026-10-13.csv", cl_rows)):
        with (out_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        write_sidecar(out_dir / name, meta)
    (out_dir / "rest_spectra_2026-10-13.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
