"""Decompose the walking-vs-rest high-frequency cohort offset into aperiodic and periodic parts.

Purpose:  Test whether the baseline-normalised high-beta / gamma offset of PD
          below HC in the gait TFRs sits in the aperiodic (broadband)
          component of the spectrum rather than in oscillatory power above
          the aperiodic fit. Per subject x region: (a) the TFR walking-vs-rest
          offset (gait-cycle mean of the basenorm map, dB vs rest, side
          "Both"); (b) FOOOF variant A walking-minus-rest change in aperiodic
          offset, exponent and aperiodic-adjusted band power; (c) from the
          same raw ROI spectra the walking-minus-rest change in total log
          band power, its exact split into aperiodic-fit change and periodic
          (adjusted) change, and the aperiodic-only prediction at the band
          centre. Group means with CIs, PD-HC differences, correlations with
          the TFR offset, robustness without P4-2007, and a figure. The mixed
          models run in aperiodic_offset_lmm.R on the long table written here.
Inputs:   data/derivatives/tfr/<subj>_pre8walk-eeg-tfr-gait.h5
              (compute_tfr_gait.py; per-subject gait-warped basenorm TFR,
              dB = 10 log10(walk / mean rest); primary source of the TFR offset)
          data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg.nc
              (aggregate_tfr_gavg.py; per_subject slab; cross-check only, see
              "Deviations" below)
          outputs/stats/revision_2026-10/fooof_unwarped_2026-10-13.csv
              (fooof_refit.py; variant A, min_peak_height 0.15, eeg, side Both)
          data/derivatives/psd_unwarped/<subj>_{baseline,pre8walk}-eeg-psd.csv
              (psd_unwarped.py; variant A ROI-mean Welch spectra 4-45 Hz)
          outputs/stats/revision_2026-10/dbs_artefact_spectra_2026-10-13.csv
              (dbs_artefact_check.py; whole-block 2-100 Hz spectra, cross-check)
          code/analysis/config.yaml (freq_bands, repro.seed)
Outputs:  outputs/stats/revision_2026-10/aperiodic_offset_per_subject_2026-10-14.csv
              (wide: one row per subject x region x band)
          outputs/stats/revision_2026-10/aperiodic_offset_long_2026-10-14.csv
              (long: subject x region x condition x measure x value, for R)
          outputs/stats/revision_2026-10/aperiodic_offset_decomposition_2026-10-14.csv
              (group means, CIs, PD-HC differences, correlations, robustness)
          outputs/stats/revision_2026-10/aperiodic_offset_decomposition_2026-10-14.txt
          outputs/figures/revision_2026-10/aperiodic_offset_decomposition.png
Author:   Lena
Date:     2026-10-14
Seed:     42 (config repro.seed; no randomness in the computation)

Conventions:
  All changes are walking minus rest. TFR offset in dB (negative = lower
  during walking). FOOOF offset, exponent and adjusted power are log10
  units; log10 quantities are multiplied by 10 to report dB. Band bins
  lo <= f < hi, Gamma inclusive of 45 Hz (as in band_power_above_aperiodic.py).
  Identity used for the decomposition (exact for the same spectrum):
      mean_band(log10 P) = mean_band(offset - exponent * log10 f) + adj_power
  so  d_total = d_aperiodic + d_periodic  (residual reported as a check).

Deviations:
  The aggregated NetCDF (written 2026-05-19) predates the per-subject gait
  TFR files (2026-05-20). Its per_subject slab differs from the current
  files for P4-2003 (motor side Right about +95 dB in the NetCDF, i.e. a
  bad channel in an earlier TFR version; motor Both +34 dB vs -6 dB now),
  P4-2012 (1.6-4.3 dB) and P4-2010 (< 0.1 dB); the other 17 subjects agree
  to within 0.02 dB. The TFR offset is therefore read from the per-subject
  files (ROI mean of channels, non-finite cells treated as missing, as in
  aggregate_tfr_gavg.py); the NetCDF value is kept as `tfr_offset_nc_db`.

Usage:
  python code/analysis/revision/aperiodic_offset_decomposition.py
"""
from __future__ import annotations

import csv
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"
LABEL = {PD: "PD", HC: "HC"}
BLOCKS = {"baseline": "rest", "pre8walk": "walking"}
REGIONS = ("frontal", "motor")
BANDS = ("High Beta", "Gamma")
EXCLUDE_ROBUST = "P4-2007"  # near-flat rest spectrum (dbs_artefact_check_2026-10-13.txt)
DATE = "2026-10-14"


def band_mask(fr: np.ndarray, band: str, lo: float, hi: float) -> np.ndarray:
    return (fr >= lo) & ((fr <= hi) if band == "Gamma" else (fr < hi))


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h, se


def welch_diff(a, b):
    """PD-HC difference, Welch t, p, Welch-Satterthwaite 95 % CI, Hedges g."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    return d, d - h, d + h, t, df, p, hedges_g(a, b)


def main() -> int:
    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("aperiodic_offset_decomposition", config); seed = int(config["repro"]["seed"])
    meta["versions"] = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "xarray", "matplotlib")}
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"; fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    bands = {k: tuple(v) for k, v in config["freq_bands"].items() if k in BANDS}

    # ---- (a) TFR walking-vs-rest offset: gait-cycle mean of the basenorm map ----
    # Cross-check slab from the aggregated NetCDF (stale for 3 subjects, see header).
    ds = xr.open_dataset(paths.tfr_gavg_dir / "pre8walk-eeg-tfr-gavg.nc")
    subjects = [str(s) for s in ds.subject.values]
    cohort = {str(s): str(c) for s, c in zip(ds.subject.values, ds.subject_cohort.values)}
    tfr_nc = {}  # (subj, region, band) -> dB
    fr_tfr = ds.freq.values
    for region in REGIONS:
        cyc = ds["per_subject"].sel(region=region, side="Both").mean("time")  # (subject, freq), dB
        for band, (lo, hi) in bands.items():
            m = band_mask(fr_tfr, band, lo, hi)
            for s, val in zip(subjects, cyc.values[:, m].mean(axis=1)):
                tfr_nc[(s, region, band)] = float(val)
    ds.close()
    # Primary: per-subject gait TFR files, one subject at a time (18 GB machine).
    import gc
    import mne
    coi = config["connectivity"]["channels_of_interest"]
    roi_chans = {r: [c for side in ("Left", "Right", "Central") for c in coi[r][side]] for r in REGIONS}
    tfr = {}  # (subj, region, band) -> dB, gait-cycle mean, ROI "Both"
    for s in subjects:
        t = mne.time_frequency.read_tfrs(paths.tfr_dir / f"{s}_pre8walk-eeg-tfr-gait.h5", verbose="ERROR")
        t = t[0] if isinstance(t, list) else t
        d = np.array(t.data, dtype=float); d[~np.isfinite(d)] = np.nan
        for region in REGIONS:
            idx = [t.ch_names.index(c) for c in roi_chans[region] if c in t.ch_names]
            rm = np.nanmean(d[idx], axis=0)  # (freq, time)
            for band, (lo, hi) in bands.items():
                m = band_mask(np.asarray(t.freqs), band, lo, hi)
                tfr[(s, region, band)] = float(np.nanmean(rm[m]))
        del t, d; gc.collect()

    # ---- (b) FOOOF variant A, mph 0.15, eeg, side Both ----
    fo = {}  # (subj, region, block) -> dict
    for r in csv.DictReader((out_dir / "fooof_unwarped_2026-10-13.csv").open()):
        if (r["modality"] == "eeg" and r["spectrum_variant"] == "A" and r["side"] == "Both"
                and abs(float(r["min_peak_height"]) - 0.15) < 1e-9 and r["region"] in REGIONS):
            fo[(r["subject"], r["region"], r["block"])] = {
                "offset": float(r["aperiodic_offset"]), "exponent": float(r["aperiodic_exponent"]),
                "r2": float(r["fit_r_squared"]),
                **{f"{b}_adj": float(r[f"{b}_adj_power"]) for b in BANDS}}

    # ---- (c) raw ROI spectra (same spectra FOOOF was fitted on) ----
    spec = {}  # (subj, region, block) -> (freqs, log10 power)
    for f in sorted((paths.derivatives_root / "psd_unwarped").glob("*-eeg-psd.csv")):
        rows = [r for r in csv.DictReader(f.open()) if r["spectrum_variant"] == "A" and r["side"] == "Both"]
        if not rows:
            continue
        subj, block = rows[0]["subject"], rows[0]["block"]
        for region in REGIONS:
            rr = [r for r in rows if r["region"] == region]
            if rr:
                spec[(subj, region, block)] = (np.array([float(r["freq"]) for r in rr]),
                                               np.log10(np.array([float(r["power"]) for r in rr])))
    # cross-check: whole-block 2-100 Hz spectra (turns included, no mask)
    screen = {}  # (subj, region, block) -> (freqs, log10 power)
    acc = {}
    for r in csv.DictReader((out_dir / "dbs_artefact_spectra_2026-10-13.csv").open()):
        acc.setdefault((r["subject"], r["region"], r["block"]), []).append((float(r["freq"]), float(r["log10_power"])))
    for k, v in acc.items():
        v.sort(); screen[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))

    # ---- per-subject table ----
    wide, long = [], []
    for s in subjects:
        for region in REGIONS:
            fr_r, p_r = spec[(s, region, "baseline")]; fr_w, p_w = spec[(s, region, "pre8walk")]
            assert np.allclose(fr_r, fr_w)
            fo_r, fo_w = fo[(s, region, "baseline")], fo[(s, region, "pre8walk")]
            d_off = fo_w["offset"] - fo_r["offset"]; d_exp = fo_w["exponent"] - fo_r["exponent"]
            for cond, f_ in (("rest", fo_r), ("walking", fo_w)):
                long.append(dict(subject=s, cohort=LABEL[cohort[s]], region=region, condition=cond, band="all",
                                 measure="aperiodic_offset", value=f_["offset"]))
                long.append(dict(subject=s, cohort=LABEL[cohort[s]], region=region, condition=cond, band="all",
                                 measure="aperiodic_exponent", value=f_["exponent"]))
            for band, (lo, hi) in bands.items():
                m = band_mask(fr_r, band, lo, hi); mlf = np.log10(fr_r[m]).mean(); fc = (lo + hi) / 2
                tot_r, tot_w = p_r[m].mean(), p_w[m].mean()                       # band-mean log10 power
                ap_r = fo_r["offset"] - fo_r["exponent"] * mlf                    # band-mean aperiodic fit
                ap_w = fo_w["offset"] - fo_w["exponent"] * mlf
                apc_r = fo_r["offset"] - fo_r["exponent"] * np.log10(fc)          # aperiodic at band centre
                apc_w = fo_w["offset"] - fo_w["exponent"] * np.log10(fc)
                per_r, per_w = fo_r[f"{band}_adj"], fo_w[f"{band}_adj"]           # periodic (adjusted) power
                d_total, d_ap, d_per = 10 * (tot_w - tot_r), 10 * (ap_w - ap_r), 10 * (per_w - per_r)
                # cross-check from the whole-block screen spectra
                fs_, ps_r = screen[(s, region, "baseline")]; _, ps_w = screen[(s, region, "pre8walk")]
                ms = band_mask(fs_, band, lo, hi); d_total_screen = 10 * (ps_w[ms].mean() - ps_r[ms].mean())
                wide.append(dict(
                    subject=s, cohort=LABEL[cohort[s]], region=region, band=band,
                    tfr_offset_db=tfr[(s, region, band)],
                    tfr_offset_nc_db=tfr_nc[(s, region, band)],
                    tfr_nc_minus_files_db=tfr_nc[(s, region, band)] - tfr[(s, region, band)],
                    d_aperiodic_offset=d_off, d_aperiodic_exponent=d_exp,
                    d_total_db=d_total, d_aperiodic_db=d_ap, d_periodic_db=d_per,
                    identity_residual_db=d_total - (d_ap + d_per),
                    d_aperiodic_centre_db=10 * (apc_w - apc_r),
                    frac_aperiodic=d_ap / d_total if abs(d_total) > 1e-9 else np.nan,
                    d_total_screen_db=d_total_screen,
                    rest_total_db=10 * tot_r, walk_total_db=10 * tot_w,
                    rest_aperiodic_db=10 * ap_r, walk_aperiodic_db=10 * ap_w,
                    rest_periodic_db=10 * per_r, walk_periodic_db=10 * per_w,
                    rest_offset=fo_r["offset"], walk_offset=fo_w["offset"],
                    rest_exponent=fo_r["exponent"], walk_exponent=fo_w["exponent"],
                    rest_r2=fo_r["r2"], walk_r2=fo_w["r2"]))
                for cond, tot, ap, per in (("rest", tot_r, ap_r, per_r), ("walking", tot_w, ap_w, per_w)):
                    for meas, val in (("total_band_db", 10 * tot), ("aperiodic_band_db", 10 * ap), ("periodic_band_db", 10 * per)):
                        long.append(dict(subject=s, cohort=LABEL[cohort[s]], region=region, condition=cond,
                                         band=band, measure=meas, value=val))

    max_resid = max(abs(w["identity_residual_db"]) for w in wide)
    lines = [f"Aperiodic vs periodic decomposition of the walking-vs-rest offset (variant A, mph 0.15), seed {seed}",
             "All changes = walking minus rest; dB; negative = lower during walking.",
             f"Identity check  d_total = d_aperiodic + d_periodic: max |residual| = {max_resid:.2e} dB", ""]
    summ = []

    def rows_of(region, band, exclude=None):
        return [w for w in wide if w["region"] == region and w["band"] == band and w["subject"] != exclude]

    stale = sorted({w["subject"] for w in wide if abs(w["tfr_nc_minus_files_db"]) > 0.02})
    stale_desc = []
    for s in stale:
        parts = ["%s %s %+.2f" % (w["region"], w["band"], w["tfr_nc_minus_files_db"])
                 for w in wide if w["subject"] == s and abs(w["tfr_nc_minus_files_db"]) > 0.02]
        stale_desc.append("%s (%s)" % (s, ", ".join(parts)))
    lines.append("TFR offset read from the per-subject gait TFR files; the aggregated NetCDF (minus files) differs by "
                 "> 0.02 dB for: " + ("; ".join(stale_desc) or "none"))
    lines.append("")
    comps = [("tfr_offset_db", "TFR offset (gait-cycle mean, dB vs rest; per-subject files)"),
             ("tfr_offset_nc_db", "TFR offset from the aggregated NetCDF (stale for 3 subjects)"),
             ("d_total_db", "PSD total band power change"),
             ("d_aperiodic_db", "  aperiodic component (band-mean fit)"),
             ("d_periodic_db", "  periodic component (adjusted power)"),
             ("d_aperiodic_centre_db", "  aperiodic-only prediction at band centre"),
             ("d_total_screen_db", "PSD total change, whole-block 2-100 Hz screen spectra (cross-check)"),
             ("d_aperiodic_offset", "FOOOF offset change (log10)"),
             ("d_aperiodic_exponent", "FOOOF exponent change")]

    # ---- group means, PD-HC differences ----
    for region in REGIONS:
        for band in BANDS:
            lines.append(f"=== {region} | {band} ===")
            lines.append(f"  {'quantity':66s} {'PD mean [95% CI]':26s} {'HC mean [95% CI]':26s} {'PD-HC [95% CI]':26s} Welch p   g")
            for key, lab in comps:
                rw = rows_of(region, band)
                a = [w[key] for w in rw if w["cohort"] == "PD"]; b = [w[key] for w in rw if w["cohort"] == "HC"]
                ma, la, ua, _ = mean_ci(a); mb, lb, ub, _ = mean_ci(b)
                d, dl, du, t, df, p, g = welch_diff(a, b)
                lines.append(f"  {lab:66s} {ma:+6.2f} [{la:+6.2f}, {ua:+6.2f}]  {mb:+6.2f} [{lb:+6.2f}, {ub:+6.2f}]  "
                             f"{d:+6.2f} [{dl:+6.2f}, {du:+6.2f}]  {p:.3f}  {g:+.2f}")
                summ.append(dict(section="group", region=region, band=band, quantity=key, subset="all",
                                 n_pd=len(a), n_hc=len(b), pd_mean=ma, pd_ci_lo=la, pd_ci_hi=ua,
                                 hc_mean=mb, hc_ci_lo=lb, hc_ci_hi=ub, diff_pd_hc=d, diff_ci_lo=dl, diff_ci_hi=du,
                                 welch_t=t, welch_df=df, welch_p=p, hedges_g=g))
            # share of the total change carried by the aperiodic component (group level and per subject)
            for coh in ("PD", "HC"):
                rw = [w for w in rows_of(region, band) if w["cohort"] == coh]
                tot = np.mean([w["d_total_db"] for w in rw]); ap = np.mean([w["d_aperiodic_db"] for w in rw])
                fr_sub = np.array([w["frac_aperiodic"] for w in rw])
                lines.append(f"  share of mean total change carried by aperiodic, {coh}: {ap / tot:.2f} "
                             f"(per-subject median {np.nanmedian(fr_sub):.2f}, IQR {np.nanpercentile(fr_sub, 25):.2f}-{np.nanpercentile(fr_sub, 75):.2f})")
                summ.append(dict(section="share", region=region, band=band, quantity="frac_aperiodic_of_total", subset=coh,
                                 n_pd=len(rw) if coh == "PD" else 0, n_hc=len(rw) if coh == "HC" else 0,
                                 pd_mean=ap / tot if coh == "PD" else np.nan, hc_mean=ap / tot if coh == "HC" else np.nan,
                                 diff_pd_hc=np.nanmedian(fr_sub)))
            # PD-HC difference of the components: how much of the cohort difference in d_total is aperiodic
            rw = rows_of(region, band)
            dd = {k: np.mean([w[k] for w in rw if w["cohort"] == "PD"]) - np.mean([w[k] for w in rw if w["cohort"] == "HC"])
                  for k in ("d_total_db", "d_aperiodic_db", "d_periodic_db", "tfr_offset_db")}
            lines.append(f"  PD-HC of total change {dd['d_total_db']:+.2f} dB = aperiodic {dd['d_aperiodic_db']:+.2f} + periodic {dd['d_periodic_db']:+.2f} "
                         f"(TFR offset PD-HC {dd['tfr_offset_db']:+.2f} dB)")
            # ---- correlations across the 20 subjects with the TFR offset ----
            x = np.array([w["tfr_offset_db"] for w in rw])
            for key, lab in (("d_aperiodic_db", "aperiodic change"), ("d_periodic_db", "periodic change"),
                             ("d_total_db", "total change"), ("d_aperiodic_offset", "offset change"),
                             ("d_aperiodic_exponent", "exponent change"), ("d_total_screen_db", "total change (screen spectra)")):
                y = np.array([w[key] for w in rw])
                r, pr = stats.pearsonr(x, y); rho, ps = stats.spearmanr(x, y)
                lines.append(f"  corr(TFR offset, {lab:30s}) n={len(x)}: Pearson r={r:+.2f} p={pr:.3f}; Spearman rho={rho:+.2f} p={ps:.3f}")
                summ.append(dict(section="correlation", region=region, band=band, quantity=f"tfr_offset_vs_{key}", subset="all",
                                 n_pd=10, n_hc=10, pearson_r=r, pearson_p=pr, spearman_rho=rho, spearman_p=ps))
            # within-cohort correlations with the aperiodic and periodic change
            for coh in ("PD", "HC"):
                rc = [w for w in rw if w["cohort"] == coh]; xc = np.array([w["tfr_offset_db"] for w in rc])
                for key, lab in (("d_aperiodic_db", "aperiodic change"), ("d_periodic_db", "periodic change")):
                    yc = np.array([w[key] for w in rc]); r, pr = stats.pearsonr(xc, yc)
                    lines.append(f"    within {coh}: corr(TFR offset, {lab}) r={r:+.2f} p={pr:.3f}")
                    summ.append(dict(section="correlation", region=region, band=band, quantity=f"tfr_offset_vs_{key}", subset=coh,
                                     n_pd=len(rc) if coh == "PD" else 0, n_hc=len(rc) if coh == "HC" else 0, pearson_r=r, pearson_p=pr))
            lines.append("")

    # ---- robustness: cohort comparison of the TFR offset and of the components without P4-2007 ----
    lines.append("=== Robustness: PD vs HC without P4-2007 (near-flat rest spectrum) and without P4-2003 (lowest FOOOF R2, "
                 "bad motor channel in the stale NetCDF) ===")
    for region in REGIONS:
        for band in BANDS:
            for key, lab in (("tfr_offset_db", "TFR offset (files)"), ("tfr_offset_nc_db", "TFR offset (NetCDF)"),
                             ("d_total_db", "PSD total change"),
                             ("d_aperiodic_db", "aperiodic component"), ("d_periodic_db", "periodic component")):
                for subset, excl in (("all", None), ("without_P4-2007", EXCLUDE_ROBUST), ("without_P4-2003", "P4-2003")):
                    rw = rows_of(region, band, excl)
                    a = [w[key] for w in rw if w["cohort"] == "PD"]; b = [w[key] for w in rw if w["cohort"] == "HC"]
                    d, dl, du, t, df, p, g = welch_diff(a, b)
                    lines.append(f"  {region:8s}{band:10s}{lab:22s}{subset:16s} n={len(a)}/{len(b)}: PD-HC {d:+.2f} "
                                 f"[{dl:+.2f}, {du:+.2f}] p={p:.3f} g={g:+.2f}")
                    summ.append(dict(section="robustness", region=region, band=band, quantity=key, subset=subset,
                                     n_pd=len(a), n_hc=len(b), pd_mean=np.mean(a), hc_mean=np.mean(b),
                                     diff_pd_hc=d, diff_ci_lo=dl, diff_ci_hi=du, welch_t=t, welch_df=df, welch_p=p, hedges_g=g))
    lines.append("")

    # ---- write tables ----
    wide_csv = out_dir / f"aperiodic_offset_per_subject_{DATE}.csv"
    long_csv = out_dir / f"aperiodic_offset_long_{DATE}.csv"
    summ_csv = out_dir / f"aperiodic_offset_decomposition_{DATE}.csv"
    for p_, rows in ((wide_csv, wide), (long_csv, long)):
        with p_.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        write_sidecar(p_, meta)
    fields = []
    for r in summ:
        for k in r:
            if k not in fields:
                fields.append(k)
    with summ_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summ)
    write_sidecar(summ_csv, meta)
    (out_dir / f"aperiodic_offset_decomposition_{DATE}.txt").write_text("\n".join(lines) + "\n")

    # ---- figure: scatter per region x band + group bars of the components ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"PD": "C3", "HC": "C0"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for i, region in enumerate(REGIONS):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]; rw = rows_of(region, band)
            for coh in ("PD", "HC"):
                rc = [w for w in rw if w["cohort"] == coh]
                ax.scatter([w["d_aperiodic_db"] for w in rc], [w["tfr_offset_db"] for w in rc], color=col[coh], label=f"{coh} (n={len(rc)})", s=40)
                ax.scatter([w["d_periodic_db"] for w in rc], [w["tfr_offset_db"] for w in rc], facecolors="none",
                           edgecolors=col[coh], marker="s", s=40, label=f"{coh}, periodic" if coh == "PD" else None)
            x = np.array([w["d_aperiodic_db"] for w in rw]); y = np.array([w["tfr_offset_db"] for w in rw])
            r, pr = stats.pearsonr(x, y); sl, ic = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 10)
            ax.plot(xx, sl * xx + ic, color="k", lw=0.8)
            lo_, hi_ = min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1
            ax.plot([lo_, hi_], [lo_, hi_], color="grey", lw=0.5, ls=":")
            ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel("walking - rest, PSD component (dB)\nfilled = aperiodic, open squares = periodic")
            ax.set_ylabel("TFR offset, gait-cycle mean (dB vs rest)")
            ax.set_title(f"{region}, {band}\nTFR offset vs aperiodic change: r = {r:+.2f}, p = {pr:.3f}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
        ax = axes[i, 2]; xpos = np.arange(len(BANDS)); wdt = 0.18
        for k, (key, lab) in enumerate((("d_aperiodic_db", "aperiodic"), ("d_periodic_db", "periodic"))):
            for c_i, coh in enumerate(("PD", "HC")):
                vals = [[w[key] for w in rows_of(region, b) if w["cohort"] == coh] for b in BANDS]
                ms = [mean_ci(v) for v in vals]
                off = (k * 2 + c_i - 1.5) * wdt
                ax.bar(xpos + off, [m[0] for m in ms], wdt, yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]],
                       color=col[coh], alpha=1.0 if key == "d_aperiodic_db" else 0.4, capsize=3, label=f"{coh} {lab}")
        tfr_ms = {coh: [mean_ci([w["tfr_offset_db"] for w in rows_of(region, b) if w["cohort"] == coh]) for b in BANDS] for coh in ("PD", "HC")}
        for c_i, coh in enumerate(("PD", "HC")):
            ax.scatter(xpos + (c_i - 0.5) * 0.12, [m[0] for m in tfr_ms[coh]], marker="_", s=400, color=col[coh], linewidths=2,
                       label=f"{coh} TFR offset", zorder=5)
        ax.axhline(0, color="grey", lw=0.5); ax.set_xticks(xpos); ax.set_xticklabels(BANDS)
        ax.set_ylabel("walking - rest (dB), mean and 95 % CI"); ax.set_title(f"{region}: components of the total change")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig_path = fig_dir / "aperiodic_offset_decomposition.png"
    fig.savefig(fig_path, dpi=130); plt.close(fig); write_sidecar(fig_path, meta)

    print("\n".join(lines))
    print(f"wrote {wide_csv.name}, {long_csv.name}, {summ_csv.name}, {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
