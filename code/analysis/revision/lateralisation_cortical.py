"""Stepping-related lateralisation of sensorimotor gait modulation (PD vs HC) and STN.

Purpose:  All cortical gait-cycle analyses so far pooled the motor ROI
          ("Both" side) and found similar gait-phase modulation in PD and HC.
          This script asks whether the hemispheric lateralisation of motor
          cortical activity that should alternate with the swing leg is present
          (one-sample cluster tests per cohort), whether it differs between
          cohorts (two-sample cluster test), and whether a band-wise swing-
          asymmetry index differs between cohorts. The same logic is applied
          to the unilateral STN recordings after contralateral realignment.
Inputs:   data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg.nc           (basenorm, dB vs rest)
          data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg-avggait.nc   (avggait, per-cycle-centred)
          data/derivatives/tfr_gavg/pre8walk-dbs-tfr-gavg-avggait.nc   (STN, avggait)
          data/derivatives/tfr_gavg/pre8walk-dbs-tfr-gavg.nc           (STN, basenorm)
          code/analysis/config.yaml (repro.seed, freq_bands, connectivity.channels_of_interest)
          code/analysis/variables.py (gait_phases: IDS 0-10 %, LLS 10-50 %, FDS 50-60 %, RLS 60-100 %)
          Test functions imported unchanged from code/analysis/cluster_stats_tfr.py;
          phase/band slicers imported unchanged from code/analysis/compute_gerd_table.py.
Outputs:  data/derivatives/tfr_gavg/revision_lat_<...>.nc
              per-subject lateralisation maps and cluster-test results (T_obs, sig_mask)
          outputs/stats/revision_2026-10/lateralisation_cluster_<date>.csv       one row per cluster test
          outputs/stats/revision_2026-10/lateralisation_band_subject_<date>.csv  per subject x band x phase values
          outputs/stats/revision_2026-10/lateralisation_band_tests_<date>.csv    band-wise group tests
          outputs/stats/revision_2026-10/lateralisation_report_<date>.md         methods, tables, wording
          outputs/figures/revision_2026-10/lat_<...>.png
Author:   Lena
Date:     2026-10-14
Seed:     42 (config repro.seed via _common.setup_run; passed to MNE's permutation seed)

Description:
  Cortical. For every subject and normalisation the lateralisation map is
      L(f, t) = Right motor (C2, C4) minus Left motor (C1, C3), in dB.
  Under contralateral control L should be positive-going while the left leg
  swings (LLS, 10-50 %: right cortex controls the left leg) and negative-going
  while the right leg swings (RLS, 60-100 %). A "contralateral-minus-
  ipsilateral" map C(f, t) is built by sign-realignment: C = L in LLS,
  C = -L in RLS; both double-support phases (IDS 0-10 %, FDS 50-60 %) keep
  L as is (no swing leg) and are reported separately in the band tables.
  Tests with subjects as exchangeable units (tail 0, MNE default cluster-
  forming threshold, --n-perm permutations):
      lat_eeg_<norm>_1samp_<cohort>        L != 0 per cohort
      lat_eeg_<norm>_PDvsHC                L differs between cohorts
      lat_eeg_<norm>_contraipsi_1samp_<cohort>   C != 0 per cohort
      lat_eeg_<norm>_contraipsi_PDvsHC     C differs between cohorts
  Band-wise summary (theta, alpha, low beta, high beta, gamma; phases IDS,
  LLS, FDS, RLS with the manuscript's inclusive-endpoint windows): per
  subject the lateralisation index LI = mean L per band x phase, and the
  swing-asymmetry index SAI = LI(LLS) - LI(RLS). PD vs HC per band: Welch t,
  Mann-Whitney U, Hedges g, BH across the five bands. Each cohort's SAI vs 0:
  one-sample t and Wilcoxon, BH across bands.

  STN. One hemisphere per PD subject (4 left, 6 right). Right-STN maps keep
  the right-heel-strike anchor (0 % = ipsilateral heel strike); left-STN maps
  are circularly shifted by 50 % of the cycle so that 0 % = left heel strike,
  as in the manuscript's ipsi/contra frame (build_stn_ipsicontra_master.py).
  After realignment 10-50 % = contralateral-leg swing and 60-100 % =
  ipsilateral-leg swing for every subject. One-sample cluster test on the
  pooled realigned maps (n = 10) and band-wise swing index
  SI = mean(contralateral swing) - mean(ipsilateral swing) vs 0.

  Isolated all-NaN frequency rows in a subject's map (the aggregator maps +-inf
  to NaN) would be linearly interpolated along frequency before testing and
  logged in the .txt; the regenerated aggregates of 2026-10-14 contain none.
  The EEG aggregates regenerated on 2026-10-14 (10:26) are used: the 2026-05-19
  versions carried a corrupt P4-2003 motor Right slab (about +95 dB) that also
  corrupted motor Both. The script records the input files' modification times
  in the sidecars and the report.

  The netCDF/HDF5 stack has segfaulted once when two heavy parts ran in one
  process; the script therefore runs one part per invocation.

Usage:
  P=python
  $P code/analysis/revision/lateralisation_cortical.py --part eeg --norm basenorm
  $P code/analysis/revision/lateralisation_cortical.py --part eeg --norm avggait
  $P code/analysis/revision/lateralisation_cortical.py --part stn
  $P code/analysis/revision/lateralisation_cortical.py --part report
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402
from cluster_stats_tfr import _run_one, _run_one_1samp  # noqa: E402
from compute_gerd_table import _band_freq_slice, _phase_time_slice  # noqa: E402
from variables import gait_phases as GAIT_PHASES  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"
COHORT_TAG = {PD: "PD", HC: "HC"}
PHASES = ("IDS", "LLS", "FDS", "RLS")
SWING = {"LLS": +1.0, "RLS": -1.0}          # sign that turns L = R - L into contra - ipsi
OUT_DIR = PROJECT_ROOT / "outputs/stats/revision_2026-10"
FIG_DIR = PROJECT_ROOT / "outputs/figures/revision_2026-10"
DATE = "2026-10-14"


# ----------------------------------------------------------------------------- helpers
def _pct(times: np.ndarray) -> np.ndarray:
    return 100.0 * (times - times[0]) / (times[-1] - times[0])


def _fill_nan_rows(M: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Linearly interpolate (along frequency) isolated all-NaN frequency rows of
    per-subject maps (n_subj, F, T). Needed because the aggregator sets +-inf
    cells to NaN and one subject carries whole NaN rows at single frequencies,
    which would otherwise leave NaN t-values that split clusters. Returns the
    filled array and a log of what was filled."""
    M = M.copy(); log = []
    for i in range(M.shape[0]):
        bad = np.isnan(M[i]).any(axis=1)
        if not bad.any():
            continue
        good = np.where(~bad)[0]
        for t in range(M.shape[2]):
            col = M[i, :, t]
            nan = np.isnan(col)
            if nan.any():
                col[nan] = np.interp(freqs[nan], freqs[good], col[good])
        log.append(f"subject index {i}: interpolated rows at {freqs[bad].tolist()} Hz")
    return M, log


def _contra_ipsi(L: np.ndarray, n_times: int) -> np.ndarray:
    """Sign-realign L (…, time) so that swing phases read contra - ipsi."""
    C = L.copy()
    for ph, sgn in SWING.items():
        sl = _phase_time_slice(GAIT_PHASES[ph], n_times)
        C[..., sl] = sgn * L[..., sl]
    # phase windows overlap by one sample (inclusive endpoints); the shared boundary
    # samples (idx 110 and 132 for 220 samples) are left unflipped as double support.
    sl_fds = _phase_time_slice(GAIT_PHASES["FDS"], n_times)
    C[..., sl_fds] = L[..., sl_fds]
    return C


def _band_phase_means(maps: np.ndarray, freqs: np.ndarray, bands: dict, n_times: int,
                      phase_ranges: dict) -> dict:
    """Mean over band freqs x phase samples; maps shape (n_subj, F, T)."""
    out = {}
    for band, (lo, hi) in bands.items():
        fs = _band_freq_slice((lo, hi), freqs)
        for ph, rng in phase_ranges.items():
            ts = _phase_time_slice(rng, n_times)
            out[(band, ph)] = np.nanmean(maps[:, fs, ts], axis=(1, 2))
    return out


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float(J * (a.mean() - b.mean()) / sp)


def _describe_clusters(T_obs, sig, freqs, times_pct, diff=None) -> str:
    desc = []
    if sig.any():
        # label positive and negative cells separately so adjacent clusters of
        # opposite sign are not merged into one blob
        pieces = []
        for part in (sig & (T_obs > 0), sig & (T_obs < 0)):
            lab, n = ndimage.label(part)
            pieces += [lab == k for k in range(1, n + 1)]
        for m in pieces:
            fi, ti = np.where(m)
            if diff is not None:
                d = diff[m].mean()
                sign = f"{'PD>HC' if d > 0 else 'PD<HC'} ({d:+.2f} dB)"
            else:
                sign = f"{'pos' if T_obs[m].mean() > 0 else 'neg'} (mean t {T_obs[m].mean():+.2f})"
            desc.append(f"{freqs[fi.min()]:.0f}-{freqs[fi.max()]:.0f} Hz, "
                        f"{times_pct[ti.min()]:.0f}-{times_pct[ti.max()]:.0f} % cycle, {sign}, {m.sum()} px")
    return "; ".join(desc)


def _save_cluster_nc(path: Path, T_obs, sig, freqs, times, attrs: dict) -> None:
    ds = xr.Dataset({"T_obs": (("freq", "time"), T_obs), "sig_mask": (("freq", "time"), sig)},
                    coords={"freq": freqs, "time": times}, attrs=attrs)
    ds.to_netcdf(path)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


# ----------------------------------------------------------------------------- plotting
def _imshow(ax, M, freqs, times_pct, vlim, cmap="RdBu_r", sig=None, anchors=(10, 50, 60)):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim,
                   extent=[times_pct[0], times_pct[-1], freqs[0], freqs[-1]])
    if sig is not None and sig.any():
        ax.contour(times_pct, freqs, sig.astype(float), levels=[0.5], colors="k", linewidths=1.2)
    for x in anchors:
        ax.axvline(x, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Frequency (Hz)")
    return im


def _plot_maps(panels: list[dict], freqs, times_pct, out_png, suptitle, phase_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        im = _imshow(ax, p["M"], freqs, times_pct, p["vlim"], p.get("cmap", "RdBu_r"), p.get("sig"))
        ax.set_title(p["title"], fontsize=9)
        fig.colorbar(im, ax=ax, label=p.get("cbar", "dB"))
        if phase_labels:
            for lab, x in phase_labels.items():
                ax.text(x, freqs[-1] - 1.0, lab, ha="center", va="top", fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_band_bars(df_long: pd.DataFrame, value: str, cohorts: list[str], bands: list[str],
                    phases: tuple, out_png: str, title: str, ylabel: str):
    """Band x phase bars (mean +- SEM over subjects), one panel per cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.0 * len(cohorts), 3.6), sharey=True, squeeze=False)
    w = 0.8 / len(phases)
    for ax, coh in zip(axes[0], cohorts):
        d = df_long[df_long.cohort == coh]
        for j, ph in enumerate(phases):
            m = [d[(d.band == b) & (d.phase == ph)][value].mean() for b in bands]
            s = [d[(d.band == b) & (d.phase == ph)][value].sem() for b in bands]
            ax.bar(np.arange(len(bands)) + (j - (len(phases) - 1) / 2) * w, m, w, yerr=s, capsize=2, label=ph)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels(bands, rotation=20, fontsize=8)
        ax.set_title(f"{coh} (n = {d.subject.nunique()})", fontsize=9)
        ax.set_ylabel(ylabel)
    axes[0][0].legend(fontsize=7, title="phase")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_points(df: pd.DataFrame, value: str, cohorts: list[str], bands: list[str], out_png: str,
                 title: str, ylabel: str, ptxt: dict | None = None):
    """Per-subject points per band, cohorts side by side, with cohort mean bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)       # jitter only
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    off = {c: (i - (len(cohorts) - 1) / 2) * 0.3 for i, c in enumerate(cohorts)}
    for coh in cohorts:
        d = df[df.cohort == coh]
        for i, b in enumerate(bands):
            y = d[d.band == b][value].values
            x = i + off[coh] + rng.uniform(-0.06, 0.06, len(y))
            ax.scatter(x, y, s=18, alpha=0.75, label=coh if i == 0 else None)
            ax.hlines(np.mean(y), i + off[coh] - 0.12, i + off[coh] + 0.12, colors="k", lw=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands)
    if ptxt:
        for i, b in enumerate(bands):
            ax.text(i, ax.get_ylim()[1], ptxt.get(b, ""), ha="center", va="top", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- EEG part
def run_eeg(norm: str, args, config, paths, meta, seed) -> None:
    g = paths.tfr_gavg_dir
    fn = {"basenorm": "pre8walk-eeg-tfr-gavg.nc", "avggait": "pre8walk-eeg-tfr-gavg-avggait.nc"}[norm]
    ds = xr.open_dataset(g / fn)
    import datetime as _dt
    in_mtime = _dt.datetime.fromtimestamp((g / fn).stat().st_mtime).isoformat(timespec="minutes")
    freqs, times = ds.freq.values, ds.time.values
    n_times = len(times)
    times_pct = _pct(times)
    bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    band_names = list(bands)

    right = ds.per_subject.sel(region="motor", side="Right")
    left = ds.per_subject.sel(region="motor", side="Left")
    L_da = (right - left).rename("L")               # (subject, freq, time)
    L, fill_log = _fill_nan_rows(L_da.values, freqs)
    C = _contra_ipsi(L, n_times)
    subjects = ds.subject.values
    cohorts = ds.subject_cohort.values
    keep = ~np.isnan(L).all(axis=(1, 2))
    assert keep.all(), "a subject lacks a motor hemisphere"
    tag = ""
    if args.exclude:
        drop = np.isin(subjects, args.exclude)
        L, C, subjects, cohorts = L[~drop], C[~drop], subjects[~drop], cohorts[~drop]
        tag = "_excl-" + "-".join(args.exclude)

    # persist per-subject maps
    maps = xr.Dataset(
        {"L": (("subject", "freq", "time"), L), "C": (("subject", "freq", "time"), C)},
        coords={"subject": subjects, "freq": freqs, "time": times, "subject_cohort": ("subject", cohorts)},
        attrs=dict(description="L = Right motor (C2,C4) minus Left motor (C1,C3), dB; "
                               "C = contra-minus-ipsi (L in LLS, -L in RLS, L in double support)",
                   normalisation=norm, source=fn, seed=seed, date=DATE))
    maps.to_netcdf(g / f"revision_lat_eeg_{norm}{tag}_maps.nc")

    rows: list[dict] = []
    lines: list[str] = [f"Lateralisation cluster tests, EEG {norm}; n_perm={args.n_perm}, alpha={args.alpha}, seed={seed}",
                        f"input {fn} (modified {in_mtime}); {n_times} samples per cycle; "
                        f"P4-2003 motor Right mean {float(right.sel(subject='P4-2003').mean()):+.2f} dB, "
                        f"Left {float(left.sel(subject='P4-2003').mean()):+.2f} dB"]
    lines += [f"NaN handling: {m.replace('subject index ', '').split(':')[0]} -> {subjects[int(m.split()[2].rstrip(':'))]} {m.split(':', 1)[1].strip()}" for m in fill_log]
    clusters: dict[str, np.ndarray] = {}

    def record(test, data_name, n1, n2, T, sig, cp, diff=None):
        desc = _describe_clusters(T, sig, freqs, times_pct, diff)
        minp = float(min(cp)) if cp else float("nan")
        n_sig = int(sum(p < args.alpha for p in cp))
        rows.append(dict(test=test, normalisation=norm, region="motor", map_name=data_name, n_group1=n1, n_group2=n2,
                         n_clusters=len(cp), n_significant=n_sig, min_cluster_p=minp, significant_clusters=desc))
        lines.append(f"{test:40s} n={n1}/{n2 or '-'} clusters={len(cp):3d} sig={n_sig} min p={minp:.4f}  {desc}")
        _save_cluster_nc(g / f"revision_lat_cluster_{test}{tag}.nc", T, sig, freqs, times,
                         dict(test=test, normalisation=norm, map_name=data_name, excluded=",".join(args.exclude), n_permutations=args.n_perm,
                              alpha=args.alpha, seed=seed, cluster_p=str(cp)))
        clusters[test] = sig

    for data_name, M in (("L", L), ("C", C)):
        ctag = "" if data_name == "L" else "contraipsi_"
        d = {coh: M[cohorts == coh] for coh in (PD, HC)}
        for coh in (PD, HC):
            T, sig, cp = _run_one_1samp(d[coh], args.n_perm, args.alpha, seed)
            record(f"lat_eeg_{norm}_{ctag}1samp_{COHORT_TAG[coh]}", data_name, len(d[coh]), 0, T, sig, cp)
        T, sig, cp = _run_one(d[PD], d[HC], args.n_perm, args.alpha, seed)
        record(f"lat_eeg_{norm}_{ctag}PDvsHC", data_name, len(d[PD]), len(d[HC]), T, sig, cp,
               diff=d[PD].mean(0) - d[HC].mean(0))

    # ---- band-wise summary ----
    subj_rows = []
    for data_name, M in (("L", L), ("C", C)):
        bp = _band_phase_means(M, freqs, bands, n_times, GAIT_PHASES)
        for (band, ph), vals in bp.items():
            for s, coh, v in zip(subjects, cohorts, vals):
                subj_rows.append(dict(normalisation=norm, map_name=data_name, subject=s, cohort=COHORT_TAG[coh],
                                      band=band, phase=ph, value=float(v)))
    df_subj = pd.DataFrame(subj_rows)
    # swing-asymmetry index from L: LI(LLS) - LI(RLS)
    piv = df_subj[df_subj["map_name"] == "L"].pivot_table(index=["subject", "cohort", "band"], columns="phase", values="value").reset_index()
    piv["SAI"] = piv["LLS"] - piv["RLS"]
    piv["DS_diff"] = piv["IDS"] - piv["FDS"]      # double-support asymmetry (post-RHS minus post-LHS)
    piv.insert(0, "normalisation", norm)

    test_rows = []
    for metric in ("SAI", "DS_diff", "IDS", "LLS", "FDS", "RLS"):
        # PD vs HC per band
        tmp = []
        for b in band_names:
            a = piv[(piv.cohort == "PD") & (piv.band == b)][metric].values
            c = piv[(piv.cohort == "HC") & (piv.band == b)][metric].values
            tw = stats.ttest_ind(a, c, equal_var=False)
            mw = stats.mannwhitneyu(a, c, alternative="two-sided")
            tmp.append(dict(normalisation=norm, metric=metric, comparison="PD_vs_HC", band=b,
                            n_PD=len(a), n_HC=len(c), mean_PD=a.mean(), sd_PD=a.std(ddof=1),
                            mean_HC=c.mean(), sd_HC=c.std(ddof=1),
                            welch_t=tw.statistic, welch_df=tw.df, p_welch=tw.pvalue,
                            mann_whitney_U=mw.statistic, p_mw=mw.pvalue, hedges_g=_hedges_g(a, c)))
        pw = _bh([r["p_welch"] for r in tmp]); pm = _bh([r["p_mw"] for r in tmp])
        for r, q1, q2 in zip(tmp, pw, pm):
            r["p_welch_BH"] = q1; r["p_mw_BH"] = q2
        test_rows += tmp
        # each cohort vs zero
        for coh in ("PD", "HC"):
            tmp = []
            for b in band_names:
                a = piv[(piv.cohort == coh) & (piv.band == b)][metric].values
                t1 = stats.ttest_1samp(a, 0.0)
                try:
                    wz = stats.wilcoxon(a)
                    wstat, wp = wz.statistic, wz.pvalue
                except ValueError:
                    wstat, wp = np.nan, np.nan
                tmp.append(dict(normalisation=norm, metric=metric, comparison=f"{coh}_vs_0", band=b,
                                n_PD=len(a) if coh == "PD" else np.nan, n_HC=len(a) if coh == "HC" else np.nan,
                                mean_PD=a.mean() if coh == "PD" else np.nan, sd_PD=a.std(ddof=1) if coh == "PD" else np.nan,
                                mean_HC=a.mean() if coh == "HC" else np.nan, sd_HC=a.std(ddof=1) if coh == "HC" else np.nan,
                                welch_t=t1.statistic, welch_df=len(a) - 1, p_welch=t1.pvalue,
                                mann_whitney_U=wstat, p_mw=wp, hedges_g=float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) else np.nan))
            pw = _bh([r["p_welch"] for r in tmp]); pm = _bh([r["p_mw"] for r in tmp])
            for r, q1, q2 in zip(tmp, pw, pm):
                r["p_welch_BH"] = q1; r["p_mw_BH"] = q2
            test_rows += tmp
    df_tests = pd.DataFrame(test_rows)

    # ---- write ----
    pd.DataFrame(rows).to_csv(OUT_DIR / f"lateralisation_cluster_eeg_{norm}{tag}_{DATE}.csv", index=False)
    df_subj.to_csv(OUT_DIR / f"lateralisation_band_subject_eeg_{norm}{tag}_{DATE}.csv", index=False)
    piv.to_csv(OUT_DIR / f"lateralisation_band_index_eeg_{norm}{tag}_{DATE}.csv", index=False)
    df_tests.to_csv(OUT_DIR / f"lateralisation_band_tests_eeg_{norm}{tag}_{DATE}.csv", index=False)
    (OUT_DIR / f"lateralisation_cluster_eeg_{norm}{tag}_{DATE}.txt").write_text("\n".join(lines) + "\n")
    write_sidecar(OUT_DIR / f"lateralisation_cluster_eeg_{norm}{tag}_{DATE}.csv",
                  {**meta, "part": "eeg", "normalisation": norm, "excluded": args.exclude, "nan_rows_interpolated": fill_log, "input_modified": in_mtime, "n_times": int(n_times), "n_permutations": args.n_perm, "alpha": args.alpha,
                   "input": fn, "n_PD": int((cohorts == PD).sum()), "n_HC": int((cohorts == HC).sum())})
    print("\n".join(lines))

    # ---- figures ----
    if args.no_figs:
        return
    phase_labels = {"IDS": 5, "LLS": 30, "FDS": 55, "RLS": 80}
    mPD, mHC = np.nanmean(L[cohorts == PD], 0), np.nanmean(L[cohorts == HC], 0)
    vlim = float(np.nanpercentile(np.abs(np.concatenate([mPD, mHC])), 99))
    _plot_maps([dict(M=mPD, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_1samp_PD"], title="PD: L = Right - Left motor"),
                dict(M=mHC, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_1samp_HC"], title="HC: L = Right - Left motor"),
                dict(M=mPD - mHC, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_PDvsHC"], title="PD - HC (outline: two-sample cluster)")],
               freqs, times_pct, FIG_DIR / f"lat_eeg_{norm}{tag}_maps_L.png",
               f"Motor-cortex lateralisation maps, {norm} (outlines: p < {args.alpha}, {args.n_perm} perm.)", phase_labels)
    cPD, cHC = np.nanmean(C[cohorts == PD], 0), np.nanmean(C[cohorts == HC], 0)
    _plot_maps([dict(M=cPD, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_contraipsi_1samp_PD"], title="PD: contra - ipsi (swing-realigned)"),
                dict(M=cHC, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_contraipsi_1samp_HC"], title="HC: contra - ipsi (swing-realigned)"),
                dict(M=cPD - cHC, vlim=vlim, sig=clusters[f"lat_eeg_{norm}_contraipsi_PDvsHC"], title="PD - HC")],
               freqs, times_pct, FIG_DIR / f"lat_eeg_{norm}{tag}_maps_C.png",
               f"Contra-minus-ipsi motor maps, {norm} (double support unflipped)", phase_labels)
    _plot_band_bars(df_subj[df_subj["map_name"] == "L"], "value", ["PD", "HC"], band_names, PHASES,
                    FIG_DIR / f"lat_eeg_{norm}{tag}_band_phase_bars.png",
                    f"Lateralisation index (Right - Left motor) by band and phase, {norm}", "L (dB), mean +- SEM")
    ptxt = {r.band: f"p={r.p_welch:.2f}" for r in df_tests[(df_tests.metric == "SAI") & (df_tests.comparison == "PD_vs_HC")].itertuples()}
    _plot_points(piv, "SAI", ["PD", "HC"], band_names, FIG_DIR / f"lat_eeg_{norm}{tag}_swing_asymmetry_points.png",
                 f"Swing-asymmetry index L(LLS) - L(RLS) per subject, {norm} (Welch p, uncorrected)", "SAI (dB)", ptxt)


# ----------------------------------------------------------------------------- STN part
def run_stn(args, config, paths, meta, seed) -> None:
    g = paths.tfr_gavg_dir
    bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    band_names = list(bands)
    rows, lines, subj_rows, test_rows = [], [f"STN realigned cluster tests; n_perm={args.n_perm}, alpha={args.alpha}, seed={seed}"], [], []
    realigned_phases = {"post_HSI": GAIT_PHASES["IDS"], "contra_swing": GAIT_PHASES["LLS"],
                        "post_HSC": GAIT_PHASES["FDS"], "ipsi_swing": GAIT_PHASES["RLS"]}

    for norm in ("avggait", "basenorm"):
        fn = {"basenorm": "pre8walk-dbs-tfr-gavg.nc", "avggait": "pre8walk-dbs-tfr-gavg-avggait.nc"}[norm]
        ds = xr.open_dataset(g / fn)
        freqs, times = ds.freq.values, ds.time.values
        n_times = len(times)
        times_pct = _pct(times)
        shift = int(round(0.5 * n_times))          # 275 samples -> 138 (50.2 % of the cycle)
        subs, sides, arr = [], [], []
        for s in ds.subject.values:
            if str(ds.subject_cohort.sel(subject=s).values) != PD:
                continue
            for side in ("Left", "Right"):
                v = ds.per_subject.sel(subject=s, region="STN", side=side).values
                if np.isnan(v).all():
                    continue
                if side == "Left":
                    v = np.roll(v, -shift, axis=-1)     # new 0 % = left heel strike (HSI for a left STN)
                subs.append(str(s)); sides.append(side); arr.append(v)
        M = np.stack(arr)
        xr.Dataset({"realigned": (("subject", "freq", "time"), M)},
                   coords={"subject": subs, "freq": freqs, "time": times, "hemisphere": ("subject", sides)},
                   attrs=dict(description="STN maps in the ipsi/contra frame: 0 % = ipsilateral heel strike; "
                                          "left-STN maps rolled by -shift samples", shift_samples=shift,
                              normalisation=norm, seed=seed, date=DATE)).to_netcdf(g / f"revision_lat_stn_{norm}_realigned.nc")

        T, sig, cp = _run_one_1samp(M, args.n_perm, args.alpha, seed)
        desc = _describe_clusters(T, sig, freqs, times_pct)
        minp = float(min(cp)) if cp else float("nan"); n_sig = int(sum(p < args.alpha for p in cp))
        test = f"lat_stn_{norm}_realigned_1samp_pooled"
        rows.append(dict(test=test, normalisation=norm, region="STN", map_name="realigned", n_group1=len(M), n_group2=0,
                         n_clusters=len(cp), n_significant=n_sig, min_cluster_p=minp, significant_clusters=desc))
        lines.append(f"{test:40s} n={len(M)} (L={sides.count('Left')}, R={sides.count('Right')}) clusters={len(cp):3d} sig={n_sig} min p={minp:.4f}  {desc}")
        _save_cluster_nc(g / f"revision_lat_cluster_{test}.nc", T, sig, freqs, times,
                         dict(test=test, normalisation=norm, n_permutations=args.n_perm, alpha=args.alpha, seed=seed, cluster_p=str(cp)))

        # band x realigned phase, swing index
        bp = _band_phase_means(M, freqs, bands, n_times, realigned_phases)
        for (band, ph), vals in bp.items():
            for s, sd, v in zip(subs, sides, vals):
                subj_rows.append(dict(normalisation=norm, subject=s, hemisphere=sd, band=band, phase=ph, value=float(v)))
        df = pd.DataFrame([r for r in subj_rows if r["normalisation"] == norm])
        piv = df.pivot_table(index=["subject", "hemisphere", "band"], columns="phase", values="value").reset_index()
        piv["SI"] = piv["contra_swing"] - piv["ipsi_swing"]
        piv["DS_diff"] = piv["post_HSI"] - piv["post_HSC"]
        piv.insert(0, "normalisation", norm)
        piv.to_csv(OUT_DIR / f"lateralisation_band_index_stn_{norm}_{DATE}.csv", index=False)
        for metric in ("SI", "DS_diff"):
            tmp = []
            for b in band_names:
                a = piv[piv.band == b][metric].values
                t1 = stats.ttest_1samp(a, 0.0)
                try:
                    wz = stats.wilcoxon(a); wstat, wp = wz.statistic, wz.pvalue
                except ValueError:
                    wstat, wp = np.nan, np.nan
                tmp.append(dict(normalisation=norm, metric=metric, comparison="STN_vs_0", band=b, n=len(a),
                                mean=a.mean(), sd=a.std(ddof=1), t=t1.statistic, df=len(a) - 1, p_t=t1.pvalue,
                                wilcoxon_W=wstat, p_wilcoxon=wp, d=float(a.mean() / a.std(ddof=1))))
            pt = _bh([r["p_t"] for r in tmp]); pw = _bh([r["p_wilcoxon"] for r in tmp])
            for r, q1, q2 in zip(tmp, pt, pw):
                r["p_t_BH"] = q1; r["p_wilcoxon_BH"] = q2
            test_rows += tmp

        if not args.no_figs:
            mean_map = M.mean(0)
            vlim = float(np.nanpercentile(np.abs(mean_map), 99))
            _plot_maps([dict(M=mean_map, vlim=vlim, sig=sig, title=f"STN realigned mean (n = {len(M)}), {norm}"),
                        dict(M=T, vlim=float(np.nanpercentile(np.abs(T), 99)), sig=sig, title="t (one-sample)", cbar="t")],
                       freqs, times_pct, FIG_DIR / f"lat_stn_{norm}_realigned_map.png",
                       "STN, ipsi/contra frame: 0 % = ipsilateral heel strike, 10-50 % contra swing, 60-100 % ipsi swing",
                       {"pHSI": 5, "contra": 30, "pHSC": 55, "ipsi": 80})
            dfl = df.copy(); dfl["cohort"] = "PD (STN)"; dfl["subject"] = dfl["subject"]
            _plot_band_bars(dfl, "value", ["PD (STN)"], band_names, tuple(realigned_phases),
                            FIG_DIR / f"lat_stn_{norm}_band_phase_bars.png",
                            f"STN band x realigned phase, {norm}", "dB, mean +- SEM")
            pv = piv.copy(); pv["cohort"] = "PD (STN)"
            ptxt = {r["band"]: f"p={r['p_t']:.2f}" for r in test_rows if r["metric"] == "SI" and r["normalisation"] == norm}
            _plot_points(pv, "SI", ["PD (STN)"], band_names, FIG_DIR / f"lat_stn_{norm}_swing_index_points.png",
                         f"STN swing index contra - ipsi swing per subject, {norm} (t-test p, uncorrected)", "SI (dB)", ptxt)

    pd.DataFrame(rows).to_csv(OUT_DIR / f"lateralisation_cluster_stn_{DATE}.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(OUT_DIR / f"lateralisation_band_subject_stn_{DATE}.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / f"lateralisation_band_tests_stn_{DATE}.csv", index=False)
    (OUT_DIR / f"lateralisation_cluster_stn_{DATE}.txt").write_text("\n".join(lines) + "\n")
    write_sidecar(OUT_DIR / f"lateralisation_cluster_stn_{DATE}.csv",
                  {**meta, "part": "stn", "n_permutations": args.n_perm, "alpha": args.alpha, "shift_samples": shift})
    print("\n".join(lines))


# ----------------------------------------------------------------------------- report part
def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt.get(c, "{:.3f}").format(v) if not np.isnan(v) else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def run_report(args, config, meta) -> None:
    bands = list(config["freq_bands"])
    cl = pd.concat([pd.read_csv(OUT_DIR / f"lateralisation_cluster_eeg_{n}_{DATE}.csv") for n in ("basenorm", "avggait")]
                   + [pd.read_csv(OUT_DIR / f"lateralisation_cluster_stn_{DATE}.csv")], ignore_index=True)
    bt = pd.concat([pd.read_csv(OUT_DIR / f"lateralisation_band_tests_eeg_{n}_{DATE}.csv") for n in ("basenorm", "avggait")], ignore_index=True)
    st = pd.read_csv(OUT_DIR / f"lateralisation_band_tests_stn_{DATE}.csv")
    cl_all = pd.concat([cl], ignore_index=True)
    cl_all.to_csv(OUT_DIR / f"lateralisation_cluster_{DATE}.csv", index=False)
    bt.to_csv(OUT_DIR / f"lateralisation_band_tests_{DATE}.csv", index=False)

    f3 = {"min_cluster_p": "{:.4f}"}
    fb = {"p_welch": "{:.3f}", "p_welch_BH": "{:.3f}", "p_mw": "{:.3f}", "p_mw_BH": "{:.3f}", "hedges_g": "{:.2f}",
          "mean_PD": "{:.2f}", "sd_PD": "{:.2f}", "mean_HC": "{:.2f}", "sd_HC": "{:.2f}", "welch_t": "{:.2f}", "welch_df": "{:.1f}"}
    fs = {"mean": "{:.2f}", "sd": "{:.2f}", "t": "{:.2f}", "p_t": "{:.3f}", "p_t_BH": "{:.3f}", "p_wilcoxon": "{:.3f}",
          "p_wilcoxon_BH": "{:.3f}", "d": "{:.2f}"}
    n_perm = args.n_perm

    md = [f"# Stepping-related lateralisation of motor-cortical gait modulation, and STN contralateral alignment ({DATE})",
          "",
          "Script: `code/analysis/revision/lateralisation_cortical.py` (seed 42; cluster tests imported from "
          "`code/analysis/cluster_stats_tfr.py`; phase and band windows from `compute_gerd_table.py` / `variables.py`).",
          "",
          "## Data provenance",
          "",
          *[f"- `{l}`" for n in ("basenorm", "avggait")
            for l in (OUT_DIR / f"lateralisation_cluster_eeg_{n}_{DATE}.txt").read_text().splitlines()[1:2]],
          "- The EEG aggregates regenerated on 2026-10-14 (10:26) were used. The previous versions (2026-05-19) held a corrupt "
          "P4-2003 motor Right slab (about +95 dB, bad channel before the C2 fix) that also corrupted motor Both; "
          "P4-2003 motor Right now averages within a few dB of zero like the other participants (value above). "
          "DBS aggregates were not regenerated and were used as is.",
          "",
          "## Methods paragraph (draft)",
          "",
          "To test whether sensorimotor gait modulation alternated between hemispheres with the swing leg, we computed for "
          "every participant a lateralisation map L(f, t) = right motor cortex (C2, C4) minus left motor cortex (C1, C3), in dB, "
          "from the gait-warped time-frequency representations (1-45 Hz, 1100 samples over the 1.1 s cycle; right heel strike "
          "at 0 %, left toe-off 10 %, left heel strike 50 %, right toe-off 60 %). This was done for the baseline-normalised "
          "maps (dB relative to rest) and for the average-gait-cycle-normalised maps (per-cycle mean removed). Under "
          "contralateral control L is expected to be positive-going during left-leg swing (10-50 %) and negative-going during "
          "right-leg swing (60-100 %). A contralateral-minus-ipsilateral map C was obtained by keeping L during left-leg swing, "
          "negating it during right-leg swing and leaving the double-support phases unchanged. With participants as units, we "
          f"ran cluster-based permutation tests (MNE, two-tailed, default cluster-forming threshold, {args.n_perm} permutations, "
          "seed 42) of L and C against zero within each cohort and between cohorts (PD vs HC). For a band-wise summary we "
          "averaged L within each frequency band (theta 4-8, alpha 8-12, low beta 12-20, high beta 20-30, gamma 30-45 Hz) and "
          "gait phase (IDS 0-10 %, LLS 10-50 %, FDS 50-60 %, RLS 60-100 %, inclusive-endpoint windows as in the main analysis) "
          "and defined the swing-asymmetry index SAI = L(LLS) - L(RLS). SAI was compared between cohorts per band with Welch's "
          "t-test and the Mann-Whitney U test (Hedges' g as effect size) and against zero within each cohort with one-sample "
          "t and Wilcoxon tests; p-values were Benjamini-Hochberg corrected across the five bands. For the unilateral STN "
          "recordings (4 left, 6 right) the maps were realigned to an ipsilateral/contralateral frame (right STN unchanged; "
          "left STN circularly shifted by half a cycle, so 0 % = ipsilateral heel strike, 10-50 % = contralateral-leg swing, "
          "60-100 % = ipsilateral-leg swing), pooled over patients (n = 10) and tested against zero with the same cluster test; "
          "the band-wise swing index SI = mean(contralateral swing) - mean(ipsilateral swing) was tested against zero.",
          "",
          "## Cluster tests (subjects as units)",
          "",
          "`map_name` L = Right - Left motor; C = contra - ipsi after swing realignment; `realigned` = STN ipsi/contra frame. "
          "Two-sample tests report the direction of PD - HC inside each significant cluster.",
          "",
          _md_table(cl_all, ["test", "normalisation", "map_name", "n_group1", "n_group2", "n_clusters", "n_significant",
                             "min_cluster_p", "significant_clusters"], f3),
          ""]

    for norm in ("basenorm", "avggait"):
        md += [f"## Band-wise cortical results, {norm}", ""]
        d = bt[(bt.normalisation == norm) & (bt.metric == "SAI") & (bt.comparison == "PD_vs_HC")]
        md += [f"### Swing-asymmetry index SAI = L(LLS) - L(RLS), PD vs HC ({norm})", "",
               _md_table(d, ["band", "mean_PD", "sd_PD", "mean_HC", "sd_HC", "welch_t", "welch_df", "p_welch", "p_welch_BH",
                             "p_mw", "p_mw_BH", "hedges_g"], fb), ""]
        for coh in ("PD", "HC"):
            d = bt[(bt.normalisation == norm) & (bt.metric == "SAI") & (bt.comparison == f"{coh}_vs_0")]
            d = d.rename(columns={f"mean_{coh}": "mean", f"sd_{coh}": "sd", "welch_t": "t", "welch_df": "df", "p_welch": "p_t",
                                  "p_welch_BH": "p_t_BH", "p_mw": "p_wilcoxon", "p_mw_BH": "p_wilcoxon_BH", "hedges_g": "d"})
            md += [f"### SAI vs 0, {coh} ({norm}); d = mean/SD", "",
                   _md_table(d, ["band", "mean", "sd", "t", "df", "p_t", "p_t_BH", "p_wilcoxon", "p_wilcoxon_BH", "d"], fs), ""]
        md += [f"### Lateralisation index L per phase, PD vs HC ({norm}); Welch p BH-corrected across bands within phase", ""]
        d = bt[(bt.normalisation == norm) & (bt.metric.isin(PHASES)) & (bt.comparison == "PD_vs_HC")].copy()
        d = d.rename(columns={"metric": "phase"})
        md += [_md_table(d, ["phase", "band", "mean_PD", "sd_PD", "mean_HC", "sd_HC", "p_welch", "p_welch_BH", "hedges_g"], fb), ""]
        d = bt[(bt.normalisation == norm) & (bt.metric == "DS_diff") & (bt.comparison == "PD_vs_HC")]
        md += [f"### Double-support asymmetry L(IDS) - L(FDS), PD vs HC ({norm})", "",
               _md_table(d, ["band", "mean_PD", "sd_PD", "mean_HC", "sd_HC", "p_welch", "p_welch_BH", "hedges_g"], fb), ""]

    # sensitivity runs (any lateralisation_cluster_eeg_<norm>_excl-*.csv present)
    for f in sorted(OUT_DIR.glob(f"lateralisation_cluster_eeg_*_excl-*_{DATE}.csv")):
        stem = f.name[len("lateralisation_cluster_eeg_"):-len(f"_{DATE}.csv")]
        norm_x, excl = stem.split("_excl-")
        md += [f"## Sensitivity: {norm_x} without {excl}", "",
               _md_table(pd.read_csv(f), ["test", "n_group1", "n_group2", "n_clusters", "n_significant", "min_cluster_p", "significant_clusters"], f3), ""]
        bx = pd.read_csv(OUT_DIR / f"lateralisation_band_tests_eeg_{stem}_{DATE}.csv")
        d = bx[(bx.metric == "SAI") & (bx.comparison == "PD_vs_HC")]
        md += [f"### SAI PD vs HC ({norm_x}, without {excl})", "",
               _md_table(d, ["band", "mean_PD", "sd_PD", "mean_HC", "sd_HC", "welch_t", "p_welch", "p_welch_BH", "p_mw", "p_mw_BH", "hedges_g"], fb), ""]
        d = bx[(bx.metric.isin(PHASES)) & (bx.comparison == "PD_vs_HC")].rename(columns={"metric": "phase"})
        md += [f"### L per phase PD vs HC ({norm_x}, without {excl})", "",
               _md_table(d, ["phase", "band", "mean_PD", "sd_PD", "mean_HC", "sd_HC", "p_welch", "p_welch_BH", "hedges_g"], fb), ""]
    md += ["## STN, ipsi/contra frame (n = 10; 4 left, 6 right)", ""]
    for norm in ("avggait", "basenorm"):
        d = st[(st.normalisation == norm) & (st.metric == "SI")]
        md += [f"### Swing index SI = contra swing - ipsi swing vs 0 ({norm})", "",
               _md_table(d, ["band", "n", "mean", "sd", "t", "df", "p_t", "p_t_BH", "p_wilcoxon", "p_wilcoxon_BH", "d"], fs), ""]
        d = st[(st.normalisation == norm) & (st.metric == "DS_diff")]
        md += [f"### Double-support difference post-HSI - post-HSC vs 0 ({norm})", "",
               _md_table(d, ["band", "n", "mean", "sd", "t", "df", "p_t", "p_t_BH", "p_wilcoxon", "p_wilcoxon_BH", "d"], fs), ""]

    md += ["## Figures", "",
           "- `outputs/figures/revision_2026-10/lat_eeg_<norm>_maps_L.png` cohort-mean L maps with one-sample cluster outlines and PD - HC with two-sample outline",
           "- `outputs/figures/revision_2026-10/lat_eeg_<norm>_maps_C.png` contra-minus-ipsi maps",
           "- `outputs/figures/revision_2026-10/lat_eeg_<norm>_band_phase_bars.png` band x phase lateralisation index per cohort",
           "- `outputs/figures/revision_2026-10/lat_eeg_<norm>_swing_asymmetry_points.png` per-subject SAI",
           "- `outputs/figures/revision_2026-10/lat_stn_<norm>_realigned_map.png`, `lat_stn_<norm>_band_phase_bars.png`, `lat_stn_<norm>_swing_index_points.png`",
           "", "## How to state the finding", "",
           "TODO(verify): filled in after inspection of the tables above (see section appended below).", ""]
    (OUT_DIR / f"lateralisation_report_{DATE}.md").write_text("\n".join(md))
    write_sidecar(OUT_DIR / f"lateralisation_report_{DATE}.md", {**meta, "part": "report", "n_permutations": n_perm})
    print(f"wrote {OUT_DIR / f'lateralisation_report_{DATE}.md'}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=["eeg", "stn", "report"], required=True)
    parser.add_argument("--norm", choices=["basenorm", "avggait"], default="basenorm")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-figs", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="subject codes dropped before the EEG tests (sensitivity run; outputs get an _excl-<codes> tag)")
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("lateralisation_cortical", config)
    seed = int(config["repro"]["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.part == "eeg":
        run_eeg(args.norm, args, config, paths, meta, seed)
    elif args.part == "stn":
        run_stn(args, config, paths, meta, seed)
    else:
        run_report(args, config, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
