"""Cluster-based permutation tests on gait-warped TFRs with subjects as units.

Purpose:  Test, at full time-frequency resolution and with participants as the
          exchangeable units, (i) whether gait-cycle-locked cortical activity
          differs between PD and HC, (ii) whether its within-cycle modulation
          (average-gait-cycle normalisation, offsets removed) differs between
          cohorts, and (iii) whether cortical and STN modulation differs from
          zero within each cohort. Complements the random-slope LMMs, which
          bin the cycle into four phases and five bands.
Inputs:   data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg.nc           (basenorm)
          data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg-avggait.nc   (avggait)
          data/derivatives/tfr_gavg/pre8walk-dbs-tfr-gavg-avggait.nc   (STN, avggait)
          data/derivatives/tfr_gavg/pre8walk-dbs-tfr-gavg.nc           (STN, basenorm)
          code/analysis/config.yaml (repro.seed)
          Test functions are imported unchanged from code/analysis/cluster_stats_tfr.py.
Outputs:  data/derivatives/tfr_gavg/revision_cluster_<test>.nc  (T_obs, sig_mask per test)
          outputs/stats/revision_2026-10/cluster_perm_2026-10-13.csv  (one row per test)
          outputs/stats/revision_2026-10/cluster_perm_2026-10-13.txt
          outputs/figures/revision_2026-10/cluster_<test>_<region>_<side>.png
Author:   Lena
Date:     2026-10-13
Seed:     42 (config repro.seed, passed to MNE's permutation seed)

Description:
  Tests (tail = 0, cluster-forming threshold = MNE default, i.e. the
  t-distribution quantile for p = 0.05 given the group sizes; n permutations
  from --n-perm, default 5000):
    eeg_basenorm_PDvsHC   : independent-samples test, PD vs HC, baseline-
                            normalised dB maps (offset + modulation)
    eeg_avggait_PDvsHC    : same on average-gait-cycle-normalised maps
                            (modulation only; the time-frequency analogue of
                            Cohort x GaitPhase)
    eeg_avggait_1samp_<cohort> : one-sample test, modulation != 0 per cohort
    eeg_basenorm_1samp_<cohort>: one-sample test, walking != rest per cohort
    dbs_avggait_1samp     : STN modulation != 0, all PD (hemispheres pooled,
                            in the canonical RHS-anchored alignment) and per side
    dbs_basenorm_1samp    : STN walking != rest, all PD and per side
  Regions: motor and frontal; sides: Both (primary), Left, Right, Central.

Usage:
  python code/analysis/revision/cluster_perm_revision.py
  python code/analysis/revision/cluster_perm_revision.py --n-perm 1000
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402
from cluster_stats_tfr import _run_one, _run_one_1samp, _subject_data  # noqa: E402

REGIONS = ("motor", "frontal")
SIDES = ("Both", "Left", "Right", "Central")
PD, HC = "Patient_NFB", "AgeMatched_NFB"


def _dbs_data(ds: xr.Dataset, side: str | None) -> np.ndarray:
    """STN per-subject slab, pooled over sides (each subject has one) or one side."""
    arr = ds.per_subject.sel(region="STN")
    if side is None:
        # each subject has exactly one non-NaN side; take it
        arr = arr.max(dim="side", skipna=True)        # max of one value = the value
        vals = arr.values
    else:
        vals = arr.sel(side=side).values
    keep = ~np.isnan(vals).all(axis=(1, 2))
    return vals[keep]


def _plot(T_obs, sig_mask, freqs, times, title, out_png, stat="t"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    vmax = np.nanpercentile(np.abs(T_obs), 99)
    vmin, cmap = (0, "Reds") if stat == "F" else (-vmax, "RdBu_r")
    im = ax.imshow(T_obs, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[times[0] * 100, times[-1] * 100, freqs[0], freqs[-1]])
    if sig_mask.any():
        ax.contour(np.linspace(times[0] * 100, times[-1] * 100, T_obs.shape[1]), freqs,
                   sig_mask.astype(float), levels=[0.5], colors="k", linewidths=1.2)
    for x in (10, 50, 60):
        ax.axvline(x, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Gait cycle (%)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label=stat)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("cluster_perm_revision", config)
    seed = int(config["repro"]["seed"])
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    g = paths.tfr_gavg_dir
    ds_eeg = {"basenorm": xr.open_dataset(g / "pre8walk-eeg-tfr-gavg.nc"),
              "avggait": xr.open_dataset(g / "pre8walk-eeg-tfr-gavg-avggait.nc")}
    ds_dbs = {"basenorm": xr.open_dataset(g / "pre8walk-dbs-tfr-gavg.nc"),
              "avggait": xr.open_dataset(g / "pre8walk-dbs-tfr-gavg-avggait.nc")}
    freqs = ds_eeg["basenorm"].freq.values

    rows: list[dict] = []
    lines = [f"Cluster-based permutation tests, {args.date}; n_perm={args.n_perm}, alpha={args.alpha}, seed={seed}", ""]

    def record(test, norm, region, side, n1, n2, T_obs, sig, cluster_p, times, diff=None):
        # Two-sample tests return an F statistic (unsigned); direction comes from
        # `diff` = mean(PD) - mean(HC). One-sample tests return a signed t.
        # time axis: seconds over the 1.1 s warped cycle -> fraction of cycle
        times_pct = (times - times[0]) / (times[-1] - times[0])
        n_sig = int(sum(p < args.alpha for p in cluster_p))
        minp = float(min(cluster_p)) if cluster_p else float("nan")
        # describe significant clusters: freq range, time range, sign
        desc = []
        if sig.any():
            from scipy import ndimage
            lab, n = ndimage.label(sig)
            for k in range(1, n + 1):
                m = lab == k
                fi, ti = np.where(m)
                if diff is not None:
                    sign = f"PD<HC ({diff[m].mean():+.1f} dB)" if diff[m].mean() < 0 else f"PD>HC ({diff[m].mean():+.1f} dB)"
                else:
                    sign = "neg" if T_obs[m].mean() < 0 else "pos"
                desc.append(f"{freqs[fi.min()]:.0f}-{freqs[fi.max()]:.0f} Hz, "
                            f"{times_pct[ti.min()]*100:.0f}-{times_pct[ti.max()]*100:.0f} % cycle, {sign}, "
                            f"{m.sum()} px")
        row = dict(test=test, normalisation=norm, region=region, side=side, n_group1=n1, n_group2=n2,
                   n_clusters=len(cluster_p), n_significant=n_sig, min_cluster_p=minp,
                   significant_clusters="; ".join(desc))
        rows.append(row)
        lines.append(f"{test:28s} {region:8s} {side:8s} n={n1}/{n2 if n2 else '-'}  clusters={len(cluster_p):3d}  "
                     f"sig={n_sig}  min p={minp:.4f}  {'; '.join(desc)}")
        da = xr.Dataset({"T_obs": (("freq", "time"), T_obs), "sig_mask": (("freq", "time"), sig)},
                        coords={"freq": freqs, "time": times},
                        attrs=dict(test=test, region=region, side=side, n_permutations=args.n_perm,
                                   alpha=args.alpha, seed=seed, cluster_p=str(cluster_p)))
        nc = g / f"revision_cluster_{test}_{region}_{side}.nc"
        da.to_netcdf(nc)
        _plot(T_obs, sig, freqs, times_pct, f"{test} {region} {side} (min cluster p={minp:.3f})",
              fig_dir / f"cluster_{test}_{region}_{side}.png", stat="F" if diff is not None else "t")

    # ---- EEG between-cohort tests ----
    for norm in ("basenorm", "avggait"):
        ds = ds_eeg[norm]
        for region in REGIONS:
            for side in SIDES:
                d_pd = _subject_data(ds, PD, region, side); d_hc = _subject_data(ds, HC, region, side)
                T, sig, cp = _run_one(d_pd, d_hc, args.n_perm, args.alpha, seed)
                record(f"eeg_{norm}_PDvsHC", norm, region, side, len(d_pd), len(d_hc), T, sig, cp, ds.time.values,
                       diff=np.nanmean(d_pd, axis=0) - np.nanmean(d_hc, axis=0))
        # ---- EEG one-sample tests per cohort (Both side only) ----
        for cohort, tag in ((PD, "PD"), (HC, "HC")):
            for region in REGIONS:
                d = _subject_data(ds, cohort, region, "Both")
                T, sig, cp = _run_one_1samp(d, args.n_perm, args.alpha, seed)
                record(f"eeg_{norm}_1samp_{tag}", norm, region, "Both", len(d), 0, T, sig, cp, ds.time.values)

    # ---- STN one-sample tests ----
    for norm in ("basenorm", "avggait"):
        ds = ds_dbs[norm]
        for side in (None, "Left", "Right"):
            d = _dbs_data(ds, side)
            if len(d) < 3:
                lines.append(f"dbs_{norm}_1samp STN {side}: n={len(d)}, skipped"); continue
            T, sig, cp = _run_one_1samp(d, args.n_perm, args.alpha, seed)
            record(f"dbs_{norm}_1samp", norm, "STN", side or "pooled", len(d), 0, T, sig, cp, ds.time.values)

    csv_path = out_dir / f"cluster_perm_{args.date}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    txt_path = out_dir / f"cluster_perm_{args.date}.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    write_sidecar(csv_path, {**meta, "n_permutations": args.n_perm, "alpha": args.alpha, "n_tests": len(rows)})
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
