"""Cluster-based permutation tests on the post8walk block (replication of cluster_perm_revision.py).

Purpose:  Repeat the revision cluster tests (cluster_perm_revision.py, which
          hardcodes the pre8walk block) on the second walking block
          (post8walk) of the same session, to check whether (i) the
          within-cycle gait modulation is present and does not differ between
          PD and HC, (ii) the baseline-normalised PD-vs-HC offset in high
          beta / gamma is present, and (iii) the STN shows no within-cycle
          modulation cluster. Test functions, region/side scheme, tail,
          threshold, number of permutations and seed are identical to the
          pre8walk run; only the input block differs (--block).
Inputs:   data/derivatives/tfr_gavg/<block>-eeg-tfr-gavg.nc           (basenorm)
          data/derivatives/tfr_gavg/<block>-eeg-tfr-gavg-avggait.nc   (avggait)
          data/derivatives/tfr_gavg/<block>-dbs-tfr-gavg.nc           (STN, basenorm)
          data/derivatives/tfr_gavg/<block>-dbs-tfr-gavg-avggait.nc   (STN, avggait)
          code/analysis/config.yaml (repro.seed)
          Test functions imported unchanged from code/analysis/cluster_stats_tfr.py;
          STN selection and plotting imported from cluster_perm_revision.py.
Outputs:  data/derivatives/tfr_gavg/revision_cluster_<block>_<test>_<region>_<side>.nc
          outputs/stats/revision_2026-10/cluster_perm_<block>_<date>.csv  (one row per test)
          outputs/stats/revision_2026-10/cluster_perm_<block>_<date>.txt
          outputs/figures/revision_2026-10/cluster_<block>_<test>_<region>_<side>.png
Author:   Lena
Date:     2026-10-14
Seed:     42 (config repro.seed, passed to MNE's permutation seed)

Caveats:
  post8walk derivatives were built from the server `_proc.fif` export (a
  later preprocessing run than the pickle-era pre8walk data), so absolute
  values are not expected to match pre8walk. P4-2012 has no post8walk
  recording (EEG n = 9 PD / 10 HC, STN n = 9).
  HC scaling: the HC post8walk proc fifs store microvolts as volts (1e6 x
  amplitude; outputs/stats/post8walk_quality_2026-08-14.txt), whereas the
  baseline block used for normalisation is the pickle-era export in volts.
  The baseline-normalised HC maps therefore carry a constant
  10*log10(1e12) = +120 dB (observed per-subject medians 116.6-124.0 dB,
  PD -8.8 to +0.7 dB). --hc-correction-db (default 120) is subtracted from
  every HC subject's basenorm map before testing; avggait maps are
  unaffected (the constant is removed by construction).

Usage:
  python code/analysis/revision/cluster_perm_post8walk.py
  python code/analysis/revision/cluster_perm_post8walk.py --block post8walk --n-perm 5000
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
sys.path.insert(0, str(HERE))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402
from cluster_stats_tfr import _run_one, _run_one_1samp, _subject_data  # noqa: E402
from cluster_perm_revision import REGIONS, SIDES, PD, HC, _dbs_data, _plot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", default="post8walk")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--date", default="2026-10-14")
    parser.add_argument("--hc-correction-db", type=float, default=120.0,
                        help="Constant subtracted from HC basenorm EEG maps (microvolt-as-volt "
                             "scaling of the HC post8walk export; see Caveats).")
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("cluster_perm_post8walk", config)
    seed = int(config["repro"]["seed"])
    np.random.seed(seed)
    block = args.block
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    g = paths.tfr_gavg_dir
    ds_eeg = {"basenorm": xr.open_dataset(g / f"{block}-eeg-tfr-gavg.nc"),
              "avggait": xr.open_dataset(g / f"{block}-eeg-tfr-gavg-avggait.nc")}
    ds_dbs = {"basenorm": xr.open_dataset(g / f"{block}-dbs-tfr-gavg.nc"),
              "avggait": xr.open_dataset(g / f"{block}-dbs-tfr-gavg-avggait.nc")}
    freqs = ds_eeg["basenorm"].freq.values
    if args.hc_correction_db != 0:
        ds = ds_eeg["basenorm"].load()
        is_hc = (ds.subject_cohort == HC)
        ds["per_subject"] = ds.per_subject.where(~is_hc, ds.per_subject - args.hc_correction_db)
        ds_eeg["basenorm"] = ds

    rows: list[dict] = []
    lines = [f"Cluster-based permutation tests, block={block}, {args.date}; "
             f"n_perm={args.n_perm}, alpha={args.alpha}, seed={seed}; "
             f"HC basenorm EEG maps corrected by -{args.hc_correction_db:g} dB (microvolt-as-volt export)", ""]

    def record(test, norm, region, side, n1, n2, T_obs, sig, cluster_p, times, diff=None):
        times_pct = (times - times[0]) / (times[-1] - times[0])
        n_sig = int(sum(p < args.alpha for p in cluster_p))
        minp = float(min(cluster_p)) if cluster_p else float("nan")
        desc = []
        if sig.any():
            from scipy import ndimage
            lab, n = ndimage.label(sig)
            for k in range(1, n + 1):
                m = lab == k
                fi, ti = np.where(m)
                if diff is not None:
                    sign = (f"PD<HC ({diff[m].mean():+.1f} dB)" if diff[m].mean() < 0
                            else f"PD>HC ({diff[m].mean():+.1f} dB)")
                else:
                    sign = "neg" if T_obs[m].mean() < 0 else "pos"
                desc.append(f"{freqs[fi.min()]:.0f}-{freqs[fi.max()]:.0f} Hz, "
                            f"{times_pct[ti.min()]*100:.0f}-{times_pct[ti.max()]*100:.0f} % cycle, {sign}, "
                            f"{m.sum()} px")
        rows.append(dict(block=block, test=test, normalisation=norm, region=region, side=side,
                         n_group1=n1, n_group2=n2, n_clusters=len(cluster_p), n_significant=n_sig,
                         min_cluster_p=minp, significant_clusters="; ".join(desc)))
        lines.append(f"{test:28s} {region:8s} {side:8s} n={n1}/{n2 if n2 else '-'}  "
                     f"clusters={len(cluster_p):3d}  sig={n_sig}  min p={minp:.4f}  {'; '.join(desc)}")
        da = xr.Dataset({"T_obs": (("freq", "time"), T_obs), "sig_mask": (("freq", "time"), sig)},
                        coords={"freq": freqs, "time": times},
                        attrs=dict(block=block, test=test, region=region, side=side,
                                   n_permutations=args.n_perm, alpha=args.alpha, seed=seed,
                                   cluster_p=str(cluster_p)))
        da.to_netcdf(g / f"revision_cluster_{block}_{test}_{region}_{side}.nc")
        _plot(T_obs, sig, freqs, times_pct, f"{block} {test} {region} {side} (min cluster p={minp:.3f})",
              fig_dir / f"cluster_{block}_{test}_{region}_{side}.png", stat="F" if diff is not None else "t")

    for norm in ("basenorm", "avggait"):
        ds = ds_eeg[norm]
        for region in REGIONS:
            for side in SIDES:
                d_pd = _subject_data(ds, PD, region, side); d_hc = _subject_data(ds, HC, region, side)
                T, sig, cp = _run_one(d_pd, d_hc, args.n_perm, args.alpha, seed)
                record(f"eeg_{norm}_PDvsHC", norm, region, side, len(d_pd), len(d_hc), T, sig, cp,
                       ds.time.values, diff=np.nanmean(d_pd, axis=0) - np.nanmean(d_hc, axis=0))
        for cohort, tag in ((PD, "PD"), (HC, "HC")):
            for region in REGIONS:
                d = _subject_data(ds, cohort, region, "Both")
                T, sig, cp = _run_one_1samp(d, args.n_perm, args.alpha, seed)
                record(f"eeg_{norm}_1samp_{tag}", norm, region, "Both", len(d), 0, T, sig, cp, ds.time.values)

    for norm in ("basenorm", "avggait"):
        ds = ds_dbs[norm]
        for side in (None, "Left", "Right"):
            d = _dbs_data(ds, side)
            if len(d) < 3:
                lines.append(f"dbs_{norm}_1samp STN {side}: n={len(d)}, skipped"); continue
            T, sig, cp = _run_one_1samp(d, args.n_perm, args.alpha, seed)
            record(f"dbs_{norm}_1samp", norm, "STN", side or "pooled", len(d), 0, T, sig, cp, ds.time.values)

    csv_path = out_dir / f"cluster_perm_{block}_{args.date}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    txt_path = out_dir / f"cluster_perm_{block}_{args.date}.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    write_sidecar(csv_path, {**meta, "block": block, "n_permutations": args.n_perm,
                             "alpha": args.alpha, "n_tests": len(rows)})
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
