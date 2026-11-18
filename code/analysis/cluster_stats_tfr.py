"""Permutation cluster test on cohort-aggregated TFRs.

Inputs:
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-gavg.nc  (aggregate_tfr_gavg.py)
  - code/analysis/config.yaml (repro.seed; n_permutations from --n-perm)
Outputs:
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-cluster.nc
      xr.Dataset with vars:
        T_obs(region, side, freq, time)     — observed t-statistic
        sig_mask(region, side, freq, time)  — boolean, True where p < --alpha
      attrs: alpha, n_permutations, tail, comparison, n_clusters_significant
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-cluster.nc.meta.json
Description:
  Between-group cluster-based permutation test per (region, side) on the
  per-subject TFRs from the aggregator. Independent (PD vs HC) by
  default; matches the legacy `permutation_cluster_test` call in
  scripts/psd_fooof.py:_run_permutation_cluster_test but generalized
  from 1-D (freq) to 2-D (freq × time).

  EEG: 8 (region, side) tests = {motor, frontal} × {Left, Right, Central, Both}.
  DBS: skipped — only PD has STN, so no between-group test.

  Each test:
    data1 = per_subject(PD, region, side)  shape (n_pd, n_freqs, n_times)
    data2 = per_subject(HC, region, side)  shape (n_hc, n_freqs, n_times)
    NaN rows are dropped (subjects without that region's channels).
    `mne.stats.permutation_cluster_test([data1, data2], n_permutations=N,
      tail=0, out_type='mask', seed=cfg.repro.seed)`
    → T_obs (n_freqs, n_times), clusters (list of bool masks),
      cluster_p_values

  sig_mask is the union of clusters whose p < --alpha (default 0.05).
  Cluster-level correction is built into MNE's test; no extra FDR/
  Bonferroni across (region, side) — each is a separate hypothesis.

Usage:
  python cluster_stats_tfr.py --block pre8walk --modality eeg
  python cluster_stats_tfr.py --block pre8walk --n-perm 5000 --alpha 0.05
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from _common import (
    atomic_write,
    load_config,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    write_sidecar,
)

REGIONS_EEG = ("motor", "frontal")
SIDES_EEG = ("Left", "Right", "Central", "Both")


def _subject_data(ds: xr.Dataset, cohort: str, region: str, side: str
                  ) -> np.ndarray:
    """Per-cohort, per-region slab. Returns shape (n_kept_subjects, n_freqs, n_times)
    with all-NaN subjects dropped (couldn't contribute to that region)."""
    arr = ds.per_subject.sel(region=region, side=side).where(
        ds.subject_cohort == cohort, drop=True,
    ).values                                              # (n_cohort, n_freqs, n_times)
    keep = ~np.isnan(arr).all(axis=(1, 2))
    return arr[keep]


def _run_one(data_pd: np.ndarray, data_hc: np.ndarray,
             n_permutations: int, alpha: float, seed: int
             ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Independent-samples cluster test between two groups (3-D data: (n_obs, F, T)).

    Returns (T_obs, sig_mask, cluster_p_values).
    sig_mask is the union of clusters with p < alpha.
    """
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        T_obs, clusters, cluster_p, _H0 = mne.stats.permutation_cluster_test(
            [data_pd, data_hc],
            n_permutations=n_permutations, tail=0, out_type="mask",
            seed=seed, verbose="WARNING",
        )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for cluster, p in zip(clusters, cluster_p):
        if p < alpha:
            sig_mask |= cluster
    return T_obs, sig_mask, [float(p) for p in cluster_p]


def _run_one_1samp(data: np.ndarray, n_permutations: int, alpha: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """One-sample cluster test against H0: mean = 0 (3-D data: (n_obs, F, T)).

    Tests whether the gait-warped TFR (already in dB vs baseline) deviates
    significantly from zero — i.e. is there significant suppression or
    enhancement anywhere in the gait cycle, after multiple-comparison
    correction via cluster permutation.

    Returns (T_obs, sig_mask, cluster_p_values).
    """
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        T_obs, clusters, cluster_p, _H0 = mne.stats.permutation_cluster_1samp_test(
            data, n_permutations=n_permutations, tail=0, out_type="mask",
            seed=seed, verbose="WARNING",
        )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for cluster, p in zip(clusters, cluster_p):
        if p < alpha:
            sig_mask |= cluster
    return T_obs, sig_mask, [float(p) for p in cluster_p]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--n-perm", type=int, default=5000,
                        help="Number of permutations. 5000 matches the legacy "
                             "paper default; 1000 is faster for development.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    block = validate_block(args.block, config)

    in_path = paths.tfr_gavg_dir / f"{block}-{args.modality}-tfr-gavg.nc"
    out_path = paths.tfr_gavg_dir / f"{block}-{args.modality}-tfr-cluster.nc"
    if not in_path.exists():
        print(f"missing aggregator NetCDF {in_path}", file=sys.stderr)
        return 2
    if should_skip(out_path, args.force):
        print(f"skip {out_path.name} (exists)")
        return 0

    meta = setup_run("cluster_stats_tfr", config)
    seed = int(config["repro"]["seed"])

    ds = xr.open_dataset(in_path)
    cohorts = list(ds.cohort.values)

    if args.modality == "dbs":
        return _main_dbs_1samp(ds, args, paths, block, out_path,
                               meta, seed)

    if "Patient_NFB" not in cohorts or "AgeMatched_NFB" not in cohorts:
        print(f"need both PD and HC in cohorts; got {cohorts}", file=sys.stderr)
        return 3

    freqs = ds.freq.values
    times = ds.time.values
    n_freqs, n_times = len(freqs), len(times)

    T_obs_all = np.full((len(REGIONS_EEG), len(SIDES_EEG), n_freqs, n_times),
                        np.nan, dtype=np.float64)
    sig_mask_all = np.zeros((len(REGIONS_EEG), len(SIDES_EEG), n_freqs, n_times),
                            dtype=bool)
    summary: dict[str, dict] = {}
    n_sig_total = 0

    for r_i, region in enumerate(REGIONS_EEG):
        for sd_i, side in enumerate(SIDES_EEG):
            data_pd = _subject_data(ds, "Patient_NFB", region, side)
            data_hc = _subject_data(ds, "AgeMatched_NFB", region, side)
            if data_pd.shape[0] < 2 or data_hc.shape[0] < 2:
                print(f"  skip {region}.{side}: n_pd={data_pd.shape[0]} "
                      f"n_hc={data_hc.shape[0]} (need ≥2 per group)",
                      file=sys.stderr)
                continue
            print(f"  cluster test {region}.{side}: n_pd={data_pd.shape[0]} "
                  f"n_hc={data_hc.shape[0]}")
            T_obs, sig_mask, cluster_p = _run_one(
                data_pd, data_hc,
                n_permutations=args.n_perm,
                alpha=args.alpha, seed=seed,
            )
            T_obs_all[r_i, sd_i] = T_obs
            sig_mask_all[r_i, sd_i] = sig_mask
            n_sig = sum(1 for p in cluster_p if p < args.alpha)
            n_sig_total += n_sig
            summary[f"{region}.{side}"] = {
                "n_pd": int(data_pd.shape[0]),
                "n_hc": int(data_hc.shape[0]),
                "n_clusters": len(cluster_p),
                "n_significant": n_sig,
                "min_cluster_p": float(min(cluster_p)) if cluster_p else None,
            }

    out_ds = xr.Dataset(
        data_vars={
            "T_obs":    (("region", "side", "freq", "time"), T_obs_all),
            "sig_mask": (("region", "side", "freq", "time"), sig_mask_all),
        },
        coords={
            "region": list(REGIONS_EEG),
            "side":   list(SIDES_EEG),
            "freq":   freqs,
            "time":   times,
        },
        attrs={
            "block":         block,
            "modality":      args.modality,
            "comparison":    "Patient_NFB vs AgeMatched_NFB",
            "test":          "mne.stats.permutation_cluster_test",
            "tail":          0,
            "alpha":         float(args.alpha),
            "n_permutations": int(args.n_perm),
            "seed":          seed,
            "n_clusters_significant_total": int(n_sig_total),
        },
    )

    with atomic_write(out_path) as tmp:
        out_ds.to_netcdf(tmp)

    meta.update({
        "block": block, "modality": args.modality,
        "input": str(in_path), "output": str(out_path),
        "n_permutations": int(args.n_perm),
        "alpha": float(args.alpha),
        "summary": summary,
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}")
    print(f"  significant clusters (across all region.sides): {n_sig_total}")
    for key, s in summary.items():
        flag = "*" if s["n_significant"] else " "
        mp = f"{s['min_cluster_p']:.4f}" if s['min_cluster_p'] is not None else "n/a"
        print(f"   {flag} {key:15s} n_clusters={s['n_clusters']:2d}  "
              f"n_sig={s['n_significant']}  min_p={mp}")
    return 0


def _main_dbs_1samp(ds, args, paths, block, out_path, meta, seed) -> int:
    """One-sample cluster test on PD STN: significant deviation from 0 dB.

    Tests `mean(per_subject(STN.<side>, PD)) ≠ 0` per (region=STN, side).
    Inputs already in dB vs baseline, so the test asks: is there
    significant suppression or enhancement at any (freq, time) cell?

    Writes the same NetCDF shape as the EEG path (region × side × freq × time),
    with `region=['STN']` and `side=['Left', 'Right']`. HC cohort is
    skipped — only PD has an implant.
    """
    freqs = ds.freq.values
    times = ds.time.values
    n_freqs, n_times = len(freqs), len(times)
    regions = ("STN",)
    sides = ("Left", "Right")

    T_obs_all = np.full((len(regions), len(sides), n_freqs, n_times),
                        np.nan, dtype=np.float64)
    sig_mask_all = np.zeros((len(regions), len(sides), n_freqs, n_times),
                            dtype=bool)
    summary: dict[str, dict] = {}
    n_sig_total = 0

    for sd_i, side in enumerate(sides):
        data_pd = _subject_data(ds, "Patient_NFB", "STN", side)
        # Drop any all-finite rows that contain inf (defensive; aggregator
        # already replaces inf with NaN).
        if data_pd.shape[0] < 3:
            print(f"  skip STN.{side}: n_pd={data_pd.shape[0]} (need ≥3 for 1-sample test)",
                  file=sys.stderr)
            continue
        print(f"  1-sample cluster test STN.{side}: n_pd={data_pd.shape[0]}")
        T_obs, sig_mask, cluster_p = _run_one_1samp(
            data_pd, n_permutations=args.n_perm,
            alpha=args.alpha, seed=seed,
        )
        T_obs_all[0, sd_i] = T_obs
        sig_mask_all[0, sd_i] = sig_mask
        n_sig = sum(1 for p in cluster_p if p < args.alpha)
        n_sig_total += n_sig
        summary[f"STN.{side}"] = {
            "n_pd":          int(data_pd.shape[0]),
            "n_clusters":    len(cluster_p),
            "n_significant": n_sig,
            "min_cluster_p": float(min(cluster_p)) if cluster_p else None,
        }

    out_ds = xr.Dataset(
        data_vars={
            "T_obs":    (("region", "side", "freq", "time"), T_obs_all),
            "sig_mask": (("region", "side", "freq", "time"), sig_mask_all),
        },
        coords={
            "region": list(regions),
            "side":   list(sides),
            "freq":   freqs,
            "time":   times,
        },
        attrs={
            "block":         block,
            "modality":      "dbs",
            "comparison":    "PD STN vs 0 (1-sample)",
            "test":          "mne.stats.permutation_cluster_1samp_test",
            "tail":          0,
            "alpha":         float(args.alpha),
            "n_permutations": int(args.n_perm),
            "seed":          seed,
            "n_clusters_significant_total": int(n_sig_total),
        },
    )

    with atomic_write(out_path) as tmp:
        out_ds.to_netcdf(tmp)

    meta.update({
        "block": block, "modality": "dbs",
        "output": str(out_path),
        "n_permutations": int(args.n_perm),
        "alpha": float(args.alpha),
        "summary": summary,
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}")
    print(f"  significant clusters (across STN.Left + STN.Right): {n_sig_total}")
    for key, s in summary.items():
        flag = "*" if s["n_significant"] else " "
        mp = f"{s['min_cluster_p']:.4f}" if s['min_cluster_p'] is not None else "n/a"
        print(f"   {flag} {key:10s} n_pd={s['n_pd']}  n_clusters={s['n_clusters']:2d}  "
              f"n_sig={s['n_significant']}  min_p={mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
