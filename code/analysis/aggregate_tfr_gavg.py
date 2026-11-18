"""Cohort × region × side grand averages from per-subject gait-warped TFRs.

Inputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-gait.h5         (--norm basenorm, default)
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-gait-avggait.h5 (--norm avggait)
  - code/analysis/config.yaml (cohorts, freq_bands, channels_of_interest)
Outputs:
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-gavg[-<norm>].nc
      xr.Dataset with vars:
        gavg(cohort, region, side, freq, time)   — mean across subjects
        per_subject(subject, region, side, freq, time)  — one subject's regional TFR
      coords: cohort, region (motor/frontal), side (Left/Right/Central/Both),
              subject, freq, time
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-gavg.nc.meta.json
Description:
  Reads every per-subject gait-warped TFR for the given block, averages
  channels into the regions defined in
  `config.yaml:connectivity.channels_of_interest`, stacks subjects per
  cohort, and emits both per-subject regional TFRs and per-cohort grand
  averages in a single NetCDF.

  Region members come from `motor` and `frontal` only; STN is not yet
  aggregated here (would need the DBS gait-warped TFRs first).

  Channel availability: a subject contributes to a region if at least
  one of that region's canonical channels is present in its gait TFR.
  Skipped subjects appear in the `per_subject` array as NaN.

Usage:
  python aggregate_tfr_gavg.py --block pre8walk
  python aggregate_tfr_gavg.py --block pre8walk --modality eeg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from _common import (
    atomic_write,
    load_config,
    output_path,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    write_sidecar,
)


EEG_REGIONS = ("motor", "frontal")
EEG_SIDES = ("Left", "Right", "Central", "Both")
DBS_REGIONS = ("STN",)
DBS_SIDES = ("Left", "Right")


def _region_channels(config: dict, region: str, side: str) -> list[str]:
    coi = config["connectivity"]["channels_of_interest"][region]
    if side == "Both":
        return list(dict.fromkeys(
            coi.get("Left", []) + coi.get("Right", []) + coi.get("Central", [])
        ))
    return list(coi.get(side, []))


def _dbs_side(tfr) -> str | None:
    """Return 'Left' or 'Right' based on the TFR's info description string
    (built by `data_loader.convert_dbs_to_mne` via 'DBS side: <side>')."""
    desc = (tfr.info.get("description") or "").lower()
    if "left" in desc:
        return "Left"
    if "right" in desc:
        return "Right"
    return None


def _subject_region_tfr(tfr, target_channels: list[str]) -> np.ndarray | None:
    """Return nanmean of `target_channels` present in `tfr` along the channel
    axis, or None if no listed channel is available.

    Converts ±inf → NaN before averaging. Inf cells arise upstream when a
    task-block TFR has zero power at a (freq, time) bin (flat channel
    segment); `10*log10(0/baseline) = -inf`. Treating those cells as
    missing here means one bad channel doesn't propagate through the
    region mean, which would otherwise pin the whole regional cell to
    -inf and then the cohort grand average too.
    """
    present = [c for c in target_channels if c in tfr.ch_names]
    if not present:
        return None
    idx = [tfr.ch_names.index(c) for c in present]
    data = tfr.data[idx].copy()
    data[~np.isfinite(data)] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(data, axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True,
                        help="Task block to aggregate (gait-warp is undefined for baseline).")
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--method", choices=["median", "mean"], default="median",
                        help="Aggregation across subjects within a cohort. "
                             "Default 'median' matches the legacy paper figures "
                             "(filenames suffixed `_median_`).")
    parser.add_argument("--norm", choices=["basenorm", "avggait"], default="basenorm",
                        help="Which per-subject gait-warped TFR to aggregate: "
                             "'basenorm' reads `-tfr-gait.h5` (manuscript Fig 3a/3b); "
                             "'avggait' reads `-tfr-gait-avggait.h5` (Fig 7a/7b).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.block == "baseline":
        parser.error("aggregate_tfr_gavg: gait-warped TFR is only defined for task blocks")

    config = load_config()
    paths = resolve_paths(config)
    block = validate_block(args.block, config)
    norm_suffix = "" if args.norm == "basenorm" else f"-{args.norm}"
    out_path = paths.tfr_gavg_dir / f"{block}-{args.modality}-tfr-gavg{norm_suffix}.nc"

    if should_skip(out_path, args.force):
        print(f"skip {block} {args.modality} (exists: {out_path.name})")
        return 0

    meta = setup_run("aggregate_tfr_gavg", config)

    import mne

    cohorts = list(config["cohorts"].keys())
    subjects_by_cohort = config["cohorts"]
    all_subjects = [s for c in cohorts for s in subjects_by_cohort[c]]
    cohort_of_subject = {s: c for c in cohorts for s in subjects_by_cohort[c]}

    # Per-modality region/side schema. DBS is single-channel STN with the
    # implanted side recorded in tfr.info['description']; HC cohort never
    # contributes (no implant), and PD subjects contribute to exactly one
    # side per cohort grand-average.
    if args.modality == "eeg":
        regions, sides = EEG_REGIONS, EEG_SIDES
    else:  # dbs
        regions, sides = DBS_REGIONS, DBS_SIDES

    # Load every per-subject gait TFR once; record subject grids.
    per_subject_tfrs: dict[str, "mne.time_frequency.AverageTFR"] = {}
    missing: list[str] = []
    for subj in all_subjects:
        gait_suffix = f"-{args.modality}-tfr-gait{norm_suffix}.h5"
        p = output_path(paths.tfr_dir, subj, block, gait_suffix)
        if not p.exists():
            missing.append(subj)
            continue
        t = mne.time_frequency.read_tfrs(p)
        per_subject_tfrs[subj] = t[0] if isinstance(t, list) else t

    if not per_subject_tfrs:
        print(f"no per-subject gait TFRs found for block={block} modality={args.modality};"
              f" run compute_tfr_gait.py first", file=sys.stderr)
        return 2

    # Use the first available subject's freq/time grid as canonical.
    canonical = next(iter(per_subject_tfrs.values()))
    freqs = canonical.freqs
    times = canonical.times
    n_freqs, n_times = len(freqs), len(times)

    # per_subject(subject, region, side, freq, time)
    per_subject = np.full(
        (len(all_subjects), len(regions), len(sides), n_freqs, n_times),
        np.nan, dtype=np.float64,
    )
    n_contributing = np.zeros((len(regions), len(sides), len(cohorts)), dtype=int)

    for s_i, subj in enumerate(all_subjects):
        tfr = per_subject_tfrs.get(subj)
        if tfr is None:
            continue
        if not np.allclose(tfr.freqs, freqs) or not np.allclose(tfr.times, times):
            print(f"WARN {subj} freq/time grid differs from canonical; skipping",
                  file=sys.stderr)
            continue
        c_i = cohorts.index(cohort_of_subject[subj])

        if args.modality == "eeg":
            for r_i, region in enumerate(regions):
                for sd_i, side in enumerate(sides):
                    chans = _region_channels(config, region, side)
                    arr = _subject_region_tfr(tfr, chans)
                    if arr is not None:
                        per_subject[s_i, r_i, sd_i] = arr
                        n_contributing[r_i, sd_i, c_i] += 1
        else:  # dbs: single STN region, side from tfr.info description
            side_label = _dbs_side(tfr)
            if side_label is None or side_label not in sides:
                print(f"WARN {subj} no DBS side in TFR info; skipping", file=sys.stderr)
                continue
            sd_i = sides.index(side_label)
            data = tfr.data[0].copy()                          # (n_freqs, n_times)
            data[~np.isfinite(data)] = np.nan
            per_subject[s_i, 0, sd_i] = data
            n_contributing[0, sd_i, c_i] += 1

    # gavg(cohort, region, side, freq, time): aggregate over subjects within
    # cohort (default median, matches the legacy paper figures); NaN-aware.
    gavg = np.full(
        (len(cohorts), len(regions), len(sides), n_freqs, n_times),
        np.nan, dtype=np.float64,
    )
    agg = np.nanmedian if args.method == "median" else np.nanmean
    for c_i, cohort in enumerate(cohorts):
        s_idx = [all_subjects.index(s) for s in subjects_by_cohort[cohort]]
        slab = per_subject[s_idx]                        # (n_cohort_subjects, regions, sides, freqs, times)
        with np.errstate(invalid="ignore"):
            gavg[c_i] = agg(slab, axis=0)

    ds = xr.Dataset(
        data_vars={
            "gavg":        (("cohort", "region", "side", "freq", "time"), gavg),
            "per_subject": (("subject", "region", "side", "freq", "time"), per_subject),
        },
        coords={
            "cohort":  list(cohorts),
            "region":  list(regions),
            "side":    list(sides),
            "subject": list(all_subjects),
            "freq":    freqs,
            "time":    times,
        },
        attrs={
            "block":    block,
            "modality": args.modality,
            "n_subjects_total": len(all_subjects),
            "n_subjects_missing": len(missing),
            "missing_subjects": ",".join(missing),
        },
    )
    # Map cohort membership onto the subject coord for downstream filtering.
    ds = ds.assign_coords(
        subject_cohort=("subject", [cohort_of_subject[s] for s in all_subjects])
    )

    paths.tfr_gavg_dir.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        ds.to_netcdf(tmp)

    meta.update({
        "block": block,
        "modality": args.modality,
        "input_dir": str(paths.tfr_dir),
        "output": str(out_path),
        "n_subjects_loaded": len(per_subject_tfrs),
        "n_subjects_missing": len(missing),
        "missing_subjects": missing,
        "regions": list(regions),
        "sides": list(sides),
        "n_contributing": {
            f"{regions[r]}.{sides[s]}.{cohorts[c]}": int(n_contributing[r, s, c])
            for r in range(len(regions))
            for s in range(len(sides))
            for c in range(len(cohorts))
        },
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}")
    print(f"  per_subject shape: {per_subject.shape}")
    print(f"  gavg shape:        {gavg.shape}")
    print(f"  missing:           {missing or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
