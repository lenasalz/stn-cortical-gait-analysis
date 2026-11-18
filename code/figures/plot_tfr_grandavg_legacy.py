"""Render paper-style grand-average TFR figures via the legacy plotter.

Inputs:
  - data/derivatives/tfr_gavg/<block>-eeg-tfr-gavg.nc  (aggregate_tfr_gavg.py --modality eeg)
  - data/derivatives/tfr_gavg/<block>-dbs-tfr-gavg.nc  (aggregate_tfr_gavg.py --modality dbs)
  - code/analysis/config.yaml (freq_bands)
Outputs:
  - outputs/figures/legacy_style/...
      PDF/PNGs in the layout `compute_and_plot_grand_averages_flexible`
      writes — one figure per (block, comparison) defined in --comparisons.
Description:
  Thin adapter from the new per-subject NetCDFs to the legacy
  dict-of-lists structure that `code/analysis/tfr.py:
  compute_and_plot_grand_averages_flexible` consumes. Lets you produce
  the exact figures the manuscript was built on without abandoning the
  new pipeline's clean per-subject data layout.

  Reconstructed dict shape (per the legacy contract):
    tfr_group_results[<group>][<block>] = {
      'ID':          [subjects...],
      'EEG_motor':   {'Left': [arr|None ...], 'Right': [...], 'Central': [...], 'Both': [...]},
      'EEG_frontal': {same shape},
      'STN':         {'Left':  [arr|None ...], 'Right': [arr|None ...]},
    }
  where each `arr` is a 2-D `(n_freqs, n_times)` slice from
  `per_subject` and `None` marks subjects that did not contribute to
  that region/side (e.g. an HC subject for STN, or a PD subject whose
  DBS was implanted contralaterally).

Usage:
  python plot_tfr_grandavg_legacy.py --block pre8walk
  python plot_tfr_grandavg_legacy.py --block pre8walk --method median
  python plot_tfr_grandavg_legacy.py --block pre8walk --scale -2 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
    load_config,
    resolve_paths,
    setup_run,
    validate_block,
)
from tfr import (  # noqa: E402
    compute_and_plot_grand_averages_flexible,
    compute_and_plot_stn_ipsicontra_flexible,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "legacy_style"


def _subject_slab_or_none(ds: xr.Dataset, subject: str, region: str, side: str):
    """Return `(n_freqs, n_times)` numpy slice for one subject's region.side,
    or None if the entire slab is NaN (subject did not contribute)."""
    arr = ds.per_subject.sel(subject=subject, region=region, side=side).values
    if np.isnan(arr).all():
        return None
    return arr


def build_legacy_dict(ds_eeg: xr.Dataset, ds_dbs: xr.Dataset | None,
                      block: str) -> dict:
    """Reshape the per-subject NetCDFs into the legacy dict-of-lists."""
    cohorts = list(ds_eeg.cohort.values)
    out: dict = {}
    for cohort in cohorts:
        cohort_subjects = [
            str(s) for s in ds_eeg.subject.values
            if str(ds_eeg.subject_cohort.sel(subject=s).item()) == cohort
        ]
        block_data: dict = {"ID": cohort_subjects}

        for legacy_region, region in (("EEG_motor", "motor"),
                                      ("EEG_frontal", "frontal")):
            block_data[legacy_region] = {}
            for side in ("Left", "Right", "Central", "Both"):
                block_data[legacy_region][side] = [
                    _subject_slab_or_none(ds_eeg, subj, region, side)
                    for subj in cohort_subjects
                ]

        block_data["STN"] = {"Left": [], "Right": []}
        if ds_dbs is not None:
            for side in ("Left", "Right"):
                block_data["STN"][side] = [
                    _subject_slab_or_none(ds_dbs, subj, "STN", side)
                    if subj in ds_dbs.subject.values
                    else None
                    for subj in cohort_subjects
                ]
        else:
            for side in ("Left", "Right"):
                block_data["STN"][side] = [None] * len(cohort_subjects)

        out.setdefault(cohort, {})[block] = block_data
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True)
    parser.add_argument("--method", choices=["mean", "median"], default="mean",
                        help="Cohort aggregation method passed to the legacy plotter.")
    parser.add_argument("--scale", nargs=2, type=float, default=None,
                        metavar=("VMIN", "VMAX"),
                        help="dB color scale for the legacy plotter. Default depends "
                             "on --norm: basenorm uses (-8, 8) and avggait uses "
                             "(-2, 2), matching the legacy notebook calls in "
                             "analysis_paper.ipynb cells 17/19/28/29.")
    parser.add_argument("--norm", choices=["basenorm", "avggait"], default="basenorm",
                        help="Drives the default --scale (-8,8 for basenorm; -2,2 for avggait).")
    parser.add_argument("--save-format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--ipsicontra", action="store_true",
                        help="Also render the single-panel ipsi/contra-aligned STN "
                             "figure (supplementary Fig.; STN-Left phases shifted by "
                             "-50%% so 0%% = ipsilateral heel-strike).")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    block = validate_block(args.block, config)
    meta = setup_run("plot_tfr_grandavg_legacy", config)

    norm_suffix = "" if args.norm == "basenorm" else f"-{args.norm}"
    eeg_nc = paths.tfr_gavg_dir / f"{block}-eeg-tfr-gavg{norm_suffix}.nc"
    dbs_nc = paths.tfr_gavg_dir / f"{block}-dbs-tfr-gavg{norm_suffix}.nc"
    if not eeg_nc.exists():
        print(f"missing {eeg_nc}; run "
              f"`aggregate_tfr_gavg.py --modality eeg --norm {args.norm}` first",
              file=sys.stderr)
        return 2

    ds_eeg = xr.open_dataset(eeg_nc)
    ds_dbs = xr.open_dataset(dbs_nc) if dbs_nc.exists() else None

    tfr_group_results = build_legacy_dict(ds_eeg, ds_dbs, block)

    freqs = ds_eeg.freq.values
    time_axis = ds_eeg.time.values            # EEG: 220 samples for 1.1s
    if ds_dbs is not None:
        time_axis_stn = ds_dbs.time.values    # DBS: 275 samples for 1.1s
    else:
        time_axis_stn = time_axis

    # Convert config freq_bands to the (low, high) tuple form the legacy expects.
    freq_bands = {name: tuple(edges) for name, edges in config["freq_bands"].items()}

    # One main comparison: PD vs HC, full cohorts.
    comparisons = [{
        "name": "PD_vs_HC",
        "cohorts": {
            "PD": {"group": "Patient_NFB",    "ids": None},
            "HC": {"group": "AgeMatched_NFB", "ids": None},
        },
    }]

    # One sub-dir per normalization keeps basenorm and avggait PDFs from
    # overwriting each other (Aline's notebook writes
    # ERSPs/basenorm and ERSPs/avggait — we mirror that layout).
    out_dir = LEGACY_OUT_DIR / args.norm
    out_dir.mkdir(parents=True, exist_ok=True)

    # Norm-specific default scale (matches analysis_paper.ipynb cells 17,28).
    scale = tuple(args.scale) if args.scale else (
        (-8.0, 8.0) if args.norm == "basenorm" else (-2.0, 2.0)
    )
    print(f"=== legacy plotter — block={block} norm={args.norm} "
          f"method={args.method} scale={scale} ===")
    grand_averages = compute_and_plot_grand_averages_flexible(
        tfr_group_results=tfr_group_results,
        blocks=[block],
        freqs=freqs,
        freq_bands=freq_bands,
        time_axis=time_axis,
        time_axis_stn=time_axis_stn,
        results_dir=str(out_dir),
        expected_time_samples_eeg=len(time_axis),
        expected_time_samples_stn=len(time_axis_stn),
        comparisons=comparisons,
        method=args.method,
        save_format=args.save_format,
        scale=scale,
    )

    print(f"wrote PDFs to {out_dir.relative_to(PROJECT_ROOT)}")
    print(f"cohorts produced: {list(grand_averages.keys())}")

    if args.ipsicontra:
        if ds_dbs is None:
            print("--ipsicontra requested but no DBS NetCDF found; skipping.",
                  file=sys.stderr)
        else:
            print(f"=== ipsi/contra-aligned STN — block={block} norm={args.norm} "
                  f"method={args.method} scale={scale} ===")
            # comparisons_to_run=None -> single-panel "Both" per main group
            # (the supplementary manuscript figure; HC has no STN and is skipped).
            compute_and_plot_stn_ipsicontra_flexible(
                tfr_group_results=tfr_group_results,
                blocks=[block],
                freqs=freqs,
                freq_bands=freq_bands,
                time_axis_stn=time_axis_stn,
                results_dir=str(out_dir),
                expected_time_samples_stn=len(time_axis_stn),
                save_format=args.save_format,
                scale=scale,
                comparisons_to_run=None,
                method=args.method,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
