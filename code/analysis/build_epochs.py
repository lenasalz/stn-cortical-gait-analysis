"""Materialize per-subject epoch files from the existing ngb_study pickle cache.

Inputs:
  - $RESIDUAL_DATA_ROOT/Final_filtered/all_data_epoched_dico.pkl   (legacy cache)
  - code/analysis/config.yaml (cohorts, blocks)
Outputs:
  - data/derivatives/epochs/<subj>_<block>-epo.fif      (one MNE Epochs per file)
  - data/derivatives/epochs/<subj>_<block>-epo.fif.meta.json (sidecar)
Description:
  One-shot import bridge. Reads the monolithic
  `all_data_epoched_dico.pkl` produced by the legacy pipeline,
  extracts each (subject, block) slice, and writes it out as a
  standalone `-epo.fif`. After this runs once, every downstream
  compute script (compute_psd, compute_tfr, …) reads only its own
  subject's `.fif` file via `mne.read_epochs(..., preload=False)`.

  This script exists for the migration from the dict-based pickle
  cache to the per-subject layout. Once `code/preprocessing/` is in
  place, the same `-epo.fif` files can be produced upstream and this
  bridge is retired.

Usage:
  python build_epochs.py --subject P4-2002 --block pre8walk
  python build_epochs.py --all                        # iterate every (subj, block)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from itertools import product
from pathlib import Path

from _common import (
    cohort_of,
    load_config,
    output_path,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    validate_subject,
    write_sidecar,
)


def _legacy_pickle_path(data_root: Path) -> Path:
    # TODO(verify): the legacy pickle lives under "Final_filtered" on the
    # production server. Confirm subdir name (variants seen in
    # ngb_study/scripts/variables.py: "Final_filtered",
    # "Final_noshift_bandpassadj") before first run.
    return data_root / "Final_filtered" / "all_data_epoched_dico.pkl"


def extract_one(dico, cohort: str, block: str, subject: str):
    """Return the MNE Epochs object for one (cohort, block, subject)."""
    block_entry = dico[cohort][block]
    idx = block_entry["ID"].index(subject)
    return block_entry["EEG"][idx], block_entry["DBS"][idx]


def write_one(epochs, out_path: Path) -> None:
    if epochs is None:
        # Preserve the "missing data" signal as an empty sentinel file so
        # downstream scripts can distinguish "not yet computed" from
        # "no data for this subject/block".
        out_path.with_suffix(out_path.suffix + ".missing").touch()
        return
    patch_legacy_annotations(epochs)
    epochs.save(out_path, overwrite=True, fmt="single")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Process every (subject, block) combination from config.")
    g.add_argument("--subject", help="Single anonymized subject code.")
    parser.add_argument("--block", help="Required when --subject is given.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    setup_run("build_epochs", config)

    pkl_path = _legacy_pickle_path(paths.data_root)
    if not pkl_path.exists():
        print(f"legacy pickle not found at {pkl_path}", file=sys.stderr)
        return 2

    print(f"loading {pkl_path} (this is the only full-cache load)...")
    with pkl_path.open("rb") as f:
        dico = pickle.load(f)

    if args.all:
        jobs = [
            (subj, block)
            for cohort, members in config["cohorts"].items()
            for subj in members
            for block in config["blocks"]
        ]
    else:
        if not args.block:
            parser.error("--block required with --subject")
        jobs = [(validate_subject(args.subject), validate_block(args.block, config))]

    n_written = n_skipped = n_missing = 0
    for subject, block in jobs:
        cohort = cohort_of(subject, config)
        # Per-modality output: keep EEG and DBS separate so downstream
        # scripts pick the modality they need without loading both. The
        # skip check uses the EEG output explicitly (the `-epo.fif` legacy
        # name is not produced by this script — checking it would cause
        # every run to re-extract from the source pickle).
        eeg_out = output_path(paths.epochs_dir, subject, block, "-eeg-epo.fif")
        dbs_out = output_path(paths.epochs_dir, subject, block, "-dbs-epo.fif")
        if should_skip(eeg_out, args.force):
            n_skipped += 1
            continue
        eeg_epochs, dbs_epochs = extract_one(dico, cohort, block, subject)
        write_one(eeg_epochs, eeg_out)
        write_one(dbs_epochs, dbs_out)
        meta = {
            "subject": subject, "cohort": cohort, "block": block,
            "source_pickle": str(pkl_path),
            "eeg_output": str(eeg_out),
            "dbs_output": str(dbs_out),
            "eeg_present": eeg_epochs is not None,
            "dbs_present": dbs_epochs is not None,
        }
        # TODO(verify): confirm the marker file approach (`*.missing`) is
        # the right way to encode "no data for this subject/block" once
        # we know which (subj, block) combinations are expected to be
        # absent. Alternative: skip the (subject, block) entry entirely
        # and rely on index.csv.
        write_sidecar(eeg_out, meta)
        if eeg_epochs is None and dbs_epochs is None:
            n_missing += 1
        else:
            n_written += 1

    print(f"wrote={n_written} skipped={n_skipped} missing={n_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
