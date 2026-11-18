"""Convert a cycle-averaged gait-warped TFR (basenorm) into its avggait variant.

Inputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-gait.h5  (compute_tfr_gait.py)
Outputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-gait-avggait.h5
  - sidecar JSON
Description:
  avggait normalization: subtracts the time-mean per (channel, freq)
  from the gait-warped, cycle-averaged dB TFR. Recenters each
  (channel, freq) around zero across the gait cycle so the figure
  reflects within-cycle modulation rather than the slow DC offset
  carried over from the baseline.

  Matches `_normalize_avg_tfr_whole_cycle_inplace` in
  final_analysis/scripts/tfr.py: `data -= mean(data, axis=-1, keepdims=True)`.

  Whether starting from basenorm-dB or absolute-dB doesn't matter —
  the baseline constant cancels in the subtraction.

Usage:
  python apply_avggait_norm.py --subject P4-2002 --block pre8walk --modality eeg
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

from _common import (
    atomic_write,
    cohort_of,
    load_config,
    output_path,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    validate_subject,
    write_sidecar,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.block == "baseline":
        parser.error("baseline has no gait-warped TFR")

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    in_path = output_path(paths.tfr_dir, subject, block,
                          f"-{args.modality}-tfr-gait.h5")
    out_path = output_path(paths.tfr_dir, subject, block,
                           f"-{args.modality}-tfr-gait-avggait.h5")
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    if not in_path.exists():
        print(f"missing {in_path}; run compute_tfr_gait.py first", file=sys.stderr)
        return 2

    meta = setup_run("apply_avggait_norm", config)

    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        tfr = mne.time_frequency.read_tfrs(in_path)
    tfr = tfr[0] if isinstance(tfr, list) else tfr

    # Subtract time-mean per (channel, freq).
    time_mean = tfr.data.mean(axis=-1, keepdims=True)        # (n_chans, n_freqs, 1)
    tfr.data = tfr.data - time_mean

    with atomic_write(out_path) as tmp:
        mne.time_frequency.write_tfrs(tmp, tfr, overwrite=True)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality,
        "input": str(in_path), "output": str(out_path),
        "norm": "avggait",
        "method": "tfr -= mean(tfr, axis=time)  per (channel, freq)",
        "n_channels": int(tfr.data.shape[0]),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  shape={tfr.data.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
