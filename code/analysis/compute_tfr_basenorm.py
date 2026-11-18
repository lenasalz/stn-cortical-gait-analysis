"""Baseline-normalize a continuous task TFR against the same subject's continuous baseline TFR.

Inputs:
  - data/derivatives/tfr/<subj>_baseline-{eeg,dbs}-tfr-cont.h5  (compute_tfr_continuous.py)
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-cont.h5    (compute_tfr_continuous.py)
  - code/analysis/config.yaml
Outputs:
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-cont-basenorm.h5
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-cont-basenorm.h5.meta.json
Description:
  Per-channel dB normalization:
      normalized[c, f, t] = 10 * log10(task[c, f, t] / mean_t(baseline)[c, f])

  Replicates the legacy normalization in
  final_analysis/scripts/tfr.py:compute_tfr_on_subset_and_normalize_optimized
  lines 116-174:
      base_power_vectors = tfr_eeg_base.data.mean(axis=2)
      normalized_data[i, :, :] = 10 * log10(tfr_eeg_task.data[i, :, :]
                                            / baseline[:, np.newaxis])

  Both inputs must be continuous TFRs produced by `compute_tfr_continuous.py`
  on Raw input — NOT the gait-warped epoched TFRs from `compute_tfr.py`.
  This is the path the paper figures use (basenorm is applied to the
  continuous TFR; gait-warping is a downstream step).

  Channel intersection: only channels present in both TFR files are kept.

Usage:
  python compute_tfr_basenorm.py --subject P4-2002 --block pre8walk --modality eeg
  python compute_tfr_basenorm.py --subject P4-2002 --block pre8walk --modality dbs
"""
from __future__ import annotations

import argparse
import sys

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
    parser.add_argument("--block", required=True,
                        help="Task block to normalize (must not be 'baseline').")
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--baseline-block", default="baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.block == args.baseline_block:
        parser.error("--block must differ from --baseline-block")

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    base_block = validate_block(args.baseline_block, config)
    cohort = cohort_of(subject, config)

    base_path = output_path(paths.tfr_dir, subject, base_block,
                            f"-{args.modality}-tfr-cont.h5")
    task_path = output_path(paths.tfr_dir, subject, block,
                            f"-{args.modality}-tfr-cont.h5")
    out_path = output_path(paths.tfr_dir, subject, block,
                           f"-{args.modality}-tfr-cont-basenorm.h5")

    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    for p, name in ((base_path, "baseline"), (task_path, "task")):
        if not p.exists():
            print(f"missing {name} continuous TFR {p}; "
                  f"run compute_tfr_continuous.py first", file=sys.stderr)
            return 2

    meta = setup_run("compute_tfr_basenorm", config)

    import mne

    base_tfr = mne.time_frequency.read_tfrs(base_path)
    task_tfr = mne.time_frequency.read_tfrs(task_path)
    base_tfr = base_tfr[0] if isinstance(base_tfr, list) else base_tfr
    task_tfr = task_tfr[0] if isinstance(task_tfr, list) else task_tfr

    common = [ch for ch in task_tfr.ch_names if ch in base_tfr.ch_names]
    if not common:
        print(f"no shared channels between baseline and {block}", file=sys.stderr)
        return 3
    if not np.allclose(base_tfr.freqs, task_tfr.freqs):
        print("baseline and task TFR freq grids differ", file=sys.stderr)
        return 4

    base_tfr = base_tfr.copy().pick(common, verbose="WARNING")
    task_tfr = task_tfr.copy().pick(common, verbose="WARNING")

    # Time-average baseline → (n_chans, n_freqs); broadcast over task time.
    # `10*log10(0/baseline) = -inf` and `10*log10(0/0) = NaN` arise when a
    # task-block channel has a flat segment in some (freq, time) cells.
    # Replace ±Inf and NaN with NaN so downstream consumers can drop the
    # affected rows (R's na.omit) rather than carry synthetic floor values
    # forward — matches the implicit behavior of Aline's pipeline (no
    # eps clamp on her basenorm path, line 172 in final_analysis tfr.py).
    # `compute_gerd_table.py`'s avggait branch uses np.nanmean defensively
    # so a single NaN cell does not poison the entire (channel, freq)
    # subject-mean plane.
    base_power = base_tfr.data.mean(axis=-1)        # (n_chans, n_freqs)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = 10.0 * np.log10(task_tfr.data / base_power[:, :, np.newaxis])
    norm[~np.isfinite(norm)] = np.nan
    task_tfr.data = norm

    with atomic_write(out_path) as tmp:
        mne.time_frequency.write_tfrs(tmp, task_tfr, overwrite=True)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality,
        "baseline_block": base_block,
        "baseline_input": str(base_path),
        "task_input": str(task_path),
        "output": str(out_path),
        "n_channels": len(common),
        "method": "10*log10(task / mean_t(baseline)) per channel × freq",
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  shape={task_tfr.data.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
