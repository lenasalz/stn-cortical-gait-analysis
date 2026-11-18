"""Build gait-warped epochs from per-subject Raw files (no autoreject yet).

Inputs:
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif       (build_raw_per_subject.py)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-proc-raw.fif  (Patient_NFB only)
  - code/analysis/config.yaml + variables.py (gait sequence + event %)
Outputs:
  - data/derivatives/epochs/<subj>_<block>-eeg-epo.fif      (un-filtered: all valid cycles)
  - data/derivatives/epochs/<subj>_<block>-dbs-epo.fif      (Patient_NFB only)
  - sidecar JSONs
Description:
  Per-subject port of `create_epoched_data_dictionary` in
  ngb_study/notebooks/final_analysis/scripts/epoching.py. Produces
  the SAME data structure as the legacy notebook before its inline
  `clean_epochs_autoreject_paired` step.

  Why this replaces `build_epochs.py`: that bridge read
  `all_data_epoched_dico.pkl`, which on this machine already had a
  rejection mask applied (113 of 119 cycles for P4-2002 pre8walk).
  This script starts from the un-filtered Raw and produces the full
  119 cycles, which is what `clean_epochs_autoreject_paired` expects
  as input. After this stage: `compute_autoreject_mask.py` runs
  against the full cycle set and the kept-count matches the legacy
  notebook (118/119 for P4-2002).

Usage:
  python build_epochs_from_raw.py --subject P4-2002 --block pre8walk
  python build_epochs_from_raw.py --subject P4-2002 --block baseline
"""
from __future__ import annotations

import argparse
import sys
import warnings

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
from epoching import (
    create_epochs_array,
    filter_gait_events,
    replace_events_in_mne,
    resample_to_fixed_length,
    segment_gait_cycles,
    warp_dbs_or_eeg_data,
)
from variables import (
    event_percentages as DEFAULT_EVENT_PERCENTAGES,
    gait_events_correct_sequence as DEFAULT_GAIT_SEQ,
)

EPOCH_DURATION_S = 1.1


def _build_baseline_epochs(raw):
    import mne
    events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION_S)
    return mne.Epochs(raw, events, tmin=0, tmax=EPOCH_DURATION_S,
                      baseline=None, preload=True, verbose="WARNING")


def _build_task_epochs(raw, gait_seq, event_pct):
    filt = replace_events_in_mne(raw, filter_gait_events(raw, gait_seq), gait_seq)
    cycles = segment_gait_cycles(filt, gait_seq)
    if not cycles:
        return None, 0
    warped = warp_dbs_or_eeg_data(filt, cycles, event_pct)
    desired_len = int(round(EPOCH_DURATION_S * raw.info["sfreq"]))
    resampled = resample_to_fixed_length(warped, desired_length=desired_len)
    epochs = create_epochs_array(resampled, filt.info)
    return epochs, len(cycles)


def _save(epochs, out_path):
    if epochs is None:
        out_path.with_suffix(out_path.suffix + ".missing").touch()
        return
    if hasattr(epochs, "annotations") and epochs.annotations is not None:
        patch_legacy_annotations(epochs)
    epochs.save(out_path, overwrite=True, fmt="single")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    eeg_in = output_path(paths.raw_clean_dir, subject, block, "-eeg-raw.fif")
    dbs_in = output_path(paths.raw_clean_dir, subject, block, "-dbs-proc-raw.fif")
    eeg_out = output_path(paths.epochs_dir, subject, block, "-eeg-epo.fif")
    dbs_out = output_path(paths.epochs_dir, subject, block, "-dbs-epo.fif")

    if should_skip(eeg_out, args.force) and (
        cohort != "Patient_NFB" or should_skip(dbs_out, args.force)
    ):
        print(f"skip {subject} {block} (exists)")
        return 0
    if not eeg_in.exists():
        print(f"missing {eeg_in}; run build_raw_per_subject first", file=sys.stderr)
        return 2

    meta = setup_run("build_epochs_from_raw", config)

    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        raw_eeg = mne.io.read_raw_fif(eeg_in, preload=True, verbose="WARNING")
        raw_dbs = (
            mne.io.read_raw_fif(dbs_in, preload=True, verbose="WARNING")
            if (cohort == "Patient_NFB" and dbs_in.exists()) else None
        )
    patch_legacy_annotations(raw_eeg)
    if raw_dbs is not None:
        patch_legacy_annotations(raw_dbs)

    if block == "baseline":
        eeg_ep = _build_baseline_epochs(raw_eeg)
        dbs_ep = _build_baseline_epochs(raw_dbs) if raw_dbs is not None else None
        n_cycles_eeg = len(eeg_ep)
    else:
        eeg_ep, n_cycles_eeg = _build_task_epochs(
            raw_eeg, DEFAULT_GAIT_SEQ, DEFAULT_EVENT_PERCENTAGES
        )
        if raw_dbs is not None:
            dbs_ep, _ = _build_task_epochs(
                raw_dbs, DEFAULT_GAIT_SEQ, DEFAULT_EVENT_PERCENTAGES
            )
        else:
            dbs_ep = None

    # Legacy truncation: keep the shorter modality so cycles line up.
    if eeg_ep is not None and dbs_ep is not None:
        min_n = min(len(eeg_ep), len(dbs_ep))
        if len(eeg_ep) > min_n:
            eeg_ep = eeg_ep[:min_n]
        if len(dbs_ep) > min_n:
            dbs_ep = dbs_ep[:min_n]

    _save(eeg_ep, eeg_out)
    if cohort == "Patient_NFB":
        _save(dbs_ep, dbs_out)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "eeg_input": str(eeg_in),
        "dbs_input": str(dbs_in) if raw_dbs is not None else None,
        "eeg_output": str(eeg_out),
        "dbs_output": str(dbs_out) if cohort == "Patient_NFB" else None,
        "n_eeg_epochs": int(len(eeg_ep)) if eeg_ep is not None else 0,
        "n_dbs_epochs": int(len(dbs_ep)) if dbs_ep is not None else 0,
        "n_cycles_segmented": int(n_cycles_eeg),
    })
    write_sidecar(eeg_out, meta)
    print(f"wrote {eeg_out.name}  eeg={len(eeg_ep) if eeg_ep is not None else 0}  "
          f"dbs={len(dbs_ep) if dbs_ep is not None else 0}  segmented={n_cycles_eeg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
