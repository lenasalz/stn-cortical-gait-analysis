"""Build per-subject cleaned-Raw .fif files without ever holding the cohort in RAM.

Inputs:
  - Upstream raw EEG/DBS at $RESIDUAL_DATA_ROOT/<group>/<subj>/{ProcessedData,RawData}/...
    (file layout matches ngb_study/notebooks/analysis/scripts/data_loader.py:
     load_eeg_data + load_dbs_data)
  - code/analysis/config.yaml (cohorts, blocks)
Outputs:
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif       (always)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-filt-raw.fif  (Patient_NFB, sync ok; bandpass-filtered)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-unfilt-raw.fif (Patient_NFB, sync ok; raw with gait events, before bandpass)
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif.meta.json (sidecar; records `eeg_sync_cropped` flag because PD-cohort EEG is cropped to the DBS sync pulse while HC EEG is not)
Description:
  Per-subject replacement for ngb_study scripts/data_loader.py:create_data_dictionary.
  The legacy function builds an in-RAM dict of every subject's mne.io.Raw +
  DBS DataFrame, then pickles it (`all_data_PD.pkl` ~11 GB,
  `all_data_HAM.pkl` ~5.4 GB on this machine). Building those pickles
  was OOM-ing on the user's MacBook because peak RAM is ~= pickle size.

  This script does the same per-(subject, block) work but writes one .fif
  per subject inside the loop, then `del`s the references so Python releases
  the memory before the next subject. Peak RAM stays at ~one subject's data
  (~150-300 MB) regardless of cohort size.

  Note: on this machine the existing pickles live under
  `<data_root>/Final_filtered/`. To regenerate from raw, point
  $RESIDUAL_DATA_ROOT at the upstream raw layout (one level up from
  `Final_filtered/`) where load_eeg_data expects
  `<group_var>/<subject>/ProcessedData/EEG/<block>/<block>_proc.fif`.

Usage:
  python build_raw_per_subject.py --all
  python build_raw_per_subject.py --subject P4-2002 --block pre8walk
  python build_raw_per_subject.py --all --force        # recompute existing files
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

# Default DBS sampling rate used as fallback when sync fails. Matches the
# in-line `1 / 250` constant in data_loader.crop_to_walking_part_only.
_DEFAULT_DBS_SFREQ_HZ = 250

# Reuse helpers from code/analysis/_common.py and the migrated data_loader.
# Earlier versions imported data_loader from the ngb_study checkout; that
# version still had hardcoded `/home/jovyan/...` paths and `os.makedirs`
# side effects at module import (via variables.py), which crashed on any
# host that wasn't the original JupyterHub. Always use the migrated copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
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


def process_one(group_var: str, subject: str, block: str,
                data_root: Path, raw_clean_dir: Path,
                gait_event_colors: dict, resample_eeg: bool,
                crop_to_walking: bool, force: bool) -> str:
    """Load, align, optionally crop one (subject, block); write to disk; release.

    Returns one of: 'wrote', 'skipped', 'no_eeg', 'eeg_only', 'no_dbs_sync'.
    """
    from data_loader import (  # type: ignore  # noqa: E402
        add_gait_events_to_dbs_mne_file,
        align_dbs_with_eeg,
        convert_dbs_to_mne,
        crop_to_walking_part_only,
        load_dbs_data,
        load_eeg_data,
        preprocess_dbs_data,
    )

    eeg_out          = output_path(raw_clean_dir, subject, block, "-eeg-raw.fif")
    # Naming matches build_raw_from_pickle.py + downstream consumers
    # (compute_tfr_continuous reads `-dbs-proc-raw.fif`).
    dbs_proc_out = output_path(raw_clean_dir, subject, block, "-dbs-proc-raw.fif")
    dbs_raw_out  = output_path(raw_clean_dir, subject, block, "-dbs-raw.fif")
    no_dbs_sentinel  = output_path(raw_clean_dir, subject, block, "-dbs-raw.fif.missing")

    # Skip-if-exists must check every artifact this stage emits. For HC
    # cohort: only the EEG file. For PD: EEG + both DBS files (or the
    # `.missing` sentinel that records sync-failure / no-DBS).
    eeg_done = should_skip(eeg_out, force)
    pd_dbs_done = (
        should_skip(dbs_proc_out, force) and should_skip(dbs_raw_out, force)
    ) or should_skip(no_dbs_sentinel, force)
    if eeg_done and (group_var != "Patient_NFB" or pd_dbs_done):
        return "skipped"

    eeg = load_eeg_data(str(data_root), group_var, subject, block)
    if eeg is None:
        eeg_out.with_suffix(eeg_out.suffix + ".missing").touch()
        return "no_eeg"

    dbs_raw = dbs_proc = None
    dbs_side = None
    status = "eeg_only"
    eeg_sync_cropped = False

    if group_var == "Patient_NFB":
        dbs_df, dbs_side = load_dbs_data(str(data_root), subject, block)
        if dbs_df is not None:
            # align_dbs_with_eeg returns either (eeg_cropped, sync_time) on
            # success or a single None when the underlying DataFrame is
            # treated as empty. Defend against both shapes.
            align_result = align_dbs_with_eeg(eeg, dbs_df, block)
            if align_result is None:
                dbs_sync_pulse = None
            elif isinstance(align_result, tuple) and len(align_result) == 2:
                eeg, dbs_sync_pulse = align_result
                eeg_sync_cropped = True
            else:
                raise RuntimeError(
                    f"align_dbs_with_eeg returned unexpected shape: {align_result!r}"
                )

            dbs_raw = convert_dbs_to_mne(dbs_df, dbs_side)
            if dbs_sync_pulse is not None:
                dbs_raw = add_gait_events_to_dbs_mne_file(
                    dbs_raw, eeg, gait_event_colors, resample_dbs=False
                )
                dbs_proc = preprocess_dbs_data(dbs_raw)
                status = "wrote"
            elif block == "baseline":
                # Baseline DBS for subjects whose baseline was a separate
                # recording (e.g. P4-2010) has no sync pulse with EEG. Sync
                # is not needed for baseline — downstream only uses it for
                # time-averaged spectrum. Keep the bandpass-filtered Raw;
                # skip gait-events (none in baseline anyway).
                dbs_proc = preprocess_dbs_data(dbs_raw)
                status = "wrote"
            else:
                # Task block requires sync to align gait events. Drop DBS.
                dbs_raw = None
                status = "no_dbs_sync"

            if resample_eeg:
                dbs_fs = dbs_proc.info["sfreq"] if dbs_proc is not None else _DEFAULT_DBS_SFREQ_HZ
                eeg = eeg.resample(dbs_fs)
        else:
            status = "eeg_only"

        if crop_to_walking and block != "baseline":
            # Legacy signature: (eeg, raw_dbs, dbs) → (eeg, raw_dbs, dbs)
            # where `raw_dbs` is the unfiltered+events stream and `dbs` is
            # the bandpass-filtered one.
            eeg, dbs_raw, dbs_proc = crop_to_walking_part_only(
                eeg, dbs_raw, dbs_proc
            )

    eeg.save(eeg_out, overwrite=True, fmt="single")
    if dbs_proc is not None:
        dbs_proc.save(dbs_proc_out, overwrite=True, fmt="single")
    if dbs_raw is not None:
        dbs_raw.save(dbs_raw_out, overwrite=True, fmt="single")
    # Sentinel: records "DBS expected but absent" so the next run skips
    # this subject instead of re-loading the source pickle every time.
    if group_var == "Patient_NFB" and dbs_proc is None:
        no_dbs_sentinel.touch()

    write_sidecar(eeg_out, {
        "stage": "build_raw_per_subject",
        "subject": subject, "cohort": group_var, "block": block,
        "eeg_output": str(eeg_out),
        "dbs_proc_output": str(dbs_proc_out) if dbs_proc is not None else None,
        "dbs_raw_output": str(dbs_raw_out) if dbs_raw is not None else None,
        "dbs_side": dbs_side,
        "eeg_sync_cropped": eeg_sync_cropped,
        "status": status,
    })

    # Release before next subject. Peak RAM = one subject's data.
    del eeg, dbs_raw, dbs_proc
    gc.collect()
    return status if status != "eeg_only" else "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Process every (subject, block) combination from config.")
    g.add_argument("--subject", help="Single anonymized subject code.")
    parser.add_argument("--block", help="Required when --subject is given.")
    parser.add_argument("--no-resample-eeg", action="store_true",
                        help="Skip EEG resample to DBS sfreq.")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip cropping non-baseline blocks to walking part.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    setup_run("build_raw_per_subject", config)

    # Gait-event colors from the migrated variables.py (no side effects).
    from variables import gait_event_colors  # type: ignore  # noqa: E402

    if args.all:
        jobs = [
            (cohort, subj, block)
            for cohort, members in config["cohorts"].items()
            for subj in members
            for block in config["blocks"]
        ]
    else:
        if not args.block:
            parser.error("--block required with --subject")
        s = validate_subject(args.subject)
        b = validate_block(args.block, config)
        jobs = [(cohort_of(s, config), s, b)]

    counts = {
        "wrote": 0, "skipped": 0, "no_eeg": 0,
        "eeg_only": 0, "no_dbs_sync": 0, "failed": 0,
    }
    for cohort, subj, block in jobs:
        try:
            status = process_one(
                cohort, subj, block,
                data_root=paths.data_root,
                raw_clean_dir=paths.raw_clean_dir,
                gait_event_colors=gait_event_colors,
                resample_eeg=not args.no_resample_eeg,
                crop_to_walking=not args.no_crop,
                force=args.force,
            )
        except Exception:
            print(f"FAIL {subj} {block} ({cohort}):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            counts["failed"] += 1
            continue
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:11s}  {subj} {block} ({cohort})")

    print("---")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
