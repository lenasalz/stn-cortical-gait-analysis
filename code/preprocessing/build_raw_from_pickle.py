"""Materialize per-subject continuous Raw .fif files from the legacy cohort pickles.

Inputs:
  - $RESIDUAL_DATA_ROOT/Final_filtered/all_data_PD.pkl   (~11 GB, Patient_NFB)
  - $RESIDUAL_DATA_ROOT/Final_filtered/all_data_HAM.pkl  (~5.4 GB, AgeMatched_NFB)
  - code/analysis/config.yaml
Outputs:
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif
  - data/derivatives/raw_clean/<subj>_<block>-dbs-raw.fif       (Patient_NFB only)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-proc-raw.fif  (Patient_NFB only)
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif.meta.json
Description:
  Bridge from the legacy monolithic cohort pickles (one per cohort) to
  per-subject continuous Raw .fif files. Counterpart to build_epochs.py
  which extracts gait-warped epochs from `all_data_epoched_dico.pkl`.
  Continuous Raw is required by the basenorm pipeline
  (compute_tfr_continuous + compute_tfr_basenorm) which has to compute
  TFR on the full continuous baseline before any gait-cycle warping.

  Memory: loads one cohort pickle at a time (peak ~12 GB for PD), writes
  each subject's .fif inside the loop, then `del`s the dict.

  MNE compatibility: the legacy pickles were saved with an older MNE
  whose RawArray lived at `mne.io.array.array.RawArray`; MNE 1.12 moved
  it to `mne.io.array._array.RawArray`. The script aliases the old
  module path before unpickling.

Usage:
  python build_raw_from_pickle.py --cohort Patient_NFB
  python build_raw_from_pickle.py --all
  python build_raw_from_pickle.py --all --force
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import traceback
from pathlib import Path

# Reuse helpers from code/analysis/_common.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
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


COHORT_PICKLE = {
    "Patient_NFB":    "all_data_PD.pkl",   # ~11 GB
    "AgeMatched_NFB": "all_data_HAM.pkl",  # ~5.4 GB
}

# Per-subject channels that are corrupt in the source recording and must be
# dropped before TFR computation. See P7 diagnostic
# (reproduction_tests_matthias/report/p7_motor_basenorm_2x_2026-05-19.html)
# for the C2 / P4-2003 trace: baseline power ~10⁻¹⁰, task power ~10⁴⁴ →
# +200 dB basenorm outlier on a single channel, doubled the Fig 4 Motor bar.
BAD_CHANNEL_OVERRIDES = {
    "P4-2003": {"C2"},
}

# After dropping a bad channel, optionally rename a healthy neighbour to take
# its slot in any downstream channel-of-interest list (motor_areas /
# frontal_areas in variables.py). For P4-2003: FC2 is the anatomically
# nearest healthy channel to C2 in the 10-20 layout (right pre/motor cortex)
# and is NOT used in any other ROI (frontal_areas uses AF3/AF4/F1/F2/Fz,
# not the FC row), so the substitution doesn't disturb the frontal-region
# average.
CHANNEL_SUBSTITUTIONS = {
    "P4-2003": {"FC2": "C2"},
}


def _alias_legacy_mne_modules() -> None:
    """Map the old `mne.io.array.array` import path used in pickles to the
    current `mne.io.array._array`. No-op if the new module is unavailable.
    """
    import importlib

    try:
        new_mod = importlib.import_module("mne.io.array._array")
    except ImportError:
        return
    sys.modules.setdefault("mne.io.array.array", new_mod)


def _process_cohort(cohort: str, pickle_path: Path,
                    raw_clean_dir: Path, force: bool,
                    cohort_members: list[str],
                    side_overrides: dict[str, str] | None = None,
                    block_filter: str | None = None) -> dict[str, int]:
    """Load one cohort pickle, write each subject's .fif inside the loop."""
    counts = {"wrote": 0, "skipped": 0, "no_eeg": 0, "no_dbs": 0, "failed": 0}

    if not pickle_path.exists():
        print(f"missing {pickle_path}", file=sys.stderr)
        counts["failed"] = len(cohort_members) * 2
        return counts

    print(f"loading {pickle_path} (cohort={cohort})...")
    with pickle_path.open("rb") as f:
        dico = pickle.load(f)

    for block, payload in dico.items():
        if block_filter is not None and block != block_filter:
            continue
        ids = payload.get("ID", [])
        eeg_list = payload.get("EEG", [])
        dbs_list = payload.get("DBS", [])
        dbs_proc_list = payload.get("DBS_proc", [])

        for subject in cohort_members:
            try:
                idx = ids.index(subject)
            except ValueError:
                # Subject in config but not in pickle for this block.
                counts["failed"] += 1
                print(f"FAIL {subject} {block} ({cohort}): not in pickle ID list",
                      file=sys.stderr)
                continue

            eeg_out = output_path(raw_clean_dir, subject, block, "-eeg-raw.fif")
            dbs_out = output_path(raw_clean_dir, subject, block, "-dbs-raw.fif")
            dbs_proc_out = output_path(raw_clean_dir, subject, block, "-dbs-proc-raw.fif")

            if should_skip(eeg_out, force) and (
                cohort != "Patient_NFB"
                or (should_skip(dbs_out, force) and should_skip(dbs_proc_out, force))
            ):
                counts["skipped"] += 1
                continue

            try:
                eeg = eeg_list[idx] if idx < len(eeg_list) else None
                dbs = dbs_list[idx] if idx < len(dbs_list) else None
                dbs_proc = dbs_proc_list[idx] if idx < len(dbs_proc_list) else None

                if eeg is None:
                    eeg_out.with_suffix(eeg_out.suffix + ".missing").touch()
                    counts["no_eeg"] += 1
                else:
                    patch_legacy_annotations(eeg)
                    # Per-subject bad-channel overrides (channels that are
                    # corrupt in the source recording and need to be dropped
                    # before downstream TFR / basenorm runs). Verified by
                    # P7 diagnostic 2026-05-19: C2 for P4-2003 had baseline
                    # power ~1e-10 and task power ~1e+44 → +200 dB basenorm
                    # outlier amplified into 2x bar-height in Fig 4 Motor.
                    bad_overrides = BAD_CHANNEL_OVERRIDES.get(subject, set())
                    drop_now = [ch for ch in bad_overrides if ch in eeg.ch_names]
                    if drop_now:
                        eeg = eeg.copy().drop_channels(drop_now)
                        print(f"  dropped corrupt channels {drop_now} from {subject} {block}")
                    sub_map = CHANNEL_SUBSTITUTIONS.get(subject, {})
                    rename_now = {old: new for old, new in sub_map.items()
                                  if old in eeg.ch_names}
                    if rename_now:
                        eeg.rename_channels(rename_now)
                        print(f"  renamed {rename_now} for {subject} {block}")
                    eeg.save(eeg_out, overwrite=True, fmt="single")

                # The pickle's DBS Raw objects carry the side label baked in
                # at pickle-creation time, so config.yaml dbs_side_overrides
                # must be re-applied here (the load_dbs_data path is not
                # involved in this bridge).
                override_side = (side_overrides or {}).get(subject)
                if cohort == "Patient_NFB":
                    if dbs is not None:
                        patch_legacy_annotations(dbs)
                        if override_side:
                            dbs.info["description"] = f"DBS side: {override_side}"
                        dbs.save(dbs_out, overwrite=True, fmt="single")
                    if dbs_proc is not None:
                        patch_legacy_annotations(dbs_proc)
                        if override_side:
                            dbs_proc.info["description"] = f"DBS side: {override_side}"
                        dbs_proc.save(dbs_proc_out, overwrite=True, fmt="single")
                    if dbs is None and dbs_proc is None:
                        counts["no_dbs"] += 1
                    if override_side and (dbs is not None or dbs_proc is not None):
                        print(f"  applied DBS side override for {subject}: "
                              f"{override_side!r} (config.yaml dbs_side_overrides)")

                if eeg is not None:
                    counts["wrote"] += 1
                    write_sidecar(eeg_out, {
                        "stage": "build_raw_from_pickle",
                        "subject": subject, "cohort": cohort, "block": block,
                        "source_pickle": str(pickle_path),
                        "eeg_output": str(eeg_out),
                        "dbs_output": str(dbs_out) if dbs is not None else None,
                        "dbs_proc_output": str(dbs_proc_out) if dbs_proc is not None else None,
                        "eeg_n_times": int(eeg.n_times),
                        "eeg_sfreq": float(eeg.info["sfreq"]),
                        "dbs_side_override": override_side,
                    })
                    print(f"wrote {subject} {block} ({cohort})")
            except Exception:
                counts["failed"] += 1
                print(f"FAIL {subject} {block} ({cohort}):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue

    del dico
    gc.collect()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Process every cohort listed in config.yaml.")
    g.add_argument("--cohort", choices=list(COHORT_PICKLE.keys()),
                   help="Process a single cohort.")
    g.add_argument("--subject",
                   help="Single anonymized subject code (loads only that "
                        "subject's cohort pickle).")
    parser.add_argument("--block",
                        help="With --subject: restrict to one block.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    _alias_legacy_mne_modules()

    config = load_config()
    paths = resolve_paths(config)
    setup_run("build_raw_from_pickle", config)

    subject_filter = block_filter = None
    if args.subject:
        subject_filter = validate_subject(args.subject)
        if args.block:
            block_filter = validate_block(args.block, config)
        cohorts_to_run = [cohort_of(subject_filter, config)]
    else:
        cohorts_to_run = (list(config["cohorts"].keys()) if args.all
                          else [args.cohort])

    side_overrides = config.get("dbs_side_overrides") or {}

    totals = {"wrote": 0, "skipped": 0, "no_eeg": 0, "no_dbs": 0, "failed": 0}
    for cohort in cohorts_to_run:
        if cohort not in COHORT_PICKLE:
            print(f"unknown cohort {cohort}; expected {list(COHORT_PICKLE.keys())}",
                  file=sys.stderr)
            return 2
        pickle_path = paths.data_root / "Final_filtered" / COHORT_PICKLE[cohort]
        members = ([subject_filter] if subject_filter
                   else config["cohorts"][cohort])
        counts = _process_cohort(cohort, pickle_path, paths.raw_clean_dir,
                                 args.force, members,
                                 side_overrides=side_overrides,
                                 block_filter=block_filter)
        for k, v in counts.items():
            totals[k] += v

    print("---")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
