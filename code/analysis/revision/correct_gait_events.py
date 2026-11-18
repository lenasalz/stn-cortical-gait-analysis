"""Apply the documented gait-event corrections to the pickle-era raw_clean export.

Purpose:  Shift (and, where documented, truncate) the gait-event annotations
          of walks whose events were placed at the wrong Vicon start marker,
          so that the unchanged pipeline scripts read correctly placed events.
Inputs:   data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif
              (pickle-era export; first run only, then kept as the original)
          data/derivatives/raw_clean/original_events/<subj>_<block>-eeg-raw.fif
              (untouched copy made by this script; always the source)
          code/analysis/config.yaml  (gait_event_corrections, repro.seed)
          docs/vicon_start_check_2026-11-10.md  (evidence for the values)
Outputs:  data/derivatives/raw_clean/original_events/<subj>_<block>-eeg-raw.fif
              (+ the original .meta.json), written once, never overwritten
          data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif
              (same signal, corrected annotations)
          data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif.meta.json
              (original sidecar content plus a `gait_event_correction` entry)
Author:   Lena
Date:     2026-11-16
Seed:     config.yaml:repro.seed (42) via _common.setup_run; the correction
          itself is deterministic.

Description:
  For every subject in config.yaml:gait_event_corrections the script
    1. copies the export to raw_clean/original_events/ if no copy exists yet,
    2. checks that `shift_s` equals the difference between the first two
       "Enter pressed" annotations (to 0.01 s) and that the stored
       `viconStart` is the first press,
    3. removes gait events whose original onset is >= `drop_from_s`,
    4. adds `shift_s` to the onset of every remaining gait event
       (HS/FF/TO left/right) and moves `viconStart` to the second press,
    5. saves the Raw with the corrected annotations; the signal is untouched.
  The script always reads the copy in original_events/, so running it twice
  gives the same result. Only EEG files exist for the corrected subjects
  (healthy controls); a subject with a DBS file is rejected.

Usage:
  python code/analysis/revision/correct_gait_events.py            # all listed
  python code/analysis/revision/correct_gait_events.py --subject P4-1019
  python code/analysis/revision/correct_gait_events.py --dry-run
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    load_config,
    output_path,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    validate_block,
    validate_subject,
    write_sidecar,
)

GAIT_EVENTS = (
    "HSrightlocs", "HSleftlocs",
    "FFrightlocs", "FFleftlocs",
    "TOrightlocs", "TOleftlocs",
)
PRESS = "Enter pressed"
VICON_START = "viconStart"


def correct_subject(subject: str, spec: dict, paths, meta: dict, dry_run: bool) -> dict:
    import mne

    block = spec["block"]
    shift = float(spec["shift_s"])
    drop_from = spec.get("drop_from_s")
    drop_from = None if drop_from is None else float(drop_from)

    target = output_path(paths.raw_clean_dir, subject, block, "-eeg-raw.fif")
    dbs = output_path(paths.raw_clean_dir, subject, block, "-dbs-proc-raw.fif")
    if dbs.exists():
        raise RuntimeError(f"{subject}: a DBS file exists; this script corrects EEG-only subjects.")
    orig_dir = paths.raw_clean_dir / "original_events"
    orig = orig_dir / target.name
    target_meta = target.with_suffix(target.suffix + ".meta.json")
    orig_meta = orig.with_suffix(orig.suffix + ".meta.json")

    if not orig.exists():
        if not target.exists():
            raise FileNotFoundError(target)
        if dry_run:
            source = target
        else:
            orig_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, orig)
            if target_meta.exists():
                shutil.copy2(target_meta, orig_meta)
            source = orig
    else:
        source = orig

    raw = mne.io.read_raw_fif(source, preload=True, verbose="error")
    patch_legacy_annotations(raw)
    ann = raw.annotations
    onset = np.asarray(ann.onset, dtype=float)
    desc = np.asarray(ann.description)

    presses = np.sort(onset[desc == PRESS])
    if len(presses) < 2:
        raise RuntimeError(f"{subject}: fewer than two '{PRESS}' annotations.")
    press_diff = float(presses[1] - presses[0])
    if abs(press_diff - shift) > 0.01:
        raise RuntimeError(
            f"{subject}: shift_s {shift} does not match the press difference {press_diff:.3f} s.")
    vs = onset[desc == VICON_START]
    if len(vs) != 1 or abs(float(vs[0]) - float(presses[0])) > 0.01:
        raise RuntimeError(f"{subject}: stored viconStart is not the first press; already corrected?")

    is_gait = np.isin(desc, GAIT_EVENTS)
    drop = is_gait & (onset >= drop_from) if drop_from is not None else np.zeros(len(onset), bool)
    new_onset = onset.copy()
    new_onset[is_gait] += shift
    new_onset[desc == VICON_START] = presses[1]
    keep = ~drop

    is_hs = np.isin(desc, ("HSrightlocs", "HSleftlocs"))
    info = {
        "config_entry": {"block": block, "shift_s": shift, "drop_from_s": drop_from},
        "press_times_s": [round(float(p), 3) for p in presses[:2]],
        "press_difference_s": round(press_diff, 3),
        "n_gait_events_original": int(is_gait.sum()),
        "n_gait_events_dropped": int(drop.sum()),
        "n_heel_strikes_original": int(is_hs.sum()),
        "n_heel_strikes_kept": int((is_hs & keep).sum()),
        "source": str(orig.relative_to(paths.derivatives_root.parent.parent)),
        "evidence": "docs/vicon_start_check_2026-11-10.md",
    }
    print(f"{subject} {block}: shift {shift:+.2f} s (presses {presses[0]:.2f}, {presses[1]:.2f}); "
          f"heel strikes {info['n_heel_strikes_original']} -> {info['n_heel_strikes_kept']}; "
          f"gait events dropped {info['n_gait_events_dropped']}")
    if dry_run:
        return info

    new_ann = mne.Annotations(
        onset=new_onset[keep],
        duration=np.asarray(ann.duration, dtype=float)[keep],
        description=desc[keep],
        orig_time=ann.orig_time,
    )
    raw.set_annotations(new_ann, verbose="error")
    patch_legacy_annotations(raw)
    raw.save(target, overwrite=True, verbose="error")

    sidecar = json.loads(orig_meta.read_text()) if orig_meta.exists() else {}
    sidecar["gait_event_correction"] = {
        **info,
        "script": "code/analysis/revision/correct_gait_events.py",
        "seed": meta["seed"],
        "mne": meta["mne"],
        "numpy": meta["numpy"],
    }
    write_sidecar(target, sidecar)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", help="one subject code; default: all listed in config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("correct_gait_events", config)
    print(json.dumps({"seed": meta["seed"],
                      **{p: importlib.metadata.version(p) for p in ("numpy", "mne")}}))

    corrections = config.get("gait_event_corrections") or {}
    subjects = [args.subject] if args.subject else list(corrections)
    for subject in subjects:
        validate_subject(subject)
        if subject not in corrections:
            raise ValueError(f"{subject} has no entry in config.yaml:gait_event_corrections.")
        validate_block(corrections[subject]["block"], config)
        correct_subject(subject, corrections[subject], paths, meta, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
