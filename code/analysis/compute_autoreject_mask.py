"""Per-subject autoreject (EEG) + DBS-threshold artifact mask.

Inputs:
  - data/derivatives/epochs/<subj>_<block>-eeg-epo.fif  (build_epochs.py)
  - data/derivatives/epochs/<subj>_<block>-dbs-epo.fif  (Patient_NFB only)
  - code/analysis/config.yaml (repro.seed)
Outputs:
  - data/derivatives/autoreject/<subj>_<block>-mask.npz
      keep_mask (bool array, n_cycles)
      bad_eeg, bad_dbs (component bools)
  - data/derivatives/autoreject/<subj>_<block>-mask.npz.meta.json
Description:
  Port of `clean_epochs_autoreject_paired` from
  ngb_study/notebooks/final_analysis/scripts/epoching.py. For each
  (subject, block):
    - Run `autoreject.AutoReject(random_state=42)` on EEG epochs to get
      `bad_eeg` per cycle.
    - Run `find_bad_dbs_epochs(dbs, threshold=auto_threshold_single_channel(dbs))`
      on DBS epochs to get `bad_dbs` per cycle (Patient_NFB only).
    - keep_mask = NOT (bad_eeg OR bad_dbs).
    - HC subjects (no DBS) get keep_mask = NOT bad_eeg.

  Replaces the dependency on the cached `rejection_mask_dico.pkl` (which
  is from an older autoreject run and produces different kept counts
  than the legacy notebook's last-run stdout). After this stage:
  `compute_tfr_gait.py` and `compute_gerd_table.py` read per-subject
  masks here in preference to the legacy pkl.

Usage:
  python compute_autoreject_mask.py --subject P4-2002 --block pre8walk
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
from epoching import auto_threshold_single_channel, find_bad_dbs_epochs


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
    seed = int(config["repro"]["seed"])

    autoreject_dir = paths.derivatives_root / "autoreject"
    autoreject_dir.mkdir(parents=True, exist_ok=True)
    out_path = autoreject_dir / f"{subject}_{block}-mask.npz"
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} (exists)")
        return 0

    eeg_in = output_path(paths.epochs_dir, subject, block, "-eeg-epo.fif")
    dbs_in = output_path(paths.epochs_dir, subject, block, "-dbs-epo.fif")
    if not eeg_in.exists():
        print(f"missing {eeg_in}; run build_epochs.py first", file=sys.stderr)
        return 2

    meta = setup_run("compute_autoreject_mask", config)

    import mne
    from autoreject import AutoReject

    eeg = mne.read_epochs(eeg_in, preload=True, verbose="WARNING")
    n_orig = len(eeg)

    # EEG autoreject (matches the legacy call:
    # AutoReject(random_state=42, n_jobs=-1, verbose=False))
    print(f"running AutoReject on {subject} {block} EEG (n_orig={n_orig}) …")
    ar = AutoReject(random_state=seed, n_jobs=-1, verbose=False)
    _, log = ar.fit(eeg).transform(eeg, return_log=True)
    bad_eeg = np.asarray(log.bad_epochs, dtype=bool)

    # DBS threshold-based rejection (Patient_NFB only)
    if dbs_in.exists() and cohort == "Patient_NFB":
        dbs = mne.read_epochs(dbs_in, preload=True, verbose="WARNING")
        if len(dbs) != n_orig:
            print(f"WARN dbs n={len(dbs)} != eeg n={n_orig}; using EEG mask only",
                  file=sys.stderr)
            bad_dbs = np.zeros(n_orig, dtype=bool)
        else:
            thr = auto_threshold_single_channel(dbs)
            bad_dbs = find_bad_dbs_epochs(dbs, threshold=thr)
    else:
        bad_dbs = np.zeros(n_orig, dtype=bool)

    keep_mask = ~(bad_eeg | bad_dbs)

    # Open as a binary file handle so np.savez writes to *this* path
    # instead of auto-appending '.npz' (which would land beside the
    # atomic_write tmp and leave atomic_write to rename an empty
    # mkstemp placeholder over the destination).
    with atomic_write(out_path) as tmp:
        with open(tmp, "wb") as fh:
            np.savez(fh, keep_mask=keep_mask, bad_eeg=bad_eeg, bad_dbs=bad_dbs)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "output": str(out_path),
        "n_orig": int(n_orig),
        "n_kept": int(keep_mask.sum()),
        "n_bad_eeg": int(bad_eeg.sum()),
        "n_bad_dbs": int(bad_dbs.sum()),
        "ar_random_state": seed,
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  n_kept={keep_mask.sum()}/{n_orig}  "
          f"(bad_eeg={bad_eeg.sum()}, bad_dbs={bad_dbs.sum()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
