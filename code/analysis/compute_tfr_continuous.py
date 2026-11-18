"""Continuous Morlet TFR on Raw input — required by basenorm.

Inputs:
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif       (build_raw_from_pickle.py)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-proc-raw.fif  (Patient_NFB only)
  - code/analysis/config.yaml (tfr.* parameters)
Outputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-cont.h5   (AverageTFR)
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-cont.h5.meta.json
Description:
  Runs `tfr_morlet` on the **raw continuous recording** (not on epochs).
  Counterpart of `compute_tfr.py`, which reads gait-warped epochs.
  The continuous TFR is the input to the basenorm pipeline
  (`compute_tfr_basenorm.py`): the legacy paper computes baseline power
  by time-averaging this object before normalizing the task TFR.

  Channel set: MNE's default (excludes bads + stim). Use `--regional`
  to restrict to motor + frontal channels listed in
  `config.yaml:connectivity.channels_of_interest` — matches the
  legacy `compute_tfr_on_subset_and_normalize_optimized` channel pick.

  Disk: at sfreq=2000 Hz with `continuous_decim=10`, EEG produces
  ~150-450 MB per subject per block depending on duration. The full
  cohort sweep (40 jobs) lands in the 5-15 GB range.

  DBS modality reads `-dbs-proc-raw.fif` (the bandpass-filtered DBS,
  matching the legacy `dbs_data_proc` slot in `all_data_dico`).

Usage:
  python compute_tfr_continuous.py --subject P4-2002 --block baseline --modality eeg
  python compute_tfr_continuous.py --subject P4-2002 --block pre8walk --modality eeg --regional
  python compute_tfr_continuous.py --subject P4-2002 --block pre8walk --modality dbs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _regional_picks(config: dict) -> list[str]:
    """Union of motor + frontal channel groups from config.yaml."""
    coi = config["connectivity"]["channels_of_interest"]
    chans: list[str] = []
    for region in ("motor", "frontal"):
        for side in ("Left", "Right", "Central"):
            chans.extend(coi[region].get(side, []))
    # de-duplicate while preserving order
    seen = set()
    out = []
    for c in chans:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def compute_continuous_tfr(raw_path: Path, freqs: np.ndarray,
                           n_cycles: np.ndarray, decim: int,
                           target_sfreq: int | None, regional_picks: list[str] | None,
                           n_jobs: int):
    """Run tfr_morlet on a Raw object. Returns AverageTFR.

    Mirrors the legacy call in
    final_analysis/scripts/tfr.py:compute_tfr_on_subset_and_normalize_optimized:
        tfr_morlet(raw, freqs, n_cycles, return_itc=False, average=False,
                   n_jobs=n_jobs, decim=tfr_decim, use_fft=True)
    For Raw input the `average` flag is irrelevant (no epoch axis); MNE
    returns AverageTFR shape `(n_chans, n_freqs, n_times)`.
    """
    import mne

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="WARNING")
    if regional_picks is not None:
        present = [ch for ch in regional_picks if ch in raw.ch_names]
        if not present:
            raise RuntimeError(
                f"none of the regional picks {regional_picks!r} are present in {raw_path.name}"
            )
        raw = raw.pick(present, verbose="WARNING")
    if target_sfreq:
        raw = raw.resample(target_sfreq, n_jobs=n_jobs)

    tfr = mne.time_frequency.tfr_morlet(
        raw, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=False,
        decim=decim, n_jobs=n_jobs, use_fft=True,
        verbose="WARNING",
    )
    return tfr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--regional", action="store_true",
                        help="Pick only motor + frontal channels per config "
                             "(matches legacy compute_tfr_on_subset_*).")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    in_suffix = "-eeg-raw.fif" if args.modality == "eeg" else "-dbs-proc-raw.fif"
    in_path = output_path(paths.raw_clean_dir, subject, block, in_suffix)
    out_path = output_path(paths.tfr_dir, subject, block, f"-{args.modality}-tfr-cont.h5")

    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    if not in_path.exists():
        print(f"missing input {in_path}; run build_raw_from_pickle.py first",
              file=sys.stderr)
        return 2

    meta = setup_run("compute_tfr_continuous", config)
    tfr_cfg = config["tfr"]
    f = tfr_cfg["freqs"]
    freqs = np.arange(f["fmin"], f["fmax"] + f["fstep"], f["fstep"])
    fwhm_s = float(tfr_cfg["fwhm_s"])
    sigma = fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    n_cycles = 2 * np.pi * freqs * sigma

    decim_cfg = tfr_cfg.get("continuous_decim", 10)
    if isinstance(decim_cfg, dict):
        decim = int(decim_cfg.get(args.modality, decim_cfg.get("eeg", 10)))
    else:
        decim = int(decim_cfg)
    picks = _regional_picks(config) if args.regional else None

    tfr = compute_continuous_tfr(
        in_path, freqs=freqs, n_cycles=n_cycles, decim=decim,
        target_sfreq=tfr_cfg.get("target_sfreq"),
        regional_picks=picks,
        n_jobs=args.n_jobs,
    )

    import mne
    with atomic_write(out_path) as tmp:
        mne.time_frequency.write_tfrs(tmp, tfr, overwrite=True)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality,
        "input": str(in_path), "output": str(out_path),
        "n_freqs": int(freqs.size),
        "fmin": float(freqs[0]), "fmax": float(freqs[-1]),
        "fwhm_s": fwhm_s,
        "decim": decim,
        "regional_picks": picks,
        "n_channels": len(tfr.ch_names),
        "n_times": int(tfr.data.shape[-1]),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  shape={tfr.data.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
