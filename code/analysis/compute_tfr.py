"""Compute Morlet TFR for one subject × block × modality, write HDF5.

Inputs:
  - data/derivatives/epochs/<subj>_<block>-{eeg,dbs}-epo.fif  (build_epochs.py)
  - code/analysis/config.yaml (tfr.* parameters)
Outputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr.h5      (AverageTFR by default)
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr.h5.meta.json (sidecar)
Description:
  Single Morlet pass per (subject, block, modality). Defaults to
  `average=True` → AverageTFR (n_chans × n_freqs × n_times), ~30 MB
  per subject. Pass `--per-epoch` for `average=False` → EpochsTFR
  (n_epochs × n_chans × n_freqs × n_times), ~3 GB per subject — what
  the legacy process_and_save_tfr_data produces, needed only for
  per-epoch permutation cluster stats.

  Channel selection: MNE's default (excludes bads + stim). Matches the
  legacy tfr.py behavior in scripts/tfr.py.

Usage:
  python compute_tfr.py --subject P4-2002 --block pre8walk --modality eeg
  python compute_tfr.py --subject P4-2002 --block pre8walk --modality eeg --per-epoch
  python compute_tfr.py --subject P4-2002 --block pre8walk --modality dbs
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


def compute_tfr_one(epochs_path: Path, freqs: np.ndarray, n_cycles: np.ndarray,
                    decim: int, target_sfreq: int | None, average: bool,
                    n_jobs: int):
    """Single Morlet TFR via mne.time_frequency.tfr_morlet on per-subject Epochs.

    Matches the legacy call in scripts/tfr.py: `return_itc=False,
    use_fft=True, decim=tfr_decim`. The only knob exposed at the CLI
    is `average` — legacy uses False for EEG (per-epoch needed for
    permutation stats) and True for DBS.
    """
    import mne

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")
    if target_sfreq:
        epochs = epochs.resample(target_sfreq, n_jobs=n_jobs)

    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=average,
        decim=decim,
        n_jobs=n_jobs,
        use_fft=True,
        verbose="WARNING",
    )
    return tfr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--per-epoch", action="store_true",
                        help="Save EpochsTFR (per-epoch). Default: AverageTFR.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallelism for tfr_morlet (default: 1).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    suffix = "-tfr-epochs.h5" if args.per_epoch else "-tfr.h5"
    in_path = output_path(paths.epochs_dir, subject, block, f"-{args.modality}-epo.fif")
    out_path = output_path(paths.tfr_dir, subject, block, f"-{args.modality}{suffix}")

    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    if not in_path.exists():
        print(f"missing input {in_path}; run build_epochs.py first", file=sys.stderr)
        return 2

    meta = setup_run("compute_tfr", config)
    tfr_cfg = config["tfr"]
    f = tfr_cfg["freqs"]
    freqs = np.arange(f["fmin"], f["fmax"] + f["fstep"], f["fstep"])
    # Evolving wavelets with target FWHM, matching the legacy formula in
    # tfr.py:compute_tfr_on_subset_and_normalize_optimized.
    fwhm_s = float(tfr_cfg["fwhm_s"])
    sigma = fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    n_cycles = 2 * np.pi * freqs * sigma

    tfr = compute_tfr_one(
        in_path, freqs=freqs, n_cycles=n_cycles,
        decim=int(tfr_cfg["decim"]),
        target_sfreq=tfr_cfg.get("target_sfreq"),
        average=not args.per_epoch,
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
        "fwhm_s": float(tfr_cfg["fwhm_s"]),
        "decim": int(tfr_cfg["decim"]),
        "average": not args.per_epoch,
        "n_channels": len(tfr.ch_names),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  shape={tfr.data.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
