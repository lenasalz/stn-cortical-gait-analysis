"""Compute Welch PSD for one subject × block, write per-subject NetCDF.

Inputs:
  - data/derivatives/epochs/<subj>_<block>-epo.fif  (produced by build_epochs.py)
  - code/analysis/config.yaml (psd.* parameters)
Outputs:
  - data/derivatives/psd/<subj>_<block>-psd.nc      (xarray DataArray: chan × freq)
  - data/derivatives/psd/<subj>_<block>-psd.nc.meta.json (sidecar)
Description:
  Median-across-epochs Welch PSD per channel. Replaces the
  per-cell `compute_psd().plot()` loops in
  ngb_study/notebooks/analysis/analysis.ipynb cells 70–73 (which
  recomputed PSD on every notebook execution and embedded per-subject
  figures into the saved notebook).

Reference port:
  ngb_study/notebooks/analysis/scripts/calculations.py:get_averaged_psd
  — Welch with hann window, 2-D input (n_epochs, n_samples), median
  across epochs (axis=0). Generalized here to N-channel input.

Usage:
  python compute_psd.py --subject P4-2002 --block pre8walk
  python compute_psd.py --subject P4-2002 --block pre8walk --modality dbs
  python compute_psd.py --subject P4-2002 --block pre8walk --force
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


def compute_psd_one(
    epochs_path: Path,
    fmin: float,
    fmax: float,
    nperseg: int,
    average: str,
):
    """Median (or mean) Welch PSD across epochs, per channel.
    Returns (freqs, psd[chan, freq], ch_names, sfreq).
    """
    import mne
    from scipy.signal import welch

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")
    data = epochs.get_data()  # (n_epochs, n_chans, n_samples)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3-D epochs array (n_epochs, n_chans, n_samples), "
            f"got shape {data.shape}."
        )
    sfreq = epochs.info["sfreq"]

    # nperseg may exceed segment length on short epochs — cap defensively.
    nperseg_eff = min(nperseg, data.shape[-1])

    freqs, pxx = welch(
        data, fs=sfreq, window="hann", nperseg=nperseg_eff, axis=-1
    )  # pxx: (n_epochs, n_chans, n_freqs)

    if average == "median":
        psd = np.median(pxx, axis=0)
    elif average == "mean":
        psd = np.mean(pxx, axis=0)
    else:
        raise ValueError(f"Unknown average={average!r}; expected 'median' or 'mean'.")

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[band_mask], psd[:, band_mask], epochs.ch_names, sfreq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg",
                        help="Which modality's epochs to read (default: eeg).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output exists.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    in_path = output_path(paths.epochs_dir, subject, block, f"-{args.modality}-epo.fif")
    out_path = output_path(paths.psd_dir, subject, block, f"-{args.modality}-psd.nc")

    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} (exists: {out_path.name})")
        return 0
    if not in_path.exists():
        print(f"missing input {in_path}; run build_epochs.py first", file=sys.stderr)
        return 2

    meta = setup_run("compute_psd", config)
    psd_cfg = config["psd"]

    freqs, psd, ch_names, sfreq = compute_psd_one(
        in_path,
        fmin=float(psd_cfg["fmin"]),
        fmax=float(psd_cfg["fmax"]),
        nperseg=int(psd_cfg["nperseg"]),
        average=psd_cfg["average"],
    )

    import xarray as xr  # lazy import keeps env requirements local to this script

    da = xr.DataArray(
        psd,
        dims=("channel", "freq"),
        coords={"channel": ch_names, "freq": freqs},
        attrs={
            "subject": subject,
            "cohort": cohort,
            "block": block,
            "sfreq": float(sfreq),
            "method": "welch",
            "average": psd_cfg["average"],
            "nperseg": int(psd_cfg["nperseg"]),
            "fmin": float(psd_cfg["fmin"]),
            "fmax": float(psd_cfg["fmax"]),
        },
    )
    with atomic_write(out_path) as tmp:
        da.to_netcdf(tmp)

    meta.update({
        "subject": subject,
        "cohort": cohort,
        "block": block,
        "modality": args.modality,
        "input": str(in_path),
        "output": str(out_path),
        "n_channels": len(ch_names),
        "n_freqs": int(freqs.size),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.relative_to(out_path.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
