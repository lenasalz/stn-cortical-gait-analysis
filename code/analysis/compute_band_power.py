"""Compute relative band power for one subject × block from cached PSD.

Inputs:
  - data/derivatives/psd/<subj>_<block>-psd.nc   (compute_psd.py)
  - code/analysis/config.yaml (freq_bands, psd.norm_freq_range)
Outputs:
  - data/derivatives/band_power/<subj>_<block>-bp.csv  (one row per channel × band)
  - data/derivatives/band_power/<subj>_<block>-bp.csv.meta.json
Description:
  Vectorized relative band power. Reads the cached PSD NetCDF, masks
  freq bins per band, and writes a tidy CSV (subject, cohort, block,
  channel, band, abs_power, rel_power). Replaces the
  4-deep nested `(region, group, freq_band, subject)` Python loop
  pattern in ngb_study/notebooks/analysis/scripts/calculations.py
  (`process_band_power_data_epochs`).

Usage:
  python compute_band_power.py --subject P4-2002 --block pre8walk

Reference port:
  ngb_study/notebooks/analysis/scripts/calculations.py:
    - get_averaged_psd  (already in compute_psd.py)
    - calculate_relative_band_power
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


def relative_power(freqs: np.ndarray, psd: np.ndarray,
                   band: tuple[float, float],
                   norm_range: tuple[float, float]) -> np.ndarray:
    """Median-relative band power per channel.

    Port of calculate_relative_band_power for vectorized N-channel input.
    Differs from the legacy reference in `scripts/calculations.py:412` in
    one place: the legacy version computes a *single scalar* normalization
    over the 1-D averaged PSD, while this function computes one
    normalization per channel. This matches the per-channel paradigm used
    elsewhere in the new pipeline; if the manuscript stats expect the
    pooled-PSD normalization, aggregate first then call this helper.

    Note on the norm range: when `norm_range == (psd.fmin, psd.fmax)` the
    relative spectrum is identical to the cached PSD normalized by its
    own median — i.e. the band-mask is the only operation that matters.
    The default norm range in config.yaml (1–45 Hz) equals the cached
    PSD bounds, so today this helper is functionally a band-median ratio.
    Widen `psd.norm_freq_range` if a wider normalization band is needed.

    Returns an array of shape (n_channels,).
    """
    norm_mask = (freqs >= norm_range[0]) & (freqs <= norm_range[1])
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not norm_mask.any() or not band_mask.any():
        return np.full(psd.shape[0], np.nan)

    norm_power = np.median(psd[:, norm_mask], axis=1)
    norm_power = np.where(norm_power == 0, np.nan, norm_power)
    rel_psd = psd / norm_power[:, np.newaxis]
    return np.median(rel_psd[:, band_mask], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg",
                        help="Which modality's PSD to read (default: eeg).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    in_path = output_path(paths.psd_dir, subject, block, f"-{args.modality}-psd.nc")
    out_path = output_path(paths.band_power_dir, subject, block, f"-{args.modality}-bp.csv")
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} (exists)")
        return 0
    if not in_path.exists():
        print(f"missing input {in_path}; run compute_psd.py first", file=sys.stderr)
        return 2

    meta = setup_run("compute_band_power", config)

    import pandas as pd
    import xarray as xr

    da = xr.open_dataarray(in_path)
    freqs = da["freq"].values
    psd = da.values  # (n_channels, n_freqs)
    ch_names = list(da["channel"].values)

    norm_range = tuple(config["psd"]["norm_freq_range"])
    rows = []
    for band_name, (lo, hi) in config["freq_bands"].items():
        rel = relative_power(freqs, psd, (float(lo), float(hi)), norm_range)
        band_mask = (freqs >= lo) & (freqs <= hi)
        # `median_psd_in_band`: median PSD value within the band, per channel.
        # NOT integrated band power — see calculations.py for the
        # trapezoidal-integration variant if the manuscript Methods report
        # absolute band power as ∫ PSD df.
        median_psd = np.median(psd[:, band_mask], axis=1) if band_mask.any() \
                     else np.full(psd.shape[0], np.nan)
        for i, ch in enumerate(ch_names):
            rows.append({
                "subject": subject, "cohort": cohort, "block": block,
                "channel": ch, "band": band_name,
                "median_psd_in_band": float(median_psd[i]),
                "rel_power": float(rel[i]),
            })
    df = pd.DataFrame(rows)
    with atomic_write(out_path) as tmp:
        df.to_csv(tmp, index=False)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "input": str(in_path), "output": str(out_path),
        "n_rows": len(df),
        "norm_range": list(norm_range),
        "bands": list(config["freq_bands"].keys()),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
