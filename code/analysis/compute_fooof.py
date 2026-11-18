"""Per-subject FOOOF spectral parameterization on ROI-averaged PSDs.

Inputs:
  - data/derivatives/epochs/<subj>_<block>-eeg-epo.fif       (build_epochs.py)
  - data/derivatives/raw_clean/<subj>_<block>-dbs-proc-raw.fif (build_raw_from_pickle.py; --modality dbs)
  - code/analysis/config.yaml (fooof.*, freq_bands, channels_of_interest)
Outputs:
  - data/derivatives/fooof/<subj>_<block>-<modality>-fooof.csv
      one row per (region, side) with:
        subject, cohort, block, modality, region, side,
        n_channels, n_epochs, fit_r_squared, fit_error,
        aperiodic_offset, aperiodic_exponent,
        and per-band columns:
          <band>_peak_present (bool)
          <band>_cf            (center frequency, Hz; NaN if no peak)
          <band>_pw            (peak power, FOOOF model units; NaN)
          <band>_bw            (peak bandwidth; NaN)
  - data/derivatives/fooof/<subj>_<block>-<modality>-fooof.csv.meta.json
Description:
  Per-subject orchestrator over the migrated `psd_fooof.py` library. For
  each region.side defined in config.connectivity.channels_of_interest
  (motor, frontal × Left/Right/Central/Both) or for STN (DBS):
    1. Load epochs (EEG) or Raw (DBS), pick the ROI channels.
    2. Compute Welch PSD (`mne.Epochs.compute_psd` for EEG epochs;
       direct `scipy.signal.welch` for DBS Raw) over the FOOOF freq range.
    3. Average across epochs/channels → single 1-D PSD per region.side.
    4. Fit `FOOOFGroup`/`FOOOF` with config params
       (peak_width_limits=[2, 12.0], max_n_peaks=3, min_peak_height=0.15,
       aperiodic_mode='fixed', freq_range=[4.0, 45.0] — verbatim from
       analysis.ipynb cell 77 FOOOFGroup call).
    5. Extract aperiodic parameters + per-band peak (closest peak in
       each `freq_bands` window).

  Paper tables `tab1_stn_fooof_stats`, `tab2_cortical_psd_between_cohort`,
  `tab9_aperiodic_stats` are computed downstream from this output (LMM /
  Welch's t-test / Mann-Whitney with FDR correction) — that step is a
  separate aggregator script.

Usage:
  python compute_fooof.py --subject P4-2002 --block pre8walk --modality eeg
  python compute_fooof.py --subject P4-2002 --block baseline --modality dbs
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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


def _region_channels(config: dict, region: str, side: str) -> list[str]:
    coi = config["connectivity"]["channels_of_interest"][region]
    if side == "Both":
        return list(dict.fromkeys(
            coi.get("Left", []) + coi.get("Right", []) + coi.get("Central", [])
        ))
    return list(coi.get(side, []))


def _fit_fooof_one(freqs: np.ndarray, psd: np.ndarray, fcfg: dict
                   ) -> dict:
    """Fit a single FOOOF model on a 1-D averaged PSD.

    Returns dict with aperiodic params, fit quality, and per-band peak
    summaries (cf/pw/bw, NaN where no peak fell in the band).
    """
    from fooof import FOOOF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = FOOOF(
            peak_width_limits=tuple(fcfg["peak_width_limits"]),
            max_n_peaks=int(fcfg["max_n_peaks"]),
            min_peak_height=float(fcfg["min_peak_height"]),
            aperiodic_mode=fcfg["aperiodic_mode"],
            verbose=False,
        )
        fm.fit(freqs, psd, tuple(fcfg["freq_range"]))

    aperiodic = fm.aperiodic_params_                      # [offset, exponent]
    out: dict = {
        "aperiodic_offset":   float(aperiodic[0]),
        "aperiodic_exponent": float(aperiodic[-1]),
        "fit_r_squared":      float(fm.r_squared_),
        "fit_error":          float(fm.error_),
    }
    # `peak_params_`: array of shape (n_peaks, 3): (cf, pw, bw)
    peaks = fm.peak_params_
    return out, peaks


def _assign_peaks_to_bands(peaks: np.ndarray, freq_bands: dict
                           ) -> dict[str, dict[str, float]]:
    """For each band, pick the peak whose CF falls in the band with the
    highest power. Returns {band: {cf, pw, bw} or NaN's}."""
    out: dict[str, dict[str, float]] = {}
    for name, (lo, hi) in freq_bands.items():
        if peaks.size == 0:
            out[name] = {"present": False,
                         "cf": np.nan, "pw": np.nan, "bw": np.nan}
            continue
        in_band = (peaks[:, 0] >= float(lo)) & (peaks[:, 0] <= float(hi))
        if not in_band.any():
            out[name] = {"present": False,
                         "cf": np.nan, "pw": np.nan, "bw": np.nan}
            continue
        winners = peaks[in_band]
        top = winners[np.argmax(winners[:, 1])]
        out[name] = {"present": True,
                     "cf": float(top[0]), "pw": float(top[1]),
                     "bw": float(top[2])}
    return out


def _psd_for_eeg_region(epochs_path: Path, channels: list[str],
                        fmin: float, fmax: float):
    """Average Welch PSD across epochs and ROI channels → 1-D `(n_freqs,)` PSD."""
    import mne

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")
    present = [c for c in channels if c in epochs.ch_names
               and c not in epochs.info["bads"]]
    if not present:
        return None, None, 0, 0
    epochs = epochs.copy().pick(present, verbose="WARNING")
    n_fft = len(epochs.times)
    spectrum = epochs.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose="WARNING",
    )
    data = spectrum.get_data(return_freqs=False)           # (n_epochs, n_chans, n_freqs)
    psd = data.mean(axis=(0, 1))                            # (n_freqs,)
    return spectrum.freqs, psd, len(present), data.shape[0]


def _psd_for_dbs(raw_path: Path, fmin: float, fmax: float):
    """Welch PSD on the single DBS channel. Returns (freqs, psd_1d, 1, 0)."""
    import mne

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="WARNING")
    n_fft = min(int(raw.info["sfreq"] * 4), raw.n_times)    # 4 s window
    spectrum = raw.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", n_fft=n_fft, verbose="WARNING",
    )
    data = spectrum.get_data(return_freqs=False).mean(axis=0)
    return spectrum.freqs, data, 1, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    out_path = output_path(paths.fooof_dir, subject, block,
                           f"-{args.modality}-fooof.csv")
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0

    fcfg = config["fooof"]
    freq_bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    fmin, fmax = float(fcfg["freq_range"][0]), float(fcfg["freq_range"][1])

    meta = setup_run("compute_fooof", config)
    rows: list[dict] = []

    if args.modality == "eeg":
        epochs_path = output_path(paths.epochs_dir, subject, block, "-eeg-epo.fif")
        if not epochs_path.exists():
            print(f"missing input {epochs_path}; run build_epochs.py first",
                  file=sys.stderr)
            return 2
        targets = [
            (region, side)
            for region in ("motor", "frontal")
            for side in ("Left", "Right", "Central", "Both")
        ]
        for region, side in targets:
            channels = _region_channels(config, region, side)
            freqs, psd, n_ch, n_ep = _psd_for_eeg_region(
                epochs_path, channels, fmin, fmax
            )
            if psd is None:
                continue
            ap, peaks = _fit_fooof_one(freqs, psd, fcfg)
            band_peaks = _assign_peaks_to_bands(peaks, freq_bands)
            row = {
                "subject": subject, "cohort": cohort, "block": block,
                "modality": "eeg", "region": region, "side": side,
                "n_channels": n_ch, "n_epochs": n_ep,
                **ap,
            }
            for band, vals in band_peaks.items():
                row[f"{band}_peak_present"] = vals["present"]
                row[f"{band}_cf"] = vals["cf"]
                row[f"{band}_pw"] = vals["pw"]
                row[f"{band}_bw"] = vals["bw"]
            rows.append(row)
    else:  # dbs
        raw_path = output_path(paths.raw_clean_dir, subject, block,
                               "-dbs-proc-raw.fif")
        if not raw_path.exists():
            print(f"missing input {raw_path}; run build_raw_from_pickle.py first",
                  file=sys.stderr)
            return 2
        freqs, psd, n_ch, _ = _psd_for_dbs(raw_path, fmin, fmax)
        ap, peaks = _fit_fooof_one(freqs, psd, fcfg)
        band_peaks = _assign_peaks_to_bands(peaks, freq_bands)
        # DBS "region" is STN; side derives from raw.info description
        import mne
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="WARNING")
        desc = (raw.info.get("description") or "").lower()
        side = "Left" if "left" in desc else "Right" if "right" in desc else "Unknown"
        row = {
            "subject": subject, "cohort": cohort, "block": block,
            "modality": "dbs", "region": "STN", "side": side,
            "n_channels": 1, "n_epochs": 0,
            **ap,
        }
        for band, vals in band_peaks.items():
            row[f"{band}_peak_present"] = vals["present"]
            row[f"{band}_cf"] = vals["cf"]
            row[f"{band}_pw"] = vals["pw"]
            row[f"{band}_bw"] = vals["bw"]
        rows.append(row)

    df = pd.DataFrame(rows)
    with atomic_write(out_path) as tmp:
        df.to_csv(tmp, index=False)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality,
        "output": str(out_path),
        "n_rows": len(df),
        "fooof_params": dict(fcfg),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  rows={len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
