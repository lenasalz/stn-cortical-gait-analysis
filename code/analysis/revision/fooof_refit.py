"""FOOOF refit on the unwarped ROI spectra with a min_peak_height grid (R1.2, R1.3).

Purpose:
  Fit the spectral parameterisation model to every ROI spectrum produced
  by psd_unwarped.py (both spectrum variants A and B) for a sensitivity
  grid of min_peak_height in {0.05, 0.09, 0.15}. All other settings are
  the submitted ones (config.yaml:fooof): peak_width_limits [2, 12],
  max_n_peaks 3, aperiodic_mode fixed, fit range 4-45 Hz; the remaining
  arguments are the fooof 1.1.1 library defaults (peak_threshold 2.0 SD,
  reported in the output). Besides the submitted peak summaries the table
  carries two peak-free measures per band:
    (i)  <Band>_adj_power: aperiodic-adjusted band power = mean over the
         band bins of the FOOOF flattened spectrum, i.e.
         log10(power) - aperiodic fit, recomputed as
         fm.power_spectrum - fm._ap_fit (identical to fm._spectrum_flat
         in fooof 1.1.1; the recomputation is used so that the value does
         not depend on a private attribute).
    (ii) <Band>_rel_power: relative band power from the raw spectrum =
         sum of linear power in the band / sum over 4-45 Hz.
  Band bins: lo <= f < hi for Theta, Alpha, Low Beta, High Beta and
  lo <= f <= hi for Gamma, so that shared band edges (8, 12, 20, 30 Hz)
  are counted once.

Inputs:
  - data/derivatives/psd_unwarped/<subj>_<block>-<modality>-psd.csv
    (psd_unwarped.py)
  - code/analysis/config.yaml (fooof, freq_bands, repro.seed)
  - code/analysis/compute_fooof.py (_assign_peaks_to_bands; same band rule
    as the submitted pipeline: peak CF in [lo, hi], highest peak wins)
Outputs:
  - outputs/stats/revision_2026-10/fooof_unwarped_2026-10-13.csv
      one row per subject x block x modality x region x spectrum_variant x
      min_peak_height  (+ .meta.json)
Author: Lena
Date: 2026-10-13
Seed: config.yaml:repro.seed (42) via _common.setup_run (FOOOF fitting is
  deterministic; seed set for completeness).

Usage:
  python fooof_refit.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _common import (  # noqa: E402
    PROJECT_ROOT,
    atomic_write,
    load_config,
    setup_run,
    write_sidecar,
)
from compute_fooof import _assign_peaks_to_bands  # noqa: E402

# Output date tag; override with the environment variable REVISION_DATE for a rerun.
DATE = __import__("os").environ.get("REVISION_DATE", "2026-10-13")
STAGE = "fooof_refit"
IN_DIR = PROJECT_ROOT / "data" / "derivatives" / "psd_unwarped"
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
OUT = STATS_DIR / f"fooof_unwarped_{DATE}.csv"
MIN_PEAK_HEIGHTS = (0.05, 0.09, 0.15)


def band_mask(freqs: np.ndarray, lo: float, hi: float, last: bool) -> np.ndarray:
    if last:
        return (freqs >= lo) & (freqs <= hi)
    return (freqs >= lo) & (freqs < hi)


def fit_one(freqs: np.ndarray, psd: np.ndarray, fcfg: dict, mph: float,
            freq_bands: dict) -> dict:
    from fooof import FOOOF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = FOOOF(
            peak_width_limits=tuple(fcfg["peak_width_limits"]),
            max_n_peaks=int(fcfg["max_n_peaks"]),
            min_peak_height=float(mph),
            aperiodic_mode=fcfg["aperiodic_mode"],
            verbose=False,
        )
        fm.fit(freqs, psd, tuple(fcfg["freq_range"]))

    ap = fm.aperiodic_params_
    out = {
        "min_peak_height": float(mph),
        "peak_threshold": float(fm.peak_threshold),
        "n_peaks": int(fm.n_peaks_),
        "aperiodic_offset": float(ap[0]),
        "aperiodic_exponent": float(ap[-1]),
        "fit_r_squared": float(fm.r_squared_),
        "fit_error": float(fm.error_),
    }
    band_peaks = _assign_peaks_to_bands(fm.peak_params_, freq_bands)
    # flattened spectrum: log10 power minus aperiodic fit (== fm._spectrum_flat)
    flat = fm.power_spectrum - fm._ap_fit
    ffit = fm.freqs
    lin = 10 ** fm.power_spectrum
    total = lin.sum()
    names = list(freq_bands)
    for i, (band, (lo, hi)) in enumerate(freq_bands.items()):
        m = band_mask(ffit, float(lo), float(hi), last=(i == len(names) - 1))
        vals = band_peaks[band]
        out[f"{band}_peak_present"] = bool(vals["present"])
        out[f"{band}_cf"] = vals["cf"]
        out[f"{band}_pw"] = vals["pw"]
        out[f"{band}_bw"] = vals["bw"]
        out[f"{band}_adj_power"] = float(flat[m].mean())
        out[f"{band}_rel_power"] = float(lin[m].sum() / total)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)

    config = load_config()
    meta = setup_run(STAGE, config)
    fcfg = config["fooof"]
    freq_bands = {k: tuple(v) for k, v in config["freq_bands"].items()}

    files = sorted(IN_DIR.glob("*-psd.csv"))
    if not files:
        print(f"no spectra under {IN_DIR}; run psd_unwarped.py first", file=sys.stderr)
        return 2
    rows = []
    for f in files:
        df = pd.read_csv(f)
        keys = ["subject", "cohort", "block", "modality", "spectrum_variant",
                "region", "side", "n_channels"]
        for key_vals, g in df.groupby(keys, sort=False):
            g = g.sort_values("freq")
            freqs = g["freq"].to_numpy(float)
            psd = g["power"].to_numpy(float)
            base = dict(zip(keys, key_vals))
            for mph in MIN_PEAK_HEIGHTS:
                rows.append({**base, **fit_one(freqs, psd, fcfg, mph, freq_bands)})

    out = pd.DataFrame(rows)
    out = out.sort_values(["modality", "region", "cohort", "subject", "block",
                           "spectrum_variant", "min_peak_height"]).reset_index(drop=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(OUT) as tmp:
        out.to_csv(tmp, index=False)

    import fooof
    meta.update({
        "author": "Lena", "date": DATE, "fooof_version": fooof.__version__,
        "inputs": [str(p) for p in files], "output": str(OUT),
        "n_rows": int(len(out)), "min_peak_heights": list(MIN_PEAK_HEIGHTS),
        "peak_threshold_default": float(out["peak_threshold"].iloc[0]),
        "fooof_params_fixed": {k: v for k, v in fcfg.items() if k != "min_peak_height"},
        "adj_power_definition": "mean over band bins of log10(power) - aperiodic fit (fm.power_spectrum - fm._ap_fit)",
        "rel_power_definition": "sum linear power in band / sum linear power 4-45 Hz",
        "band_bin_rule": "lo <= f < hi, last band lo <= f <= hi",
    })
    write_sidecar(OUT, meta)
    print(f"wrote {OUT}  rows={len(out)}  files={len(files)}  "
          f"peak_threshold={out['peak_threshold'].iloc[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
