"""Where are the FOOOF peaks when a band has no detected peak? (R1.2)

Purpose:  For every spectrum in which a band (alpha, low beta, high beta) has
          no detected peak, check whether a fitted peak lies just outside the
          band limits, overlaps the band with its width, or was dropped by the
          limit on the number of peaks. Bands are assigned by the centre
          frequency of the peak (compute_fooof._assign_peaks_to_bands), so a
          peak at 12.1 Hz counts as low beta and leaves alpha empty.
Inputs:   data/derivatives/psd_unwarped/<subj>_<block>-<modality>-psd.csv
              (psd_unwarped.py; spectrum variant A, ROI side "Both" for EEG)
          data/derivatives/fooof/<subj>_<block>-<modality>-fooof.csv
              (submitted pipeline; only with --submitted; per-band peaks, as
              stored: highest peak per band)
          code/analysis/config.yaml (fooof, freq_bands, repro.seed)
Outputs:  outputs/stats/revision_2026-10/fooof_all_peaks_<date>.csv
              every fitted peak: subject, cohort, block, modality, region,
              fit (submitted settings | max_n_peaks 8), cf, pw, bw
          outputs/stats/revision_2026-10/fooof_missing_band_peaks_<date>.csv
              one row per spectrum x band without a peak: nearest peak, its
              distance to the band, overlap, what the 8-peak fit finds
          outputs/stats/revision_2026-10/fooof_missing_band_peaks_<date>.txt
          (+ .meta.json sidecars)
Author:   Lena
Date:     2026-11-17
Seed:     config.yaml:repro.seed (42) via _common.setup_run (fits are
          deterministic).

Usage:
  REVISION_DATE=2026-11-16 python code/analysis/revision/fooof_peak_locations.py
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PROJECT_ROOT, atomic_write, load_config, setup_run, write_sidecar  # noqa: E402

DATE = os.environ.get("REVISION_DATE", "2026-11-16")
IN_DIR = PROJECT_ROOT / "data" / "derivatives" / "psd_unwarped"
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
BANDS_CHECKED = ("Alpha", "Low Beta", "High Beta")
WIDE_MAX_PEAKS = 8


def fit_peaks(freqs: np.ndarray, psd: np.ndarray, fcfg: dict, max_n_peaks: int) -> np.ndarray:
    from fooof import FOOOF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = FOOOF(peak_width_limits=tuple(fcfg["peak_width_limits"]), max_n_peaks=int(max_n_peaks),
                   min_peak_height=float(fcfg["min_peak_height"]), aperiodic_mode=fcfg["aperiodic_mode"],
                   verbose=False)
        fm.fit(freqs, psd, tuple(fcfg["freq_range"]))
    return np.atleast_2d(fm.peak_params_) if fm.n_peaks_ else np.empty((0, 3))


def main() -> int:
    config = load_config()
    meta = setup_run("fooof_peak_locations", config)
    fcfg, bands = config["fooof"], config["freq_bands"]
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "fooof")}
    print(json.dumps({"seed": meta["seed"], **versions}))

    peak_rows, miss_rows = [], []
    files = sorted(IN_DIR.glob("*-psd.csv"))
    for f in files:
        d = pd.read_csv(f)
        d = d[d.spectrum_variant == "A"]
        if "side" in d and (d.modality.iloc[0] == "eeg"):
            d = d[d.side == "Both"]
        for (subject, cohort, block, modality, region), g in d.groupby(
                ["subject", "cohort", "block", "modality", "region"]):
            g = g.sort_values("freq")
            freqs, psd = g.freq.to_numpy(float), g.power.to_numpy(float)
            key = dict(subject=subject, cohort=cohort, block=block, modality=modality, region=region)
            sub = fit_peaks(freqs, psd, fcfg, fcfg["max_n_peaks"])
            wide = fit_peaks(freqs, psd, fcfg, WIDE_MAX_PEAKS)
            for name, pk in (("submitted_settings", sub), (f"max_n_peaks_{WIDE_MAX_PEAKS}", wide)):
                for cf, pw, bw in pk:
                    peak_rows.append({**key, "fit": name, "cf": cf, "pw": pw, "bw": bw})
            for band in BANDS_CHECKED:
                lo, hi = map(float, bands[band])
                if ((sub[:, 0] >= lo) & (sub[:, 0] <= hi)).any():
                    continue
                row = {**key, "band": band, "band_lo": lo, "band_hi": hi, "n_peaks_fit": len(sub),
                       "at_peak_limit": len(sub) >= int(fcfg["max_n_peaks"])}
                if len(sub):
                    dist = np.where(sub[:, 0] < lo, lo - sub[:, 0], sub[:, 0] - hi)
                    j = int(np.argmin(dist))
                    cf, pw, bw = sub[j]
                    row.update(nearest_cf=cf, nearest_pw=pw, nearest_bw=bw, distance_hz=float(dist[j]),
                               nearest_side="below" if cf < lo else "above",
                               overlaps_band=bool((cf + bw / 2 >= lo) and (cf - bw / 2 <= hi)))
                else:
                    row.update(nearest_cf=np.nan, nearest_pw=np.nan, nearest_bw=np.nan, distance_hz=np.nan,
                               nearest_side="", overlaps_band=False)
                inw = wide[(wide[:, 0] >= lo) & (wide[:, 0] <= hi)] if len(wide) else wide
                row.update(wide_fit_has_peak_in_band=bool(len(inw)),
                           wide_fit_cf=float(inw[np.argmax(inw[:, 1]), 0]) if len(inw) else np.nan,
                           wide_fit_pw=float(inw[:, 1].max()) if len(inw) else np.nan)
                miss_rows.append(row)

    peaks, miss = pd.DataFrame(peak_rows), pd.DataFrame(miss_rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_peaks = STATS_DIR / f"fooof_all_peaks_{DATE}.csv"
    out_miss = STATS_DIR / f"fooof_missing_band_peaks_{DATE}.csv"
    out_txt = STATS_DIR / f"fooof_missing_band_peaks_{DATE}.txt"
    for out, df in ((out_peaks, peaks), (out_miss, miss)):
        with atomic_write(out) as tmp:
            df.to_csv(tmp, index=False)

    n_spec = peaks[["subject", "block", "modality", "region"]].drop_duplicates().shape[0]
    L = [f"FOOOF peaks outside the band limits, {DATE} (unwarped spectra, variant A; settings: "
         f"min_peak_height {fcfg['min_peak_height']}, max_n_peaks {fcfg['max_n_peaks']}, "
         f"peak_width_limits {fcfg['peak_width_limits']})", "",
         f"spectra fitted: {n_spec}; spectrum x band cells checked: {n_spec * len(BANDS_CHECKED)}; "
         f"cells without a peak in the band: {len(miss)}", ""]
    for band in BANDS_CHECKED:
        m = miss[miss.band == band]
        L.append(f"{band} ({m.band_lo.iloc[0]:g}-{m.band_hi.iloc[0]:g} Hz): no peak in {len(m)} spectra")
        L.append(f"  nearest peak within 1 Hz of the band: {(m.distance_hz <= 1).sum()}; "
                 f"within 2 Hz: {(m.distance_hz <= 2).sum()}; nearest peak overlaps the band "
                 f"(CF +/- BW/2): {m.overlaps_band.sum()}; no peak in the whole spectrum: {(m.n_peaks_fit == 0).sum()}")
        L.append(f"  fit already at the limit of {fcfg['max_n_peaks']} peaks: {m.at_peak_limit.sum()}; "
                 f"fit with max_n_peaks {WIDE_MAX_PEAKS} finds a peak in the band: {m.wide_fit_has_peak_in_band.sum()}")
        for (coh, blk), g in m.groupby(["cohort", "block"]):
            L.append(f"    {coh} {blk}: {len(g)} without peak; within 1 Hz: {(g.distance_hz <= 1).sum()}, "
                     f"within 2 Hz: {(g.distance_hz <= 2).sum()}")
        L.append("")
    out_txt.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    for out in (out_peaks, out_miss, out_txt):
        m2 = dict(meta)
        m2.update({"author": "Lena", "date": DATE, "inputs": [str(IN_DIR)], "output": str(out),
                   "versions": versions, "fooof_settings": fcfg, "wide_max_n_peaks": WIDE_MAX_PEAKS})
        write_sidecar(out, m2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
