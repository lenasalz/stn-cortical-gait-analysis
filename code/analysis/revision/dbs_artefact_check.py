"""Screen cortical EEG spectra up to 100 Hz for stimulation-related artefact.

Purpose:  Test whether the higher resting high-beta/gamma power in PD could be
          a DBS stimulation artefact: compare PD and HC ROI spectra over the
          full 4-100 Hz passband of the clean data, look for narrow spectral
          lines (stimulation sub-harmonics, beat frequencies between the two
          stimulators) and for a power profile rising towards high frequencies.
Inputs:   data/derivatives/raw_clean/<subj>_{baseline,pre8walk}-eeg-raw.fif
              (pickle-era clean continuous EEG, 4-100 Hz FIR, post-ICA)
          code/analysis/config.yaml (ROI channels, cohorts)
Outputs:  outputs/stats/revision_2026-10/dbs_artefact_spectra_2026-10-13.csv
              (ROI-mean log10 PSD per subject x block x region, 2-100 Hz)
          outputs/stats/revision_2026-10/dbs_artefact_check_2026-10-13.txt
          outputs/figures/revision_2026-10/dbs_artefact_spectra.png
Author:   Lena
Date:     2026-10-13
Seed:     42 (config repro.seed; no randomness in the computation)

Description:
  Welch PSD (4 s Hann windows, 50 % overlap) on the whole continuous block,
  ROI channels only (bads excluded), averaged within ROI, log10. The
  walking block includes turns here; this is an artefact screen, not an
  analysis. Narrow peaks are detected on the PD-minus-HC mean difference
  (scipy.signal.find_peaks, prominence 1 dB, width <= 2 Hz) and on each
  PD subject's own spectrum above 45 Hz.

Usage:
  python code/analysis/revision/dbs_artefact_check.py
"""
from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"


def roi_channels(config):
    coi = config["connectivity"]["channels_of_interest"]
    return {reg: sum((list(v) for v in sides.values()), []) for reg, sides in coi.items()}


def main() -> int:
    import mne
    mne.set_log_level("WARNING")
    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("dbs_artefact_check", config)
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"; fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    rois = roi_channels(config)
    subjects = [s for co in (PD, HC) for s in config["cohorts"][co]] if "cohorts" in config else None
    if subjects is None:
        subjects = sorted({p.name.split("_")[0] for p in paths.raw_clean_dir.glob("*_baseline-eeg-raw.fif")})

    rows = []; spectra = {}
    for subj in subjects:
        for block in ("baseline", "pre8walk"):
            f = paths.raw_clean_dir / f"{subj}_{block}-eeg-raw.fif"
            if not f.exists():
                continue
            raw = mne.io.read_raw_fif(f, preload=False)
            for reg, chans in rois.items():
                picks = [c for c in chans if c in raw.ch_names and c not in raw.info["bads"]]
                if not picks:
                    continue
                data = raw.get_data(picks=picks)
                sf = raw.info["sfreq"]
                fr, p = signal.welch(data, fs=sf, window="hann", nperseg=int(4 * sf), noverlap=int(2 * sf), axis=-1)
                m = (fr >= 2) & (fr <= 100)
                roi = np.log10(p[:, m].mean(axis=0))
                spectra[(subj, block, reg)] = (fr[m], roi)
                for fq, v in zip(fr[m], roi):
                    rows.append(dict(subject=subj, cohort=PD if subj.startswith("P4-2") else HC, block=block, region=reg,
                                     n_channels=len(picks), freq=fq, log10_power=v))
                del data
            del raw; gc.collect()

    csv_path = out_dir / "dbs_artefact_spectra_2026-10-13.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_sidecar(csv_path, meta)

    lines = ["Stimulation-artefact screen, cortical EEG 2-100 Hz (ROI mean, log10 power)", ""]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for r, block in enumerate(("baseline", "pre8walk")):
        for c, reg in enumerate(("frontal", "motor")):
            fr = None; X = {PD: [], HC: []}
            for (s, b, rg), (f_, p_) in spectra.items():
                if b == block and rg == reg:
                    fr = f_; X[PD if s.startswith("P4-2") else HC].append(p_)
            Xpd, Xhc = np.array(X[PD]), np.array(X[HC])
            ax = axes[r, c]
            for Xg, col, lab in ((Xpd, "C3", "PD"), (Xhc, "C0", "HC")):
                for s_ in Xg:
                    ax.plot(fr, 10 * s_, color=col, alpha=0.2, lw=0.6)
                ax.plot(fr, 10 * Xg.mean(0), color=col, lw=2, label=f"{lab} mean (n={len(Xg)})")
            ax.set_title(f"{reg}, {block}"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(fontsize=8)
            diff = 10 * (Xpd.mean(0) - Xhc.mean(0))
            axd = axes[r, 2]; axd.plot(fr, diff, label=f"{reg}"); axd.axhline(0, color="grey", lw=0.5)
            axd.set_title(f"PD minus HC, {block}"); axd.set_xlabel("Hz"); axd.set_ylabel("dB"); axd.legend(fontsize=8)
            # narrow peaks in the difference curve
            pk, props = signal.find_peaks(diff, prominence=1.0, width=(None, 8))   # width in bins (0.25 Hz) -> <= 2 Hz
            pk_txt = "; ".join(f"{fr[i]:.2f} Hz (+{props['prominences'][k]:.1f} dB)" for k, i in enumerate(pk)) or "none"
            # band summary of the difference
            def band_mean(lo, hi):
                mm = (fr >= lo) & (fr < hi); return diff[mm].mean()
            lines.append(f"{block:9s}{reg:8s} PD-HC mean: 4-12 Hz {band_mean(4,12):+.2f} dB | 12-30 {band_mean(12,30):+.2f} | 30-45 {band_mean(30,45):+.2f} | 45-70 {band_mean(45,70):+.2f} | 70-100 {band_mean(70,100):+.2f}")
            lines.append(f"{'':17s} narrow peaks (<=2 Hz wide, >=1 dB) in PD-HC difference: {pk_txt}")
            # per-PD-subject lines above 45 Hz
            per_subj = []
            for s_, sp in zip([s for (s, b, rg) in spectra if b == block and rg == reg and s.startswith("P4-2")],
                              Xpd):
                # detrend with a 5 Hz median filter to isolate narrow lines
                base = signal.medfilt(sp, kernel_size=41)
                resid = 10 * (sp - base)
                hi = fr >= 45
                pk2, pr2 = signal.find_peaks(resid, prominence=2.0)
                pk2 = [i for i in pk2 if hi[i]]
                if pk2:
                    per_subj.append(f"{s_}: " + ", ".join(f"{fr[i]:.1f}" for i in pk2[:6]))
            lines.append(f"{'':17s} PD subjects with narrow lines > 45 Hz (>= 2 dB above local median): " + ("; ".join(per_subj) or "none"))
            lines.append("")
    fig.tight_layout(); fig.savefig(fig_dir / "dbs_artefact_spectra.png", dpi=130); plt.close(fig)
    (out_dir / "dbs_artefact_check_2026-10-13.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
