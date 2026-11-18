"""Group power spectra computed without time warping (supplementary figure).

Purpose:  Plot the spectra that the revised spectral statistics are based on:
          cortical ROI spectra (motor, prefrontal; PD vs HC) and the STN-LFP
          spectrum (PD), at rest and during walking, computed without
          gait-cycle time warping.
Inputs:   data/derivatives/psd_unwarped/<subj>_<block>-<modality>-psd.csv
              (code/analysis/revision/psd_unwarped.py; spectrum variant A,
              EEG ROI side "Both")
          code/analysis/config.yaml (cohorts, freq_bands, repro.seed)
Outputs:  outputs/figures/psd_unwarped/PSD_unwarped_<date>.pdf
          outputs/figures/psd_unwarped/PSD_unwarped_<date>.png  (300 dpi)
          (+ .meta.json)
Author:   Lena
Date:     2026-11-18
Seed:     config.yaml:repro.seed (42) via _common.setup_run (no randomness).

Description:
  Per group, condition and frequency: median across participants of the
  linear power and the standard error of the median (median absolute
  deviation / sqrt(n)), both converted to decibels (10 log10), as for the
  spectra in the main text. PD red, HC blue; rest solid, walking dashed;
  vertical dashed lines at the band limits (8, 12, 20, 30 Hz).

Usage:
  python code/figures/plot_psd_unwarped.py --date 2026-11-18
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "analysis"))
from _common import atomic_write, load_config, setup_run, write_sidecar  # noqa: E402

IN_DIR = PROJECT_ROOT / "data" / "derivatives" / "psd_unwarped"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "psd_unwarped"
COLORS = {"PD": "#CD5F66", "HC": "#5794A0"}
STYLE = {"baseline": ("-", "rest"), "pre8walk": ("--", "walking")}


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(IN_DIR.glob("*-psd.csv")):
        d = pd.read_csv(f)
        d = d[d.spectrum_variant == "A"]
        d = d[(d.modality == "dbs") | (d.side == "Both")]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["group"] = np.where(d.cohort == "Patient_NFB", "PD", "HC")
    return d


def summary(d: pd.DataFrame):
    piv = d.pivot_table(index="subject", columns="freq", values="power")
    med = piv.median(axis=0)
    mad = (piv - med).abs().median(axis=0)
    sem = mad / np.sqrt(piv.shape[0])
    freqs = med.index.to_numpy(float)
    lo = np.clip(med - sem, np.finfo(float).tiny, None)
    return freqs, 10 * np.log10(med), 10 * np.log10(lo), 10 * np.log10(med + sem), piv.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    config = load_config()
    meta = setup_run("plot_psd_unwarped", config)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")}
    print(json.dumps({"seed": meta["seed"], **versions}))

    d = load()
    edges = sorted({float(e) for lo_hi in config["freq_bands"].values() for e in lo_hi})[1:-1]
    panels = [("Motor cortex", (d.modality == "eeg") & (d.region == "motor"), ("PD", "HC")),
              ("Prefrontal cortex", (d.modality == "eeg") & (d.region == "frontal"), ("PD", "HC")),
              ("STN", d.modality == "dbs", ("PD",))]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    counts = {}
    for ax, (title, sel, groups) in zip(axes, panels):
        for grp in groups:
            for blk, (ls, cond) in STYLE.items():
                s = d[sel & (d.group == grp) & (d.block == blk)]
                if s.empty:
                    continue
                f, m, lo, hi, n = summary(s)
                counts[f"{title}|{grp}|{cond}"] = int(n)
                ax.plot(f, m, ls, color=COLORS[grp], lw=1.8, label=f"{grp} {cond}")
                ax.fill_between(f, lo, hi, color=COLORS[grp], alpha=0.18, lw=0)
        for e in edges:
            ax.axvline(e, color="0.5", ls=":", lw=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(4, 45)
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Power (dB)")
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=13)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = OUT_DIR / f"PSD_unwarped_{args.date}.{ext}"
        with atomic_write(out) as tmp:
            fig.savefig(tmp, format=ext, **kw)
        m = dict(meta)
        m.update({"author": "Lena", "date": args.date, "input_dir": str(IN_DIR), "output": str(out),
                  "n_participants": counts, "versions": versions})
        write_sidecar(out, m)
        print(f"wrote {out}")
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
