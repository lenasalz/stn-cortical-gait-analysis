"""Side-by-side cohort gERSP maps, their difference, and the cluster-test outline.

Purpose:  Show what the cluster-based permutation test quantifies: cohort
          median gERSP maps (PD, HC), the PD minus HC difference, and the
          significant-cluster outline from cluster_perm_revision.py, for the
          baseline-normalised and the average-gait-cycle-normalised maps.
          Also a per-subject panel (gamma and high-beta time courses) to show
          between-subject spread.
Inputs:   data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg{,-avggait}.nc
          data/derivatives/tfr_gavg/revision_cluster_eeg_{basenorm,avggait}_PDvsHC_<region>_Both.nc
Outputs:  outputs/figures/revision_2026-10/gersp_overlay_<region>.png
Author:   Lena
Date:     2026-10-13
Seed:     not applicable (plotting only)

Usage:
  for r in frontal motor; do python code/analysis/revision/plot_gersp_cluster_overlay.py --region $r; done
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"


def main() -> int:
    # One region per process: the netCDF/HDF5 stack segfaults when a second
    # region is plotted in the same interpreter, so the caller loops.
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--region", choices=("frontal", "motor"), required=True)
    region_arg = ap.parse_args().region
    paths = resolve_paths(load_config())
    g = paths.tfr_gavg_dir
    fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    for region in (region_arg,):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for row, (norm, fname, scale) in enumerate((("basenorm", "pre8walk-eeg-tfr-gavg.nc", 6),
                                                    ("avggait", "pre8walk-eeg-tfr-gavg-avggait.nc", 2))):
            ds = xr.open_dataset(g / fname)
            cl = xr.open_dataset(g / f"revision_cluster_eeg_{norm}_PDvsHC_{region}_Both.nc")
            freqs = ds.freq.values; t = np.linspace(0, 100, ds.sizes["time"])
            ext = [0, 100, freqs[0], freqs[-1]]
            per = ds.per_subject.sel(region=region, side="Both")
            pd_sub = per.where(ds.subject_cohort == PD, drop=True).values
            hc_sub = per.where(ds.subject_cohort == HC, drop=True).values
            med_pd, med_hc = np.nanmedian(pd_sub, 0), np.nanmedian(hc_sub, 0)
            panels = [(med_pd, f"PD median ({norm})", -scale, scale, "RdBu_r"),
                      (med_hc, f"HC median ({norm})", -scale, scale, "RdBu_r"),
                      (np.nanmean(pd_sub, 0) - np.nanmean(hc_sub, 0), "PD minus HC (mean)", -scale, scale, "RdBu_r"),
                      (cl.T_obs.values, "F map + significant cluster", 0, np.nanpercentile(cl.T_obs.values, 99), "Reds")]
            for col, (arr, title, vmin, vmax, cmap) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(arr, aspect="auto", origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
                if col == 3 and cl.sig_mask.values.any():
                    ax.contour(t, freqs, cl.sig_mask.values.astype(float), levels=[0.5], colors="k", linewidths=1.5)
                if col == 2 and cl.sig_mask.values.any():
                    ax.contour(t, freqs, cl.sig_mask.values.astype(float), levels=[0.5], colors="k", linewidths=1.0)
                for x in (10, 50, 60):
                    ax.axvline(x, color="k", lw=0.5, ls="--")
                ax.set_title(title, fontsize=10); ax.set_xlabel("Gait cycle (%)"); ax.set_ylabel("Hz")
                fig.colorbar(im, ax=ax, fraction=0.04, label="dB" if col < 3 else "F")
        # per-subject time courses, basenorm, gamma and high beta
        ds = xr.open_dataset(g / "pre8walk-eeg-tfr-gavg.nc"); t = np.linspace(0, 100, ds.sizes["time"])
        per = ds.per_subject.sel(region=region, side="Both")
        for col, (band, lo, hi) in enumerate((("gamma 30-45 Hz", 30, 45), ("high beta 20-30 Hz", 20, 30))):
            ax = axes[2, col]
            for cohort, color in ((PD, "C3"), (HC, "C0")):
                sub = per.where(ds.subject_cohort == cohort, drop=True).sel(freq=slice(lo, hi)).mean("freq").values
                for s in sub:
                    ax.plot(t, s, color=color, alpha=0.35, lw=0.8)
                ax.plot(t, np.nanmean(sub, 0), color=color, lw=2.5, label=f"{'PD' if cohort == PD else 'HC'} mean")
            ax.axhline(0, color="k", lw=0.5); ax.set_title(f"per subject, basenorm, {band}", fontsize=10)
            ax.set_xlabel("Gait cycle (%)"); ax.set_ylabel("dB vs rest"); ax.legend(fontsize=8)
        for col, (band, lo, hi) in enumerate((("gamma 30-45 Hz", 30, 45), ("high beta 20-30 Hz", 20, 30))):
            ax = axes[2, col + 2]
            dsa = xr.open_dataset(g / "pre8walk-eeg-tfr-gavg-avggait.nc")
            pera = dsa.per_subject.sel(region=region, side="Both")
            for cohort, color in ((PD, "C3"), (HC, "C0")):
                sub = pera.where(dsa.subject_cohort == cohort, drop=True).sel(freq=slice(lo, hi)).mean("freq").values
                for s in sub:
                    ax.plot(t, s, color=color, alpha=0.35, lw=0.8)
                ax.plot(t, np.nanmean(sub, 0), color=color, lw=2.5, label=f"{'PD' if cohort == PD else 'HC'} mean")
            ax.axhline(0, color="k", lw=0.5); ax.set_title(f"per subject, avggait, {band}", fontsize=10)
            ax.set_xlabel("Gait cycle (%)"); ax.set_ylabel("dB vs cycle mean"); ax.legend(fontsize=8)
        fig.suptitle(f"{region} ROI: cohort maps, difference, cluster test, per-subject spread", fontsize=12)
        fig.tight_layout()
        out = fig_dir / f"gersp_overlay_{region}.png"
        fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
