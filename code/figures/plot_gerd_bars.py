"""Render manuscript Fig 4 (basenorm) and Fig 8 (avggait): gERD/ERS line plots
+ per-(band,region,cohort) statistic bar panels.

Inputs:
  - data/derivatives/tfr_gavg/<block>-eeg-tfr-gavg[-<norm>].nc  (aggregate_tfr_gavg.py)
  - data/derivatives/tfr_gavg/<block>-dbs-tfr-gavg[-<norm>].nc
  - code/analysis/config.yaml (freq_bands)
  - code/analysis/variables.py (event_percentages, gait_phases)
Outputs:
  - outputs/figures/gerd_bars/<norm>/ERSDs_lineplot_8walk_PD_vs_HC_<norm>.pdf
  - outputs/figures/gerd_bars/<norm>/stats_metrics_PD_vs_HC_pre8walk_<band>_<region>_<norm>.pdf
Description:
  Thin adapter from our per-subject NetCDFs to the legacy dict-of-lists
  consumed by `final_analysis/scripts/ersds_analysis.generate_ersd_analysis`.
  Same plumbing pattern as `plot_tfr_grandavg_legacy.py`, applied to the
  gERD/ERS analysis instead of the TFR grandavg plotter.

  Cohorts and color palette match the manuscript ("PD" / "HC", with
  Aline's #CD5F66 / #5794A0). PD_vs_HC is the only comparison emitted.

Usage:
  python code/figures/plot_gerd_bars.py --block pre8walk --norm basenorm
  python code/figures/plot_gerd_bars.py --block pre8walk --norm avggait
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _common import load_config, resolve_paths, setup_run, validate_block

# Plot module (code/analysis/ersds_analysis.py) + the gait variables it needs.
import ersds_analysis as ersa  # noqa: E402
generate_ersd_analysis = ersa.generate_ersd_analysis
get_freq_indices = ersa.get_freq_indices

# Our analysis vars (manuscript event % + phase definitions).
from variables import event_percentages, gait_phases  # noqa: E402

# Manuscript per-band Y-limits (symmetric ±) for the ERS/ERD line plot.
# Theta is -6/+6 in the published Fig 4 caption; other bands keep moderate
# defaults so the panels look comparable. Override via --ylim or --ylim-band.
DEFAULT_YLIM_PER_BAND = {
    "Theta":     6.0,
    "Alpha":     6.0,
    "Low Beta":  4.0,
    "High Beta": 4.0,
    "Gamma":     4.0,
}

OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "gerd_bars"

# Manuscript palette (analysis_paper.ipynb cell 7).
COLOR_DICT = {"PD": "#CD5F66", "HC": "#5794A0"}

# One main comparison + one line plot (PD vs HC), per cell 7.
COMPARISONS_TO_RUN = [{
    "name": "PD_vs_HC",
    "cohorts": {
        "PD": {"group": "Patient_NFB",    "ids": None, "color": COLOR_DICT["PD"]},
        "HC": {"group": "AgeMatched_NFB", "ids": None, "color": COLOR_DICT["HC"]},
    },
}]
LINE_PLOTS_TO_GENERATE = [{"name": "8walk_PD_vs_HC", "cohorts": ["PD", "HC"]}]


def _make_fixed_ylim_plotter(ylim_per_band: dict[str, float]):
    """Return a drop-in replacement for ersds_analysis.plot_erds_unified that
    forces a symmetric Y-limit per band instead of auto-scaling. Mirrors
    Aline's plotter line-for-line; the only change is line 224's set_ylim.
    """
    subarea_styles = {"Left": ":", "Both": "-", "Right": "--"}
    area_labels = {"EEG_motor": "Motor", "EEG_frontal": "Frontal", "STN": "STN"}
    block_name_map = {"pre8walk": "8walk", "post8walk": "Post-8walk"}
    font_settings = {"title": 28, "label": 26, "ticks": 22, "legend": 30,
                     "gait_ticks": 22}
    areas_to_plot = ["EEG_motor", "EEG_frontal", "STN"]

    def plot_erds_unified_fixed_y(
        tfr_group_results, freqs, freq_bands, blocks, event_percentages,
        results_dir, all_cohort_definitions, cohorts_to_plot, filename_suffix,
        subareas=("Left", "Both", "Right"), normalization="",
    ):
        n_bands, n_areas = len(freq_bands), len(areas_to_plot)
        for block in blocks:
            if block == "baseline":
                continue
            fig, axes = plt.subplots(n_bands, n_areas, figsize=(28, 7 * n_bands),
                                      sharex=True, constrained_layout=False)
            if n_bands == 1:
                axes = np.array([axes])
            if n_areas == 1:
                axes = axes.reshape(-1, 1)
            for row_idx, (band, band_lim) in enumerate(freq_bands.items()):
                for col_idx, area_type in enumerate(areas_to_plot):
                    ax = axes[row_idx, col_idx]
                    for cohort_name in cohorts_to_plot:
                        if cohort_name not in all_cohort_definitions:
                            continue
                        ci = all_cohort_definitions[cohort_name]
                        base_group = ci["group"]
                        id_filter = set(ci["ids"]) if ci.get("ids") else None
                        if not (base_group in tfr_group_results
                                and block in tfr_group_results[base_group]
                                and area_type in tfr_group_results[base_group][block]):
                            continue
                        for subarea in subareas:
                            if subarea not in tfr_group_results[base_group][block][area_type]:
                                continue
                            all_arrays = tfr_group_results[base_group][block][area_type][subarea]
                            all_ids = tfr_group_results[base_group][block]["ID"]
                            idx = [i for i, sid in enumerate(all_ids)
                                   if not id_filter or sid in id_filter]
                            subj_arrays = [all_arrays[i] for i in idx
                                           if i < len(all_arrays) and all_arrays[i] is not None]
                            if not subj_arrays:
                                continue
                            arr_stack = np.stack(subj_arrays)
                            f_idx = get_freq_indices(freqs, band_lim)
                            data = arr_stack[:, f_idx, :].mean(axis=1)
                            mean = np.nanmean(data, axis=0)
                            sem = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
                            x = np.linspace(0, 100, data.shape[1])
                            ax.plot(x, mean, color=ci["color"],
                                    linestyle=subarea_styles.get(subarea, "-"),
                                    lw=2.5, label=f"{cohort_name} {subarea}")
                            ax.fill_between(x, mean - sem, mean + sem,
                                             color=ci["color"], alpha=0.15)
                    # x-axis ticks (gait events)
                    ticks = [0, 10, 50, 60, 100]
                    tick_lbls = ["0%: HSR", "10%: TOL", "50%: HSL",
                                  "60%: TOR", "100%: HSR"]
                    event_times = [10, 50, 60]
                    ax.set_xticks(ticks)
                    ax.xaxis.set_minor_locator(plt.NullLocator())
                    if row_idx == n_bands - 1:
                        ax.set_xticklabels(tick_lbls, rotation=45, ha="right",
                                            rotation_mode="anchor",
                                            fontsize=font_settings["gait_ticks"])
                        ax.set_xlabel("Gait Cycle (%)",
                                       fontsize=font_settings["label"])
                    else:
                        ax.tick_params(axis="x", which="both", bottom=False,
                                        top=False, labelbottom=False)
                    for et in event_times:
                        ax.axvline(et, color="black", linestyle="-", linewidth=1.5)
                    ax.axhline(0, color="black", linestyle="-", lw=0.7)
                    ax.set_xlim(0, 100)

                    # ----- KEY DIFFERENCE: fixed per-band Y limit -----
                    yl = ylim_per_band.get(band)
                    if yl is not None and yl > 0:
                        ax.set_ylim(-yl, yl)
                    # --------------------------------------------------

                    if row_idx == 0:
                        ax.set_title(area_labels[area_type],
                                      fontsize=font_settings["title"])
                    if col_idx == 0:
                        ax.text(-0.35, 0.5,
                                f"{band.title()}\n({band_lim[0]}-{band_lim[1]} Hz)",
                                transform=ax.transAxes, ha="center", va="center",
                                rotation="vertical",
                                fontsize=font_settings["title"], weight="bold")
                        ax.set_ylabel("ERS/ERD (dB)",
                                       fontsize=font_settings["label"], weight="bold")
                    ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
                    ax.tick_params(axis="y", which="major",
                                    labelsize=font_settings["ticks"])
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=font_settings["legend"],
                        loc="upper center", bbox_to_anchor=(0.56, 0.09),
                        ncol=len(cohorts_to_plot) * len(subareas), frameon=False)
            fig.tight_layout(pad=1.5, rect=[0.05, 0.1, 0.98, 0.95])
            block_short = block_name_map.get(block, block)
            fname = f"ERSDs_lineplot_{block_short}_{filename_suffix}_{normalization}.pdf"
            fig.savefig(os.path.join(results_dir, fname), dpi=300,
                         bbox_inches="tight")
            plt.close(fig)
    return plot_erds_unified_fixed_y


def build_legacy_dict(ds_eeg: xr.Dataset, ds_dbs: xr.Dataset | None,
                       block: str) -> dict:
    """Reconstruct the legacy dict-of-lists structure from per-subject NetCDFs.

    Mirrors build_legacy_dict in plot_tfr_grandavg_legacy.py. For each
    cohort/block, returns only that cohort's subjects (filtered by the
    NetCDF's subject_cohort coord). Without this filter, both cohort
    buckets receive every subject, and the downstream cohort averages
    collapse to the same global mean.
    """
    out: dict[str, dict] = {}
    for cohort in ds_eeg.cohort.values.tolist():
        cohort_subjs = [
            str(s) for s in ds_eeg.subject.values
            if str(ds_eeg.subject_cohort.sel(subject=s).item()) == cohort
        ]
        block_data: dict = {"ID": cohort_subjs}
        for region in ds_eeg.region.values.tolist():
            key = f"EEG_{region}"
            block_data[key] = {}
            for side in ds_eeg.side.values.tolist():
                lst: list = []
                for subj in cohort_subjs:
                    arr = ds_eeg["per_subject"].sel(
                        subject=subj, region=region, side=side
                    ).values
                    if np.all(np.isnan(arr)):
                        lst.append(None)
                    else:
                        lst.append(arr)
                block_data[key][side] = lst
        block_data["STN"] = {"Left": [], "Right": []}
        if ds_dbs is not None:
            for side in ("Left", "Right"):
                lst = []
                for subj in cohort_subjs:
                    if subj not in ds_dbs.subject.values:
                        lst.append(None)
                        continue
                    arr = ds_dbs["per_subject"].sel(
                        subject=subj, region="STN", side=side
                    ).values
                    if np.all(np.isnan(arr)):
                        lst.append(None)
                    else:
                        lst.append(arr)
                block_data["STN"][side] = lst
        else:
            block_data["STN"]["Left"]  = [None] * len(cohort_subjs)
            block_data["STN"]["Right"] = [None] * len(cohort_subjs)
        out.setdefault(cohort, {})[block] = block_data
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True)
    parser.add_argument("--norm", choices=["basenorm", "avggait"], default="basenorm",
                        help="basenorm → Fig 4; avggait → supplementary Fig 8.")
    parser.add_argument("--ylim", type=float, default=None,
                        help="Override all bands to symmetric [-ylim, +ylim]. "
                             "If unset, uses manuscript per-band defaults "
                             "(Theta/Alpha ±6, Beta/Gamma ±4). "
                             "Pass --ylim 0 to fall back to Aline's auto-scale.")
    parser.add_argument("--ylim-band", action="append", default=[],
                        metavar="BAND=VALUE",
                        help="Per-band override, repeatable. "
                             "Example: --ylim-band Theta=8 --ylim-band Gamma=3.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    block = validate_block(args.block, config)
    setup_run("plot_gerd_bars", config)

    norm_suffix = "" if args.norm == "basenorm" else f"-{args.norm}"
    eeg_nc = paths.tfr_gavg_dir / f"{block}-eeg-tfr-gavg{norm_suffix}.nc"
    dbs_nc = paths.tfr_gavg_dir / f"{block}-dbs-tfr-gavg{norm_suffix}.nc"
    if not eeg_nc.exists():
        print(f"missing {eeg_nc}; run "
              f"`aggregate_tfr_gavg.py --modality eeg --norm {args.norm}` first",
              file=sys.stderr)
        return 2

    ds_eeg = xr.open_dataset(eeg_nc)
    ds_dbs = xr.open_dataset(dbs_nc) if dbs_nc.exists() else None

    tfr_group_results = build_legacy_dict(ds_eeg, ds_dbs, block)

    freqs = ds_eeg.freq.values
    freq_bands = {name: tuple(edges) for name, edges in config["freq_bands"].items()}

    # Resolve final per-band ylim dict from CLI + defaults.
    ylim_per_band: dict[str, float] = {}
    if args.ylim == 0:
        # 0 → keep Aline's auto-scale (no patch)
        pass
    elif args.ylim is not None:
        for b in DEFAULT_YLIM_PER_BAND:
            ylim_per_band[b] = args.ylim
    else:
        ylim_per_band.update(DEFAULT_YLIM_PER_BAND)
    for spec in args.ylim_band:
        if "=" not in spec:
            parser.error(f"--ylim-band must look like Theta=8, got {spec!r}")
        band, val = spec.split("=", 1)
        ylim_per_band[band.strip()] = float(val)
    if ylim_per_band:
        print(f"  Y-axis scale (per band, symmetric): {ylim_per_band}")
        ersa.plot_erds_unified = _make_fixed_ylim_plotter(ylim_per_band)

    out_dir = OUT_DIR / args.norm
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== gERD/ERS bars — block={block} norm={args.norm} → {out_dir} ===")

    generate_ersd_analysis(
        tfr_group_results=tfr_group_results,
        freqs=freqs,
        freq_bands=freq_bands,
        blocks=[block],
        event_percentages=event_percentages,
        gait_phases=gait_phases,
        results_dir=str(out_dir),
        comparisons_to_run=COMPARISONS_TO_RUN,
        line_plots_to_generate=LINE_PLOTS_TO_GENERATE,
        palette=COLOR_DICT,
        normalization=args.norm,
    )

    print(f"wrote outputs to {out_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
