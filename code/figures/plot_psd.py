"""Render manuscript Fig 1 (STN PSD: rest vs walk, PD) and Fig 2 (cortical PSD: PD vs HC).

Inputs:
  - data/derivatives/epochs/<subj>_<block>-{eeg,dbs}-epo.fif   (build_epochs.py)
  - code/analysis/config.yaml (cohorts, blocks, freq_bands)
  - code/analysis/variables.py (motor_areas, frontal_areas)
Outputs:
  - outputs/figures/psd/PSD_PD_vs_HC_Motor_Both.pdf       (Fig 2 — motor)
  - outputs/figures/psd/PSD_PD_vs_HC_Frontal_Both.pdf     (Fig 2 — frontal)
  - outputs/figures/psd/PSD_DBS_PD_epoched.pdf            (Fig 1 — STN rest vs walk)
Description:
  Computes per-subject Welch PSD on gait-warped epochs (4-45 Hz,
  n_fft=epoch_len, matching Aline's psd_fooof.py:122-126), aggregates
  per cohort, and uses Aline's `plot_psd_comparison_multi_block` to
  render the manuscript-style PSD figures.

  ROIs (cortical): "Motor_Both" = motor_areas['Both'],
                   "Frontal_Both" = frontal_areas['Both'].
  STN: single channel, PD cohort only, comparison = rest vs walk.

Usage:
  python code/figures/plot_psd.py
  python code/figures/plot_psd.py --modality eeg     # only Fig 2
  python code/figures/plot_psd.py --modality dbs     # only Fig 1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _common import load_config, resolve_paths, setup_run  # noqa: E402

# Region channel definitions (kept in our analysis vars; bit-identical to Aline's).
from variables import motor_areas, frontal_areas  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import median_abs_deviation  # noqa: E402


def _get_band_edges(freq_bands: dict, freqs: np.ndarray) -> list[float]:
    edges = {v for pair in freq_bands.values() for v in pair}
    return sorted(e for e in edges if freqs[0] <= e <= freqs[-1])


def plot_psd_comparison_multi_block(
    psd_data_for_type: dict, blocks_to_plot: list[str],
    cohorts_to_plot: list[str], cohort_definitions: dict,
    colors: list[str], title: str, filename: str, freq_bands: dict,
    font_scale: float = 1.0,
) -> None:
    """Inlined copy of final_analysis/scripts/psd_fooof.plot_psd_comparison_multi_block.

    Median + MAD-based standard error, dB scale. We inline rather than import to
    avoid pulling in the specparam dependency of psd_fooof.py.

    ``font_scale`` multiplies every font size; use >1 for figures placed at a
    small \\textwidth fraction (Fig 2 sits at ~0.49\\textwidth, so its text must
    be enlarged to survive the down-scaling) and 1.0 for near-full-width figures.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    block_styles = {"baseline": "-", "pre8walk": "--", "post8walk": ":"}
    block_name_map = {"pre8walk": "Walk", "post8walk": "Post-8walk", "baseline": "Rest"}
    cohort_colors = {c: col for c, col in zip(cohorts_to_plot, colors)}

    freqs = None
    for b in blocks_to_plot:
        for c in cohorts_to_plot:
            if b in psd_data_for_type and c in psd_data_for_type[b]:
                freqs = psd_data_for_type[b][c].get("freqs")
                if freqs is not None:
                    break
        if freqs is not None:
            break
    if freqs is None:
        print(f"  no PSDs to plot for {filename}")
        return

    for cohort_name in cohorts_to_plot:
        color = cohort_colors.get(cohort_name, "k")
        for block in blocks_to_plot:
            if block not in psd_data_for_type or cohort_name not in psd_data_for_type[block]:
                continue
            psds_array = psd_data_for_type[block][cohort_name]["psds"]
            n_subs = psds_array.shape[0]
            median_psd = np.median(psds_array, axis=0)
            mad = median_abs_deviation(psds_array, axis=0, scale="normal")
            se_median = mad / np.sqrt(n_subs)
            median_psd_db = 10 * np.log10(np.maximum(1e-18, median_psd))
            lo_db = 10 * np.log10(np.maximum(1e-18, median_psd - se_median))
            hi_db = 10 * np.log10(np.maximum(1e-18, median_psd + se_median))
            ax.plot(freqs, median_psd_db, color=color,
                    linestyle=block_styles.get(block, "-"),
                    label=f"{cohort_name} - {block_name_map.get(block, block)}", lw=2)
            ax.fill_between(freqs, lo_db, hi_db, color=color, alpha=0.1)

    if freq_bands:
        for edge in _get_band_edges(freq_bands, freqs):
            ax.axvline(x=edge, color="k", linestyle="--", linewidth=1.5, alpha=0.6)
        # Keep the full band names at their base size so the narrow theta/alpha
        # labels fit without overlapping, even though the axis fonts are enlarged.
        band_fs = round(18 * min(font_scale, 1.0))
        for band_name, (low, high) in freq_bands.items():
            x_center = (low + high) / 2
            ax.text(x_center, 0.03, band_name, transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=band_fs, fontweight="bold")
    ax.tick_params(axis="x", labelsize=round(18 * font_scale))
    ax.tick_params(axis="y", labelsize=round(18 * font_scale))
    ax.set_xlabel("Frequency (Hz)", fontsize=round(22 * font_scale))
    ax.set_ylabel("Power Spectral Density (dB)", fontsize=round(22 * font_scale))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=round(22 * font_scale), loc="upper right")
    ax.set_xlim(4, 45)
    # Title suppressed for manuscript output (claim is in the caption).
    plt.tight_layout()
    # Save in both PDF and PNG (manuscript includegraphics uses .png).
    fig.savefig(filename, dpi=200)
    png_path = str(Path(filename).with_suffix(".png"))
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "psd"

# Manuscript cohort palette (analysis_paper.ipynb cell 7).
COLOR_DICT = {"PD": "#CD5F66", "HC": "#5794A0"}
COHORT_GROUP = {"PD": "Patient_NFB", "HC": "AgeMatched_NFB"}


def _compute_psd_one(epochs_path: Path, channels: list[str] | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch PSD on epochs, fmin=4, fmax=45, n_fft=epoch_length.
    Returns (psd_mean_over_epochs_and_chans, freqs) or None if missing."""
    if not epochs_path.exists():
        return None
    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ep = mne.read_epochs(str(epochs_path), preload=True, verbose="ERROR")
    if channels is not None:
        present = [c for c in channels if c in ep.info["ch_names"]]
        if not present:
            return None
        ep = ep.pick(present, verbose="ERROR")
    n_fft = len(ep.times)
    spectrum = ep.compute_psd(fmin=4, fmax=45, method="welch", n_fft=n_fft, verbose="ERROR")
    psd_values = spectrum.get_data(return_freqs=False).mean(axis=(0, 1))
    return psd_values, spectrum.freqs


def _aggregate_cohort_psds(epochs_dir: Path, subjects: list[str], block: str,
                            modality: str, channels: list[str] | None
                            ) -> tuple[np.ndarray, np.ndarray] | None:
    """Aggregate per-subject PSDs for one cohort × block × modality × ROI."""
    psds, freqs = [], None
    for sid in subjects:
        p = epochs_dir / f"{sid}_{block}-{modality}-epo.fif"
        out = _compute_psd_one(p, channels)
        if out is None:
            continue
        psd, f = out
        psds.append(psd)
        if freqs is None:
            freqs = f
    if not psds:
        return None
    return np.asarray(psds), freqs


def _plot_one_roi(out_path: Path, psd_data_by_block: dict, freqs: np.ndarray,
                  cohorts_to_plot: list[str], freq_bands: dict, title: str,
                  font_scale: float = 1.0) -> None:
    """Wrap our cohort-aggregated PSDs in Aline's dict layout and call her plotter."""
    psd_data_for_type: dict = {}
    for block, by_cohort in psd_data_by_block.items():
        psd_data_for_type[block] = {}
        for cohort_name, psds in by_cohort.items():
            if psds is None:
                continue
            psd_data_for_type[block][cohort_name] = {
                "psds": psds, "freqs": freqs,
            }
    plot_psd_comparison_multi_block(
        psd_data_for_type=psd_data_for_type,
        blocks_to_plot=list(psd_data_by_block.keys()),
        cohorts_to_plot=cohorts_to_plot,
        cohort_definitions={c: {"group": COHORT_GROUP[c], "ids": None,
                                 "color": COLOR_DICT[c]} for c in cohorts_to_plot},
        colors=[COLOR_DICT[c] for c in cohorts_to_plot],
        title=title,
        filename=str(out_path),
        freq_bands=freq_bands,
        font_scale=font_scale,
    )
    print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modality", choices=["eeg", "dbs", "both"], default="both",
                        help="'eeg' → Fig 2 only; 'dbs' → Fig 1 only; 'both' (default) → both.")
    parser.add_argument("--block", default="pre8walk",
                        help="Task block (in addition to 'baseline', always included).")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    setup_run("plot_psd", config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blocks = ["baseline", args.block]
    freq_bands = {name: tuple(edges) for name, edges in config["freq_bands"].items()}
    subjects_by_cohort = {
        "PD": config["cohorts"]["Patient_NFB"],
        "HC": config["cohorts"]["AgeMatched_NFB"],
    }

    rois_eeg = {
        "Motor_Both":   motor_areas["Both"],
        "Frontal_Both": frontal_areas["Both"],
    }

    if args.modality in ("eeg", "both"):
        for roi_name, ch_list in rois_eeg.items():
            print(f"\n=== EEG · ROI {roi_name} ({len(ch_list)} channels) ===")
            psd_data_by_block: dict = {b: {} for b in blocks}
            freqs_ref = None
            for block in blocks:
                for cohort_name, subs in subjects_by_cohort.items():
                    res = _aggregate_cohort_psds(paths.epochs_dir, subs, block, "eeg", ch_list)
                    if res is None:
                        psd_data_by_block[block][cohort_name] = None
                    else:
                        psd_data_by_block[block][cohort_name], freqs_ref = res
            if freqs_ref is None:
                print(f"  no PSDs for ROI {roi_name} — skip")
                continue
            _plot_one_roi(
                OUT_DIR / f"PSD_PD_vs_HC_{roi_name}.pdf",
                psd_data_by_block, freqs_ref, ["PD", "HC"],
                freq_bands, title=f"Cortical PSD: PD vs HC ({roi_name})",
                font_scale=1.7,  # Fig 2 sits at ~0.49\textwidth; enlarge to stay legible
            )

    if args.modality in ("dbs", "both"):
        print("\n=== DBS · STN (PD cohort, rest vs walk) ===")
        psd_data_by_block: dict = {b: {} for b in blocks}
        freqs_ref = None
        for block in blocks:
            res = _aggregate_cohort_psds(paths.epochs_dir, subjects_by_cohort["PD"],
                                          block, "dbs", channels=None)
            if res is not None:
                psd_data_by_block[block]["PD"], freqs_ref = res
            else:
                psd_data_by_block[block]["PD"] = None
        if freqs_ref is None:
            print("  no DBS PSDs found")
            return 2
        _plot_one_roi(
            OUT_DIR / "PSD_DBS_PD_epoched.pdf",
            psd_data_by_block, freqs_ref, ["PD"],
            freq_bands, title="STN PSD: PD cohort (rest vs walk)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
