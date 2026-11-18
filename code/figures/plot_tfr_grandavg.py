"""Plot gait-warped grand-average TFR per cohort × region from the aggregator NetCDF.

Inputs:
  - data/derivatives/tfr_gavg/<block>-<modality>-tfr-gavg.nc  (aggregate_tfr_gavg.py)
  - code/analysis/config.yaml
Outputs:
  - outputs/figures/<block>_<modality>_<side>_tfr_grandavg.pdf
  - outputs/figures/<block>_<modality>_<side>_tfr_grandavg.pdf.meta.json
Description:
  EEG layout: 3 rows × 2 cols. Rows 1-2 are per-cohort grand averages
  (cols=motor, frontal); row 3 is PD − HC difference. `--side` selects
  motor/frontal sides (default `Both`).

  DBS layout: 1 row × 2 cols. Patient_NFB STN.Left and STN.Right. HC
  cohort has no implant, so no comparison row.

  Time axis 0–1.1 s (one warped gait cycle); freq 1–45 Hz.

Usage:
  python plot_tfr_grandavg.py --block pre8walk --modality eeg
  python plot_tfr_grandavg.py --block pre8walk --modality dbs
  python plot_tfr_grandavg.py --block pre8walk --modality eeg --side Both
  python plot_tfr_grandavg.py --block pre8walk --vmin -5 --vmax 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
    atomic_write,
    load_config,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    write_sidecar,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _tfr_panel(ax, data: np.ndarray, freqs: np.ndarray, times: np.ndarray,
               vmin: float, vmax: float, title: str, cmap: str,
               show_x: bool, show_y: bool, paper_style: bool = False):
    im = ax.pcolormesh(times, freqs, data, vmin=vmin, vmax=vmax, cmap=cmap,
                       shading="auto")
    fs = _FONT_PAPER if paper_style else _FONT_DEFAULT
    ax.set_title(title, fontsize=fs["title"])
    if show_x:
        ax.set_xlabel("Gait Cycle (%)" if paper_style else "time in gait cycle [s]",
                      fontsize=fs["label"])
    if show_y:
        ax.set_ylabel("Frequency (Hz)" if paper_style else "frequency [Hz]",
                      fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])

    # gait-event guides at the canonical percentages of the 1.1 s cycle
    duration = float(times[-1] - times[0])
    line_style = "-" if paper_style else ":"
    line_width = 1.0 if paper_style else 0.6
    line_alpha = 0.7 if paper_style else 0.6
    for pct, name in [(10, "TO-L"), (50, "HS-L"), (60, "TO-R")]:
        x = times[0] + pct / 100.0 * duration
        ax.axvline(x, color="k", linestyle=line_style, linewidth=line_width,
                   alpha=line_alpha)
        ax.text(x, freqs[-1] * 0.97, name, fontsize=fs["gait_marker"],
                ha="center", va="top", color="k", alpha=line_alpha)
    return im


_FONT_DEFAULT = {"title": 10, "label": 10, "ticks": 9,  "gait_marker": 7,  "figure_title": 12}
_FONT_PAPER   = {"title": 14, "label": 12, "ticks": 11, "gait_marker": 9,  "figure_title": 16}

_COHORT_DISPLAY = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}
_BLOCK_DISPLAY  = {"pre8walk": "Walk", "post8walk": "Post-walk", "baseline": "Rest"}


def _cohort_label(cohort: str, paper_style: bool) -> str:
    return _COHORT_DISPLAY.get(cohort, cohort) if paper_style else cohort


def _block_label(block: str, paper_style: bool) -> str:
    return _BLOCK_DISPLAY.get(block, block) if paper_style else block


def _plot_dbs(ds, freqs, times, out_path, args, meta, in_path,
              paths, block) -> int:
    """PD STN.Left / STN.Right grand averages, side-by-side. No HC.

    If a `<block>-dbs-tfr-cluster.nc` exists alongside the aggregator
    NetCDF, overlays the 1-sample cluster-test sig_mask as black contours
    on each panel (`(p<α cluster)` suffix added to that panel's title).
    """
    left  = ds.gavg.sel(cohort="Patient_NFB", region="STN", side="Left").values
    right = ds.gavg.sel(cohort="Patient_NFB", region="STN", side="Right").values
    finite = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    vmin = args.vmin if args.vmin is not None else float(np.percentile(finite, 1))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(finite, 99))
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    cluster_path = paths.tfr_gavg_dir / f"{block}-dbs-tfr-cluster.nc"
    cluster_ds = xr.open_dataset(cluster_path) if cluster_path.exists() else None
    fs = _FONT_PAPER if args.paper_style else _FONT_DEFAULT
    cohort_label = _cohort_label("Patient_NFB", args.paper_style)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"wspace": 0.20})
    mesh = None
    for col_i, (side, panel) in enumerate([("Left", left), ("Right", right)]):
        ax = axes[col_i]
        suffix = ""
        if cluster_ds is not None:
            sig = cluster_ds.sig_mask.sel(region="STN", side=side).values
            if sig.any():
                lw = 1.6 if args.paper_style else 1.2
                ax.contour(times, freqs, sig.astype(float), levels=[0.5],
                           colors="black", linewidths=lw)
                suffix = f"  (p<{float(cluster_ds.attrs['alpha']):.2g} cluster)"
        m = _tfr_panel(
            ax, panel, freqs, times, vmin, vmax,
            title=f"{cohort_label} — STN.{side}{suffix}",
            cmap="jet" if args.paper_style else "RdBu_r",
            show_x=True, show_y=(col_i == 0),
            paper_style=args.paper_style,
        )
        if mesh is None:
            mesh = m
    fig.subplots_adjust(right=0.88, top=0.83, bottom=0.18)
    cax = fig.add_axes([0.90, 0.18, 0.020, 0.65])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("dB vs baseline", fontsize=fs["label"])
    cbar.ax.tick_params(labelsize=fs["ticks"])
    suptitle_block = _block_label(block, args.paper_style)
    fig.suptitle(
        f"Gait-warped grand-average STN TFR — {suptitle_block} (dbs)",
        fontsize=fs["figure_title"], y=0.95,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        fig.savefig(tmp, format="pdf", bbox_inches="tight")
    plt.close(fig)

    meta.update({
        "modality": "dbs", "input": str(in_path), "output": str(out_path),
        "vmin": float(vmin), "vmax": float(vmax),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--side", default="Both",
                        choices=["Left", "Right", "Central", "Both"])
    parser.add_argument("--vmin", type=float, default=None,
                        help="dB color scale lower bound (default: 1st pct of finite data)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="dB color scale upper bound (default: 99th pct)")
    parser.add_argument("--paper-style", action="store_true",
                        help="Larger fonts, paper cohort labels (PD/HC), "
                             "block names (Walk/Rest); same data, same layout.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    block = validate_block(args.block, config)

    in_path = paths.tfr_gavg_dir / f"{block}-{args.modality}-tfr-gavg.nc"
    style_suffix = "_paper" if args.paper_style else ""
    out_path = FIGURES_DIR / (
        f"{block}_{args.modality}_{args.side}_tfr_grandavg{style_suffix}.pdf"
    )
    if not in_path.exists():
        print(f"missing {in_path}; run aggregate_tfr_gavg.py first", file=sys.stderr)
        return 2
    if should_skip(out_path, args.force):
        print(f"skip {out_path.name} (exists)")
        return 0

    meta = setup_run("plot_tfr_grandavg", config)

    ds = xr.open_dataset(in_path)
    cohorts = list(ds.cohort.values)
    side = args.side
    freqs = ds.freq.values
    times = ds.time.values

    if args.modality == "dbs":
        return _plot_dbs(ds, freqs, times, out_path, args, meta, in_path,
                         paths, block)

    if len(cohorts) != 2:
        print(f"expected 2 cohorts for eeg, got {cohorts}", file=sys.stderr)
        return 3

    # Per-cohort dB grids and the difference grid.
    panels: dict[str, np.ndarray] = {}
    for cohort in cohorts:
        for region in ("motor", "frontal"):
            panels[(cohort, region)] = ds.gavg.sel(
                cohort=cohort, region=region, side=side
            ).values
    diff_motor   = panels[(cohorts[0], "motor")]   - panels[(cohorts[1], "motor")]
    diff_frontal = panels[(cohorts[0], "frontal")] - panels[(cohorts[1], "frontal")]

    # Color scales: shared across cohort rows; symmetric for the diff row.
    cohort_finite = np.concatenate([p[np.isfinite(p)] for p in panels.values()])
    if args.vmin is None:
        vmin = float(np.percentile(cohort_finite, 1))
    else:
        vmin = args.vmin
    if args.vmax is None:
        vmax = float(np.percentile(cohort_finite, 99))
    else:
        vmax = args.vmax
    # Make the cohort scale symmetric around 0 if it straddles zero (dB-of-ratio).
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    diff_concat = np.concatenate([
        diff_motor[np.isfinite(diff_motor)],
        diff_frontal[np.isfinite(diff_frontal)],
    ])
    dm = float(np.percentile(np.abs(diff_concat), 99))

    fig, axes = plt.subplots(3, 2, figsize=(10, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.20})

    cohort_mesh = None
    cohort_cmap = "jet" if args.paper_style else "RdBu_r"
    for row_i, cohort in enumerate(cohorts):
        for col_i, region in enumerate(("motor", "frontal")):
            ax = axes[row_i, col_i]
            mesh = _tfr_panel(
                ax, panels[(cohort, region)], freqs, times, vmin, vmax,
                title=f"{_cohort_label(cohort, args.paper_style)} — "
                      f"{region}.{side}",
                cmap=cohort_cmap,
                show_x=False, show_y=(col_i == 0),
                paper_style=args.paper_style,
            )
            if cohort_mesh is None:
                cohort_mesh = mesh

    # Difference row
    cluster_path = paths.tfr_gavg_dir / f"{block}-{args.modality}-tfr-cluster.nc"
    cluster_ds = xr.open_dataset(cluster_path) if cluster_path.exists() else None
    diff_mesh = None
    for col_i, (region, dpanel) in enumerate([("motor", diff_motor),
                                              ("frontal", diff_frontal)]):
        ax = axes[2, col_i]
        suffix = ""
        if cluster_ds is not None:
            sig = cluster_ds.sig_mask.sel(region=region, side=side).values
            if sig.any():
                contour_lw = 1.6 if args.paper_style else 1.2
                ax.contour(times, freqs, sig.astype(float), levels=[0.5],
                           colors="black", linewidths=contour_lw)
                suffix = f"  (p<{float(cluster_ds.attrs['alpha']):.2g} cluster)"
        diff_title = (
            f"{_cohort_label(cohorts[0], args.paper_style)} − "
            f"{_cohort_label(cohorts[1], args.paper_style)}  "
            f"({region}.{side}){suffix}"
        )
        mesh = _tfr_panel(
            ax, dpanel, freqs, times, -dm, dm,
            title=diff_title, cmap="PuOr_r",
            show_x=True, show_y=(col_i == 0),
            paper_style=args.paper_style,
        )
        if diff_mesh is None:
            diff_mesh = mesh

    # Shared colorbars: one for the cohort rows, one for the diff row.
    fig.subplots_adjust(right=0.88, top=0.93, bottom=0.07)
    cax_top = fig.add_axes([0.90, 0.43, 0.020, 0.50])
    cax_bot = fig.add_axes([0.90, 0.07, 0.020, 0.28])
    cbar_top = fig.colorbar(cohort_mesh, cax=cax_top)
    cbar_bot = fig.colorbar(diff_mesh, cax=cax_bot)
    fs = _FONT_PAPER if args.paper_style else _FONT_DEFAULT
    cbar_top.set_label("dB vs baseline", fontsize=fs["label"])
    cbar_bot.set_label("Δ dB", fontsize=fs["label"])
    cbar_top.ax.tick_params(labelsize=fs["ticks"])
    cbar_bot.ax.tick_params(labelsize=fs["ticks"])

    suptitle_block = _block_label(block, args.paper_style)
    fig.suptitle(
        f"Gait-warped grand-average TFR — {suptitle_block} "
        f"({args.modality}, side={side})",
        fontsize=fs["figure_title"], y=0.97,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        fig.savefig(tmp, format="pdf", bbox_inches="tight")
    plt.close(fig)

    meta.update({
        "block": block, "modality": args.modality, "side": side,
        "input": str(in_path), "output": str(out_path),
        "cohorts": cohorts,
        "regions": ["motor", "frontal"],
        "vmin": float(vmin), "vmax": float(vmax),
        "diff_vmax_symmetric": float(dm),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
