"""Bar plot per region pair: connectivity rest vs walk × cohort × band.

Inputs:
  - data/derivatives/connectivity/connectivity_master.csv  (aggregate_connectivity.py)
  - outputs/tables/connectivity_stats.csv                 (stats_connectivity.py)
  - code/analysis/config.yaml (freq_bands ordering)
Outputs:
  - outputs/figures/supplementary/connectivity_<metric>.pdf  (+ .png)
  - outputs/figures/supplementary/connectivity_<metric>.pdf.meta.json
Description:
  One figure per connectivity metric (coh, plv, wpli). Three panels
  (one per region pair: motor-frontal, dbs-motor, dbs-frontal), each
  with grouped bars: cohort × condition (rest, walk) on the x-axis,
  hue = condition, x-major = band. Error bars are SEM across subjects.
  Significance asterisks read from connectivity_stats.csv use the
  FDR-corrected p-values (* p<0.05, ** p<0.01, *** p<0.001, ° p<0.1).

  Bars are sized per the thesis figure-9 style (rebuttal supplement);
  the table is the source of truth for the rebuttal, the figure is
  there for visual comparison.

Usage:
  python code/figures/plot_connectivity.py
  python code/figures/plot_connectivity.py --metric coh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    resolve_paths,
    setup_run,
    write_sidecar,
)

METRICS = ("coh", "plv", "wpli")
COHORT_LABEL = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}
COND_LABEL = {"baseline": "Rest", "pre8walk": "Walk"}
OUT_REL = Path("outputs") / "figures" / "supplementary"


def _stars(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "°"
    return ""


def _plot_one(metric: str, df: pd.DataFrame, stats_df: pd.DataFrame,
              bands: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    region_pairs = ["motor-frontal", "dbs-motor", "dbs-frontal"]
    region_pairs = [rp for rp in region_pairs
                    if rp in df["region_pair"].unique()]
    n_rp = len(region_pairs)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_rp,
        figsize=(4.5 * n_rp, 3.6),
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]

    width = 0.18
    series = [
        ("Patient_NFB", "baseline", "#c7d6f0", "PD Rest"),
        ("Patient_NFB", "pre8walk", "#264e9a", "PD Walk"),
        ("AgeMatched_NFB", "baseline", "#f4cfcf", "HC Rest"),
        ("AgeMatched_NFB", "pre8walk", "#b13a3a", "HC Walk"),
    ]

    for ax, rp in zip(axes, region_pairs):
        sub = df[(df["region_pair"] == rp)]
        x = np.arange(len(bands))
        for j, (cohort, block, color, label) in enumerate(series):
            offsets_j = (j - (len(series) - 1) / 2) * width
            means, sems = [], []
            for band in bands:
                vals = sub[(sub["cohort"] == cohort)
                           & (sub["block"] == block)
                           & (sub["band"] == band)][metric].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    means.append(np.nan)
                    sems.append(0.0)
                else:
                    means.append(float(np.mean(vals)))
                    sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                if len(vals) > 1 else 0.0)
            if rp.startswith("dbs-") and cohort != "Patient_NFB":
                continue
            ax.bar(x + offsets_j, means, width=width, color=color,
                   edgecolor="black", linewidth=0.4, label=label)
            ax.errorbar(x + offsets_j, means, yerr=sems, fmt="none",
                        ecolor="black", elinewidth=0.7, capsize=2)

        # Significance markers: within-cohort rest vs walk, plotted above the walk bar
        if rp.startswith("dbs-"):
            cohorts_to_mark = ["Patient_NFB"]
        else:
            cohorts_to_mark = ["Patient_NFB", "AgeMatched_NFB"]
        for c in cohorts_to_mark:
            label_c = COHORT_LABEL[c]
            for band_i, band in enumerate(bands):
                hit = stats_df[
                    (stats_df["region_pair"] == rp)
                    & (stats_df["metric"] == metric)
                    & (stats_df["band"] == band)
                    & (stats_df["comparison"] == f"Rest vs Walk ({label_c})")
                ]
                if hit.empty:
                    continue
                pf = hit["p_fdr"].iloc[0]
                star = _stars(pf)
                if not star:
                    continue
                # Walk bar offset:
                if c == "Patient_NFB":
                    walk_offset = (1 - (len(series) - 1) / 2) * width
                    color = "#264e9a"
                else:
                    walk_offset = (3 - (len(series) - 1) / 2) * width
                    color = "#b13a3a"
                walk_val = sub[(sub["cohort"] == c)
                               & (sub["block"] == "pre8walk")
                               & (sub["band"] == band)][metric]
                walk_val = walk_val.to_numpy()
                walk_val = walk_val[np.isfinite(walk_val)]
                if len(walk_val) == 0:
                    continue
                ymax = ax.get_ylim()[1]
                y = max(np.mean(walk_val) * 1.10, np.mean(walk_val) + 0.02 * ymax)
                ax.text(band_i + walk_offset, y, star, ha="center", va="bottom",
                        fontsize=10, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=15)
        ax.set_title(rp)
        ax.set_ylabel(metric.upper())
        ax.spines[["top", "right"]].set_visible(False)
        if rp == region_pairs[0]:
            ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=2)

    fig.suptitle(
        f"Spectral connectivity — {metric.upper()} "
        f"(mean ± SEM across subjects; stars: rest vs walk within cohort, FDR-BH)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metric", choices=METRICS, default=None,
                        help="If given, render only this metric. Default: all three.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    bands = list(config["freq_bands"].keys())

    master_csv = paths.connectivity_dir / "connectivity_master.csv"
    stats_csv = PROJECT_ROOT / "outputs" / "tables" / "connectivity_stats.csv"
    for p, name in ((master_csv, "connectivity_master.csv"),
                    (stats_csv, "connectivity_stats.csv")):
        if not p.exists():
            print(f"missing {name}: {p}", file=sys.stderr)
            return 2

    meta = setup_run("plot_connectivity", config)
    df = pd.read_csv(master_csv)
    stats_df = pd.read_csv(stats_csv)

    out_dir = PROJECT_ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [args.metric] if args.metric else list(METRICS)
    outputs: list[Path] = []
    for metric in metrics:
        out_path = out_dir / f"connectivity_{metric}.pdf"
        if out_path.exists() and not args.force:
            print(f"skip (exists): {out_path}")
            continue
        _plot_one(metric, df, stats_df, bands, out_path)
        outputs.append(out_path)
        print(f"wrote {out_path.relative_to(PROJECT_ROOT)}")

    meta.update({
        "metrics_rendered": metrics,
        "outputs": [str(o) for o in outputs],
    })
    if outputs:
        write_sidecar(outputs[0], meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
