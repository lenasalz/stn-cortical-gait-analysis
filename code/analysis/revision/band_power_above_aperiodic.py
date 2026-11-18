"""Tidy long table of peak-free band power from the unwarped spectra (R1.2).

Purpose:
  Reshape the per-band aperiodic-adjusted band power (mean of the FOOOF
  flattened spectrum over the band, log10 units) and the relative band
  power (band sum / 4-45 Hz sum of the raw spectrum) from fooof_refit.py
  into one long table, one row per subject x block x modality x region x
  spectrum_variant x min_peak_height x band x measure. The primary setting
  for the Cohort x Condition model (psd_mixed_anova.R, separate step) is
  spectrum_variant A with min_peak_height 0.15 (flag `is_primary`); the
  other settings are kept for sensitivity analyses. Note that the relative
  power does not depend on min_peak_height, and the adjusted power depends
  on it only through the aperiodic fit.

Inputs:
  - outputs/stats/revision_2026-10/fooof_unwarped_2026-10-13.csv (fooof_refit.py)
  - code/analysis/config.yaml (freq_bands, repro.seed)
Outputs:
  - outputs/stats/revision_2026-10/band_power_unwarped_2026-10-13.csv
      columns: subject, cohort, block, modality, region, side,
      spectrum_variant, min_peak_height, is_primary, band, measure
      (adj_power | rel_power), value  (+ .meta.json)
  - outputs/stats/revision_2026-10/band_power_unwarped_descriptives_2026-10-13.csv
      mean, SD, n per cohort x block x region x band x measure x
      spectrum_variant x min_peak_height  (+ .meta.json)
Author: Lena
Date: 2026-10-13
Seed: config.yaml:repro.seed (42) via _common.setup_run (reshaping only).

Usage:
  python band_power_above_aperiodic.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

# Output date tag; override with the environment variable REVISION_DATE for a rerun.
DATE = __import__("os").environ.get("REVISION_DATE", "2026-10-13")
STAGE = "band_power_above_aperiodic"
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
REFIT = STATS_DIR / f"fooof_unwarped_{DATE}.csv"
OUT_LONG = STATS_DIR / f"band_power_unwarped_{DATE}.csv"
OUT_DESC = STATS_DIR / f"band_power_unwarped_descriptives_{DATE}.csv"
PRIMARY = ("A", 0.15)
ID_COLS = ["subject", "cohort", "block", "modality", "region", "side",
           "spectrum_variant", "min_peak_height"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    config = load_config()
    meta = setup_run(STAGE, config)
    bands = list(config["freq_bands"])

    df = pd.read_csv(REFIT)
    parts = []
    for measure in ("adj_power", "rel_power"):
        cols = {f"{b}_{measure}": b for b in bands}
        long = df[ID_COLS + list(cols)].melt(
            id_vars=ID_COLS, var_name="band", value_name="value")
        long["band"] = long["band"].map(cols)
        long["measure"] = measure
        parts.append(long)
    long = pd.concat(parts, ignore_index=True)
    long["is_primary"] = ((long["spectrum_variant"] == PRIMARY[0])
                          & (long["min_peak_height"] == PRIMARY[1]))
    long["band"] = pd.Categorical(long["band"], categories=bands, ordered=True)
    long = long.sort_values(
        ["measure", "modality", "region", "spectrum_variant", "min_peak_height",
         "cohort", "subject", "block", "band"]).reset_index(drop=True)

    desc = (long.groupby(["measure", "modality", "region", "spectrum_variant",
                          "min_peak_height", "cohort", "block", "band"],
                         observed=True)["value"]
            .agg(n="size", mean="mean", sd="std").reset_index())

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(OUT_LONG) as tmp:
        long.to_csv(tmp, index=False)
    with atomic_write(OUT_DESC) as tmp:
        desc.to_csv(tmp, index=False)
    base = {"author": "Lena", "date": DATE, "input": str(REFIT),
            "primary": {"spectrum_variant": PRIMARY[0], "min_peak_height": PRIMARY[1]},
            "adj_power": "mean over band of log10(power) - aperiodic fit (FOOOF flattened spectrum)",
            "rel_power": "band linear power sum / 4-45 Hz sum"}
    write_sidecar(OUT_LONG, {**meta, **base, "output": str(OUT_LONG), "n_rows": int(len(long))})
    write_sidecar(OUT_DESC, {**meta, **base, "output": str(OUT_DESC), "n_rows": int(len(desc))})

    show = desc[(desc["spectrum_variant"] == PRIMARY[0])
                & (desc["min_peak_height"] == PRIMARY[1])
                & (desc["measure"] == "adj_power")]
    print(show.pivot_table(index=["region", "band"], columns=["cohort", "block"],
                           values="mean").round(3).to_string())
    print(f"wrote {OUT_LONG} rows={len(long)}; {OUT_DESC} rows={len(desc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
