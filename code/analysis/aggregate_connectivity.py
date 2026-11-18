"""Concatenate per-subject connectivity NetCDFs into a long master CSV.

Inputs:
  - data/derivatives/connectivity/*-conn.nc  (compute_connectivity.py)
  - code/analysis/config.yaml (cohort lookup)
Outputs:
  - data/derivatives/connectivity/connectivity_master.csv
      Long format; one row per (subject, block, region_pair, band, metric).
  - data/derivatives/connectivity/connectivity_master.csv.meta.json
Description:
  Reads every `<subj>_<block>-conn.nc`, melts it from the per-subject
  (region_pair × band) variable matrix into a long-format table that's
  easier to feed into stats / plots. Columns:
    subject, cohort, block, region_pair, band, coh, plv, wpli,
    n_kept, n_orig, mask_source

  This is the analogue of `fooof_master.csv` for the connectivity
  pipeline. The master frame is the input expected by
  `stats_connectivity.py` (rest-vs-walk + PD-vs-HC with Shapiro-Wilk →
  paired t / Wilcoxon / Mann-Whitney + FDR-BH) and by the supplementary
  table / figure renderers.

Usage:
  python aggregate_connectivity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from _common import (
    atomic_write,
    cohort_of,
    load_config,
    resolve_paths,
    setup_run,
    should_skip,
    write_sidecar,
)


def _rows_from_nc(nc_path: Path, config) -> list[dict]:
    """Melt one per-subject NetCDF into long-format rows."""
    ds = xr.open_dataset(nc_path)
    subject = ds.attrs.get("subject")
    block = ds.attrs.get("block")
    if subject is None or block is None:
        # Fall back to filename parse.
        stem = nc_path.stem.replace("-conn", "")
        subject, block = stem.split("_", 1)
    cohort = cohort_of(subject, config)

    rows: list[dict] = []
    for rp in ds.region_pair.values:
        for band in ds.band.values:
            rows.append({
                "subject": subject,
                "cohort": cohort,
                "block": block,
                "region_pair": str(rp),
                "band": str(band),
                "coh": float(ds.coh.sel(region_pair=rp, band=band)),
                "plv": float(ds.plv.sel(region_pair=rp, band=band)),
                "wpli": float(ds.wpli.sel(region_pair=rp, band=band)),
                "n_kept": int(ds.attrs.get("n_kept", -1)),
                "n_orig": int(ds.attrs.get("n_orig", -1)),
                "mask_source": str(ds.attrs.get("mask_source", "")),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    out_path = paths.connectivity_dir / "connectivity_master.csv"
    if should_skip(out_path, args.force):
        print(f"skip (exists): {out_path}")
        return 0

    meta = setup_run("aggregate_connectivity", config)
    files = sorted(paths.connectivity_dir.glob("*-conn.nc"))
    if not files:
        print(f"missing connectivity NetCDFs under {paths.connectivity_dir}; "
              f"run compute_connectivity.py first", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for f in files:
        rows.extend(_rows_from_nc(f, config))
    df = pd.DataFrame(rows)
    df.sort_values(["cohort", "subject", "block", "region_pair", "band"],
                   inplace=True, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        df.to_csv(tmp, index=False)
    meta.update({
        "output": str(out_path),
        "n_rows": int(len(df)),
        "n_files_scanned": int(len(files)),
        "blocks_present": sorted(df["block"].unique().tolist()),
        "cohorts_present": sorted(df["cohort"].unique().tolist()),
        "region_pairs_present": sorted(df["region_pair"].unique().tolist()),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path}  rows={len(df)}  files={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
