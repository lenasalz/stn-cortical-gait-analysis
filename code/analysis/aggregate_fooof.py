"""Concatenate per-subject FOOOF CSVs into a master DataFrame.

Inputs:
  - data/derivatives/fooof/*-fooof.csv  (compute_fooof.py)
  - code/analysis/config.yaml (cohorts, blocks)
Outputs:
  - data/derivatives/fooof/fooof_master.csv
      All per-(subject, block, modality, region, side) rows in one file.
  - data/derivatives/fooof/fooof_master.csv.meta.json
Description:
  Reads every `<subj>_<block>-<modality>-fooof.csv` under
  `data/derivatives/fooof/`, concatenates them, and writes a single
  master CSV. The master CSV is the input expected by the legacy
  `psd_fooof.py:run_fooof_statistics` (Welch's t-test / Mann-Whitney
  + FDR per parameter family) and by `code/analysis/R/stats_lmm_*.R`
  for the manuscript's LMM tables (tab1, tab2, tab9).

  The master frame's primary keys are
  (subject, block, modality, region, side); each row has aperiodic
  params + per-band peak summaries. NaN in a peak column means no
  peak fell in that band for that subject.

Usage:
  python aggregate_fooof.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _common import (
    atomic_write,
    load_config,
    resolve_paths,
    setup_run,
    should_skip,
    write_sidecar,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("aggregate_fooof", config)

    csvs = sorted(paths.fooof_dir.glob("*-fooof.csv"))
    # Skip the master itself if it already exists.
    csvs = [p for p in csvs if not p.name.startswith("fooof_master")]
    if not csvs:
        print(f"no per-subject fooof CSVs under {paths.fooof_dir}; "
              "run compute_fooof.py first", file=sys.stderr)
        return 2

    out_path = paths.fooof_dir / "fooof_master.csv"
    if should_skip(out_path, args.force):
        print(f"skip {out_path.name} (exists)")
        return 0

    frames = [pd.read_csv(p) for p in csvs]
    master = pd.concat(frames, ignore_index=True)

    # Sort for stability of downstream stats.
    master = master.sort_values(
        ["modality", "region", "side", "cohort", "subject", "block"]
    ).reset_index(drop=True)

    with atomic_write(out_path) as tmp:
        master.to_csv(tmp, index=False)

    counts = (master
              .groupby(["modality", "cohort", "block"])
              .size()
              .reset_index(name="n_rows"))
    meta.update({
        "input_dir": str(paths.fooof_dir),
        "n_per_subject_csvs": len(csvs),
        "output": str(out_path),
        "n_rows": int(len(master)),
        "rows_by_modality_cohort_block": counts.to_dict(orient="records"),
    })
    write_sidecar(out_path, meta)

    print(f"wrote {out_path.name}  rows={len(master)}  "
          f"from {len(csvs)} per-subject files")
    print(counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
