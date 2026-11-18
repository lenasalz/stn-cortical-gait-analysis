"""Build the ipsi/contra-aligned STN master CSVs from the canonical DBS masters.

Inputs:
  - data/derivatives/gerd/master_dbs_basenorm.csv   (aggregate_gerd.py)
  - data/derivatives/gerd/master_dbs_avggait.csv

Outputs:
  - data/derivatives/gerd/master_dbs_ipsicontra_basenorm.csv
  - data/derivatives/gerd/master_dbs_ipsicontra_avggait.csv

Description:
  The supplementary STN LMM in `stats_LMM_STN_ipsicontra_final.R` analyses
  STN-LFP gait modulation under a frame in which 0% of the gait cycle is
  always the heel-strike ipsilateral to the recording site (HSI), and 50%
  is the heel-strike of the contralateral foot (HSC).

  For the Right STN, the canonical gait cycle is already HSI-anchored
  (gait cycle starts at right heel-strike at 0%, the ipsi side for the
  right STN). For the Left STN, the canonical 0% is right heel-strike,
  which is the *contra* side; the data therefore need to be circularly
  shifted by -50% of the cycle so that the new 0% sits at the left
  heel-strike (HSI for the left STN).

  Because the master CSVs hold per-cycle gERD values aggregated within
  each gait-phase window (IDS, LLS, FDS, RLS), shifting the time-series
  by -50% is mathematically equivalent to relabelling the Left-STN gait
  phases:
       IDS (0--10%, post-HS-right)  <->  FDS (50--60%, post-HS-left)
       LLS (10--50%, left swing)    <->  RLS (60--100%, right swing)

  After relabelling, the four GaitPhase levels are interpretable in a
  side-relative frame for both hemispheres:
       IDS = post-HSI; LLS = contra-leg swing;
       FDS = post-HSC; RLS = ipsi-leg swing.

  No data values are modified; only the labels for Left-STN rows are
  swapped. The Hemisphere column is preserved (Left / Right) so that
  the LMM's Hemisphere main effect tests for residual ipsi-vs-contra
  power asymmetry across all phases.

Usage:
  python build_stn_ipsicontra_master.py
  python build_stn_ipsicontra_master.py --force
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import load_config, resolve_paths, setup_run  # noqa: E402

# Symmetric swap mapping: each phase exchanges with its 50%-shifted partner.
PHASE_SWAP = {"IDS": "FDS", "FDS": "IDS", "LLS": "RLS", "RLS": "LLS"}


def build_one(in_path: Path, out_path: Path, force: bool) -> int:
    if out_path.exists() and not force:
        print(f"skip {out_path.name} (exists; use --force to overwrite)")
        return 0
    if not in_path.exists():
        print(f"missing input {in_path}", file=sys.stderr)
        return 2
    df = pd.read_csv(in_path)
    n_before = len(df)
    left_mask = df["Hemisphere"] == "Left"
    n_left = int(left_mask.sum())
    df.loc[left_mask, "GaitPhase"] = df.loc[left_mask, "GaitPhase"].map(PHASE_SWAP)
    if df["GaitPhase"].isna().any():
        # Some Left row had an unexpected phase label; abort.
        bad = df[df["GaitPhase"].isna()]
        print(f"unexpected GaitPhase values in Left rows of {in_path.name}: "
              f"{bad.shape[0]} rows", file=sys.stderr)
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path.name}  rows={n_before}  left_relabelled={n_left}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    setup_run("build_stn_ipsicontra_master", config)

    gerd = paths.derivatives_root / "gerd"
    rc = 0
    for norm in ("basenorm", "avggait"):
        in_p = gerd / f"master_dbs_{norm}.csv"
        out_p = gerd / f"master_dbs_ipsicontra_{norm}.csv"
        rc |= build_one(in_p, out_p, args.force)
    return rc


if __name__ == "__main__":
    sys.exit(main())
