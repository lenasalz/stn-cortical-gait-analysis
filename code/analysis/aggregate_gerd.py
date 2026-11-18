"""Concatenate per-subject gERD/ERS CSVs into masters for the R LMM scripts.

Inputs:
  - data/derivatives/gerd/*-gerd.csv  (compute_gerd_table.py)
Outputs:
  - data/derivatives/gerd/master_eeg_basenorm.csv   (cortical R script input)
  - data/derivatives/gerd/master_dbs_basenorm.csv   (STN R script input)
  - sidecar JSONs
Description:
  Reads every per-subject `<subj>_<block>-<modality>-gerd.csv`, splits
  by modality, writes two master CSVs whose schemas match what
  `code/analysis/R/stats_lmm_cortical_complex_final.R` and
  `stats_lmm_STN_complex_final.R` expect:
    Cortical master columns: SubjectID, Cohort, Group, Modality, Block,
      Area (motor|frontal), Hemisphere (Left|Right|Central|Both),
      GaitPhase (IDS|LLS|FDS|RLS), Epoch, ERD_ERS_<band>×5
    STN master columns: SubjectID, Cohort, Group, Block,
      Hemisphere (Left|Right — implant side), GaitPhase, Epoch,
      ERD_ERS_<band>×5

  The cortical R script filters `Hemisphere == "Both"` internally;
  the master keeps all sides so different analyses can re-filter.

Usage:
  python aggregate_gerd.py
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
    meta = setup_run("aggregate_gerd", config)

    gerd_dir = paths.derivatives_root / "gerd"
    csvs = sorted(gerd_dir.glob("*-gerd.csv"))
    csvs = [p for p in csvs if not p.name.startswith("master_")]
    if not csvs:
        print(f"no per-subject gerd CSVs under {gerd_dir}", file=sys.stderr)
        return 2

    # Per-file filenames encode the norm:
    #   `<subj>_<block>-{modality}-{norm}-gerd.csv` (current)
    #   legacy single-norm name (pre-avggait): `<subj>_<block>-{modality}-gerd.csv`
    # Group by inferred norm; legacy unspecified treated as basenorm.
    def _infer_norm(p):
        stem = p.stem  # e.g. P4-2002_pre8walk-eeg-avggait-gerd
        if stem.endswith("-avggait-gerd"): return "avggait"
        if stem.endswith("-basenorm-gerd"): return "basenorm"
        return "basenorm"  # legacy filename without explicit norm

    frames = []
    for p in csvs:
        d = pd.read_csv(p)
        d["Norm"] = _infer_norm(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # The masters feed the manuscript LMMs, so they carry the manuscript
    # blocks only (config["blocks"]). Per-subject tables of
    # config["extra_blocks"] (post8walk, written for exploration since
    # 2026-10-14) share the gerd/ folder and must not enter the masters.
    manuscript_blocks = set(config["blocks"])
    dropped = sorted(set(df["Block"].unique()) - manuscript_blocks)
    if dropped:
        print(f"excluding non-manuscript blocks from the masters: {dropped}")
    df = df[df["Block"].isin(manuscript_blocks)].reset_index(drop=True)

    counts = {}
    for modality in ("eeg", "dbs"):
        for norm in ("basenorm", "avggait"):
            sub = df[(df["Modality"] == modality) & (df["Norm"] == norm)].copy()
            if sub.empty:
                continue
            out_path = gerd_dir / f"master_{modality}_{norm}.csv"
            if should_skip(out_path, args.force):
                print(f"skip {out_path.name} (exists)")
                continue
            sub = sub.drop(columns=["Norm"]).sort_values(
                ["Cohort", "SubjectID", "Block", "Area", "Hemisphere",
                 "GaitPhase", "Epoch"]
            ).reset_index(drop=True)
            with atomic_write(out_path) as tmp:
                sub.to_csv(tmp, index=False)
            counts[f"{modality}_{norm}"] = {
                "rows": int(len(sub)),
                "subjects": int(sub["SubjectID"].nunique()),
                "blocks":   sorted(sub["Block"].unique().tolist()),
            }
            print(f"wrote {out_path.name}  rows={len(sub)}  subjects={sub['SubjectID'].nunique()}")

    meta.update({
        "input_dir": str(gerd_dir),
        "n_per_subject_csvs": len(csvs),
        "masters": counts,
    })
    for key in counts:
        out_path = gerd_dir / f"master_{key}.csv"
        if out_path.exists():
            write_sidecar(out_path, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
