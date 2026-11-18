"""Per-(subject, block) trial-count summary table for reviewer queries.

Inputs:
  - data/derivatives/autoreject/<subj>_<block>-mask.npz.meta.json
      (sidecars written by compute_autoreject_mask.py; carry n_orig, n_kept,
      n_bad_eeg, n_bad_dbs per subject/block)
  - code/analysis/config.yaml (cohort lookup)
Outputs:
  - outputs/tables/trial_counts.csv
  - outputs/tables/trial_counts.csv.meta.json
Description:
  Reviewer asks (response_matrix rows 25 / 35 / 79) want to know how many
  trials per condition each subject contributed and how much data was
  removed during artefact rejection. Both numbers fall out of the
  autoreject sidecars — no fresh computation. One row per
  (subject, block); columns: cohort, n_orig, n_kept, pct_kept,
  n_bad_eeg, n_bad_dbs. Sorted by cohort then subject so the table is
  copy-paste ready into the rebuttal/supplement.

Usage:
  python build_trial_counts_table.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from _common import (
    PROJECT_ROOT,
    cohort_of,
    load_config,
    resolve_paths,
    setup_run,
    write_sidecar,
)

OUT_REL = Path("outputs") / "tables" / "trial_counts.csv"


def _row_from_sidecar(meta_path: Path, config) -> dict | None:
    """Read one autoreject sidecar JSON and return a flat row dict.
    Returns None if the JSON is missing the required fields (older masks)."""
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"WARN skip {meta_path.name}: {type(e).__name__}", file=sys.stderr)
        return None
    required = {"subject", "block", "n_orig", "n_kept"}
    if not required.issubset(meta):
        return None
    subj, block = meta["subject"], meta["block"]
    n_orig, n_kept = int(meta["n_orig"]), int(meta["n_kept"])
    pct = (100.0 * n_kept / n_orig) if n_orig else 0.0
    return {
        "subject": subj,
        "cohort": cohort_of(subj, config),
        "block": block,
        "n_orig": n_orig,
        "n_kept": n_kept,
        "pct_kept": round(pct, 2),
        "n_bad_eeg": int(meta.get("n_bad_eeg", 0)),
        "n_bad_dbs": int(meta.get("n_bad_dbs", 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing trial_counts.csv.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    out_path = PROJECT_ROOT / OUT_REL
    if out_path.exists() and not args.force:
        print(f"skip (exists): {out_path}")
        return 0

    meta = setup_run("build_trial_counts_table", config)
    ar_dir = paths.derivatives_root / "autoreject"
    sidecars = sorted(ar_dir.glob("*-mask.npz.meta.json"))
    if not sidecars:
        print(f"missing sidecars under {ar_dir}; run compute_autoreject_mask.py first",
              file=sys.stderr)
        return 2

    rows = [r for r in (_row_from_sidecar(p, config) for p in sidecars) if r is not None]
    rows.sort(key=lambda r: (r["cohort"], r["subject"], r["block"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "cohort", "block", "n_orig", "n_kept",
              "pct_kept", "n_bad_eeg", "n_bad_dbs"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    blocks = sorted({r["block"] for r in rows})
    cohorts = sorted({r["cohort"] for r in rows})
    meta.update({
        "output": str(out_path),
        "n_rows": len(rows),
        "n_sidecars_scanned": len(sidecars),
        "blocks_present": blocks,
        "cohorts_present": cohorts,
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)} "
          f"({len(rows)} rows; blocks={blocks}; cohorts={cohorts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
