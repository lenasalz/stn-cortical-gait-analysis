"""Orchestrator: run a stage across all (subject, block) pairs in parallel.

Inputs:
  - code/analysis/config.yaml (cohorts, blocks)
  - the stage-specific compute script (build_epochs.py, compute_psd.py, ...)
Outputs:
  - whatever the called stage produces under data/derivatives/<stage>/
Description:
  Thin joblib.Parallel wrapper around the per-subject CLIs. Each stage
  is content-hashed by the existence of its output file (skip-if-exists
  is the per-script's responsibility) so re-running is cheap.

Usage:
  python run_all.py --stage psd
  python run_all.py --stage tfr --modality eeg --n-jobs 4
  python run_all.py --stage psd --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import load_config

HERE = Path(__file__).resolve().parent

STAGES = {
    "epochs":         ("build_epochs.py", []),
    "raw":            ("../preprocessing/build_raw_from_pickle.py", []),
    "psd":            ("compute_psd.py", []),
    "tfr":            ("compute_tfr.py", ["--modality"]),
    "tfr_continuous": ("compute_tfr_continuous.py", ["--modality"]),
    "tfr_basenorm":   ("compute_tfr_basenorm.py", ["--modality"]),
    "tfr_gait":       ("compute_tfr_gait.py", ["--modality"]),
    "fooof":          ("compute_fooof.py", []),
    "connectivity":   ("compute_connectivity.py", []),
    "band_power":     ("compute_band_power.py", []),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=list(STAGES.keys()))
    parser.add_argument("--modality", choices=["eeg", "dbs"],
                        help="Required when --stage tfr.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel subjects (joblib).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    script_name, required_extras = STAGES[args.stage]
    accepts_modality = "--modality" in required_extras
    if accepts_modality and not args.modality:
        parser.error(f"--modality required for stage {args.stage!r}")
    if args.modality and not accepts_modality:
        parser.error(
            f"--modality is only valid for stage 'tfr', not {args.stage!r}"
        )

    config = load_config()
    jobs = [
        (subj, block)
        for cohort, members in config["cohorts"].items()
        for subj in members
        for block in config["blocks"]
    ]

    # Special cases: build_epochs and build_raw_from_pickle each load
    # their cohort pickles once per process; running per-subject would
    # re-load the (~11 GB) pickle for every job. Shell out once with
    # --all instead. Other stages parallelize via joblib below.
    if args.stage in ("epochs", "raw"):
        cmd = [sys.executable, str(HERE / script_name), "--all"]
        if args.force:
            cmd.append("--force")
        return subprocess.run(cmd, check=False).returncode

    # tfr_basenorm and tfr_gait are per-task only — baseline has no
    # gait events and basenorm-against-itself is undefined.
    if args.stage in ("tfr_basenorm", "tfr_gait"):
        baseline_block = "baseline"
        jobs = [
            (subj, block)
            for cohort, members in config["cohorts"].items()
            for subj in members
            for block in config["blocks"]
            if block != baseline_block
        ]

    from joblib import Parallel, delayed

    def run_one(subject: str, block: str) -> int:
        cmd = [sys.executable, str(HERE / script_name),
               "--subject", subject, "--block", block]
        if accepts_modality and args.modality:
            cmd += ["--modality", args.modality]
        if args.force:
            cmd.append("--force")
        return subprocess.run(cmd, check=False).returncode

    rcs = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one)(s, b) for s, b in jobs
    )
    failed = sum(rc != 0 for rc in rcs)
    print(f"stage={args.stage} jobs={len(jobs)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
