"""Compare the presence of a spectral peak between cohorts (R1.2).

Purpose:  Test whether the proportion of participants with a detected FOOOF
          peak differs between PD and HC, per cortical region, band (alpha,
          beta 12-30 Hz) and condition (rest, walking). The reviewer asked for
          peak-detection rates to be reported and compared; an absent peak is
          a meaningful state (no oscillation above the aperiodic component),
          so presence is tested directly instead of imputing zero power.
Inputs:   outputs/stats/revision_2026-10/fooof_unwarped_<date>.csv
              (fooof_refit.py; spectrum variant A, min_peak_height 0.15,
              EEG ROI side "Both")
          code/analysis/config.yaml (repro.seed)
Outputs:  outputs/stats/revision_2026-10/fooof_peak_presence_tests_<date>.csv
              region, band, block, n_PD_present, n_PD, n_HC_present, n_HC,
              fisher_p, fisher_p_fdr (Benjamini-Hochberg across all rows),
              n_tests_in_family  (+ .meta.json)
Author:   Lena
Date:     2026-11-18
Seed:     config.yaml:repro.seed (42) via _common.setup_run (exact tests).

Description:
  A beta peak is present when the low- or the high-beta sub-band has one
  (one beta band for the peak summaries, see fooof_peak_locations.py).
  Fisher's exact test (two-sided) on the 2 x 2 table cohort x peak present,
  per region x band x condition: 2 x 2 x 2 = 8 tests, Benjamini-Hochberg
  correction across the eight.

Usage:
  REVISION_DATE=2026-11-16 python code/analysis/revision/fooof_peak_presence_tests.py
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PROJECT_ROOT, atomic_write, load_config, setup_run, write_sidecar  # noqa: E402

DATE = os.environ.get("REVISION_DATE", "2026-11-16")
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
PD, HC = "Patient_NFB", "AgeMatched_NFB"


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main() -> int:
    config = load_config()
    meta = setup_run("fooof_peak_presence_tests", config)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")}
    print(json.dumps({"seed": meta["seed"], **versions}))

    src = STATS_DIR / f"fooof_unwarped_{DATE}.csv"
    f = pd.read_csv(src)
    f = f[(f.spectrum_variant == "A") & (f.min_peak_height == 0.15) & (f.modality == "eeg") & (f.side == "Both")].copy()
    f["Beta_peak_present"] = f["Low Beta_peak_present"].astype(bool) | f["High Beta_peak_present"].astype(bool)

    rows = []
    for region in ("frontal", "motor"):
        for band in ("Alpha", "Beta"):
            for block in ("baseline", "pre8walk"):
                g = f[(f.region == region) & (f.block == block)]
                x, y = g[g.cohort == PD][f"{band}_peak_present"].astype(bool), g[g.cohort == HC][f"{band}_peak_present"].astype(bool)
                a, b = int(x.sum()), int(y.sum())
                p = stats.fisher_exact([[a, len(x) - a], [b, len(y) - b]])[1]
                rows.append({"region": region, "band": band, "block": block, "n_PD_present": a, "n_PD": len(x),
                             "n_HC_present": b, "n_HC": len(y), "fisher_p": p})
    d = pd.DataFrame(rows)
    d["fisher_p_fdr"] = bh(d.fisher_p.to_numpy())
    d["n_tests_in_family"] = len(d)

    out = STATS_DIR / f"fooof_peak_presence_tests_{DATE}.csv"
    with atomic_write(out) as tmp:
        d.to_csv(tmp, index=False)
    meta.update({"author": "Lena", "date": DATE, "input": str(src), "output": str(out), "versions": versions})
    write_sidecar(out, meta)
    print(d.round(4).to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
