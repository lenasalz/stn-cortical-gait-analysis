"""Peak-detection rates and fit quality: submitted (warped) vs unwarped FOOOF (R1.2, R1.3).

Purpose:
  Count how often the FOOOF model detected a peak in each frequency band,
  per cohort x block x ROI, in the submitted table (warped walking epochs,
  min_peak_height 0.15) and in the unwarped refit (spectrum variants A and
  B, min_peak_height 0.05 / 0.09 / 0.15). ROIs are motor and frontal
  (EEG, side "Both") and STN (DBS). Tests, run per ROI x band because
  ROIs are not independent subjects:
    - PD vs HC within block: Fisher exact test on
      [[PD present, PD absent], [HC present, HC absent]] (EEG only).
    - rest vs walking within cohort: exact McNemar test on the paired
      per-subject presence (two-sided binomial test on the discordant
      pairs, p = 1 when there are none).
  A fit-quality table gives mean / SD / min of r_squared and mean / SD /
  max of the fit error per cohort x block x modality for every source.

Inputs:
  - data/derivatives/fooof/fooof_master.csv (submitted; aggregate_fooof.py)
  - outputs/stats/revision_2026-10/fooof_unwarped_2026-10-13.csv (fooof_refit.py)
  - code/analysis/config.yaml (freq_bands, repro.seed)
Outputs:
  - outputs/stats/revision_2026-10/fooof_detection_rates_2026-10-13.csv
      one row per source x spectrum_variant x min_peak_height x region x
      band x cohort x block with n_present / n_total and the test p-values
      (+ .meta.json)
  - outputs/stats/revision_2026-10/fit_quality_2026-10-13.csv  (+ .meta.json)
Author: Lena
Date: 2026-10-13
Seed: config.yaml:repro.seed (42) via _common.setup_run (counting and
  exact tests are deterministic; seed set for completeness).

Usage:
  python fooof_detection_rates.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

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
STAGE = "fooof_detection_rates"
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
SUBMITTED = PROJECT_ROOT / "data" / "derivatives" / "fooof" / "fooof_master.csv"
REFIT = STATS_DIR / f"fooof_unwarped_{DATE}.csv"
OUT_RATES = STATS_DIR / f"fooof_detection_rates_{DATE}.csv"
OUT_FIT = STATS_DIR / f"fit_quality_{DATE}.csv"
KEYS = ["source", "spectrum_variant", "min_peak_height"]


def load_sources() -> pd.DataFrame:
    sub = pd.read_csv(SUBMITTED)
    # ROI rows used in the manuscript: EEG side "Both", STN (either side).
    sub = sub[((sub["modality"] == "eeg") & (sub["side"] == "Both"))
              | (sub["modality"] == "dbs")].copy()
    sub["source"] = "submitted_warped"
    sub["spectrum_variant"] = "submitted"
    sub["min_peak_height"] = 0.15
    new = pd.read_csv(REFIT)
    new["source"] = "unwarped"
    cols = ["source", "spectrum_variant", "min_peak_height", "subject", "cohort",
            "block", "modality", "region", "side", "aperiodic_offset",
            "aperiodic_exponent", "fit_r_squared", "fit_error"]
    present = [c for c in sub.columns if c.endswith("_peak_present")]
    return pd.concat([sub[cols + present], new[cols + present]], ignore_index=True)


def mcnemar_exact(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Discordant counts (b: rest only, c: walking only) and exact p."""
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    n = bb + cc
    p = 1.0 if n == 0 else float(binomtest(min(bb, cc), n, 0.5).pvalue)
    return bb, cc, p


def detection_rates(df: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS, sort=False):
        for region, gr in g.groupby("region", sort=False):
            for band in bands:
                col = f"{band}_peak_present"
                sub = gr[["subject", "cohort", "block", col]].copy()
                sub[col] = sub[col].astype(int)
                # PD vs HC within block (Fisher)
                fisher = {}
                for block, gb in sub.groupby("block"):
                    pd_ = gb[gb["cohort"] == "Patient_NFB"][col]
                    hc_ = gb[gb["cohort"] == "AgeMatched_NFB"][col]
                    if len(pd_) and len(hc_):
                        tab = [[int(pd_.sum()), int(len(pd_) - pd_.sum())],
                               [int(hc_.sum()), int(len(hc_) - hc_.sum())]]
                        fisher[block] = float(fisher_exact(tab)[1])
                    else:
                        fisher[block] = np.nan
                # rest vs walking within cohort (exact McNemar, paired)
                for cohort, gc in sub.groupby("cohort"):
                    wide = gc.pivot(index="subject", columns="block", values=col)
                    if {"baseline", "pre8walk"} <= set(wide.columns):
                        w = wide.dropna()
                        b, c, p_mc = mcnemar_exact(w["baseline"], w["pre8walk"])
                    else:
                        b = c = 0
                        p_mc = np.nan
                    for block, gb in gc.groupby("block"):
                        rows.append({
                            "source": key[0], "spectrum_variant": key[1],
                            "min_peak_height": key[2], "region": region,
                            "band": band, "cohort": cohort, "block": block,
                            "n_present": int(gb[col].sum()),
                            "n_total": int(len(gb)),
                            "rate": float(gb[col].mean()),
                            "fisher_p_PD_vs_HC_within_block": fisher.get(block, np.nan),
                            "mcnemar_rest_only": b, "mcnemar_walk_only": c,
                            "mcnemar_exact_p_rest_vs_walk_within_cohort": p_mc,
                        })
    return pd.DataFrame(rows)


def fit_quality(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEYS + ["cohort", "block", "modality"], sort=False)
    out = g.agg(
        n=("fit_r_squared", "size"),
        r2_mean=("fit_r_squared", "mean"), r2_sd=("fit_r_squared", "std"),
        r2_min=("fit_r_squared", "min"),
        error_mean=("fit_error", "mean"), error_sd=("fit_error", "std"),
        error_max=("fit_error", "max"),
        exponent_mean=("aperiodic_exponent", "mean"),
        exponent_sd=("aperiodic_exponent", "std"),
        offset_mean=("aperiodic_offset", "mean"),
        offset_sd=("aperiodic_offset", "std"),
    ).reset_index()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    config = load_config()
    meta = setup_run(STAGE, config)
    bands = list(config["freq_bands"])

    df = load_sources()
    rates = detection_rates(df, bands)
    fq = fit_quality(df)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(OUT_RATES) as tmp:
        rates.to_csv(tmp, index=False)
    with atomic_write(OUT_FIT) as tmp:
        fq.to_csv(tmp, index=False)
    base = {"author": "Lena", "date": DATE,
            "inputs": [str(SUBMITTED), str(REFIT)],
            "submitted_rows_used": "eeg side Both + dbs STN",
            "tests": "Fisher exact PD vs HC within block; exact McNemar rest vs walking within cohort",
            }
    write_sidecar(OUT_RATES, {**meta, **base, "output": str(OUT_RATES), "n_rows": int(len(rates))})
    write_sidecar(OUT_FIT, {**meta, **base, "output": str(OUT_FIT), "n_rows": int(len(fq))})

    # console summary at the manuscript setting
    show = rates[(rates["min_peak_height"] == 0.15)
                 & (rates["spectrum_variant"].isin(["submitted", "A"]))]
    piv = show.pivot_table(index=["region", "band", "cohort", "block"],
                           columns="spectrum_variant", values="n_present")
    print(piv.to_string())
    print(f"wrote {OUT_RATES} rows={len(rates)}; {OUT_FIT} rows={len(fq)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
