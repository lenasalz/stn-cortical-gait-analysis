"""Statistical comparisons of spectral connectivity per Aline's thesis.

Inputs:
  - data/derivatives/connectivity/connectivity_master.csv
      (aggregate_connectivity.py; long-format)
  - code/analysis/config.yaml
Outputs:
  - outputs/tables/connectivity_stats.csv
      One row per (region_pair, metric, band, comparison) with the test
      used, statistic, median diff, raw p, and FDR-BH-adjusted p.
  - outputs/tables/connectivity_stats.csv.meta.json
Description:
  Re-implements the statistical block of thesis §II.D.5:
    - Within-cohort (rest vs walk), per cohort: Shapiro-Wilk on the
      paired differences → paired t-test if normal, Wilcoxon signed-rank
      otherwise.
    - Between-cohort PD vs HC, on motor-frontal connectivity only, per
      block (rest, walk): Shapiro-Wilk on each group → Welch's t-test if
      both normal, Mann-Whitney U otherwise.
    - DBS pairs (dbs-motor, dbs-frontal) are PD-only by construction, so
      only the within-cohort comparison is reported for those.
    - Multiple-comparisons correction: Benjamini-Hochberg, applied
      within each (region_pair, metric, comparison) family across the
      5 bands. Mirrors the thesis convention so the resulting table is
      directly comparable to thesis Table I.

  Normality threshold = 0.05; FDR-BH alpha = 0.05.

Usage:
  python stats_connectivity.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from _common import (
    PROJECT_ROOT,
    atomic_write,
    load_config,
    resolve_paths,
    setup_run,
    should_skip,
    write_sidecar,
)

METRICS = ("coh", "plv", "wpli")
SHAPIRO_ALPHA = 0.05
FDR_ALPHA = 0.05
OUT_REL = "outputs/tables/connectivity_stats.csv"


def _within_cohort(df: pd.DataFrame, cohort: str, region_pair: str,
                   metric: str, band: str) -> dict | None:
    """Rest vs walk within one cohort. Paired test."""
    g = df[(df["cohort"] == cohort) & (df["region_pair"] == region_pair)
           & (df["band"] == band)]
    rest = g[g["block"] == "baseline"].set_index("subject")[metric]
    walk = g[g["block"] == "pre8walk"].set_index("subject")[metric]
    common = rest.index.intersection(walk.index)
    if len(common) < 3:
        return None
    rest = rest.loc[common].to_numpy()
    walk = walk.loc[common].to_numpy()
    diff = walk - rest
    if not np.isfinite(diff).all():
        return None

    sw_stat, sw_p = stats.shapiro(diff)
    if sw_p > SHAPIRO_ALPHA:
        test_name = "Paired t-test"
        t_stat, p = stats.ttest_rel(walk, rest)
        statistic = float(t_stat)
    else:
        test_name = "Wilcoxon"
        try:
            w_stat, p = stats.wilcoxon(walk, rest, zero_method="wilcox")
            statistic = float(w_stat)
        except ValueError:
            return None
    return {
        "region_pair": region_pair,
        "metric": metric,
        "band": band,
        "comparison": f"Rest vs Walk ({cohort_short(cohort)})",
        "n_pairs": int(len(common)),
        "test": test_name,
        "statistic": statistic,
        "median_diff": float(np.median(walk - rest)),
        "shapiro_p": float(sw_p),
        "p_raw": float(p),
    }


def _between_cohort(df: pd.DataFrame, block: str, region_pair: str,
                    metric: str, band: str) -> dict | None:
    """PD vs HC within one block. Independent-samples test on values."""
    g = df[(df["region_pair"] == region_pair)
           & (df["block"] == block) & (df["band"] == band)]
    pd_vals = g[g["cohort"] == "Patient_NFB"][metric].to_numpy()
    hc_vals = g[g["cohort"] == "AgeMatched_NFB"][metric].to_numpy()
    if len(pd_vals) < 3 or len(hc_vals) < 3:
        return None
    if not (np.isfinite(pd_vals).all() and np.isfinite(hc_vals).all()):
        return None

    _, sw_pd = stats.shapiro(pd_vals)
    _, sw_hc = stats.shapiro(hc_vals)
    both_normal = (sw_pd > SHAPIRO_ALPHA) and (sw_hc > SHAPIRO_ALPHA)
    if both_normal:
        test_name = "Welch t-test"
        t_stat, p = stats.ttest_ind(pd_vals, hc_vals, equal_var=False)
        statistic = float(t_stat)
    else:
        test_name = "Mann-Whitney U"
        u_stat, p = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")
        statistic = float(u_stat)
    return {
        "region_pair": region_pair,
        "metric": metric,
        "band": band,
        "comparison": f"PD vs HC ({'Rest' if block == 'baseline' else 'Walk'})",
        "n_pairs": int(len(pd_vals) + len(hc_vals)),
        "test": test_name,
        "statistic": statistic,
        "median_diff": float(np.median(pd_vals) - np.median(hc_vals)),
        "shapiro_p": min(float(sw_pd), float(sw_hc)),
        "p_raw": float(p),
    }


def cohort_short(c: str) -> str:
    return {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}.get(c, c)


def _apply_fdr(rows: list[dict]) -> list[dict]:
    """FDR-BH within each (region_pair, metric, comparison) family across
    the 5 bands. Keeps the family-wise structure thesis Table I uses."""
    rows_with_idx = list(enumerate(rows))
    by_family: dict[tuple, list[int]] = {}
    for idx, r in rows_with_idx:
        key = (r["region_pair"], r["metric"], r["comparison"])
        by_family.setdefault(key, []).append(idx)
    p_fdr = [None] * len(rows)
    reject = [None] * len(rows)
    for family_idxs in by_family.values():
        p_raws = [rows[i]["p_raw"] for i in family_idxs]
        rej, p_adj, _, _ = multipletests(p_raws, alpha=FDR_ALPHA, method="fdr_bh")
        for k, idx in enumerate(family_idxs):
            p_fdr[idx] = float(p_adj[k])
            reject[idx] = bool(rej[k])
    for i, r in enumerate(rows):
        r["p_fdr"] = p_fdr[i]
        r["significant_fdr"] = reject[i]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    in_csv = paths.connectivity_dir / "connectivity_master.csv"
    if not in_csv.exists():
        print(f"missing {in_csv}; run aggregate_connectivity.py first",
              file=sys.stderr)
        return 2
    out_path = PROJECT_ROOT / OUT_REL
    if should_skip(out_path, args.force):
        print(f"skip (exists): {out_path}")
        return 0

    meta = setup_run("stats_connectivity", config)
    df = pd.read_csv(in_csv)

    bands = list(dict.fromkeys(df["band"].tolist()))
    region_pairs = sorted(df["region_pair"].unique().tolist())
    cohorts = sorted(df["cohort"].unique().tolist())

    rows: list[dict] = []
    for rp in region_pairs:
        for metric in METRICS:
            for band in bands:
                # Within-cohort: rest vs walk for every cohort present.
                for c in cohorts:
                    if rp.startswith("dbs-") and c != "Patient_NFB":
                        continue
                    r = _within_cohort(df, c, rp, metric, band)
                    if r is not None:
                        rows.append(r)
                # Between-cohort PD vs HC: only for motor-frontal (DBS
                # pairs don't have an HC counterpart).
                if rp == "motor-frontal":
                    for block in ("baseline", "pre8walk"):
                        r = _between_cohort(df, block, rp, metric, band)
                        if r is not None:
                            rows.append(r)

    if not rows:
        print("no valid comparisons produced", file=sys.stderr)
        return 3

    rows = _apply_fdr(rows)
    cols = ["region_pair", "metric", "band", "comparison", "n_pairs",
            "test", "statistic", "median_diff", "shapiro_p",
            "p_raw", "p_fdr", "significant_fdr"]
    df_out = pd.DataFrame(rows)[cols]
    df_out.sort_values(["region_pair", "metric", "comparison", "band"],
                       inplace=True, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        df_out.to_csv(tmp, index=False)

    n_sig = int(df_out["significant_fdr"].sum())
    meta.update({
        "output": str(out_path),
        "n_comparisons": int(len(df_out)),
        "n_significant_fdr": n_sig,
        "region_pairs": region_pairs,
        "metrics": list(METRICS),
        "bands": bands,
        "shapiro_alpha": SHAPIRO_ALPHA,
        "fdr_alpha": FDR_ALPHA,
    })
    write_sidecar(out_path, meta)
    print(
        f"wrote {out_path}  comparisons={len(df_out)}  "
        f"significant (FDR<{FDR_ALPHA})={n_sig}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
