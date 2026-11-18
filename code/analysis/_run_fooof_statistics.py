"""run_fooof_statistics — verbatim port of
final_analysis/scripts/psd_fooof.py:run_fooof_statistics, extracted to
avoid the specparam import at module load time.

Purpose: stats over FOOOF parameters (CF, PW per band + aperiodic
offset/exponent), with paired t-test / Wilcoxon for within-cohort
(rest vs walk) and Welch / Mann-Whitney for between-cohort (PD vs HC),
then Benjamini-Hochberg FDR per (ROI, Test_Context, Param_Family).

Inputs:
  fooof_df: DataFrame with meta cols
            ['ROI', 'Data_Type', 'Block', 'Cohort', 'Subject_Idx', 'R2']
            plus per-band cols <Band>_CF / <Band>_PW / <Band>_BW
            plus aperiodic cols Aperiodic_OFFSET / Aperiodic_EXPONENT.
  comparisons_within:  list of (block_a, block_b) tuples
  comparisons_between: list of (cohort_a, cohort_b) tuples
  results_dir: where to drop the long-format CSV.
  narrow_analysis: if True, exclude Theta and Gamma (matches Aline's
                   default in the notebook for the manuscript tables).

Outputs:
  Long-format DataFrame with one row per (ROI, context, comparison,
  parameter) test. Also writes
  statistical_analysis_FOOOF_results_FDR_split_paper.csv into results_dir.

Author: Aline Brunner (verbatim port); Lena (no edits).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def run_fooof_statistics(fooof_df, comparisons_within, comparisons_between,
                          results_dir, narrow_analysis=True):
    print("\n--- Running Statistics on FOOOF Parameters ---")
    if narrow_analysis:
        print("    -> Narrow Analysis: Theta and Gamma excluded.")
    print("    -> Imputation: Missing Peak Power set to 0.")
    print("    -> Exclusion: Bandwidth (BW) ignored.")

    stats_results = []
    df_stats = fooof_df.copy()
    pw_cols = [c for c in df_stats.columns if c.endswith("_PW")]
    for col in pw_cols:
        df_stats[col] = df_stats[col].fillna(0)

    meta_cols = ["ROI", "Data_Type", "Block", "Cohort", "Subject_Idx", "R2"]
    all_params = [c for c in df_stats.columns if c not in meta_cols]
    param_cols = [c for c in all_params if "_BW" not in c]
    if narrow_analysis:
        param_cols = [c for c in param_cols
                      if "Theta" not in c and "Gamma" not in c]
    print(f"    -> Parameters included in stats: {param_cols}")

    rois = df_stats["ROI"].unique()
    data_types = df_stats["Data_Type"].unique()

    for roi in rois:
        for dtype in data_types:
            df_subset = df_stats[(df_stats["ROI"] == roi)
                                  & (df_stats["Data_Type"] == dtype)]
            if df_subset.empty:
                continue
            # --- WITHIN-COHORT (PAIRED) ---
            cohorts = df_subset["Cohort"].unique()
            for cohort in cohorts:
                df_cohort = df_subset[df_subset["Cohort"] == cohort]
                for (cond_a, cond_b) in comparisons_within:
                    if (cond_a not in df_cohort["Block"].values
                            or cond_b not in df_cohort["Block"].values):
                        continue
                    df_a = df_cohort[df_cohort["Block"] == cond_a].set_index("Subject_Idx")
                    df_b = df_cohort[df_cohort["Block"] == cond_b].set_index("Subject_Idx")
                    df_paired = df_a.join(df_b, lsuffix="_A", rsuffix="_B", how="inner")
                    if len(df_paired) < 3:
                        continue
                    for param in param_cols:
                        col_a, col_b = f"{param}_A", f"{param}_B"
                        valid = df_paired[[col_a, col_b]].dropna()
                        data_a, data_b = valid[col_a].values, valid[col_b].values
                        if len(data_a) < 3:
                            continue
                        diffs = data_a - data_b
                        if np.all(diffs == 0):
                            continue
                        try:
                            _, p_shapiro = shapiro(diffs)
                        except Exception:
                            p_shapiro = 0.0
                        try:
                            if p_shapiro > 0.05:
                                stat, p_val = ttest_rel(data_a, data_b)
                                test_name = "Paired T-Test"
                            else:
                                stat, p_val = wilcoxon(data_a, data_b)
                                test_name = "Wilcoxon"
                            stats_results.append({
                                "Test_Context": "Within-Cohort",
                                "ROI": roi, "Data_Type": dtype,
                                "Group_Level": cohort,
                                "Comparison": f"{cond_a} vs {cond_b}",
                                "Parameter": param,
                                "N": len(data_a),
                                "Normality_P": p_shapiro,
                                "Test_Used": test_name,
                                "Statistic": stat,
                                "P_Value": p_val,
                                "Mean_Diff": np.mean(data_a) - np.mean(data_b),
                                "Median_Diff": np.median(data_a) - np.median(data_b),
                            })
                        except Exception:
                            pass
            # --- BETWEEN-COHORT (INDEPENDENT) ---
            blocks = df_subset["Block"].unique()
            for block in blocks:
                df_block = df_subset[df_subset["Block"] == block]
                for (group_a, group_b) in comparisons_between:
                    vals_a = df_block[df_block["Cohort"] == group_a]
                    vals_b = df_block[df_block["Cohort"] == group_b]
                    if vals_a.empty or vals_b.empty:
                        continue
                    for param in param_cols:
                        data_a = vals_a[param].dropna().values
                        data_b = vals_b[param].dropna().values
                        if len(data_a) < 3 or len(data_b) < 3:
                            continue
                        try:
                            _, p_shap_a = shapiro(data_a)
                            _, p_shap_b = shapiro(data_b)
                            is_normal = (p_shap_a > 0.05) and (p_shap_b > 0.05)
                            min_shapiro = min(p_shap_a, p_shap_b)
                        except Exception:
                            is_normal, min_shapiro = False, 0.0
                        try:
                            if is_normal:
                                stat, p_val = ttest_ind(data_a, data_b, equal_var=False)
                                test_name = "Welch T-Test"
                            else:
                                stat, p_val = mannwhitneyu(data_a, data_b)
                                test_name = "Mann-Whitney"
                            stats_results.append({
                                "Test_Context": "Between-Cohort",
                                "ROI": roi, "Data_Type": dtype,
                                "Group_Level": block,
                                "Comparison": f"{group_a} vs {group_b}",
                                "Parameter": param,
                                "N": f"{len(data_a)}/{len(data_b)}",
                                "Normality_P": min_shapiro,
                                "Test_Used": test_name,
                                "Statistic": stat,
                                "P_Value": p_val,
                                "Mean_Diff": np.mean(data_a) - np.mean(data_b),
                                "Median_Diff": np.median(data_a) - np.median(data_b),
                            })
                        except Exception:
                            pass

    if not stats_results:
        print("No statistics calculated.")
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats_results)
    print("    -> Applying FDR Correction (Benjamini-Hochberg) split by Family...")
    stats_df["P_FDR"] = np.nan
    stats_df["Significance_corr"] = ""
    stats_df["Significance_raw"] = ""

    def get_family(name):
        return "Aperiodic" if name in ("Aperiodic_OFFSET", "Aperiodic_EXPONENT") else "Periodic"

    stats_df["Param_Family"] = stats_df["Parameter"].apply(get_family)
    groups = stats_df.groupby(["ROI", "Test_Context", "Param_Family"])
    for (roi, context, family), idx in groups.groups.items():
        p_vals = stats_df.loc[idx, "P_Value"].values
        # Significance_raw is per-row, NaN-safe.
        stats_df.loc[idx, "Significance_raw"] = [
            "" if pd.isna(p) else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            for p in p_vals
        ]
        # multipletests() propagates NaN across the whole family if any
        # p-value is NaN (e.g. Gamma_PW where no peak was detected). Run
        # the correction only on the valid subset and write the corrected
        # values back to those rows; NaN rows keep NaN P_FDR.
        valid_mask = ~pd.isna(p_vals)
        if valid_mask.any():
            _, pvals_corr_valid, _, _ = multipletests(
                p_vals[valid_mask], alpha=0.05, method="fdr_bh")
            full_corr = np.full(len(p_vals), np.nan)
            full_corr[valid_mask] = pvals_corr_valid
        else:
            full_corr = np.full(len(p_vals), np.nan)
        stats_df.loc[idx, "P_FDR"] = full_corr
        stats_df.loc[idx, "Significance_corr"] = [
            "" if pd.isna(p) else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            for p in full_corr
        ]
    stats_df = stats_df.sort_values(
        by=["Test_Context", "ROI", "Param_Family", "Parameter"])

    out_path = os.path.join(results_dir,
                             "statistical_analysis_FOOOF_results_FDR_split_paper.csv")
    stats_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"--- Statistics saved to: {out_path} ---")
    return stats_df
