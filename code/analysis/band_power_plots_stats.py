
# Necessary imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

from calculations import run_bandpower_analysis_relative

# ----------------------------------------------------------------------
# Absolute, Z-score normalized band power analysis for epochs data
# ----------------------------------------------------------------------

def prepare_plotting_data(band_powers_dico, groups, blocks, freq_bands, plot_baseline=True):
    """
    Restructures raw band power data into a nested dictionary suitable for plotting, with outlier filtering.
    """
    print("--- Preparing data for plotting and statistics ---")
    regions = ['EEG_motor', 'EEG_frontal', 'DBS']
    phases = (['baseline'] if plot_baseline else []) + blocks
    prepared_data = {band: {g: {p: {r: [] for r in regions} for p in phases} for g in groups} for band in freq_bands.keys()}

    for group in groups:
        for band_name in freq_bands.keys():
            motor_list = band_powers_dico['motor'][group][band_name]
            frontal_list = band_powers_dico['frontal'][group][band_name]
            for bp_motor, bp_front in zip(motor_list, frontal_list):
                motor_chans = bp_motor.get('channels_to_pick', [])
                frontal_chans = bp_front.get('channels_to_pick', [])

                if plot_baseline:
                    baseline_tuple = [
                        np.nanmean([bp_motor.get(f'EEG_baseline_{ch}', np.nan) for ch in motor_chans]),
                        np.nanmean([bp_front.get(f'EEG_baseline_{ch}', np.nan) for ch in frontal_chans]),
                        bp_motor.get('baseline_DBS', np.nan)
                    ]
                    # Filter out extreme outlier values before appending
                    if not any(x > 100 for x in baseline_tuple if np.isfinite(x)):
                        for cat, val in zip(regions, baseline_tuple):
                            prepared_data[band_name][group]['baseline'][cat].append(val)

                for block in blocks:
                    block_tuple = [
                        np.nanmean([bp_motor.get(f'EEG_{block}_{ch}', np.nan) for ch in motor_chans]),
                        np.nanmean([bp_front.get(f'EEG_{block}_{ch}', np.nan) for ch in frontal_chans]),
                        bp_motor.get(f'{block}_DBS', np.nan)
                    ]
                    # Filter out extreme outlier values before appending
                    if not any(x > 100 for x in block_tuple if np.isfinite(x)):
                        for cat, val in zip(regions, block_tuple):
                            prepared_data[band_name][group][block][cat].append(val)
    print("Data preparation complete.")
    return prepared_data

def stats_boxplot_data(
    data_to_plot: dict, groups: list, blocks: list, categories: list,
    fdr_method: str = "fdr_bh", alpha: float = 0.1, paired_within: bool = True, verbose: bool = False
):
    """Performs statistical comparisons (across-group and within-group) for boxplot data."""
    results = []
    # 1. Across-group comparisons
    for block in blocks:
        for cat in categories:
            vals_g0, vals_g1 = data_to_plot.get(groups[0], {}).get(block, {}).get(cat, []), data_to_plot.get(groups[1], {}).get(block, {}).get(cat, [])
            vals_g0, vals_g1 = np.array(vals_g0, dtype=float)[np.isfinite(vals_g0)], np.array(vals_g1, dtype=float)[np.isfinite(vals_g1)]
            if len(vals_g0) < 3 or len(vals_g1) < 3: continue
            norm0, norm1 = (shapiro(vals_g0)[1] > 0.05, shapiro(vals_g1)[1] > 0.05)
            test, stat, p = ("t-test (unpaired)", *ttest_ind(vals_g0, vals_g1, equal_var=False)) if norm0 and norm1 else ("Mann–Whitney", *mannwhitneyu(vals_g0, vals_g1))
            results.append({"test_type": "across-group", "category": cat, "block1": block, "group1": groups[0], "block2": block, "group2": groups[1], "test": test, "stat": stat, "p": p, "N1": len(vals_g0), "N2": len(vals_g1)})

    # 2. Within-group comparisons
    for group in groups:
        for cat in categories:
            for i, b1 in enumerate(blocks):
                for j, b2 in enumerate(blocks):
                    if j <= i: continue
                    vals_1, vals_2 = data_to_plot.get(group, {}).get(b1, {}).get(cat, []), data_to_plot.get(group, {}).get(b2, {}).get(cat, [])
                    vals_1, vals_2 = np.array(vals_1, dtype=float)[np.isfinite(vals_1)], np.array(vals_2, dtype=float)[np.isfinite(vals_2)]
                    if len(vals_1) < 3 or len(vals_2) < 3: continue
                    use_paired, norm1, norm2 = paired_within and len(vals_1) == len(vals_2), shapiro(vals_1)[1] > 0.05, shapiro(vals_2)[1] > 0.05
                    if use_paired:
                        if norm1 and norm2: test, stat, p = "t-test (paired)", *ttest_rel(vals_1, vals_2)
                        else:
                            try: test, stat, p = "Wilcoxon (paired)", *wilcoxon(vals_1, vals_2)
                            except ValueError: test, stat, p = "Wilcoxon (failed)", np.nan, np.nan
                    else:
                        test, stat, p = ("t-test (unpaired)", *ttest_ind(vals_1, vals_2, equal_var=False)) if norm1 and norm2 else ("Mann–Whitney", *mannwhitneyu(vals_1, vals_2))
                    results.append({"test_type": "within-group", "category": cat, "block1": b1, "group1": group, "block2": b2, "group2": group, "test": test, "stat": stat, "p": p, "N1": len(vals_1), "N2": len(vals_2)})
    df = pd.DataFrame(results)
    if not df.empty and 'p' in df.columns and df['p'].notna().any():
        reject, p_corr, _, _ = multipletests(df["p"].dropna().values, alpha=alpha, method=fdr_method)
        df.loc[df['p'].notna(), 'p_fdr'], df.loc[df['p'].notna(), 'significant'] = p_corr, reject
    else: df["p_fdr"], df["significant"] = pd.NA, pd.NA
    if verbose: print(df)
    return df

def summarize_significant_results(stats_df):
    """Extracts and separates significant within- and across-group results."""
    if 'significant' not in stats_df.columns or stats_df['significant'].isna().all(): return None, None
    sig = stats_df[stats_df["significant"] == True].copy()
    within, across = sig[sig["test_type"] == "within-group"], sig[sig["test_type"] == "across-group"]
    return (within if not within.empty else None, across if not across.empty else None)

def compute_and_save_stats(results_dir, prefix, data_per_freq_band, blocks_to_test, group_vars, categories):
    """Runs stats, saves full & significant results to CSV, and returns them."""
    print("--- Computing statistics and saving results ---")
    os.makedirs(results_dir, exist_ok=True)
    stats_df = pd.concat([
        stats_boxplot_data(data, group_vars, blocks_to_test, categories).assign(freq_band=band_name)
        for band_name, data in data_per_freq_band.items()
    ], ignore_index=True)
    stats_df.to_csv(os.path.join(results_dir, f"{prefix}_stats.csv"), index=False)
    
    within_sig, across_sig = summarize_significant_results(stats_df)
    if within_sig is not None:
        within_sig.to_csv(os.path.join(results_dir, f"{prefix}_within_group_significant.csv"), index=False)
        print(f"Found and saved {len(within_sig)} significant within-group comparisons.")
    if across_sig is not None:
        across_sig.to_csv(os.path.join(results_dir, f"{prefix}_across_group_significant.csv"), index=False)
        print(f"Found and saved {len(across_sig)} significant across-group comparisons.")
    if within_sig is None and across_sig is None: print("No significant results found.")
    return stats_df, within_sig, across_sig

# --- PLOTTING ---

def _plot_jittered_points(ax, offset, vals, box_width):
    if not len(vals): return
    jitter_width = box_width * 0.4
    x_jittered = np.random.normal(loc=offset, scale=jitter_width, size=len(vals))
    ax.scatter(x_jittered, vals, color='black', s=9, alpha=0.5, zorder=3, label='_nolegend_')

def p_to_stars(p_val):
    if p_val < 0.001: return '***'
    if p_val < 0.01: return '**'
    if p_val < 0.05: return '*'
    return 'ns'

def plot_significance_bracket(ax, x1, x2, y_max, p_val, y_levels, stack_key, font_settings):
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_height, spacing = 0.02 * y_range, 0.03 * y_range
    y_level = max(y_max, y_levels.get(stack_key, y_max)) + spacing
    ax.plot([x1, x1, x2, x2], [y_level, y_level + bracket_height, y_level + bracket_height, y_level], lw=1.2, c='red')
    ax.text((x1 + x2) / 2, y_level + bracket_height, p_to_stars(p_val), ha='center', va='bottom', fontsize=font_settings['ticks'], c='red')
    y_levels[stack_key] = y_level + bracket_height

def style_ax(ax, title, font_settings):
    ax.grid(axis='y', alpha=0.7, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylabel('Absolute Median Power$_{z}$', fontsize=font_settings['label'])
    ax.set_title(title, fontsize=20, pad=15, weight='bold')
    ax.tick_params(axis='y', labelsize=font_settings['ticks'])

def plot_power_by_cohort(ax, cohort_name, prepared_data, color_map, font_settings, within_sig_df, freq_bands, phases):
    """Creates a plot comparing conditions (phases) within a single cohort."""
    band_names = list(freq_bands.keys())
    plotted_regions = ['EEG_motor', 'EEG_frontal'] + (['DBS'] if cohort_name == 'Patient_NFB' else [])
    box_width, block_sep, region_group_width = 0.2, 0.25, (len(phases) * 0.25) + 0.3
    
    for i, band in enumerate(band_names):
        for j, region in enumerate(plotted_regions):
            for k, phase in enumerate(phases):
                pos = i * (len(plotted_regions) * region_group_width) + (j - (len(plotted_regions) - 1) / 2.0) * region_group_width + (k - (len(phases) - 1) / 2.0) * block_sep
                vals = np.array(prepared_data[band][cohort_name][phase][region]); vals = vals[~np.isnan(vals)]
                if not vals.any(): continue
                ax.boxplot(vals, positions=[pos], widths=box_width, patch_artist=True, boxprops={'facecolor': color_map.get((cohort_name, phase), 'gray'), 'edgecolor': 'black', 'lw': 1.5}, medianprops={'color': 'dimgray', 'lw': 2.5}, showfliers=False, zorder=2)
                _plot_jittered_points(ax, pos, vals, box_width)
    y_levels = {}
    if within_sig_df is not None:
        for _, row in within_sig_df[within_sig_df['group1'] == cohort_name].iterrows():
            if row['category'] not in plotted_regions: continue
            i, j, k1, k2 = band_names.index(row['freq_band']), plotted_regions.index(row['category']), phases.index(row['block1']), phases.index(row['block2'])
            band_center = i * (len(plotted_regions) * region_group_width)
            region_off = (j - (len(plotted_regions) - 1) / 2.0) * region_group_width
            x1, x2 = band_center + region_off + ((k1 - (len(phases) - 1) / 2.0) * block_sep), band_center + region_off + ((k2 - (len(phases) - 1) / 2.0) * block_sep)
            vals1, vals2 = prepared_data[row['freq_band']][cohort_name][row['block1']][row['category']], prepared_data[row['freq_band']][cohort_name][row['block2']][row['category']]
            if not np.any(vals1) and not np.any(vals2): continue
            plot_significance_bracket(ax, x1, x2, np.nanmax(np.concatenate([vals1, vals2])), row['p_fdr'], y_levels, (row['freq_band'], row['category']), font_settings)
    
    ax.set_xticks([i * (len(plotted_regions) * region_group_width) for i in range(len(band_names))])
    ax.set_xticklabels([f"{name.title()}\n({freq_bands[name][0]}-{freq_bands[name][1]} Hz)" for name in band_names], fontsize=font_settings['ticks'])
    y_pos = ax.get_ylim()[0] * 1.15
    for i in range(len(band_names)):
        for j, region in enumerate(plotted_regions):
            pos = i * (len(plotted_regions) * region_group_width) + (j - (len(plotted_regions) - 1) / 2.0) * region_group_width
            ax.text(pos, y_pos, region.replace('EEG_','').replace('Dbs','STN').title(), ha='center', va='top', fontsize=font_settings['ticks']-4, weight='semibold')
    
    display_name = 'HC' if 'AgeMatched' in cohort_name else 'PD' if 'Patient' in cohort_name else cohort_name
    style_ax(ax, f'Power by Frequency Band for {display_name}', font_settings)

def plot_power_by_block(ax, phase_name, prepared_data, color_map, font_settings, across_sig_df, freq_bands, groups):
    """Creates a plot comparing cohorts within a single condition/block."""
    band_names, regions_to_plot = list(freq_bands.keys()), ['EEG_motor', 'EEG_frontal']
    box_width, cohort_sep, region_group_width = 0.3, 0.35, (len(groups) * 0.35) + 0.4

    for i, band in enumerate(band_names):
        for j, region in enumerate(regions_to_plot):
            for k, cohort in enumerate(groups):
                pos = i * (len(regions_to_plot) * region_group_width) + (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width + (k - (len(groups) - 1) / 2.0) * cohort_sep
                vals = np.array(prepared_data[band][cohort][phase_name][region]); vals = vals[~np.isnan(vals)]
                if not vals.any(): continue
                ax.boxplot(vals, positions=[pos], widths=box_width, patch_artist=True, boxprops={'facecolor': color_map.get((cohort, phase_name), 'gray'), 'edgecolor': 'black', 'lw': 1.5}, medianprops={'color': 'dimgray', 'lw': 2.5}, showfliers=False, zorder=2)
                _plot_jittered_points(ax, pos, vals, box_width)
    
    y_levels = {}
    if across_sig_df is not None:
        for _, row in across_sig_df[across_sig_df['block1'] == phase_name].iterrows():
            if row['category'] not in regions_to_plot: continue
            i, j = band_names.index(row['freq_band']), regions_to_plot.index(row['category'])
            band_center = i * (len(regions_to_plot) * region_group_width)
            region_off = (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width
            x1, x2 = band_center + region_off - cohort_sep / 2, band_center + region_off + cohort_sep / 2
            vals1, vals2 = prepared_data[row['freq_band']][groups[0]][phase_name][row['category']], prepared_data[row['freq_band']][groups[1]][phase_name][row['category']]
            if not np.any(vals1) and not np.any(vals2): continue
            plot_significance_bracket(ax, x1, x2, np.nanmax(np.concatenate([vals1, vals2])), row['p_fdr'], y_levels, (row['freq_band'], row['category']), font_settings)
            
    ax.set_xticks([i * (len(regions_to_plot) * region_group_width) for i in range(len(band_names))])
    ax.set_xticklabels([f"{name.title()}\n({freq_bands[name][0]}-{freq_bands[name][1]} Hz)" for name in band_names], fontsize=font_settings['ticks'])
    y_pos = ax.get_ylim()[0] * 1.15
    for i, band in enumerate(band_names):
        for j, region in enumerate(regions_to_plot):
            pos = i * (len(regions_to_plot) * region_group_width) + (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width
            ax.text(pos, y_pos, region.replace('EEG_','').title(), ha='center', va='top', fontsize=font_settings['ticks']-4, weight='semibold')
    
    display_name = ('Rest' if 'baseline' in phase_name else '8walk' if 'pre8walk' in phase_name else phase_name)
    style_ax(ax, f"Power by Frequency Band during {display_name.title()}", font_settings)


def generate_all_plots(prepared_data, within_sig, across_sig, results_dir, groups, phases, freq_bands, color_map, condition_label_map, plot_title_prefix):
    """Generates and saves all cohort-centric and block-centric plots."""
    font_settings = {'label': 22, 'ticks': 18, 'legend': 20, 'legend_title': 20}
    
    print("\n--- Generating individual cohort-centric plots (within-group stats) ---")
    for cohort in groups:
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_power_by_cohort(ax, cohort, prepared_data, color_map, font_settings, within_sig, freq_bands, phases, plot_title_prefix)
        ax.autoscale(enable=True, axis='y', tight=False); ax.margins(y=0.15)
        legend_handles = [mpatches.Patch(color=color_map.get((cohort, p), 'gray'), label=condition_label_map.get(p, p)) for p in phases]
        fig.legend(handles=legend_handles, title="Condition", loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=font_settings['legend'], title_fontsize=font_settings['legend_title'])
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        save_path = os.path.join(results_dir, f"band_power_epochs_{cohort}_centric.pdf")
        plt.savefig(save_path, dpi=300); plt.show()
        print(f"Saved: {save_path}")

    print("\n--- Generating individual block-centric plots (across-group stats) ---")
    for phase in phases:
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_power_by_block(ax, phase, prepared_data, color_map, font_settings, across_sig, freq_bands, groups, plot_title_prefix)
        ax.autoscale(enable=True, axis='y', tight=False); ax.margins(y=0.15)
        # legend_handles = [mpatches.Patch(color=color_map[(g, phase)], label='HC' if 'AgeMatched' in g else 'PD') for g in groups]
        legend_handles = [mpatches.Patch(color=color_map.get((g, phase), 'gray'), label='HC' if 'AgeMatched' in g else 'PD') for g in groups]
        fig.legend(handles=legend_handles, title="Group", loc='upper right', bbox_to_anchor=(0.88, 0.98), fontsize=font_settings['legend'], title_fontsize=font_settings['legend_title'])
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        save_path = os.path.join(results_dir, f"band_power_epochs_{condition_label_map.get(phase, phase)}_centric.pdf")
        plt.savefig(save_path, dpi=300); plt.show()
        print(f"Saved: {save_path}")



def stats_boxplot_data(
    data_to_plot: dict, groups: list, blocks: list, categories: list,
    fdr_method: str = "fdr_bh", alpha: float = 0.1, paired_within: bool = True, verbose: bool = False
):
    """Performs statistical comparisons (across-group and within-group) for boxplot data."""
    results = []
    # 1. Across-group comparisons (unpaired)
    for block in blocks:
        for cat in categories:
            vals_g0 = data_to_plot.get(groups[0], {}).get(block, {}).get(cat, [])
            vals_g1 = data_to_plot.get(groups[1], {}).get(block, {}).get(cat, [])
            vals_g0 = np.array(vals_g0, dtype=float)[np.isfinite(vals_g0)]
            vals_g1 = np.array(vals_g1, dtype=float)[np.isfinite(vals_g1)]
            if len(vals_g0) < 3 or len(vals_g1) < 3: continue
            norm0 = shapiro(vals_g0)[1] > 0.05
            norm1 = shapiro(vals_g1)[1] > 0.05
            test, stat, p = ("t-test (unpaired)", *ttest_ind(vals_g0, vals_g1, equal_var=False, nan_policy='omit')) if norm0 and norm1 else ("Mann–Whitney", *mannwhitneyu(vals_g0, vals_g1, alternative='two-sided'))
            results.append({"test_type": "across-group", "category": cat, "block1": block, "group1": groups[0], "block2": block, "group2": groups[1], "test": test, "stat": stat, "p": p, "N1": len(vals_g0), "N2": len(vals_g1)})

    # 2. Within-group, between-block comparisons
    for group in groups:
        for cat in categories:
            for i, b1 in enumerate(blocks):
                for j, b2 in enumerate(blocks):
                    if j <= i: continue
                    vals_1 = np.array(data_to_plot.get(group, {}).get(b1, {}).get(cat, []), dtype=float)
                    vals_2 = np.array(data_to_plot.get(group, {}).get(b2, {}).get(cat, []), dtype=float)
                    vals_1, vals_2 = vals_1[np.isfinite(vals_1)], vals_2[np.isfinite(vals_2)]
                    if len(vals_1) < 3 or len(vals_2) < 3: continue
                    
                    use_paired = paired_within and len(vals_1) == len(vals_2)
                    norm1, norm2 = shapiro(vals_1)[1] > 0.05, shapiro(vals_2)[1] > 0.05

                    if use_paired:
                        if norm1 and norm2: test, stat, p = "t-test (paired)", *ttest_rel(vals_1, vals_2, nan_policy='omit')
                        else:
                            try: test, stat, p = "Wilcoxon (paired)", *wilcoxon(vals_1, vals_2)
                            except ValueError: test, stat, p = "Wilcoxon (paired, failed)", np.nan, np.nan
                    else:
                        if norm1 and norm2: test, stat, p = "t-test (unpaired)", *ttest_ind(vals_1, vals_2, equal_var=False, nan_policy='omit')
                        else: test, stat, p = "Mann–Whitney", *mannwhitneyu(vals_1, vals_2, alternative='two-sided')
                    results.append({"test_type": "within-group", "category": cat, "block1": b1, "group1": group, "block2": b2, "group2": group, "test": test, "stat": stat, "p": p, "N1": len(vals_1), "N2": len(vals_2)})

    df = pd.DataFrame(results)
    if not df.empty and 'p' in df.columns and df['p'].notna().any():
        reject, p_corr, _, _ = multipletests(df["p"].dropna().values, alpha=alpha, method=fdr_method)
        df.loc[df['p'].notna(), 'p_fdr'] = p_corr
        df.loc[df['p'].notna(), 'significant'] = reject
    else:
        df["p_fdr"], df["significant"] = pd.NA, pd.NA
    if verbose: print(df)
    return df

def summarize_significant_results(stats_df):
    """Extracts and separates significant within- and across-group results."""
    if 'significant' not in stats_df.columns or stats_df['significant'].isna().all():
        return None, None
    sig = stats_df[stats_df["significant"] == True].copy()
    within = sig[sig["test_type"] == "within-group"]
    across = sig[sig["test_type"] == "across-group"]
    return (within if not within.empty else None, across if not across.empty else None)

# --- PLOTTING HELPER FUNCTIONS ---

def _plot_jittered_points(ax, offset, vals, box_width):
    """Adds jittered data points to a boxplot."""
    if not len(vals): return
    jitter_width = box_width * 0.4
    x_jittered = np.random.normal(loc=offset, scale=jitter_width, size=len(vals))
    ax.scatter(x_jittered, vals, color='black', s=9, alpha=0.5, zorder=3, label='_nolegend_')

def p_to_stars(p_val):
    """Converts a p-value to a star string for significance plotting."""
    if p_val < 0.001: return '***'
    if p_val < 0.01: return '**'
    if p_val < 0.05: return '*'
    return 'ns'

def plot_significance_bracket(ax, x1, x2, y_max, p_val, y_levels, stack_key, font_settings):
    """Plots a significance bracket with stars between two points on an axis."""
    if not np.isfinite(y_max): y_max = y_levels.get(stack_key, ax.get_ylim()[1] * 0.8)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_height, spacing = 0.02 * y_range, 0.03 * y_range
    y_level = max(y_max, y_levels.get(stack_key, y_max)) + spacing
    ax.plot([x1, x1, x2, x2], [y_level, y_level + bracket_height, y_level + bracket_height, y_level], lw=1.2, c='red')
    ax.text((x1 + x2) / 2, y_level + bracket_height, p_to_stars(p_val), ha='center', va='bottom', fontsize=font_settings['ticks'], c='red')
    y_levels[stack_key] = y_level + bracket_height

def style_ax(ax, title, font_settings):
    """Applies common styling to a plot axis."""
    ax.grid(axis='y', alpha=0.7, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylabel('Relative Power', fontsize=font_settings['label'])
    ax.set_title(title, fontsize=20, pad=15, weight='bold')
    ax.tick_params(axis='y', labelsize=font_settings['ticks'])

# --- CORE PLOTTING FUNCTIONS ---

def plot_power_by_cohort(ax, cohort_name, prepared_data, color_map, font_settings, within_sig_df, freq_bands, phases, plot_title_prefix):
    """Creates a plot comparing conditions (phases) within a single cohort."""
    band_names = list(freq_bands.keys())
    plotted_regions = ['EEG_motor', 'EEG_frontal'] + (['DBS'] if cohort_name == 'Patient_NFB' else [])
    box_width, block_sep, region_group_width = 0.2, 0.25, (len(phases) * 0.25) + 0.3
    
    for i, band_name in enumerate(band_names):
        for j, region in enumerate(plotted_regions):
            for k, phase in enumerate(phases):
                band_center_pos = i * (len(plotted_regions) * region_group_width)
                region_offset = (j - (len(plotted_regions) - 1) / 2.0) * region_group_width
                block_offset = (k - (len(phases) - 1) / 2.0) * block_sep
                pos = band_center_pos + region_offset + block_offset
                vals = np.array(prepared_data[band_name][cohort_name][phase][region])
                if vals.size == 0: continue
                ax.boxplot(vals, positions=[pos], widths=box_width, patch_artist=True, boxprops={'facecolor': color_map.get((cohort_name, phase), 'gray'), 'edgecolor': 'black', 'lw': 1.5}, medianprops={'color': 'dimgray', 'lw': 2.5}, showfliers=False, zorder=2)
                _plot_jittered_points(ax, pos, vals, box_width)

    y_levels = {}
    if within_sig_df is not None:
        for _, row in within_sig_df[within_sig_df['group1'] == cohort_name].iterrows():
            if row['category'] not in plotted_regions: continue
            i, j, k1, k2 = band_names.index(row['freq_band']), plotted_regions.index(row['category']), phases.index(row['block1']), phases.index(row['block2'])
            band_center = i * (len(plotted_regions) * region_group_width)
            region_off = (j - (len(plotted_regions) - 1) / 2.0) * region_group_width
            x1 = band_center + region_off + ((k1 - (len(phases) - 1) / 2.0) * block_sep)
            x2 = band_center + region_off + ((k2 - (len(phases) - 1) / 2.0) * block_sep)
            vals1, vals2 = prepared_data[row['freq_band']][cohort_name][row['block1']][row['category']], prepared_data[row['freq_band']][cohort_name][row['block2']][row['category']]
            if not vals1 or not vals2: continue
            plot_significance_bracket(ax, x1, x2, np.nanmax(np.concatenate([vals1, vals2])), row['p_fdr'], y_levels, (row['freq_band'], row['category']), font_settings)

    ax.set_xticks([i * (len(plotted_regions) * region_group_width) for i in range(len(band_names))])
    ax.set_xticklabels([f"{name.title()}\n({freq_bands[name][0]}-{freq_bands[name][1]} Hz)" for name in band_names], fontsize=font_settings['ticks'])
    y_pos = ax.get_ylim()[0] * 1.15
    for i in range(len(band_names)):
        for j, region in enumerate(plotted_regions):
            pos = i * (len(plotted_regions) * region_group_width) + (j - (len(plotted_regions) - 1) / 2.0) * region_group_width
            ax.text(pos, y_pos, region.replace('EEG_','').replace('Dbs','STN').title(), ha='center', va='top', fontsize=font_settings['ticks']-4, weight='semibold')
    
    display_cohort_name = 'HC' if 'AgeMatched' in cohort_name else 'PD' if cohort_name == 'Patient_NFB' else cohort_name
    style_ax(ax, f'{plot_title_prefix} by Frequency Band for {display_cohort_name}', font_settings)

def plot_power_by_block(ax, phase_name, prepared_data, color_map, font_settings, across_sig_df, freq_bands, groups, plot_title_prefix):
    """Creates a plot comparing cohorts within a single condition/block."""
    band_names, regions_to_plot, cohorts = list(freq_bands.keys()), ['EEG_motor', 'EEG_frontal'], groups
    box_width, cohort_sep, region_group_width = 0.3, 0.35, (len(cohorts) * 0.35) + 0.4

    for i, band_name in enumerate(band_names):
        for j, region in enumerate(regions_to_plot):
            for k, cohort in enumerate(cohorts):
                band_center_pos = i * (len(regions_to_plot) * region_group_width)
                region_offset = (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width
                cohort_offset = (k - (len(cohorts) - 1) / 2.0) * cohort_sep
                pos = band_center_pos + region_offset + cohort_offset
                vals = np.array(prepared_data[band_name][cohort][phase_name][region])
                if vals.size == 0: continue
                ax.boxplot(vals, positions=[pos], widths=box_width, patch_artist=True, boxprops={'facecolor': color_map.get((cohort, phase_name), 'gray'), 'edgecolor': 'black', 'lw': 1.5}, medianprops={'color': 'dimgray', 'lw': 2.5}, showfliers=False, zorder=2)
                _plot_jittered_points(ax, pos, vals, box_width)
    
    y_levels = {}
    if across_sig_df is not None:
        for _, row in across_sig_df[across_sig_df['block1'] == phase_name].iterrows():
            if row['category'] not in regions_to_plot: continue
            i, j = band_names.index(row['freq_band']), regions_to_plot.index(row['category'])
            band_center = i * (len(regions_to_plot) * region_group_width)
            region_off = (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width
            x1 = band_center + region_off + ((0 - (len(cohorts) - 1) / 2.0) * cohort_sep)
            x2 = band_center + region_off + ((1 - (len(cohorts) - 1) / 2.0) * cohort_sep)
            vals1, vals2 = prepared_data[row['freq_band']][cohorts[0]][phase_name][row['category']], prepared_data[row['freq_band']][cohorts[1]][phase_name][row['category']]
            if not vals1 or not vals2: continue
            plot_significance_bracket(ax, x1, x2, np.nanmax(np.concatenate([vals1, vals2])), row['p_fdr'], y_levels, (row['freq_band'], row['category']), font_settings)

    ax.set_xticks([i * (len(regions_to_plot) * region_group_width) for i in range(len(band_names))])
    ax.set_xticklabels([f"{name.title()}\n({freq_bands[name][0]}-{freq_bands[name][1]} Hz)" for name in band_names], fontsize=font_settings['ticks'])
    y_pos = ax.get_ylim()[0] * 1.15
    for i in range(len(band_names)):
        for j, region in enumerate(regions_to_plot):
            pos = i * (len(regions_to_plot) * region_group_width) + (j - (len(regions_to_plot) - 1) / 2.0) * region_group_width
            ax.text(pos, y_pos, region.replace('EEG_','').title(), ha='center', va='top', fontsize=font_settings['ticks']-4, weight='semibold')

    display_block_name = ('Rest' if 'baseline' in phase_name else '8walk' if phase_name == 'pre8walk' else phase_name)
    style_ax(ax, f"{plot_title_prefix} by Frequency Band during {display_block_name.title()}", font_settings)

# ------------------------------------------------------------
# Relative Band Power Analysis and Plotting
# ------------------------------------------------------------

# --- HIGH-LEVEL WORKFLOW FUNCTIONS ---

def calculate_all_relative_powers(group_vars, subjects_dico, all_data_epoched_dico, blocks, freq_bands, regions_of_interest):
    """Calculates relative band power for all specified groups, bands, and regions."""
    print("\n--- Computing Relative Band Powers ---")
    relative_band_powers_dico = {}
    for name, region_channels in regions_of_interest.items():
        relative_band_powers_dico[name] = {}
        for group_var in group_vars:
            relative_band_powers_dico[name][group_var] = {}
            for freq_band, band_range in freq_bands.items():
                print(f"Calculating: Region={name}, Group={group_var}, Band={freq_band}")
                relative_band_powers_dico[name][group_var][freq_band] = run_bandpower_analysis_relative(
                    subjects=subjects_dico[group_var], all_data=all_data_epoched_dico[group_var], blocks=blocks,
                    band_range=band_range, region=region_channels, norm_freq_range=(1.0, 45.0)
                )
    print("Relative power calculations complete.")
    return relative_band_powers_dico

def prepare_data_for_plotting(relative_band_powers_dico, groups, blocks, freq_bands, plot_baseline=True):
    """Restructures the calculated band power data into a format suitable for plotting functions."""
    print("\n--- Preparing data for plotting and stats ---")
    regions, phases = ['EEG_motor', 'EEG_frontal', 'DBS'], (['baseline'] if plot_baseline else []) + blocks
    prepared_data = {band: {g: {p: {r: [] for r in regions} for p in phases} for g in groups} for band in freq_bands.keys()}
    
    for group in groups:
        for band_name in freq_bands.keys():
            motor_list, frontal_list = relative_band_powers_dico['motor'][group][band_name], relative_band_powers_dico['frontal'][group][band_name]
            for bp_motor, bp_front in zip(motor_list, frontal_list):
                motor_chans, frontal_chans = bp_motor.get('channels_to_pick', []), bp_front.get('channels_to_pick', [])
                if plot_baseline:
                    vals = [np.nanmean([bp_motor.get(f'EEG_baseline_{ch}', np.nan) for ch in motor_chans]),
                            np.nanmean([bp_front.get(f'EEG_baseline_{ch}', np.nan) for ch in frontal_chans]),
                            bp_motor.get('baseline_DBS', np.nan)]
                    for cat, val in zip(regions, vals):
                        if not np.isnan(val): prepared_data[band_name][group]['baseline'][cat].append(val)
                for block in blocks:
                    vals = [np.nanmean([bp_motor.get(f'EEG_{block}_{ch}', np.nan) for ch in motor_chans]),
                            np.nanmean([bp_front.get(f'EEG_{block}_{ch}', np.nan) for ch in frontal_chans]),
                            bp_motor.get(f'{block}_DBS', np.nan)]
                    for cat, val in zip(regions, vals):
                        if not np.isnan(val): prepared_data[band_name][group][block][cat].append(val)
    print("Data preparation complete.")
    return prepared_data

def compute_and_save_bandpower_stats(results_dir, prefix, data_per_freq_band, blocks_to_test, group_vars, categories):
    """Runs statistical analysis and saves the full and significant results to CSV files."""
    print("\n--- Running statistics and saving results ---")
    os.makedirs(results_dir, exist_ok=True)
    stats_df = pd.concat([
        stats_boxplot_data(data, group_vars, blocks_to_test, categories).assign(freq_band=band_name)
        for band_name, data in data_per_freq_band.items()
    ], ignore_index=True)
    stats_df.to_csv(os.path.join(results_dir, f"{prefix}_stats.csv"), index=False)
    
    within_sig, across_sig = summarize_significant_results(stats_df)
    if within_sig is not None:
        within_sig.to_csv(os.path.join(results_dir, f"{prefix}_within_group_significant.csv"), index=False)
        print(f"Found and saved {len(within_sig)} significant within-group comparisons.")
    if across_sig is not None:
        across_sig.to_csv(os.path.join(results_dir, f"{prefix}_across_group_significant.csv"), index=False)
        print(f"Found and saved {len(across_sig)} significant across-group comparisons.")
    if within_sig is None and across_sig is None:
        print("No significant results found.")
    return stats_df, within_sig, across_sig

def generate_plots(prepared_data, within_sig, across_sig, results_dir, plot_prefix, plot_title_prefix, groups, phases, freq_bands, color_map, condition_label_map):
    """Generates and saves all cohort-centric and block-centric plots."""
    font_settings = {'label': 22, 'ticks': 18, 'legend': 20, 'legend_title': 20}
    
    print("\n--- Generating cohort-centric plots (within-group stats) ---")
    for cohort in groups:
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_power_by_cohort(ax, cohort, prepared_data, color_map, font_settings, within_sig, freq_bands, phases, plot_title_prefix)
        ax.autoscale(enable=True, axis='y', tight=False); ax.margins(y=0.15)
        # legend_handles = [mpatches.Patch(color=color_map.get((cohort, p), 'gray'), label=condition_label_map.get(p, p)) for p in phases]
        legend_handles = [mpatches.Patch(color=color_map.get((cohort, p), 'gray'), label=condition_label_map.get(p, p)) for p in phases]
        fig.legend(handles=legend_handles, title="Condition", loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=font_settings['legend'], title_fontsize=font_settings['legend_title'])
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        save_path = os.path.join(results_dir, f"{plot_prefix}_{cohort}_centric.pdf")
        plt.savefig(save_path, dpi=300); plt.show()
        print(f"Saved: {save_path}")

    print("\n--- Generating block-centric plots (across-group stats) ---")
    for phase in phases:
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_power_by_block(ax, phase, prepared_data, color_map, font_settings, across_sig, freq_bands, groups, plot_title_prefix)
        ax.autoscale(enable=True, axis='y', tight=False); ax.margins(y=0.15)
        # legend_handles = [mpatches.Patch(color=color_map[(g, phase)], label='HC' if 'AgeMatched' in g else 'PD') for g in groups]
        legend_handles = [mpatches.Patch(color=color_map.get((g, phase), 'gray'), label='HC' if 'AgeMatched' in g else 'PD') for g in groups]
        fig.legend(handles=legend_handles, title="Group", loc='upper right', bbox_to_anchor=(0.88, 0.98), fontsize=font_settings['legend'], title_fontsize=font_settings['legend_title'])
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        save_path = os.path.join(results_dir, f"{plot_prefix}_{condition_label_map.get(phase, phase)}_centric.pdf")
        plt.savefig(save_path, dpi=300); plt.show()
        print(f"Saved: {save_path}")