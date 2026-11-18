# Make sure these are at the top of your script
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import os
import matplotlib.pyplot as plt 

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.fft import fft

from tfr import _circ_shift_percent_time, _resample_to_len_2d

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_ersd_analysis(
    tfr_group_results: Dict[str, Dict[str, Any]],
    freqs: np.ndarray,
    freq_bands: Dict[str, Tuple[float, float]],
    blocks: List[str],
    event_percentages: Dict[str, float],
    gait_phases: Dict[str, Tuple[float, float]],
    results_dir: str,
    comparisons_to_run: List[Dict[str, Any]],
    line_plots_to_generate: List[Dict[str, Any]],
    palette: Optional[Dict[str, str]] = None,
    normalization: str = "" 
):
    """
    Orchestrates the creation of ERS/ERD plots and stats with decoupled definitions
    for line plots and bar plot comparisons.

    Args:
        ... (standard arguments) ...
        results_dir: Directory to save the output plots.
        comparisons_to_run: Drives the bar plots and statistical tests. Each dict
                            defines one comparison (typically between 2 cohorts).
        line_plots_to_generate: Drives the line plots. Each dict defines one figure,
                                its name, and which cohorts to display on it.
        palette: An optional dictionary mapping cohort names to colors.
    """
    print("--- Starting Flexible ERS/ERD Analysis (Decoupled Version) ---")
    os.makedirs(results_dir, exist_ok=True)

    # --- Step 1: Define all unique cohorts from BOTH configuration lists ---
    all_cohort_definitions = {}
    all_config_items = comparisons_to_run + line_plots_to_generate

    for item in all_config_items:
        # Cohorts can be nested under a 'cohorts' key (dict) or be a direct list
        cohort_source = item.get("cohorts", {})
        if isinstance(cohort_source, dict):
            # This handles the structure from comparisons_to_run
            for cohort_name, cohort_info in cohort_source.items():
                if cohort_name not in all_cohort_definitions:
                    all_cohort_definitions[cohort_name] = cohort_info
        elif isinstance(cohort_source, list):
             # This handles the simplified structure from line_plots_to_generate
             # It assumes the full definition is in comparisons_to_run
             pass # No new definitions to add here

    # --- Step 2: Create a master definition lookup from comparisons_to_run ---
    # This ensures we have the 'group' and 'ids' for every cohort mentioned.
    master_cohort_definitions = {}
    for comp in comparisons_to_run:
        master_cohort_definitions.update(comp["cohorts"])

    # --- Step 3: Set up color palette ---
    all_cohort_names = list(master_cohort_definitions.keys())
    if palette is None:
        colors = sns.color_palette("husl", len(all_cohort_names))
        palette = {name: colors[i] for i, name in enumerate(all_cohort_names)}

    # Add color info to each cohort definition for easy access
    for name, p_color in palette.items():
        if name in master_cohort_definitions:
            master_cohort_definitions[name]['color'] = p_color

    # --- Step 4: Generate the user-defined line plots ---
    print("\n--- Generating Custom ERS/ERD Line Plots ---")
    for plot_def in line_plots_to_generate:
        plot_name = plot_def["name"]
        cohorts_for_this_plot = plot_def["cohorts"]
        print(f"  - Generating line plot: '{plot_name}'")
        plot_erds_unified(
            tfr_group_results, freqs, freq_bands, blocks,
            event_percentages, results_dir,
            master_cohort_definitions,
            cohorts_for_this_plot,
            filename_suffix=plot_name,
            normalization = normalization
        )

    # --- Step 5: Generate Bar Plots and Statistics for each defined comparison ---
    print("\n--- Generating Metric Bar Plots and Statistics for each Comparison ---")
    
    # Extract metrics for ALL defined cohorts at once. This is efficient.
    all_metrics_df = {}
    for block in blocks:
        all_metrics_df[block] = {}
        for band_name, band_lim in freq_bands.items():
            all_metrics_df[block][band_name] = extract_metrics_for_all_cohorts(
                tfr_group_results, freqs, band_lim, block,
                ['EEG_motor', 'EEG_frontal', 'STN'], ['Left', 'Both', 'Right'],
                gait_phases, master_cohort_definitions
            )

    # Now, loop through each comparison to create its specific plot and stats
    for comparison in comparisons_to_run:
        comp_name = comparison["name"]
        cohorts_to_compare = list(comparison["cohorts"].keys())
        
        if len(cohorts_to_compare) != 2:
            print(f"WARNING: Skipping stats for '{comp_name}'. Bar plot statistics currently support exactly 2 groups.")
            continue
            
        print(f"\n--- Processing Comparison: {comp_name} ({' vs '.join(cohorts_to_compare)}) ---")
        for block in blocks:
            if block == 'baseline': continue
            for band_name in freq_bands.keys():
                for area in ['EEG_motor', 'EEG_frontal', 'STN']:
                    df_full = all_metrics_df[block][band_name]
                    if df_full.empty: continue
                    
                    df_comparison = df_full[df_full['group'].isin(cohorts_to_compare)].copy()
                    if df_comparison.empty: continue

                    generate_comparison_bar_plot(
                        df_comparison, block, band_name, area,
                        cohorts_to_compare, comp_name, results_dir, palette, normalization
                    )

def plot_erds_unified(
    tfr_group_results, freqs, freq_bands, blocks, event_percentages, results_dir,
    all_cohort_definitions, cohorts_to_plot, filename_suffix, subareas=('Left', 'Both', 'Right'), normalization = ""
):
    """
    MODIFIED to use the specific, hardcoded x-axis ticks and labels from the TFR plots.
    """
    # --- Style settings ---
    font_settings = {'title': 28, 'label': 26, 'ticks': 22, 'legend': 30, 'gait_ticks': 22} # Adjusted gait_ticks to match TFR
    block_name_map = {"pre8walk": "8walk", "post8walk": "Post-8walk"}
    subarea_styles = {'Left': ':', 'Both': '-', 'Right': '--'}
    area_labels = {'EEG_motor': 'Motor', 'EEG_frontal': 'Frontal', 'STN': 'STN'}
    
    areas_to_plot = ['EEG_motor', 'EEG_frontal', 'STN']
    n_bands, n_areas = len(freq_bands), len(areas_to_plot)

    for block in blocks:
        if block == 'baseline': continue
        fig, axes = plt.subplots(n_bands, n_areas, figsize=(28, 7 * n_bands), sharex=True, constrained_layout=False)
        if n_bands == 1: axes = np.array([axes])
        if n_areas == 1: axes = axes.reshape(-1, 1)

        for row_idx, (band, band_lim) in enumerate(freq_bands.items()):
            for col_idx, area_type in enumerate(areas_to_plot):
                ax = axes[row_idx, col_idx]
                max_abs_y = 0
                
                # --- This inner loop for plotting data remains the same ---
                for cohort_name in cohorts_to_plot:
                    # ... (data plotting logic is unchanged) ...
                    if cohort_name not in all_cohort_definitions:
                        print(f"Warning: Cohort '{cohort_name}' for line plot not found. Skipping.")
                        continue
                    
                    cohort_info = all_cohort_definitions[cohort_name]
                    base_group, id_filter = cohort_info['group'], set(cohort_info['ids']) if cohort_info['ids'] else None
                    
                    if not (base_group in tfr_group_results and block in tfr_group_results[base_group] and area_type in tfr_group_results[base_group][block]):
                        continue
                    
                    for subarea in subareas:
                        if subarea not in tfr_group_results[base_group][block][area_type]: continue
                        
                        all_arrays = tfr_group_results[base_group][block][area_type][subarea]
                        all_ids = tfr_group_results[base_group][block]['ID']
                        
                        indices = [i for i, sid in enumerate(all_ids) if not id_filter or sid in id_filter]
                        subj_arrays = [all_arrays[i] for i in indices if i < len(all_arrays) and all_arrays[i] is not None]

                        if not subj_arrays: continue
                        
                        arr_stack = np.stack(subj_arrays)
                        f_idx = get_freq_indices(freqs, band_lim)
                        data = arr_stack[:, f_idx, :].mean(axis=1)
                        mean, sem = np.nanmean(data, axis=0), np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
                        x = np.linspace(0, 100, data.shape[1])
                        
                        ax.plot(x, mean, color=cohort_info['color'], linestyle=subarea_styles.get(subarea, '-'), lw=2.5, label=f"{cohort_name} {subarea}")
                        ax.fill_between(x, mean - sem, mean + sem, color=cohort_info['color'], alpha=0.15)
                        max_abs_y = max(max_abs_y, np.max(np.abs(mean) + np.abs(sem)))


                # =========================================================================
                # X-AXIS LOGIC COPIED FROM _plot_single_tfr
                # =========================================================================
                
                # --- 1. Define the hardcoded ticks and labels ---
                # The x-axis is 0-100, so the percentages are the coordinates.
                ticks = [0, 10, 50, 60, 100] #TODO: ALINE CHANGE THIS WITH EVENTS percentages .values!! + below
                tick_lbls = ['0%: RHS', '10%: LTO', '50%: LHS', '60%: RTO', '100%: RHS']  # manuscript event names
                event_times = [10, 50, 60] # Times for the vertical lines

                # --- 2. Apply formatting to the axis ---
                ax.set_xticks(ticks)
                ax.xaxis.set_minor_locator(plt.NullLocator()) # Ensure no minor ticks

                if row_idx == n_bands - 1:
                    ax.set_xticklabels(tick_lbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=font_settings['gait_ticks'])
                    ax.set_xlabel('Gait Cycle (%)', fontsize=font_settings['label'])
                else:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # --- 3. Draw vertical lines for the main events ---
                for et in event_times: 
                    ax.axvline(et, color='black', linestyle='-', linewidth=1.5)

                # --- General plot styling ---
                ax.axhline(0, color='black', linestyle='-', lw=0.7)
                ax.set_xlim(0, 100)
                if max_abs_y > 0: ax.set_ylim(-max_abs_y * 1.1, max_abs_y * 1.1)
                
                if row_idx == 0:
                    ax.set_title(area_labels[area_type], fontsize=font_settings['title'])
                if col_idx == 0:
                    ax.text(-0.35, 0.5, f"{band.title()}\n({band_lim[0]}-{band_lim[1]} Hz)", transform=ax.transAxes, ha='center', va='center', rotation='vertical', fontsize=font_settings['title'], weight='bold')
                    ax.set_ylabel('ERS/ERD (dB)', fontsize=font_settings['label'], weight='bold')
                
                ax.grid(True, linestyle='--', linewidth=0.5, axis='y')
                ax.tick_params(axis='y', which='major', labelsize=font_settings['ticks'])
                # =========================================================================

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=font_settings['legend'], loc='upper center', 
                   bbox_to_anchor=(0.56, 0.09), ncol=len(cohorts_to_plot) * len(subareas), frameon=False)
        
        fig.tight_layout(pad=1.5, rect=[0.05, 0.1, 0.98, 0.95])
        
        block_short = block_name_map.get(block, block)
        fname = f"ERSDs_lineplot_{block_short}_{filename_suffix}_{normalization}.pdf"
        fig.savefig(os.path.join(results_dir, fname), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================



def extract_metrics_for_all_cohorts(
    tfr_group_results, freqs, band, block, area_types, subareas, gait_phases,
    all_cohort_definitions
):
    """
    MODIFIED to extract metrics for all defined cohorts at once, returning a single DataFrame.
    """
    data = []
    f_idx = get_freq_indices(freqs, band)
    
    for cohort_name, cohort_info in all_cohort_definitions.items():
        base_group = cohort_info['group']
        id_filter = set(cohort_info['ids']) if cohort_info['ids'] else None
        
        if base_group not in tfr_group_results or block not in tfr_group_results[base_group]: continue
        
        for area_type in area_types:
            if area_type not in tfr_group_results[base_group][block]: continue
            
            for subarea in subareas:
                if subarea not in tfr_group_results[base_group][block][area_type]: continue
                
                all_arrays = tfr_group_results[base_group][block][area_type][subarea]
                all_ids = tfr_group_results[base_group][block]['ID']
                
                for subj_idx, arr in enumerate(all_arrays):
                    subj_id = all_ids[subj_idx]
                    if (id_filter and subj_id not in id_filter) or arr is None:
                        continue
                    
                    arr_band = arr[f_idx, :].mean(axis=0)
                    data.append({
                        'subject': subj_id,
                        'group': cohort_name,
                        'area_type': area_type,
                        'area': subarea,
                        'mean_ers': np.nanmedian(arr_band),
                        'amp_fluct': calculate_intra_stride_fluctuation(arr_band, gait_phases),
                        'gpm': np.nanmean(calculate_gpm(arr[f_idx, :]))
                    })
    return pd.DataFrame(data)

def generate_comparison_bar_plot(
    df_comparison, block, band_name, area, cohorts_to_compare, comp_name, results_dir, palette, normalizaion
):
    """
    NEW helper to create a single bar plot figure for a specific comparison.
    """
    area_labels = {'EEG_motor': 'Motor', 'EEG_frontal': 'Frontal', 'STN': 'STN'}
    subareas = ['Left', 'Both', 'Right']
    
    df_plot = df_comparison[df_comparison['area_type'] == area]
    if df_plot.empty: return

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    title = f"Gait Metrics: {comp_name} ({band_name}, {block}, {area_labels[area]})"
    #fig.suptitle(title, fontsize=18)
    
    all_stats_for_plot = []
    for i, metric in enumerate(['mean_ers', 'amp_fluct', 'gpm']):
        # Use a subset of the main palette for this plot
        plot_palette = {name: color for name, color in palette.items() if name in cohorts_to_compare}
        
        plot_figure3_bars(df_plot, metric, subareas, plot_palette, ax=axs[i])
        
        stats_df = stat_bar_comparisons_and_annotate(
            df_plot,
            metric,
            axs[i],
            area_order=subareas,
            group_vars=cohorts_to_compare,
        )
        if not stats_df.empty:
            stats_df['comparison'] = comp_name
            all_stats_for_plot.append(stats_df)
        
        axs[i].set_ylabel('')
        if axs[i].get_legend():
            axs[i].get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"stats_metrics_{comp_name}_{block}_{band_name}_{area_labels[area]}_{normalizaion}.pdf"
    fig.savefig(os.path.join(results_dir, fname), dpi=300)
    plt.close(fig)

    if all_stats_for_plot:
        final_stats_table = pd.concat(all_stats_for_plot, ignore_index=True)
        print("\n" + "="*80)
        print(f"Statistical Report for: {comp_name} | {block.upper()} | {band_name.upper()} | {area_labels[area].upper()}")
        print("="*80)
        display_cols = ['metric', 'area', 't-stat', 'p_uncorrected', 'p_corr_fdr_bh', 'significant', 'stars']
        print(final_stats_table[display_cols].to_string())
        print("="*80 + "\n")


def stat_bar_comparisons_and_annotate(
    df_metrics, metric, ax, area_order, group_vars,
    correction='fdr_bh', alpha=0.05, y_offset_factor=0.08
):
    """
    Performs t-tests and annotates significance on a bar plot axis.
    This version is simplified as it only handles one comparison at a time.
    """
    stats_results = []
    group1, group2 = group_vars[0], group_vars[1]

    for area in area_order:
        d1 = df_metrics[(df_metrics['area'] == area) & (df_metrics['group'] == group1)][metric].dropna()
        d2 = df_metrics[(df_metrics['area'] == area) & (df_metrics['group'] == group2)][metric].dropna()
        t_stat, p_val = ttest_ind(d1, d2, equal_var=False, nan_policy='omit') if len(d1) > 1 and len(d2) > 1 else (np.nan, np.nan)
        stats_results.append({'area': area, 'metric': metric, 't-stat': t_stat, 'p_uncorrected': p_val})

    stats_df = pd.DataFrame(stats_results).dropna(subset=['p_uncorrected'])
    if stats_df.empty: return stats_df

    reject, pvals_corr, _, _ = multipletests(stats_df['p_uncorrected'], alpha=alpha, method=correction)
    stats_df[f'p_corr_{correction}'] = pvals_corr
    stats_df['significant'] = reject
    
    p_to_stars = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    stats_df['stars'] = stats_df.loc[stats_df['significant'], f'p_corr_{correction}'].apply(p_to_stars)
    
    # --- Annotation Logic ---
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_step = y_offset_factor * y_range
    
    for _, row in stats_df.iterrows():
        if row['significant']:
            area = row['area']
            x_pos = area_order.index(area)
            
            # Find the top of the highest bar for this area to place the annotation above it
            d1_sub = df_metrics[(df_metrics['area'] == area) & (df_metrics['group'] == group1)][metric].dropna()
            d2_sub = df_metrics[(df_metrics['area'] == area) & (df_metrics['group'] == group2)][metric].dropna()
            
            top1 = np.nanmean(d1_sub) + d1_sub.sem() if not d1_sub.empty else 0
            top2 = np.nanmean(d2_sub) + d2_sub.sem() if not d2_sub.empty else 0
            
            y_pos = max(top1, top2) + y_step
            ax.text(x_pos, y_pos, row['stars'], ha='center', va='bottom', fontsize=16, color='k')
            
    # Adjust y-limits to make space for annotations
    ax.set_ylim(top=ax.get_ylim()[1] + y_step)

    return stats_df

def plot_erds_ipsicontra(
    tfr_group_results,
    freqs,
    freq_bands,
    blocks,
    results_dir,
    all_cohort_definitions,
    cohorts_to_plot,
    expected_time_samples, # Crucial for the -50% shift logic
    filename_suffix="ipsicontra",
    normalization=""
):
    """
    Plots STN ERD/ERS lines using Ipsi-Contralateral alignment logic.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # [Start] Style settings (Unchanged)
    font_settings = {'title': 28, 'label': 26, 'ticks': 22, 'legend': 24, 'gait_ticks': 22}
    block_name_map = {"pre8walk": "8walk", "post8walk": "Post-8walk"}
    tick_locs_pct = [0, 10, 50, 60, 100]
    tick_labels = ['0%: HSI', '10%: TOC', '50%: HSC', '60%: TOI', '100%: HSI']
    event_times = [10, 50, 60]
    n_bands = len(freq_bands)
    # [End] Style settings

    for block in blocks:
        if block == 'baseline': continue
            
        fig, axes = plt.subplots(n_bands, 1, figsize=(10, 6 * n_bands), sharex=True, constrained_layout=True)
        if n_bands == 1: axes = [axes]

        for row_idx, (band_name, band_lim) in enumerate(freq_bands.items()):
            ax = axes[row_idx]
            max_abs_y = 0
            f_idx = get_freq_indices(freqs, band_lim) # Assumes this helper exists in scope

            # --- Iterate Cohorts ---
            for cohort_name in cohorts_to_plot:
                if cohort_name not in all_cohort_definitions: continue
                
                cohort_info = all_cohort_definitions[cohort_name]
                base_group = cohort_info['group']
                id_filter = set(cohort_info['ids']) if cohort_info['ids'] else None
                color = cohort_info['color']

                if base_group not in tfr_group_results: continue
                if block not in tfr_group_results[base_group]: continue
                if 'STN' not in tfr_group_results[base_group][block]: continue

                stn_data = tfr_group_results[base_group][block]['STN']
                all_ids = tfr_group_results[base_group][block].get('ID', [])

                valid_indices = [i for i, sid in enumerate(all_ids) if id_filter is None or sid in id_filter]
                if not valid_indices: continue

                # --- Ipsi-Contra Data Collection ---
                aligned_band_power = []

                # 1. Left STN (Shift -50%)
                if 'Left' in stn_data:
                    raw_list = stn_data['Left']
                    for i in valid_indices:
                        if i < len(raw_list) and raw_list[i] is not None:
                            arr_shift = _circ_shift_percent_time(raw_list[i], shift_percent=-50.0)
                            arr_resamp = _resample_to_len_2d(arr_shift, expected_time_samples)
                            if arr_resamp is not None:
                                aligned_band_power.append(np.nanmean(arr_resamp[f_idx, :], axis=0))

                # 2. Right STN (No Shift)
                if 'Right' in stn_data:
                    raw_list = stn_data['Right']
                    for i in valid_indices:
                        if i < len(raw_list) and raw_list[i] is not None:
                            arr_resamp = _resample_to_len_2d(raw_list[i], expected_time_samples)
                            if arr_resamp is not None:
                                aligned_band_power.append(np.nanmean(arr_resamp[f_idx, :], axis=0))

                # --- Compute Mean & SEM ---
                if aligned_band_power:
                    data_stack = np.stack(aligned_band_power)
                    mean_trace = np.nanmean(data_stack, axis=0)
                    sem_trace = np.nanstd(data_stack, axis=0) / np.sqrt(data_stack.shape[0])
                    x_axis = np.linspace(0, 100, len(mean_trace))

                    ax.plot(x_axis, mean_trace, color=color, linestyle='-', lw=3, label=cohort_name)
                    ax.fill_between(x_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.15)
                    
                    current_max = np.nanmax(np.abs(mean_trace) + np.abs(sem_trace))
                    if current_max > max_abs_y: max_abs_y = current_max

            # --- Subplot Formatting ---
            ax.set_title(f"{band_name.title()} ({band_lim[0]}-{band_lim[1]} Hz)", fontsize=font_settings['title'])
            ax.set_xticks(tick_locs_pct)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            for et in event_times: ax.axvline(et, color='black', linestyle='-', linewidth=1.5)
            ax.axhline(0, color='black', linestyle='-', lw=0.8)
            ax.set_xlim(0, 100)
            if max_abs_y > 0: ax.set_ylim(-max_abs_y * 1.15, max_abs_y * 1.15)
            ax.set_ylabel('Power (dB)', fontsize=font_settings['label'])
            ax.tick_params(axis='y', labelsize=font_settings['ticks'])
            ax.grid(True, linestyle='--', linewidth=0.5, axis='y')

            if row_idx == n_bands - 1:
                ax.set_xticklabels(tick_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=font_settings['gait_ticks'])
                ax.set_xlabel('Gait Cycle (%)', fontsize=font_settings['label'])
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Legend and Save
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(cohorts_to_plot), fontsize=font_settings['legend'], frameon=False)

        block_printable = block_name_map.get(block, block)
        fname = f"STN_ERDs_IpsiContra_{block_printable}_{filename_suffix}_{normalization}.pdf"
        full_path = os.path.join(results_dir, fname)
        print(f"  - Saving STN Ipsi-Contra plot: {fname}")
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)


# --- Helper Functions ---
def get_freq_indices(freqs, band):
    """Finds indices for a given frequency band."""
    low, high = band
    return np.where((freqs >= low) & (freqs < high))[0]

def plot_figure3_bars(df_metrics, metric, area_order, groups_blocks_colormap, ax=None):
    """
    Plots a single bar chart for a given metric.
    
    CHANGELOG:
    - Replaced 'group_palette' with 'groups_blocks_colormap'.
    """
    if ax is None: fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        x='area', y=metric, hue='group', data=df_metrics, order=area_order,
        palette=groups_blocks_colormap, errorbar=('se', 1), capsize=0.12, ax=ax,
        errwidth=2, edgecolor='black'
    )
    metric_titles = {'mean_ers': 'Median ERS/ERD', 'amp_fluct': 'Intra-Stride Fluctuation', 'gpm': 'GPM'}
    ax.set_title(metric_titles.get(metric, metric.replace('_', ' ').title()))
    ax.legend(loc='best'); sns.despine(ax=ax)
    ax.set_xlabel('')

def calculate_intra_stride_fluctuation(arr_band, gait_phases):
    """Calculates the difference between mean stance and swing phase power."""
    n_times = len(arr_band)
    def get_indices_for_phase(phase_name):
        start_frac, end_frac = gait_phases.get(phase_name, (0, 0)) # Use .get for safety
        return np.arange(int(start_frac * n_times), int(end_frac * n_times))
    
    # CHANGED: Using new phase names from the provided 'gait_phases' dictionary
    stance_indices = np.concatenate([get_indices_for_phase(p) for p in ['FDS', 'IDS']])
    swing_indices = np.concatenate([get_indices_for_phase(p) for p in ['LLS', 'RLS']])
    
    if len(stance_indices) == 0 or len(swing_indices) == 0: return np.nan
    return np.nanmean(arr_band[stance_indices]) - np.nanmean(arr_band[swing_indices])

def calculate_gpm(A):
    """Calculates Gait Phase Modulation index."""
    N = A.shape[1]
    if N < 3: return np.full(A.shape[0], np.nan)
    sigma_A = np.std(A, axis=1)
    sigma_A[sigma_A == 0] = 1e-9
    return (2 / (N * np.sqrt(2) * sigma_A)) * np.abs(fft(A, axis=1)[:, 2])
    
        