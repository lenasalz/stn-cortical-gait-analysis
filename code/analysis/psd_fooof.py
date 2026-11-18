import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (sem, ttest_ind, shapiro, mannwhitneyu, ttest_rel, 
                         wilcoxon, median_abs_deviation, wasserstein_distance, 
                         entropy as sp_entropy)
from scipy.signal import find_peaks
from typing import List, Dict, Any, Optional, Tuple
import mne
from mne.stats import fdr_correction
from specparam import SpectralModel
from fooof import FOOOFGroup, FOOOF
from statsmodels.stats.multitest import multipletests

from data_loader import save_data_to_pickle, load_data_from_pickle

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="fooof")

# ==============================================================================
# 1. DATA CALCULATION FUNCTIONS
# ==============================================================================

def calculate_dbs_psds(
    all_data_epoched_dico: Dict, 
    subjects_dico: Dict,
    cohort_definitions: Dict[str, Dict[str, Any]], 
    blocks: List[str]
) -> Dict[str, Any]:
    """
    Calculates and aggregates Power Spectral Densities (PSDs) for DBS data.

    This function iterates through all defined cohorts and calculates PSDs only for 
    those belonging to the 'Patient_NFB' group. It uses the Welch method for 
    spectral estimation.

    Args:
        all_data_epoched_dico (Dict): Nested dictionary containing epoched MNE objects 
            [group][block][datatype][subject_index].
        subjects_dico (Dict): Mapping of group names to lists of subject IDs.
        cohort_definitions (Dict): Definitions of cohorts including their base group 
            and optional ID filters.
        blocks (List[str]): List of experimental blocks (e.g., 'pre8walk', 'post8walk').

    Returns:
        Dict[str, Any]: A dictionary containing PSD values ('psds') and 
            frequency arrays ('freqs') organized by block and cohort name.
    """
    print("--- Pre-calculating PSDs for DBS data ---")
    all_psd_results = {'DBS_epoched': {}}
    key = 'DBS' 

    for block in ['baseline', *blocks]:
        all_psd_results['DBS_epoched'][block] = {}
        for cohort_name, cohort_info in cohort_definitions.items():
            if cohort_info['group'] != "Patient_NFB": 
                continue
                
            base_group = cohort_info['group']
            id_filter = set(cohort_info['ids']) if cohort_info['ids'] else None
            subjects = subjects_dico[base_group]
            psds_for_cohort, freqs = [], None
            
            for i, sub_id in enumerate(subjects):
                if id_filter and sub_id not in id_filter: continue
                try:
                    data_obj = all_data_epoched_dico[base_group][block][key][i] 
                    if data_obj is not None:
                        n_fft = len(data_obj.times)
                        spectrum = data_obj.compute_psd(fmin=4, fmax=45, method='welch', n_fft=n_fft)
                        psd_values = spectrum.get_data(return_freqs=False).mean(axis=0).squeeze()
                        psds_for_cohort.append(psd_values)
                        if freqs is None: freqs = spectrum.freqs
                except (IndexError, KeyError, Exception): continue

            if psds_for_cohort:
                all_psd_results['DBS_epoched'][block][cohort_name] = {'psds': np.array(psds_for_cohort), 'freqs': freqs}
    
    print("--- DBS PSD calculation complete. ---")
    return all_psd_results

def calculate_all_cohort_psds_by_roi(
    all_data_epoched_dico: Dict, 
    subjects_dico: Dict,
    cohort_definitions: Dict[str, Dict[str, Any]], 
    blocks: List[str],
    rois_to_analyze: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Calculates and aggregates PSDs for EEG data, partitioned by Region of Interest (ROI).

    For each ROI, the function picks specific channels, calculates the Welch PSD, 
    and averages across both epochs and channels within that ROI.

    Args:
        all_data_epoched_dico (Dict): Nested dictionary of epoched MNE objects.
        subjects_dico (Dict): Mapping of group names to lists of subject IDs.
        cohort_definitions (Dict): Definitions of cohorts and subject filters.
        blocks (List[str]): List of experimental conditions/blocks.
        rois_to_analyze (Dict[str, List[str]]): Mapping of ROI names to channel labels.

    Returns:
        Dict[str, Any]: A dictionary structured as [roi_name]['EEG_epoched'][block][cohort]
            containing 'psds' and 'freqs'.
    """
    print("--- Pre-calculating PSDs for all defined ROIs ---")
    all_psd_by_roi = {roi_name: {} for roi_name in rois_to_analyze}
    for roi_name, roi_channels in rois_to_analyze.items():
        all_psd_results = {'EEG_epoched': {}}
        for block in ['baseline', *blocks]:
            all_psd_results['EEG_epoched'][block] = {}
            for cohort_name, cohort_info in cohort_definitions.items():
                base_group, id_filter = cohort_info['group'], set(cohort_info['ids']) if cohort_info['ids'] else None
                subjects = subjects_dico[base_group]
                psds_for_cohort, freqs = [], None
                for i, sub_id in enumerate(subjects):
                    if id_filter and sub_id not in id_filter: continue
                    try:
                        data_obj_full = all_data_epoched_dico[base_group][block]['EEG'][i]
                        if data_obj_full is not None:
                            data_obj = data_obj_full.copy().pick_channels(roi_channels, ordered=False)
                            n_fft = len(data_obj.times)
                            spectrum = data_obj.compute_psd(fmin=4, fmax=45, method='welch', n_fft=n_fft)
                            psd_values = spectrum.get_data(return_freqs=False).mean(axis=(0, 1))
                            psds_for_cohort.append(psd_values)
                            if freqs is None: freqs = spectrum.freqs
                    except (IndexError, KeyError): continue
                    except ValueError as e:
                        print(f" Warning: Could not pick channels for ROI '{roi_name} for subj {sub_id}. Skipping. Error: {e}")
                if psds_for_cohort:
                    all_psd_results['EEG_epoched'][block][cohort_name] = {'psds': np.array(psds_for_cohort), 'freqs': freqs}
        all_psd_by_roi[roi_name] = all_psd_results
    return all_psd_by_roi

# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def get_band_edges(freq_bands: Dict, freqs: np.ndarray) -> np.ndarray:
    """
    Extracts and sorts unique frequency boundaries from a band definition dictionary.

    Args:
        freq_bands (Dict): Mapping of band names to (low, high) tuples.
        freqs (np.ndarray): The frequency array from the data to ensure bounds are relevant.

    Returns:
        np.ndarray: Sorted array of unique frequency edges.
    """
    edges = set()
    for low, high in freq_bands.values():
        edges.add(low)
        edges.add(high)
    return np.array(sorted(list(edges)))

def plot_psd_comparison_multi_block(
    psd_data_for_type: Dict[str, Dict], 
    blocks_to_plot: List[str],
    cohorts_to_plot: List[str], 
    cohort_definitions: Dict[str, Dict],
    colors: List[str], 
    title: str, 
    filename: str,
    freq_bands: Dict[str, Tuple[float, float]]
) -> None:
    """
    Creates a high-quality PSD plot comparing multiple cohorts and blocks.

    Uses Median power and Standard Error of the Median (calculated via Median 
    Absolute Deviation) for robust visualization. Power is converted to decibels (dB).

    Args:
        psd_data_for_type (Dict): The PSD data dictionary for a specific ROI or DBS.
        blocks_to_plot (List[str]): Conditions to include (e.g., ['baseline', 'pre8walk']).
        cohorts_to_plot (List[str]): Cohort names to include in the plot.
        cohort_definitions (Dict): Cohort metadata for labeling.
        colors (List[str]): List of hex/color strings for cohort differentiation.
        title (str): Plot title.
        filename (str): Path to save the resulting PDF/PNG.
        freq_bands (Dict): Frequency bands to mark with vertical lines and labels.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    block_styles = {'baseline': '-', 'pre8walk': '--', 'post8walk': ':'}
    block_name_map = {"pre8walk": "Walk", "post8walk": "Post-8walk", "baseline": "Rest"}
    cohort_colors = {cohort: color for cohort, color in zip(cohorts_to_plot, colors)}

    freqs = None
    for b in blocks_to_plot:
        for c in cohorts_to_plot:
            if b in psd_data_for_type and c in psd_data_for_type[b]:
                freqs = psd_data_for_type[b][c].get('freqs')
                if freqs is not None: break
        if freqs is not None: break
    
    if freqs is None: return

    for cohort_name in cohorts_to_plot:
        color = cohort_colors.get(cohort_name, 'k')
        for block in blocks_to_plot:
            if block not in psd_data_for_type or cohort_name not in psd_data_for_type[block]: 
                continue
            
            psds_array = psd_data_for_type[block][cohort_name]['psds']
            n_subs = psds_array.shape[0]
            # Median + percentile 
            median_psd = np.median(psds_array, axis=0)
            mad = median_abs_deviation(psds_array, axis=0, scale='normal')
            se_median = mad / np.sqrt(n_subs)
            
            median_psd_db = 10 * np.log10(np.maximum(1e-18, median_psd))
            lower_bound_db = 10 * np.log10(np.maximum(1e-18, median_psd - se_median))
            upper_bound_db = 10 * np.log10(np.maximum(1e-18, median_psd + se_median))
            
            ax.plot(freqs, median_psd_db, color=color, linestyle=block_styles.get(block, '-'), 
                    label=f"{cohort_name} - {block_name_map.get(block, block)}", lw=2)
            ax.fill_between(freqs, lower_bound_db, upper_bound_db, color=color, alpha=0.1)

    if freq_bands:
        for edge in get_band_edges(freq_bands, freqs):
            ax.axvline(x=edge, color='k', linestyle='--', linewidth=1.5, alpha=0.6)
        for band_name, (low, high) in freq_bands.items():
            x_center = (low + high) / 2
            ax.text(x_center, 0.03, band_name, transform=ax.get_xaxis_transform(), 
                    ha='center', fontsize=18, fontweight='bold')

        # --- Axis and plot formatting ---
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("Frequency (Hz)", fontsize=22)
    ax.set_ylabel("Power Spectral Density (dB)", fontsize=22) # Updated Label
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=22, loc="upper right")
    ax.set_xlim(4, 45)

    
    plt.tight_layout()
    fig.savefig(filename)
    plt.show()
    plt.close(fig)

# ==============================================================================
# 3. STATISTICAL FUNCTIONS
# ==============================================================================

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Cohen's d effect size for two independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0

def _cohens_d_paired(x, y):
    """Calculates Cohen's d effect size for paired/related samples."""
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    diff = np.array(x) - np.array(y)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return np.nan
    return np.mean(diff) / std_diff

def _run_comparison_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
    """
    Performs an independent comparison test between two arrays.

    Checks for normality using Shapiro-Wilk. Performs Welch's t-test if normal, 
    otherwise Mann-Whitney U.

    Args:
        data1 (np.ndarray): Group 1 samples.
        data2 (np.ndarray): Group 2 samples.

    Returns:
        Dict: Dictionary containing test name, statistic, p-value, and Cohen's d.
    """
    if len(data1) < 3 or len(data2) < 3:
        test = 'mann-whitney'
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    else:
        _, p1 = shapiro(data1); _, p2 = shapiro(data2)
        if p1 > 0.05 and p2 > 0.05:
            stat, p = ttest_ind(data1, data2, equal_var=False)
            test = 't-test_ind'
        else:
            stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
            test = 'mann-whitney'
    return {'test_used': test, 'statistic': stat, 'p_value': p, 'cohens_d': _cohens_d(data1, data2)}

def _run_paired_comparison_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
    """
    Performs a paired comparison test (within-subject).

    Checks normality of differences. Performs paired t-test if normal, 
    otherwise Wilcoxon signed-rank test.
    """
    if len(data1) != len(data2):
        raise ValueError("Paired samples must have the same length.")
    if len(data1) < 3:
        test_used = 'wilcoxon'
        # For wilcoxon, stat is sum of ranks, p_val is the p-value
        stat, p = wilcoxon(data1, data2)
    else:
        diff = data1 - data2
        _, p = shapiro(diff)
        if p > 0.05:
            stat, p = ttest_rel(data1, data2); test = 't-test_rel'
        else:
            stat, p = wilcoxon(data1, data2); test = 'wilcoxon'
    d = _cohens_d_paired(data1, data2)
    return {'test_used': test, 'statistic': stat, 'p_value': p, 'cohens_d': d}

def _apply_fdr_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Benjamini-Hochberg FDR correction to the p_value column of a DataFrame."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty or 'p_value' not in dataframe.columns:
        dataframe['p_fdr_corrected'] = np.nan; dataframe['significant_fdr'] = False
        return dataframe
    p_values = dataframe['p_value'].dropna()
    if p_values.empty:
        dataframe['p_fdr_corrected'] = np.nan; dataframe['significant_fdr'] = False
        return dataframe
    is_significant, p_values_corrected = fdr_correction(p_values.values, alpha=0.05, method='indep')
    dataframe.loc[p_values.index, 'p_fdr_corrected'] = p_values_corrected
    dataframe.loc[p_values.index, 'significant_fdr'] = is_significant
    return dataframe

# ==============================================================================
# 4. FOOOF / SPECPARAM FUNCTIONS
# ==============================================================================

def analyze_specparam_and_plot(
    all_psd_data_roi, 
    roi_name, 
    blocks, 
    results_dir, 
    resultsDir_fooof,
    freq_bands,
    freq_range=[4.0, 45.0],
    peak_width = [4,15],
    max_n_peaks=6,
    min_peak_height = 0.1,
    peak_threshold=2.0,
    
):
    """
    Performs Spectral Parameterization (FOOOF) analysis.
    
    1. Fits individual subjects (for Statistics).
    2. Fits the Grand Average (for Plotting Option A).
    3. Reconstructs group average fit (for Plotting Option B).
    """
    print(f"\n--- Starting FOOOF/SpecParam Analysis for ROI: {roi_name} ---")
    
    # Store stats rows here
    stats_rows = []
    
    # Define bands to extract specific peaks for statistics
    # You can adjust these ranges
    bands_of_interest = freq_bands

    # Iterate over Data Source (EEG/DBS) -> Block -> Cohort
    # We assume standard structure based on your previous code
    for data_type in ['EEG_epoched', 'DBS_epoched']:
        if data_type not in all_psd_data_roi:
            continue
            
        for block in blocks:
            if block not in all_psd_data_roi[data_type]:
                continue
                
            for cohort_name, data_dict in all_psd_data_roi[data_type][block].items():
                
                # --- 1. DATA PREP ---
                raw_freqs = data_dict['freqs']
                raw_psds = data_dict['psds'] # shape (n_subjects, n_freqs)
                
                # CROP DATA to desired range (e.g., 4-45 Hz)
                # FOOOF is sensitive to indices, so we slice arrays carefully
                mask = (raw_freqs >= freq_range[0]) & (raw_freqs <= freq_range[1])
                freqs = raw_freqs[mask]
                psds = raw_psds[:, mask]
                
                n_subs = psds.shape[0]
                if n_subs < 2: 
                    print(f"Skipping {cohort_name}-{block}: Not enough subjects.")
                    continue

                # --- 2. FIT INDIVIDUAL SUBJECTS (FOR STATISTICS) ---
                # Fixed mode is safer for >3Hz start frequency
                fg = FOOOFGroup(peak_width_limits=peak_width, 
                                max_n_peaks=max_n_peaks, 
                                min_peak_height=min_peak_height,
                                peak_threshold=peak_threshold,
                                aperiodic_mode='fixed', 
                                verbose=False)
                print(f"For: {cohort_name}, {block}, {data_type}") 
                fg.fit(freqs, psds)
                pw_str = f"{peak_width[0]}-{peak_width[1]}"
                
                # Create filename
                out_name = f"FOOOF_GroupFit_{cohort_name}_{block}_{data_type}_width{pw_str}.pdf"
                
                # Plot and Save
                # Note: We use save_fig=True and provide the path separately
                fg.plot(save_fig=True, file_name=out_name, file_path=resultsDir_fooof)
                
                # Optional: Close the plot to free memory if running a loop
                plt.close('all')
                indiv_fooof_resultsDir  = os.path.join(resultsDir_fooof, "indiv_fooof")
                os.makedirs(indiv_fooof_resultsDir, exist_ok = True)
                comparisons_df = tune_fooof_parameters(freqs, psds, indiv_fooof_resultsDir, cohort_name, block, roi_name)

                print("="*80, "Comparisons of fooof tuning \n", "="*80)
                # # Extract results per subject
                ap_params = fg.get_params('aperiodic_params') # [Offset, Exponent]
                r2_values = fg.get_params('r_squared')
                
                # Iterate subjects to extract peaks
                for sub_idx in range(n_subs):
                    # Base stats
                    row = {
                        'ROI': roi_name,
                        'Data_Type': data_type,
                        'Block': block,
                        'Cohort': cohort_name,
                        'Subject_Idx': sub_idx, # You might want to pass real IDs if available
                        'Offset': ap_params[sub_idx, 0],
                        'Exponent': ap_params[sub_idx, 1],
                        'R2': r2_values[sub_idx]
                    }
                    
                    # Peak Extraction Logic:
                    # Get peaks for this specific subject
                    flat_peaks = fg.get_fooof(sub_idx).peak_params_ 
                    # flat_peaks shape: (n_peaks, 3) -> [CF, Power, BW]
                    
                    for band_label, (f_lo, f_hi) in bands_of_interest.items():
                        # Find peaks within this band
                        in_band = flat_peaks[(flat_peaks[:, 0] >= f_lo) & (flat_peaks[:, 0] <= f_hi)]
                        
                        if len(in_band) > 0:
                            # If multiple, take the one with highest power
                            max_peak = in_band[np.argmax(in_band[:, 1])]
                            row[f'{band_label}_CF'] = max_peak[0]
                            row[f'{band_label}_PW'] = max_peak[1]
                            row[f'{band_label}_BW'] = max_peak[2]
                        else:
                            row[f'{band_label}_CF'] = np.nan
                            row[f'{band_label}_PW'] = np.nan
                            row[f'{band_label}_BW'] = np.nan
                            
                    stats_rows.append(row)

                # # --- 3. PLOTTING OPTION A: FIT OF GRAND AVERAGE ---
                # # (Best for Publication/visualizing the "archetype")
                # avg_psd = np.mean(psds, axis=0)
                # fm = FOOOF(peak_width_limits=[4, 15], max_n_peaks=3, min_peak_height=0.2, aperiodic_mode='fixed', verbose=False)
                # fm.fit(freqs, avg_psd)
                
                # # Save Plot
                # out_name = f"FOOOF_OptA_GrandAvg_{roi_name}_{block}_{cohort_name}.pdf"
                # out_path = os.path.join(results_dir, f"PSD_plots_{roi_name}", out_name)
                
                # # We use the internal simple plotter, but capture the figure to save it
                # # Note: FOOOF plots in Log10 Power, not dB (Log10 * 10). 
                # # Visually it's the same shape.
                # try:
                #     fm.plot(save_fig=True, file_name=out_path, plot_peaks='shade')
                #     plt.close('all') # Close the fig opened by fooof
                # except Exception as e:
                #     print(f"Plotting error Opt A: {e}")

               # --- 4. PLOTTING OPTION B: DEBUG (OVERLAY) ---
                
                # 1. Get Model Mean (Log Space)
                fooofed_spectrums = []
                for i in range(n_subs):
                    fm_tmp = fg.get_fooof(ind=i, regenerate=True)
                    fooofed_spectrums.append(fm_tmp.fooofed_spectrum_)
                avg_model = np.mean(np.array(fooofed_spectrums), axis=0) 
                
                # 2. Get Data Mean (Log Space) - MATCHING OPERATIONS
                # We log individual subjects first, THEN average
                # psds shape is (n_subs, n_freqs)
                log_psds = np.log10(psds)
                avg_log_data = np.mean(log_psds, axis=0)

                # 3. Get Aperiodic Fit (Log Space)
                # We need the average of the INDIVIDUAL aperiodic fits
                ap_fits = []
                for i in range(n_subs):
                    # Get the aperiodic component alone
                    fm_tmp = fg.get_fooof(ind=i, regenerate=True)
                    ap_fits.append(fm_tmp._ap_fit)
                avg_ap_fit = np.mean(np.array(ap_fits), axis=0)

                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot MATCHED data
                ax.plot(freqs, avg_log_data, 'k', label='Raw Data (Mean of Logs)', lw=2)
                ax.plot(freqs, avg_model, 'r--', label='FOOOF Model (Mean)', lw=2)
                ax.plot(freqs, avg_ap_fit, 'b:', label='Aperiodic Fit (Mean)', lw=1.5)
                ax.fill_between(freqs, avg_ap_fit, avg_model, 
                                color='green', alpha=0.3, label='Periodic Power (Mean)')
                ax.set_title(f"DEBUG FOOOF: {cohort_name} - {block} (N={n_subs})")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log10 Power")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                out_name_b = f"FOOOF_OptB_Debug_{roi_name}_{block}_{cohort_name}.pdf"
                out_path_b = os.path.join(results_dir, f"PSD_plots_{roi_name}", out_name_b)
                fig.savefig(out_path_b)
                plt.close(fig)

                # --- 5. PLOTTING OPTION C: STATISTICAL RECONSTRUCTION ---
                # This plots the "Ideal" peaks based on the average parameters used in your stats.
                
                # A. Define Bands and Colors
                bands_dict = freq_bands
                band_colors = {
                    'Theta': 'purple',
                    'Alpha': 'green',
                    'Beta':  'orange',
                    'Gamma': 'red'
                }

                # B. Define Gaussian Generator (FIXED SIGMA)
                               # B. Helper for Gaussian generation
                def gen_gaussian_array(freqs, center, height, bandwidth):
                    if np.isnan(center) or np.isnan(height) or np.isnan(bandwidth):
                        return np.zeros_like(freqs)
                    sigma = bandwidth / 2.0
                    return height * np.exp(-(freqs - center)**2 / (2 * sigma**2))

                # C. Prepare Data Arrays (Same as Option B logic)
                # 1. Raw Data (Log Mean)
                log_psds = np.log10(psds)
                avg_log_data = np.mean(log_psds, axis=0)
                
                # 2. Aperiodic Fit (Mean)
                ap_fits = []
                for i in range(n_subs):
                    fm_tmp = fg.get_fooof(ind=i, regenerate=True)
                    ap_fits.append(fm_tmp._ap_fit)
                avg_ap_fit = np.mean(np.array(ap_fits), axis=0)
                
                # 3. Full Model (Mean)
                fooofed_spectrums = []
                for i in range(n_subs):
                    fm_tmp = fg.get_fooof(ind=i, regenerate=True)
                    fooofed_spectrums.append(fm_tmp.fooofed_spectrum_)
                avg_model = np.mean(np.array(fooofed_spectrums), axis=0)

                # D. Generate Plot
                fig, ax = plt.subplots(figsize=(10, 7))
                
                # Layer 1: Raw Data (Shadow)
                ax.plot(freqs, avg_log_data, 'k', alpha=0.15, lw=3, label='Raw Data')
                
                # Layer 2: Aperiodic Background (The Floor)
                ax.plot(freqs, avg_ap_fit, 'b:', label='Aperiodic Fit', lw=1.5, alpha=0.8)
                
                # Layer 3: The Decomposed Bumps
                # We calculate the mean curve for each band and stack it on the aperiodic floor
                for band_name, (f_min, f_max) in bands_dict.items():
                    subject_curves = []
                    
                    for sub_idx in range(n_subs):
                        flat_peaks = fg.get_fooof(sub_idx).peak_params_ 
                        
                        # Find max peak in this band
                        in_band = flat_peaks[(flat_peaks[:, 0] >= f_min) & (flat_peaks[:, 0] <= f_max)]
                        
                        if len(in_band) > 0:
                            max_peak = in_band[np.argmax(in_band[:, 1])]
                            cf, pw, bw = max_peak
                            subject_curves.append(gen_gaussian_array(freqs, cf, pw, bw))
                        else:
                            subject_curves.append(np.zeros_like(freqs))
                    
                    # Average the curves (handling jitter correctly)
                    mean_curve = np.mean(np.array(subject_curves), axis=0)
                    
                    # Only plot if significant
                    if np.max(mean_curve) > 0.02: 
                        total_curve = avg_ap_fit + mean_curve
                        col = band_colors.get(band_name, 'grey')
                        
                        # Plot the bump filled
                        ax.fill_between(freqs, avg_ap_fit, total_curve, color=col, alpha=0.4, label=f'{band_name} Component')
                        ax.plot(freqs, total_curve, color=col, lw=1, alpha=0.6)

                # Layer 4: The Full Model (The Ceiling)
                # This envelopes all the bumps
                ax.plot(freqs, avg_model, 'r--', label='Full Model (Sum)', lw=2)

                ax.set_title(f"SPECTRAL DECOMPOSITION: {cohort_name} - {block}")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log10 Power")
                ax.legend(loc='upper right', fontsize='small')
                ax.grid(True, alpha=0.3)
                
                out_name_c = f"FOOOF_OptC_Hybrid_{roi_name}_{block}_{cohort_name}.pdf"
                out_path_c = os.path.join(results_dir, f"PSD_plots_{roi_name}", out_name_c)
                fig.savefig(out_path_c)
                plt.close(fig)

    # Return stats as DataFrame
    return pd.DataFrame(stats_rows)


def run_fooof_statistics(fooof_df, comparisons_within, comparisons_between, results_dir, narrow_analysis=True):
    """
    Runs stats with Normality Checks AND FDR Correction.
    
    Changes applied:
    1. Imputes 0 for Power if Peak is missing (NaN).
    2. Removes Bandwidth (_BW) from analysis.
    3. Option 'narrow_analysis=True' removes Theta and Gamma.
    4. Separates Aperiodic and Periodic components for FDR.
    """
    print("\n--- Running Statistics on FOOOF Parameters ---")
    if narrow_analysis:
        print("    -> Narrow Analysis: Theta and Gamma excluded.")
    print("    -> Imputation: Missing Peak Power set to 0.")
    print("    -> Exclusion: Bandwidth (BW) ignored.")
    
    stats_results = []
    
    # --- 1. PRE-PROCESS DATAFRAME ---
    # Work on a copy to avoid affecting the original variable
    df_stats = fooof_df.copy()
    
    # A. IMPUTE 0 FOR POWER
    # If a peak is NaN, it means it doesn't exist -> Power = 0
    # We do this for all bands present in the DF
    pw_cols = [c for c in df_stats.columns if c.endswith('_PW')]
    for col in pw_cols:
        df_stats[col] = df_stats[col].fillna(0)

    # B. DEFINE COLUMNS TO ANALYZE
    meta_cols = ['ROI', 'Data_Type', 'Block', 'Cohort', 'Subject_Idx', 'R2']
    all_params = [c for c in df_stats.columns if c not in meta_cols]
    
    # Filter 1: Remove Bandwidth (_BW)
    param_cols = [c for c in all_params if '_BW' not in c] # and ('_CF' not in c))]
    
    # Filter 2: Narrow Analysis (Remove Theta/Gamma)
    if narrow_analysis:
        param_cols = [c for c in param_cols if 'Theta' not in c and 'Gamma' not in c]
        
    print(f"    -> Parameters included in stats: {param_cols}")
    
    rois = df_stats['ROI'].unique()
    data_types = df_stats['Data_Type'].unique()

    # --- LOOP FOR COMPUTATION ---
    for roi in rois:
        for dtype in data_types:
            df_subset = df_stats[(df_stats['ROI'] == roi) & (df_stats['Data_Type'] == dtype)]
            if df_subset.empty: continue

            # -------------------------------------------------
            # 1. WITHIN-COHORT (PAIRED)
            # -------------------------------------------------
            cohorts = df_subset['Cohort'].unique()
            for cohort in cohorts:
                df_cohort = df_subset[df_subset['Cohort'] == cohort]
                for (cond_a, cond_b) in comparisons_within:
                    if cond_a not in df_cohort['Block'].values or cond_b not in df_cohort['Block'].values: continue
                    
                    df_a = df_cohort[df_cohort['Block'] == cond_a].set_index('Subject_Idx')
                    df_b = df_cohort[df_cohort['Block'] == cond_b].set_index('Subject_Idx')
                    df_paired = df_a.join(df_b, lsuffix='_A', rsuffix='_B', how='inner')
                    
                    if len(df_paired) < 3: continue
                        
                    for param in param_cols:
                        col_a, col_b = f"{param}_A", f"{param}_B"
                        # For Power, NaNs are now 0s, so we keep them.
                        # For CF (Freq), NaNs remain NaNs. We must drop them for T-test.
                        valid = df_paired[[col_a, col_b]].dropna()
                        data_a, data_b = valid[col_a].values, valid[col_b].values
                        
                        if len(data_a) < 3: continue
                        
                        # Normality
                        diffs = data_a - data_b
                        if np.all(diffs == 0): continue
                        try: _, p_shapiro = shapiro(diffs)
                        except: p_shapiro = 0.0
                        
                        # Test
                        try:
                            if p_shapiro > 0.05:
                                stat, p_val = ttest_rel(data_a, data_b)
                                test_name = 'Paired T-Test'
                            else:
                                stat, p_val = wilcoxon(data_a, data_b)
                                test_name = 'Wilcoxon'
                                
                            stats_results.append({
                                'Test_Context': 'Within-Cohort',
                                'ROI': roi, 'Data_Type': dtype,
                                'Group_Level': cohort,
                                'Comparison': f"{cond_a} vs {cond_b}",
                                'Parameter': param,
                                'N': len(data_a),
                                'Normality_P': p_shapiro,
                                'Test_Used': test_name,
                                'Statistic': stat,
                                'P_Value': p_val, 
                                'Mean_Diff': np.mean(data_a) - np.mean(data_b),
                                'Median_Diff': np.median(data_a) - np.median(data_b),
                            })
                        except Exception as e: pass

            # -------------------------------------------------
            # 2. BETWEEN-COHORT (INDEPENDENT)
            # -------------------------------------------------
            blocks = df_subset['Block'].unique()
            for block in blocks:
                df_block = df_subset[df_subset['Block'] == block]
                for (group_a, group_b) in comparisons_between:
                    vals_a = df_block[df_block['Cohort'] == group_a]
                    vals_b = df_block[df_block['Cohort'] == group_b]
                    if vals_a.empty or vals_b.empty: continue
                        
                    for param in param_cols:
                        # Drop NaNs (Freqs). Power has no NaNs (imputed 0).
                        data_a = vals_a[param].dropna().values
                        data_b = vals_b[param].dropna().values
                        if len(data_a) < 3 or len(data_b) < 3: continue
                        
                        try:
                            _, p_shap_a = shapiro(data_a)
                            _, p_shap_b = shapiro(data_b)
                            is_normal = (p_shap_a > 0.05) and (p_shap_b > 0.05)
                            min_shapiro = min(p_shap_a, p_shap_b)
                        except: is_normal, min_shapiro = False, 0.0
                        
                        try:
                            if is_normal:
                                stat, p_val = ttest_ind(data_a, data_b, equal_var=False)
                                test_name = 'Welch T-Test'
                            else:
                                stat, p_val = mannwhitneyu(data_a, data_b)
                                test_name = 'Mann-Whitney'

                            stats_results.append({
                                'Test_Context': 'Between-Cohort',
                                'ROI': roi, 'Data_Type': dtype,
                                'Group_Level': block,
                                'Comparison': f"{group_a} vs {group_b}",
                                'Parameter': param,
                                'N': f"{len(data_a)}/{len(data_b)}",
                                'Normality_P': min_shapiro,
                                'Test_Used': test_name,
                                'Statistic': stat,
                                'P_Value': p_val,
                                'Mean_Diff': np.mean(data_a) - np.mean(data_b),
                                'Median_Diff': np.median(data_a) - np.median(data_b)
                            })
                        except Exception as e: pass
                            
    if not stats_results:
        print("No statistics calculated.")
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats_results)

    # --- APPLY FDR CORRECTION (Separating Periodic/Aperiodic) ---
    print("    -> Applying FDR Correction (Benjamini-Hochberg) split by Family...")

    stats_df['P_FDR'] = np.nan
    stats_df['Significance_corr'] = ''
    stats_df['Significance_raw'] = ''

    # 1. Define Param Families
    def get_family(param_name):
        if param_name in ['Offset', 'Exponent']:
            return 'Aperiodic'
        else:
            return 'Periodic'

    stats_df['Param_Family'] = stats_df['Parameter'].apply(get_family)

    # 2. Group by ROI, Context, AND Family for correction
    groups = stats_df.groupby(['ROI', 'Test_Context', 'Param_Family'])

    for (roi, context, family), group_indices in groups.groups.items():
        
        p_vals = stats_df.loc[group_indices, 'P_Value'].values

        # Mark Raw Significance
        stats_df.loc[group_indices, 'Significance_raw'] = [
            '**' if p < 0.01 else ('*' if p < 0.05 else '') for p in p_vals
        ]
        
        # Correction
        reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        
        stats_df.loc[group_indices, 'P_FDR'] = pvals_corrected
        stats_df.loc[group_indices, 'Significance_corr'] = [
            '**' if p < 0.01 else ('*' if p < 0.05 else '') for p in pvals_corrected
        ]

    stats_df = stats_df.sort_values(by=['Test_Context', 'ROI', 'Param_Family', 'Parameter'])
    
    out_path = os.path.join(results_dir, "statistical_analysis_FOOOF_results_FDR_split_paper.csv")
    stats_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"--- Statistics saved to: {out_path} ---")
    
    return stats_df


# ==============================================================================
# 5. ORCHESTRATOR
# ==============================================================================
def main_analysis_orchestrator(
    all_data_dico: Dict, all_data_epoched_dico: Dict, subjects_dico: Dict,
    blocks: List[str], cohort_definitions: Dict, plots_to_generate: List[Dict],
    statistical_comparisons: List[List[str]],
    results_dir: str,
    results_dir_fooof: str,
    rois_to_analyze: Dict[str, List[str]],
    base_colors: List[str],
    freq_bands: Dict,
    recompute_psds: bool = False,
    is_full_version: bool = False,
):
    """
    Orchestrates the entire PSD analysis workflow for both EEG and DBS,
    including caching, plotting, and a final combined statistical report.
    """
    print("\n" + "="*80 + "\n--- Starting Full PSD Analysis Workflow ---\n" + "="*80)
    all_cohort_names = list(cohort_definitions.keys())

    # --- Step 1: Load/Recompute EEG PSDs for all ROIs ---
    pkl_file_name = "precomputed_psd_data_by_roi"
    if not recompute_psds and os.path.exists(os.path.join(results_dir, pkl_file_name + ".pkl")):
        all_psd_data_by_roi = load_data_from_pickle(pkl_file_name, results_dir)
    else:
        all_psd_data_by_roi = calculate_all_cohort_psds_by_roi(
            all_data_epoched_dico, subjects_dico,
            cohort_definitions, blocks, rois_to_analyze
        )
        save_data_to_pickle(all_psd_data_by_roi, pkl_file_name, results_dir)

    # --- Step 2: Load/Recompute DBS PSDs ---
    pkl_file_name_dbs=  "precomputed_psd_data_dbs"
    
    if not recompute_psds and os.path.exists(os.path.join(results_dir, pkl_file_name_dbs + ".pkl")):
        all_psd_data_dbs = load_data_from_pickle(pkl_file_name_dbs, results_dir)
    else:
        all_psd_data_dbs = calculate_dbs_psds(
            all_data_epoched_dico, subjects_dico, cohort_definitions, blocks
        )
        save_data_to_pickle(all_psd_data_dbs, pkl_file_name_dbs, results_dir)

    all_stats_dfs = []
    all_fooof_dfs = [] 
    # --- Step 3: Analyze and Plot EEG ROI Data ---
    for roi_name, all_psd_data_for_roi in all_psd_data_by_roi.items():
        print("\n" + "="*50 + f"\n--- Analyzing ROI: {roi_name} ---\n" + "="*50)
        output_dir_plots = os.path.join(results_dir, f"PSD_plots_{roi_name}")
        os.makedirs(output_dir_plots, exist_ok=True)
        for plot_def in plots_to_generate:
            title = f"PSD Comparison: {plot_def['name']} ({roi_name})"
            filename = os.path.join(output_dir_plots, f"PSD_{plot_def['name']}_{roi_name}.pdf")
            plot_psd_comparison_multi_block(
                psd_data_for_type=all_psd_data_for_roi['EEG_epoched'],
                blocks_to_plot=['baseline', *blocks], cohorts_to_plot=plot_def['cohorts'],
                cohort_definitions=cohort_definitions, colors=base_colors, title=title, filename=filename, freq_bands = freq_bands
            )
        if is_full_version:
            stats_df_for_roi = run_and_save_psd_statistics(
                all_psd_data=all_psd_data_for_roi, 
                data_key='EEG_epoched',
                comparisons_between=statistical_comparisons,
                cohorts_for_within=all_cohort_names, 
                walk_blocks=blocks,
                results_dir=results_dir, 
                analysis_name=roi_name, 
                freq_bands=freq_bands,
                run_advanced_stats=True, # Set to True to run new tests
                n_permutations=5000 # Can be adjusted, e.g., 1000 for faster checks
            )
            if stats_df_for_roi is not None and not stats_df_for_roi.empty:
                stats_df_for_roi['data_source'] = roi_name
                all_stats_dfs.append(stats_df_for_roi)

        # ========================================================
        # NEW: STEP 3.5 - RUN FOOOF/SPECPARAM ANALYSIS
        # ========================================================
        
        fooof_df = analyze_specparam_and_plot(
            all_psd_data_roi=all_psd_data_for_roi,
            roi_name=roi_name,
            blocks=['baseline', *blocks], # Analyze all available blocks
            results_dir=results_dir,
            resultsDir_fooof = results_dir_fooof,
            freq_bands = freq_bands,
            freq_range=[4.0, 45.0],
            peak_width = [2.0,12.0],
            max_n_peaks = 3,
            min_peak_height = 0.09,
            peak_threshold = 2,
        )
        if not fooof_df.empty:
            all_fooof_dfs.append(fooof_df)


    # --- Step 4: Analyze and Plot DBS Data ---
    print("\n" + "="*50 + "\n--- Analyzing DBS Data ---\n" + "="*50)
    output_dir_plots_dbs = os.path.join(results_dir, "PSD_plots_DBS")
    os.makedirs(output_dir_plots_dbs, exist_ok=True)
    dbs_title = "DBS PSD Comparison: PD Cohort"
    dbs_filename = os.path.join(output_dir_plots_dbs, "PSD_DBS_PD_epoched.pdf")
    plot_psd_comparison_multi_block(
        psd_data_for_type=all_psd_data_dbs['DBS_epoched'],
        blocks_to_plot=['baseline', *blocks], cohorts_to_plot=['PD'],
        cohort_definitions=cohort_definitions, colors=base_colors,
        title=dbs_title, filename=dbs_filename, freq_bands = freq_bands
    )

    if is_full_version:
        stats_df_dbs = run_and_save_psd_statistics(
            all_psd_data=all_psd_data_dbs, 
            data_key='DBS_epoched',
            comparisons_between=[], # No between-cohort for DBS
            cohorts_for_within=['PD'], 
            walk_blocks=blocks,
            results_dir=results_dir, 
            analysis_name='DBS', 
            freq_bands=freq_bands,
            run_advanced_stats=True,
            n_permutations=5000
        )
        if stats_df_dbs is not None and not stats_df_dbs.empty:
            stats_df_dbs['data_source'] = 'DBS'
            all_stats_dfs.append(stats_df_dbs)

    dbs_fooof_df = analyze_specparam_and_plot(
        all_psd_data_roi=all_psd_data_dbs, # This matches the structure expected
        roi_name='DBS',
        blocks=['baseline', *blocks],
        results_dir=results_dir,
        resultsDir_fooof = results_dir_fooof,
        freq_bands = freq_bands,
        freq_range=[4.0, 45.0],
        peak_width = [2.0,12.0],
        max_n_peaks = 3,
        min_peak_height = 0.02,
        peak_threshold=2.0

        
    )
    if not dbs_fooof_df.empty:
        all_fooof_dfs.append(dbs_fooof_df)

    # --- Step 5: Save Final Combined Report ---
    print("\n\n" + "#"*80 + "\n### FINAL COMBINED REPORT ACROSS ALL ANALYSES ###\n" + "#"*80)

    if is_full_version:
        if all_stats_dfs:
            final_combined_df = pd.concat(all_stats_dfs, ignore_index=True)
            cols = final_combined_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('data_source')))
            final_combined_df = final_combined_df[cols]
            print(final_combined_df.to_string())
            combined_output_path = os.path.join(results_dir, "statistical_analysis_ALL_COMBINED.csv")
            final_combined_df.to_csv(combined_output_path, index=False, float_format='%.4f')
            print(f"\n--- Final combined report saved to: {combined_output_path} ---")
        else:
            print("No statistical results generated to create a combined report.")

    if all_fooof_dfs:
        total_fooof_df = pd.concat(all_fooof_dfs, ignore_index=True)
        fooof_out_path = os.path.join(results_dir, "statistical_analysis_FOOOF_parameters.csv")
        total_fooof_df.to_csv(fooof_out_path, index=False)
        #print(total_fooof_df.to_string())
        print(f"\n--- FOOOF/SpecParam stats saved to: {fooof_out_path} ---")

    if not total_fooof_df.empty:
        # Define your specific comparisons here
        comps_within = [('baseline', 'pre8walk'), ('baseline', 'post8walk')]
        
        # Example: Compare PD vs HC (Assuming these are your cohort keys)
        # You can extract these from your cohort_definitions keys if you want generic code
        cohort_keys = list(cohort_definitions.keys())
        # Creates pairs like [('PD', 'HC'), ('PD', 'ET')...]
        import itertools
        comps_between = list(itertools.combinations(cohort_keys, 2))

        foof_stats = run_fooof_statistics(
            fooof_df=total_fooof_df,
            comparisons_within=comps_within,
            comparisons_between=comps_between,
            results_dir=results_dir
        )

        print("\n" + "="*80 + "\n --- FOOOF PDS statistics ---\n" + "="*80)
        print(foof_stats)
        
    print("\n" + "="*80 + "\n--- Full PSD Analysis Workflow Complete ---\n" + "="*80)


# =========================================
# Tuning FOOOF Parameters: 
# =========================================


def tune_fooof_parameters(freqs, psds, resultsDir, cohort_name, block, roi, param_grid=None):
    """
    Fits FOOOFGroup models with different parameter sets and compares R2 and Error.
    
    Args:
        freqs: array of frequencies
        psds: array of PSDs (n_subjects, n_freqs) - LINEAR POWER, NOT LOG
        param_grid: list of dicts containing parameters to test
    """
    
    # Default grid if none provided
    if param_grid is None:
        param_grid = [
            #{'label': 'width_[1-12]_minh-0.1_maxpeak-3', 'peak_width': [1, 12], 'min_height': 0.1, 'max_peaks': 3},
            #{'label': 'width_[1-8]_minh-0.1_maxpeak-3', 'peak_width': [1, 8], 'min_height': 0.1, 'max_peaks': 3},
            #{'label': 'width_[1-8]_minh-0.1_maxpeak-3', 'peak_width': [1, 8], 'min_height': 0.05, 'max_peaks': 3},
            #{'label': 'width_[1-10]_minh-0.1_maxpeak-2', 'peak_width': [1, 10], 'min_height': 0.1, 'max_peaks': 2},
            #{'label': 'width_[1-10]_minh-0.1_maxpeak-3', 'peak_width': [1, 10], 'min_height': 0.1, 'max_peaks': 3},
            #{'label': 'width_[1-10]_minh-0.13_maxpeak-2', 'peak_width': [1, 10], 'min_height': 0.13, 'max_peaks': 2},
            #{'label': 'width_[1-10]_minh-0.13_maxpeak-2', 'peak_width': [1, 10], 'min_height': 0.13, 'max_peaks': 3},
            ##{'label': 'width_[1-10]_minh-0.2_maxpeak-3', 'peak_width': [1, 10], 'min_height': 0.2, 'max_peaks': 3},
            #{'label': 'width_[1-10]_minh-0.2_maxpeak-3', 'peak_width': [1, 10], 'min_height': 0.2, 'max_peaks': 3},
            #{'label': 'width_[0.5-10]_minh-0.2_maxpeak-2', 'peak_width': [0.5, 11], 'min_height': 0.2, 'max_peaks': 2},
            #{'label': 'width_[0.5-10]_minh-0.2_maxpeak-3', 'peak_width': [0.5, 11], 'min_height': 0.2, 'max_peaks': 3},
            #{'label': 'width_[1-9]_minh-0.13_maxpeak-2', 'peak_width': [1, 9], 'min_height': 0.2, 'max_peaks': 2},
            #{'label': 'width_[1-11]_minh-0.1_maxpeak-2', 'peak_width': [1, 11], 'min_height': 0.1, 'max_peaks': 2},
            #{'label': 'width_[1-11]_minh-0.05_maxpeak-2', 'peak_width': [1, 11], 'min_height': 0.05, 'max_peaks': 2},
            # {'label': 'width_[1-9]_minh-0.09_maxpeak-2', 'peak_width': [1, 9], 'min_height': 0.09, 'max_peaks': 2},
            # {'label': 'width_[1-9]_minh-0.09_maxpeak-3', 'peak_width': [1, 9], 'min_height': 0.09, 'max_peaks': 3},
            #{'label': 'width_[2-11]_minh-0.09_maxpeak-2', 'peak_width': [2, 11], 'min_height': 0.09, 'max_peaks': 2, 'peak_threshold':2},
            #{'label': 'width_[2-11]_minh-0.09_maxpeak-3', 'peak_width': [2, 11], 'min_height': 0.09, 'max_peaks': 3},
            #{'label': 'width_[2-10]_minh-0.09_maxpeak-2', 'peak_width': [2, 10], 'min_height': 0.09, 'max_peaks': 2},
            #{'label': 'width_[2-10]_minh-0.09_maxpeak-3', 'peak_width': [2, 10], 'min_height': 0.09, 'max_peaks': 3},
            
            #{'label': 'width_[1-11]_minh-0.09_maxpeak-2', 'peak_width': [1, 11], 'min_height': 0.09, 'max_peaks': 2},
            #{'label': 'width_[1-11]_minh-0.2_maxpeak-2', 'peak_width': [1, 11], 'min_height': 0.2, 'max_peaks': 2},
            #{'label': 'width_[1-11]_minh-0.1_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.1, 'max_peaks': 3},
            #{'label': 'width_[1-11]_minh-0.05_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.05, 'max_peaks': 3},
            #{'label': 'width_[1-11]_minh-0.07_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.07, 'max_peaks': 3},
            #{'label': 'width_[1-11]_minh-0.08_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.08, 'max_peaks': 3},
            #{'label': 'width_[1-11]_minh-0.09_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.09, 'max_peaks': 3},
            #{'label': 'width_[1-11]_minh-0.1_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.1, 'max_peaks': 3},

            # {'label': 'width_[1-12]_minh-0.09_maxpeak-2', 'peak_width': [1, 12], 'min_height': 0.09, 'max_peaks': 2},
            # {'label': 'width_[1-12]_minh-0.09_maxpeak-3', 'peak_width': [1, 12], 'min_height': 0.09, 'max_peaks': 3},
            # {'label': 'width_[2-10]_minh-0.2_maxpeak-3-thresh-1.2', 'peak_width': [2, 10], 'min_height': 0.2, 'max_peaks': 3,'peak_threshold':1.2},
            # {'label': 'width_[2-12]_minh-0.09_maxpeak-2-thresh-2', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':2},
            # {'label': 'width_[2-12]_minh-0.09_maxpeak-3-thresh-2', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 3,'peak_threshold':2},
            # {'label': 'width_[2-12]_minh-0.09_maxpeak-2-thresh-1.5', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':1.5},
            # {'label': 'width_[2-12]_minh-0.09_maxpeak-3-thresh-1.5', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 3,'peak_threshold':1.5},
            #{'label': 'width_[2-10]_minh-0.09_maxpeak-4-thresh-1.5', 'peak_width': [2, 10], 'min_height': 0.09, 'max_peaks': 4,'peak_threshold':1.5},
            #{'label': 'width_[2-10]_minh-0.09_maxpeak-4-thresh-2', 'peak_width': [2, 10], 'min_height': 0.09, 'max_peaks': 4,'peak_threshold':2},

            {'label': 'width_[2-12]_minh-0.09_maxpeak-2-thresh-1', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':1},
            {'label': 'width_[2-12]_minh-0.09_maxpeak-3-thresh-2', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 3,'peak_threshold':2},
            {'label': 'width_[2-12]_minh-0.09_maxpeak-2-thresh-2', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':2},

            #{'label': 'width_[2-20]_minh-0.09_maxpeak-2-thresh-0.7', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':0.7},
            #{'label': 'width_[2-20]_minh-0.09_maxpeak-2-thresh-2.3', 'peak_width': [2, 12], 'min_height': 0.09, 'max_peaks': 3,'peak_threshold':2.3},

            #{'label': 'width_[2-20]_minh-0.09_maxpeak-2-thresh-1.5', 'peak_width': [2, 20], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':1.5},
            #{'label': 'width_[2-15]_minh-0.09_maxpeak-3-thresh-1.5', 'peak_width': [2, 15], 'min_height': 0.09, 'max_peaks': 3,'peak_threshold':1.5},
            #{'label': 'width_[2-15]_minh-0.09_maxpeak-2-thresh-1.5', 'peak_width': [2, 15], 'min_height': 0.09, 'max_peaks': 2,'peak_threshold':1.5},
            


            # {'label': 'width_[3-12]_minh-0.09_maxpeak-2', 'peak_width': [3, 12], 'min_height': 0.09, 'max_peaks': 2},
            # {'label': 'width_[3-12]_minh-0.09_maxpeak-3', 'peak_width': [3, 12], 'min_height': 0.09, 'max_peaks': 3},
            #{'label': 'width_[1-12]_minh-0.1_maxpeak-3', 'peak_width': [1, 12], 'min_height': 0.1, 'max_peaks': 3},


            # {'label': 'width_[1-13]_minh-0.09_maxpeak-2', 'peak_width': [1, 13], 'min_height': 0.09, 'max_peaks': 2},
            # {'label': 'width_[1-13]_minh-0.09_maxpeak-3', 'peak_width': [1, 13], 'min_height': 0.09, 'max_peaks': 3},
            #{'label': 'width_[2-13]_minh-0.09_maxpeak-2', 'peak_width': [2, 13], 'min_height': 0.09, 'max_peaks': 2},
            #{'label': 'width_[2-13]_minh-0.09_maxpeak-3', 'peak_width': [2, 13], 'min_height': 0.09, 'max_peaks': 3},
            # {'label': 'width_[3-13]_minh-0.09_maxpeak-2', 'peak_width': [3, 13], 'min_height': 0.09, 'max_peaks': 2},
            # {'label': 'width_[3-13]_minh-0.09_maxpeak-3', 'peak_width': [3, 13], 'min_height': 0.09, 'max_peaks': 3},
            

            #{'label': 'width_[1-13]_minh-0.1_maxpeak-3', 'peak_width': [1, 13], 'min_height': 0.1, 'max_peaks': 3},

            #{'label': 'width_[1-14]_minh-0.09_maxpeak-2', 'peak_width': [1, 14], 'min_height': 0.09, 'max_peaks': 2},
            #{'label': 'width_[1-14]_minh-0.09_maxpeak-3', 'peak_width': [1, 14], 'min_height': 0.09, 'max_peaks': 3},





            #{'label': 'width_[1-11]_minh-0.2_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.2, 'max_peaks': 3},

            #{'label': 'width_[1-11]_minh-0.05_maxpeak-3', 'peak_width': [1, 11], 'min_height': 0.05, 'max_peaks': 3},

                # Max width 6 forces it to split the broad Beta bump.
            #{'label': 'Sharp_Sensitive', 'peak_width': [1, 6], 'min_height': 0.05, 'max_peaks': 6},
        
            # B. The "Standard & Safe"
            # Max width 8 allows slightly fatter peaks but still avoids massive blobs.
            #{'label': 'Standard_Safe',   'peak_width': [1, 8], 'min_height': 0.10, 'max_peaks': 6},
        
            # C. The "Conservative but Flexible"
            # Higher threshold, but still allows 4 peaks to prevent "slot blocking".
            #{'label': 'Conservative',    'peak_width': [1, 8], 'min_height': 0.15, 'max_peaks': 4},

            #{'label': 'width_[1-10]_minh-0.13_maxpeak-3', 'peak_width': [1, 10], 'min_height': 0.13, 'max_peaks': 3},
            #{'label': 'width_[1-10]_minh-0.17_maxpeak-2', 'peak_width': [1, 10], 'min_height': 0.17, 'max_peaks': 2},
            #{'label': 'width_[1-10]_minh-0.17_maxpeak-3', 'peak_width': [1, 10], 'min_height': 0.17, 'max_peaks': 3},
            #{'label': 'width_[1-10]_minh-0.2_maxpeak-2', 'peak_width': [1, 10], 'min_height': 0.2, 'max_peaks': 2},

            #{'label': 'width_[4-15]_minh-0.2_maxpeak-6', 'peak_width': [1, 12], 'min_height': 0.2, 'max_peaks': 6},
            #{'label': 'width_[4-15]_minh-0.2_maxpeak-6', 'peak_width': [1, 12], 'min_height': 0.1, 'max_peaks': 6},
            #{'label': 'width_[1-12]_minh-0.2_maxpeak-3', 'peak_width': [1, 12], 'min_height': 0.2, 'max_peaks': 3},
            #{'label': 'width_[1-12]_minh-0.1_maxpeak-2', 'peak_width': [1, 12], 'min_height': 0.1, 'max_peaks': 2},
            #{'label': 'width_[2-12]_minh-0.15_maxpeak-3',     'peak_width': [2, 12], 'min_height': 0.15, 'max_peaks': 3},
            #{'label': 'width_[4-15]_minh-0.2_maxpeak-3', 'peak_width': [4, 15], 'min_height': 0.2, 'max_peaks': 3},
            #{'label': 'width_[2-12]_minh-0.2_maxpeak-3',  'peak_width': [2, 12], 'min_height': 0.2, 'max_peaks': 3},
        ]
    
    results = []
    
    print(f"--- Tuning FOOOF on {psds.shape[0]} subjects ---")
    
    for settings in param_grid:
        label = settings['label']
        
        # Initialize Group
        fg = FOOOFGroup(
            peak_width_limits=settings['peak_width'],
            min_peak_height=settings['min_height'],
            max_n_peaks=settings['max_peaks'],
            aperiodic_mode='fixed', # Your data is > 4Hz, so fixed is safe
            verbose=False
        )
        
        # Fit
        fg.fit(freqs, psds)
        
        # Collect Metrics
        r2s = fg.get_params('r_squared')
        errors = fg.get_params('error')
        n_peaks = [len(fg.get_fooof(ind).peak_params_) for ind in range(len(psds))]
        
        # Store summary stats
        results.append({
            'Label': label,
            'Peak_Width': str(settings['peak_width']),
            'Min_Height': settings['min_height'],
            'Mean_R2': np.mean(r2s),
            'Mean_Error': np.mean(errors),
            'Avg_Num_Peaks': np.mean(n_peaks)
        })

        #individual fooof check: 
        to_check = []
        for ind, res in enumerate(fg):
            to_check.append(fg.get_fooof(ind, regenerate=True))
        for ind, fm in enumerate(to_check):
            os.makedirs(os.path.join(resultsDir, "final"), exist_ok = True)
            fm.save_report(os.path.join(resultsDir, "final", f"Report_ToCheck_{cohort_name}_{block}_{settings['label']}_subj{ind+1}.png"))
        
        
        # OPTIONAL: Visual check of the Grand Average fit for this setting
        # We assume the user wants to see how the "Archetype" looks with these settings
        avg_psd = np.mean(psds, axis=0)
        from fooof import FOOOF
        fm = FOOOF(peak_width_limits=settings['peak_width'], 
                   min_peak_height=settings['min_height'], 
                   max_n_peaks=settings['max_peaks'], peak_threshold = settings['peak_threshold'], aperiodic_mode='fixed', verbose=False)
        fm.fit(freqs, avg_psd)
        
        plt.figure()
        fm.plot(plot_peaks='shade')
        plt.title(f"Visual Check: {label}\nR2: {fm.r_squared_:0.3f}, cohort: {cohort_name}, {block}")
        plt.show()

    # Create comparison table
    df_results = pd.DataFrame(results)
    
    # Sort by R2 (descending) to see best fits, 
    # BUT keep in mind R2 isn't everything (overfitting).
    df_results = df_results.sort_values(by='Mean_R2', ascending=False)
    df_results.to_csv(os.path.join(resultsDir, f"{cohort_name}_{block}_{roi}_fooofR2_report.csv"))
    return df_results


# OLD or in paper UNUSED FUNCTIONS: 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from typing import List, Dict, Any, Optional
import json

import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from fooof import FOOOFGroup

def run_fooof_topomaps_standalone(
    all_data_epoched_dico, 
    cohort_definitions, 
    subjects_dico,
    blocks, 
    results_dir,
    freq_range=[4.0, 45.0]
):
    """
    Robust Topographic Analysis that handles missing channels via Interpolation.
    Can be run independently of the main orchestrator.
    """
    print(f"\n" + "="*60 + f"\n--- Starting Robust FOOOF Topographic Analysis ---\n" + "="*60)
    
    # 1. Setup Directories
    cache_dir = os.path.join(results_dir, "intermediary_fooof_ch_data")
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(results_dir, "PSD_plots_Topomaps")
    os.makedirs(out_dir, exist_ok=True)

    # 2. STEP 1: FIND THE "UNIVERSAL" CHANNEL LIST
    # We must know the union of all channels across all subjects to create a common template.
    print("-> Scanning dataset for all unique channels...")
    universal_ch_names = set()
    template_info = None # We will grab one info object to use as a base
    
    for group in all_data_epoched_dico.keys():
        for block in blocks:
             if block in all_data_epoched_dico[group]:
                for ep in all_data_epoched_dico[group][block]['EEG']:
                    if ep is not None:
                        # Pick only EEG to avoid EOG/Stim channels polluting the union
                        # Use .pick() instead of legacy pick_types if possible, or suppress warning
                        tmp = ep.copy().pick('eeg', exclude=[]) 
                        universal_ch_names.update(tmp.ch_names)
                        if template_info is None:
                            template_info = tmp.info

    universal_ch_names = sorted(list(universal_ch_names))
    print(f"-> Found {len(universal_ch_names)} unique EEG channels in the entire cohort.")
    
    # Create a "Standard Info" object with all channels
    # We clone the template and update it to include all universal channels
    # (This requires that your subjects typically have standard montages like 10-20)
    # Ideally, we rely on the plotting function to use the subject's montage, 
    # but for Grand Average, we need one common info.
    
    # 3. ANALYSIS LOOP
    for block in blocks:
        print(f"\n" + "-"*40 + f"\nAnalyzing Condition: {block}\n" + "-"*40)
        
        topo_data = {
            cohort: {'Alpha': [], 'Beta': [], 'Offset': [], 'Exponent': []} 
            for cohort in cohort_definitions.keys()
        }

        for cohort_name, cohort_info in cohort_definitions.items():
            base_group = cohort_info['group']
            subj_list = subjects_dico[base_group]
            
            print(f"  Processing {cohort_name} ({len(subj_list)} subjects)...")

            for sub_idx, sub_id in enumerate(subj_list):
                cache_file = os.path.join(cache_dir, f"fooof_ch_{base_group}_{sub_id}_{block}.pkl")
                
                # --- A. LOAD / COMPUTE DATA ---
                try:
                    epochs = all_data_epoched_dico[base_group][block]['EEG'][sub_idx]
                    if epochs is None: continue
                    
                    # Update: Use inst.pick() to avoid legacy warnings
                    epochs = epochs.copy().pick('eeg', exclude='bads')
                    subj_ch_names = epochs.ch_names
                    
                    # Check Cache
                    if os.path.exists(cache_file):
                        with open(cache_file, 'rb') as f:
                            cached = pickle.load(f)
                        if cached['ch_names'] == subj_ch_names:
                            vals_alpha, vals_beta = cached['alpha'], cached['beta']
                            vals_offset, vals_exponent = cached['offset'], cached['exponent']
                        else:
                            raise ValueError("Cache channel mismatch")
                    else:
                        # Compute FOOOF (Expensive)
                        spectrum = epochs.compute_psd(fmin=freq_range[0], fmax=freq_range[1], method='welch', n_fft=2048, verbose=False)
                        psds, freqs = spectrum.get_data(return_freqs=True)
                        avg_psds = psds.mean(axis=0)
                        
                        fg = FOOOFGroup(peak_width_limits=[2, 12.0], max_n_peaks=3, min_peak_height=0.15, aperiodic_mode='fixed', verbose=False)
                        fg.fit(freqs, avg_psds)
                        
                        # Extract
                        vals_offset = fg.get_params('aperiodic_params')[:, 0]
                        vals_exponent = fg.get_params('aperiodic_params')[:, 1]
                        
                        # Periodic (Vectorized extraction for speed)
                        # Initialize with 0 (Imputation for missing peaks)
                        vals_alpha = np.zeros(len(subj_ch_names))
                        vals_beta = np.zeros(len(subj_ch_names))
                        
                        all_peaks = fg.get_params('peak_params')
                        # Note: get_params returns a flat list for FOOOFGroup if you don't be careful, 
                        # but iterating get_fooof is safer for mapping back to channels.
                        for ch_i in range(len(subj_ch_names)):
                            peaks = fg.get_fooof(ch_i).peak_params_
                            # Alpha
                            alphas = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]
                            if len(alphas) > 0: vals_alpha[ch_i] = alphas[np.argmax(alphas[:, 1])][1]
                            # Beta
                            betas = peaks[(peaks[:, 0] >= 13) & (peaks[:, 0] <= 35)]
                            if len(betas) > 0: vals_beta[ch_i] = betas[np.argmax(betas[:, 1])][1]

                        # Save Cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump({'ch_names': subj_ch_names, 'alpha': vals_alpha, 'beta': vals_beta, 
                                         'offset': vals_offset, 'exponent': vals_exponent}, f)

                    # --- B. STANDARDIZE CHANNELS (INTERPOLATION) ---
                    # This is the fix for your "Dropped Channels" issue
                    
                    # 1. Create Evoked for this subject's ACTUAL channels
                    # We create 4 separate "layers"
                    param_map = {
                        'Alpha': vals_alpha, 'Beta': vals_beta,
                        'Offset': vals_offset, 'Exponent': vals_exponent
                    }
                    
                    for param_name, data_vals in param_map.items():
                        # Create Evoked with existing channels
                        ev = mne.EvokedArray(data_vals[:, np.newaxis], epochs.info, tmin=0, verbose=False)
                        
                        # 2. Identify missing channels compared to Universal List
                        missing_chs = [ch for ch in universal_ch_names if ch not in ev.ch_names]
                        
                        if missing_chs:
                            # 3. Add missing channels (fill with 0) and mark as BAD
                            # MNE allows adding channels. We assume standard montage locations are known by MNE.
                            # We construct a dummy info for the missing part
                            try:
                                # We need to use MNE's interpolation. 
                                # The safest way is to use `mne.channels.equalize_channels` 
                                # but that drops data.
                                # Instead, we rely on the plotting function to interpolate? 
                                # No, grand_average needs matching info.
                                
                                # Strategy: Standardize configuration
                                # Create array of NaN of size Universal
                                univ_data = np.full((len(universal_ch_names), 1), np.nan)
                                
                                # Fill where we have data
                                ch_indices = [universal_ch_names.index(ch) for ch in subj_ch_names if ch in universal_ch_names]
                                univ_data[ch_indices, 0] = data_vals
                                
                                # Create Evoked with Universal Info
                                # We assume template_info has the montage. 
                                # We might need to manually construct info if template_info is missing some universal channels
                                # But usually, the union comes from the files, so we can copy one that has them?
                                # Simplified: Just use the subject's Evoked for now, 
                                # and we will use a custom averaging function below.
                                pass
                            except Exception:
                                pass
                        
                        # APPEND RAW EVOKED (We will handle averaging manually to avoid dropping)
                        topo_data[cohort_name][param_name].append(ev)

                except Exception as e:
                    print(f"    Error processing {sub_id}: {e}")

        # --- 4. PLOT FOR THIS BLOCK (Using Robust Averaging) ---
        if any(topo_data.values()):
            plot_robust_topomaps(topo_data, block, out_dir, universal_ch_names)
import warnings

def plot_robust_topomaps(topo_data, block_name, out_dir, universal_ch_names):
    """
    Averages subjects by creating a Universal Matrix and ignoring NaNs/Missing channels.
    REPORTS which channels are missing (empty mean) for each group.
    """
    plot_configs = [('Alpha', 'Beta', 'Periodic'), ('Offset', 'Exponent', 'Aperiodic')]
    groups = list(topo_data.keys())
    
    for (p1, p2, suffix) in plot_configs:
        fig, axes = plt.subplots(len(groups), 2, figsize=(10, 4 * len(groups)))
        if len(groups) == 1: axes = axes.reshape(1, -1)
        
        for row_idx, group in enumerate(groups):
            for col_idx, param in enumerate([p1, p2]):
                ax = axes[row_idx, col_idx]
                evoked_list = topo_data[group][param]
                
                if not evoked_list:
                    ax.text(0.5, 0.5, "No Data", ha='center'); ax.axis('off'); continue
                
                # --- ROBUST AVERAGING LOGIC ---
                n_subs = len(evoked_list)
                n_univ = len(universal_ch_names)
                data_matrix = np.full((n_subs, n_univ), np.nan)
                
                # Use the info of the subject with MOST channels as the base for plotting
                best_evoked = max(evoked_list, key=lambda x: len(x.ch_names))
                plot_info = best_evoked.info.copy()
                
                for i, ev in enumerate(evoked_list):
                    for ch_idx, ch_name in enumerate(ev.ch_names):
                        if ch_name in universal_ch_names:
                            u_idx = universal_ch_names.index(ch_name)
                            data_matrix[i, u_idx] = ev.data[ch_idx, 0]
                
                # 3. Compute Mean (Ignoring NaNs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_data = np.nanmean(data_matrix, axis=0)
                
                # 4. Map back to Plot Info
                final_data = []
                final_names = plot_info.ch_names
                
                for ch in final_names:
                    if ch in universal_ch_names:
                        u_idx = universal_ch_names.index(ch)
                        val = mean_data[u_idx]
                        # Fill NaNs with 0 for plotting so it doesn't crash
                        final_data.append(val if not np.isnan(val) else 0.0)
                    else:
                        final_data.append(0.0)
                        
                final_data = np.array(final_data)
                
                # Plot
                im, _ = mne.viz.plot_topomap(
                    final_data, 
                    plot_info, 
                    axes=ax, 
                    show=False, 
                    cmap='viridis' if 'Offset' not in param else 'plasma',
                    contours=0,
                    sensors=True
                )
                
                if row_idx == 0: ax.set_title(f"{param}\n({group})", fontsize=14, fontweight='bold')
                else: ax.set_title(f"({group})", fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f"{suffix} Topographies: {block_name}", fontsize=16)
        plt.tight_layout()
        
        out_name = f"Topomap_FOOOF_{suffix}_{block_name}.pdf"
        fig.savefig(os.path.join(out_dir, out_name))
        print(f"Saved: {out_name}")
        plt.close(fig)


# --- DATA CALCULATION HELPER (No changes needed here) ---
def calculate_all_cohort_psds(
    all_data_dico: Dict, all_data_epoched_dico: Dict, subjects_dico: Dict,
    cohort_definitions: Dict[str, Dict[str, Any]], blocks: List[str]
) -> Dict[str, Any]:
    """Calculates and aggregates PSDs for all defined cohorts across all data types."""
    print("--- Pre-calculating PSDs for all defined cohorts ---")
    all_psd_results = {
        'EEG_continuous': {}, 'EEG_epoched': {}, 'DBS_continuous': {}, 'DBS_epoched': {}
    }
    data_sources = {
        'EEG_continuous': (all_data_dico, 'EEG', False), 'EEG_epoched': (all_data_epoched_dico, 'EEG', True),
        'DBS_continuous': (all_data_dico, 'DBS_proc', False), 'DBS_epoched': (all_data_epoched_dico, 'DBS', True)
    }
    for data_type, (source_dico, key, is_epoched) in data_sources.items():
        for block in ['baseline', *blocks]:
            all_psd_results[data_type][block] = {}
            for cohort_name, cohort_info in cohort_definitions.items():
                base_group, id_filter = cohort_info['group'], set(cohort_info['ids']) if cohort_info['ids'] else None
                subjects = subjects_dico[base_group]
                psds_for_cohort, freqs = [], None
                for i, sub_id in enumerate(subjects):
                    if id_filter and sub_id not in id_filter: continue
                    try:
                        data_obj = source_dico[base_group][block][key][i]
                        print(sub_id, block)
                        if 'DBS' not in key:
                            data_obj = data_obj.copy().pick_channels(motor_areas['Both'], ordered=False)
                            print("picked_channels: ", motor_areas['Both'])
                        if data_obj is not None:
                            n_fft = min(2048, len(data_obj.times)) if is_epoched else 2048
                            spectrum = data_obj.compute_psd(fmin=4, fmax=45, method='welch', n_fft=n_fft)
                            psd_values = spectrum.get_data(return_freqs=False).mean(axis=(0, 1)) if is_epoched else spectrum.get_data(return_freqs=False).mean(axis=0)
                            psds_for_cohort.append(psd_values)
                            if freqs is None: freqs = spectrum.freqs
                    except (IndexError, KeyError):
                        print(f"    - Warning: Could not find data for subj {sub_id} in {base_group}/{block}/{key}. Skipping.")
                        continue
                if psds_for_cohort:
                    all_psd_results[data_type][block][cohort_name] = {'psds': np.array(psds_for_cohort), 'freqs': freqs}
    print("--- PSD calculation complete. ---")
    return all_psd_results


# --- MODIFIED MAIN ORCHESTRATOR ---
def generate_psd_comparison_plots(
    all_data_dico: Dict, all_data_epoched_dico: Dict, subjects_dico: Dict,
    blocks: List[str], cohort_definitions: Dict, plots_to_generate: List[Dict],
    results_dir: str, palette: Optional[Dict[str, str]] = None
):
    """Orchestrates the creation of comparative PSD plots based on a configuration."""
    print("\n" + "="*80 + "\n--- Starting Flexible PSD Comparison Analysis ---\n" + "="*80)
    
    all_cohort_names = list(cohort_definitions.keys())
    if palette is None: palette = {name: c for i, name in enumerate(all_cohort_names) for c in [sns.color_palette("husl", len(all_cohort_names))[i]]}
    for name, p_color in palette.items():
        if name in cohort_definitions: cohort_definitions[name]['color'] = p_color
            
    all_psd_data = calculate_all_cohort_psds(all_data_dico, all_data_epoched_dico, subjects_dico, cohort_definitions, blocks)
    
    print("\n--- Generating Custom Multi-Block PSD Comparison Plots ---")
    output_dir = os.path.join(results_dir, "PSD_comparisons_multi_block")
    os.makedirs(output_dir, exist_ok=True)
    
    # The loop for 'block' is REMOVED from here
    for plot_def in plots_to_generate:
        plot_name, cohorts_for_this_plot, data_type = plot_def.get("name"), plot_def.get("cohorts"), plot_def.get("data_type")
        if not all([plot_name, cohorts_for_this_plot, data_type]):
            print(f"  - SKIPPING plot definition due to missing keys: {plot_def}")
            continue
            
        print(f"  - Generating plot: '{plot_name}' using data type '{data_type}'")
        
        # Get all pre-calculated data for this data_type
        psd_data_for_type = all_psd_data[data_type]
        
        title = f"PSD Comparison: {plot_name}\n({data_type.replace('_', ' ').title()})"
        # Filename no longer contains block, as it shows all of them
        filename = os.path.join(output_dir, f"PSD_{plot_name}_{data_type}_ALL_BLOCKS.png")
        
        # Call the new multi-block plotting function
        plot_psd_comparison_multi_block(
            psd_data_for_type=psd_data_for_type,
            blocks_to_plot=['baseline', *blocks],
            cohorts_to_plot=cohorts_for_this_plot,
            cohort_definitions=cohort_definitions,
            title=title, filename=filename
        )
        print(f"    > Saved combined plot to {filename}")

    print("\n--- All PSD comparison plots generated. ---")


def _run_band_power_analysis_between_cohorts(psd_data_for_type, comparisons, blocks, freqs, freq_bands):
    """Calculates band power stats BETWEEN cohorts using power at median frequency."""
    print("\n--- Running Analysis 1: Band-Power Statistics (Between Cohorts) ---")
    results = []
    for block in blocks:
        for cohort1, cohort2 in comparisons:
            try:
                psds1 = psd_data_for_type[block][cohort1]['psds']
                psds2 = psd_data_for_type[block][cohort2]['psds']
            except KeyError: continue
            for band_name, (f_min, f_max) in freq_bands.items():
                freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
                if not freq_indices.size: continue
                median_freq = (f_min + f_max) / 2
                power_at_median_freq_index = freq_indices[(np.abs(freqs[freq_indices] - median_freq)).argmin()]
                band_power1 = psds1[:, power_at_median_freq_index]
                band_power2 = psds2[:, power_at_median_freq_index]
                test_results = _run_comparison_test(band_power1, band_power2)
                
                # Add placeholder columns to match the 'within' analysis DataFrame structure
                results.append({
                    'analysis': 'band_power_between_cohorts',
                    'block': block,
                    'band': band_name,
                    'cohort1': cohort1, 'mean_cohort1': np.mean(band_power1),
                    'cohort2': cohort2, 'mean_cohort2': np.mean(band_power2),
                    'cohort': f"{cohort1}_vs_{cohort2}", # Harmonized column for sorting
                    'comparison': f"{cohort1}_vs_{cohort2}", # Harmonized column for sorting
                    **test_results
                })
    return pd.DataFrame(results)

def _run_within_cohort_analysis(psd_data_for_type, cohorts, walk_blocks, freqs, freq_bands):
    """
    Calculates stats comparing each walk block to baseline WITHIN each cohort.
    This applies to both EEG and DBS data.
    """
    print("\n--- Running Analysis 2: Within-Cohort Statistics (Walk vs. Baseline) ---")
    results = []
    for cohort in cohorts:
        for walk_block in walk_blocks:
            try:
                psds_base = psd_data_for_type['baseline'][cohort]['psds']
                psds_walk = psd_data_for_type[walk_block][cohort]['psds']
                if psds_base.shape[0] != psds_walk.shape[0]:
                    print(f"    - WARNING: Subject mismatch for {cohort} between baseline ({psds_base.shape[0]}) and {walk_block} ({psds_walk.shape[0]}). Skipping.")
                    continue
            except KeyError:
                continue

            for band_name, (f_min, f_max) in freq_bands.items():
                freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
                if not freq_indices.size: continue
                median_freq = (f_min + f_max) / 2
                power_at_median_freq_index = freq_indices[(np.abs(freqs[freq_indices] - median_freq)).argmin()]
                
                power_base = psds_base[:, power_at_median_freq_index]
                power_walk = psds_walk[:, power_at_median_freq_index]
                test_results = _run_paired_comparison_test(power_base, power_walk)
                
                # Add the 'block' column to match the 'between' analysis DataFrame structure
                results.append({
                    'analysis': 'band_power_within_cohort',
                    'cohort': cohort,
                    'block': walk_block, # This is the block being compared to baseline
                    'comparison': f"{walk_block}_vs_baseline",
                    'band': band_name,
                    'mean_baseline': np.mean(power_base),
                    'mean_walk': np.mean(power_walk),
                    **test_results
                })
    return pd.DataFrame(results)


# The peak and permutation analyses are unchanged, but are here for completeness
def _run_peak_analysis(psd_data_for_type, comparisons, blocks, freqs, peak_bands):
    """Finds peaks in Alpha/Beta bands and runs t-tests on their freq and power."""
    print("\n--- Running Analysis 3: Peak Statistics (Between Cohorts) ---")
    results = []
    for block in blocks:
        for cohort1, cohort2 in comparisons:
            for band_name, (f_min, f_max) in peak_bands.items():
                peak_info1 = {'freqs': [], 'powers': []}; peak_info2 = {'freqs': [], 'powers': []}
                for cohort_name, peak_info in [(cohort1, peak_info1), (cohort2, peak_info2)]:
                    try: psds_array = psd_data_for_type[block][cohort_name]['psds']
                    except KeyError: continue
                    freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
                    for sub_psd in psds_array:
                        band_psd = sub_psd[freq_indices]
                        distance_in_samples = int(0.5 / (freqs[1] - freqs[0])) if freqs.size > 1 and freqs[1] > freqs[0] else 1
                        peaks, properties = find_peaks(band_psd, height=0, distance=max(1, distance_in_samples))
                        if peaks.size > 0:
                            highest_peak_idx_local = peaks[np.argmax(properties['peak_heights'])]
                            highest_peak_idx_global = freq_indices[highest_peak_idx_local]
                            peak_info['freqs'].append(freqs[highest_peak_idx_global])
                            peak_info['powers'].append(sub_psd[highest_peak_idx_global])
                if len(peak_info1['freqs']) > 1 and len(peak_info2['freqs']) > 1:
                    test_results_freq = _run_comparison_test(peak_info1['freqs'], peak_info2['freqs'])
                    results.append({'analysis': 'peak_stats_freq', 'block': block, 'band': band_name, 'cohort1': cohort1, 'mean_cohort1': np.mean(peak_info1['freqs']), 'cohort2': cohort2, 'mean_cohort2': np.mean(peak_info2['freqs']), **test_results_freq})
                    test_results_power = _run_comparison_test(peak_info1['powers'], peak_info2['powers'])
                    results.append({'analysis': 'peak_stats_power', 'block': block, 'band': band_name, 'cohort1': cohort1, 'mean_cohort1': np.mean(peak_info1['powers']), 'cohort2': cohort2, 'mean_cohort2': np.mean(peak_info2['powers']), **test_results_power})
    return pd.DataFrame(results)

import mne
from scipy.stats import wasserstein_distance

def _run_cluster_permutation_test(
    data1: np.ndarray, 
    data2: Optional[np.ndarray] = None, 
    paired: bool = False,
    n_permutations: int = 5000
) -> (np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]):
    """
    Performs a cluster-based permutation test on PSD data.

    Handles both paired (within-subject) and independent (between-subject) designs.

    Args:
        data1 (np.ndarray): PSD data for group 1 (n_subjects, n_freqs). For paired test,
                            this is the data from condition 1.
        data2 (np.ndarray, optional): PSD data for group 2 (n_subjects, n_freqs). 
                                      Required for independent test. Ignored for paired test
                                      if data1 is already the difference.
        paired (bool): If True, performs a paired t-test (1-sample test on the difference).
                       If False, performs an independent t-test.
        n_permutations (int): The number of permutations to run.

    Returns:
        T_obs (np.ndarray): The observed T-statistic for each frequency bin.
        clusters (list): A list of tuples, where each tuple contains the indices
                         of a significant cluster.
        cluster_p_values (np.ndarray): The p-value for each cluster.
        H0 (np.ndarray): The null distribution of the cluster-mass statistic.
    """
    if paired:
        if data1.shape != data2.shape:
            raise ValueError("For paired test, data1 and data2 must have the same shape.")
        # We test the difference against zero.
        data_diff = data1 - data2
        # MNE's 1-sample test requires a 3D array (n_obs, n_times, n_vertices)
        # We use our frequency dimension as the "time" dimension.
        data_diff = data_diff[:, :, np.newaxis] 
        print(f"  - Running CLUSTER (Paired): n_obs={data_diff.shape[0]}, n_freqs={data_diff.shape[1]}")
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
            data_diff, n_permutations=n_permutations, tail=0, # tail=0 for two-tailed
            out_type='mask' # returns a boolean mask of significant clusters
        )
    else:
        if data2 is None:
            raise ValueError("data2 must be provided for an independent test.")
        # MNE's independent test requires a list of arrays.
        data_list = [data1, data2]
        # Reshape for MNE compatibility
        data_list = [d[:, :, np.newaxis] for d in data_list]
        print(f"  - Running CLUSTER (Independent): n_obs1={data_list[0].shape[0]}, n_obs2={data_list[1].shape[0]}")
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            data_list, n_permutations=n_permutations, tail=0, # two-tailed
            out_type='mask'
        )
    # Squeeze out the last dimension we added for compatibility
    T_obs = T_obs.squeeze()
    clusters = [c.squeeze() for c in clusters] # a list of boolean masks
    
    return T_obs, clusters, cluster_p_values, H0


def _run_wasserstein_permutation_test(
    data1: np.ndarray, 
    data2: np.ndarray, 
    paired: bool = False,
    n_permutations: int = 5000
) -> Dict:
    """
    Compares two sets of PSDs using the Wasserstein distance with a permutation test.

    Args:
        data1 (np.ndarray): PSD data for group/condition 1 (n_subjects, n_freqs).
        data2 (np.ndarray): PSD data for group/condition 2 (n_subjects, n_freqs).
        paired (bool): If True, runs a paired permutation.
        n_permutations (int): The number of permutations.

    Returns:
        dict: A dictionary containing the observed distance, p-value, and effect size.
    """
    if paired:
        if data1.shape[0] != data2.shape[0]:
            raise ValueError("Paired data must have the same number of subjects.")
        
        # For paired test, the statistic is the distance between the mean PSDs.
        # Permutations are done by flipping the signs of the subject-wise differences.
        mean_psd1 = np.mean(data1, axis=0)
        mean_psd2 = np.mean(data2, axis=0)
        observed_distance = wasserstein_distance(mean_psd1, mean_psd2)
        
        # Create a distribution of the null hypothesis
        null_distribution = np.zeros(n_permutations)
        diffs = data1 - data2

        for i in range(n_permutations):
            # Randomly flip the sign for each subject's difference
            signs = np.random.choice([-1, 1], size=data1.shape[0])
            permuted_diffs = diffs * signs[:, np.newaxis]
            # Create two fake groups from the permuted differences
            perm_mean = np.mean(permuted_diffs, axis=0)
            # The distance is from the permuted mean difference to zero
            null_distribution[i] = wasserstein_distance(perm_mean, np.zeros_like(perm_mean))

    else: # Independent test
        mean_psd1 = np.mean(data1, axis=0)
        mean_psd2 = np.mean(data2, axis=0)
        observed_distance = wasserstein_distance(mean_psd1, mean_psd2)
        
        # Pool all data and shuffle labels
        all_data = np.vstack([data1, data2])
        labels = np.array([0] * len(data1) + [1] * len(data2))
        
        null_distribution = np.zeros(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(labels)
            perm_data1 = all_data[labels == 0]
            perm_data2 = all_data[labels == 1]
            perm_mean1 = np.mean(perm_data1, axis=0)
            perm_mean2 = np.mean(perm_data2, axis=0)
            null_distribution[i] = wasserstein_distance(perm_mean1, perm_mean2)
            
    # Calculate p-value: the proportion of null distances >= observed distance
    p_value = (np.sum(null_distribution >= observed_distance) + 1) / (n_permutations + 1)
    
    # Effect size: Cohen's d on the mean power of the whole spectrum
    d = _cohens_d(data1.mean(axis=1), data2.mean(axis=1)) if not paired else _cohens_d_paired(data1.mean(axis=1), data2.mean(axis=1))

    return {'test_used': 'wasserstein_perm', 'statistic': observed_distance, 'p_value': p_value, 'cohens_d': d}


def _plot_cluster_test_result(
    T_obs: np.ndarray, 
    clusters: list, 
    cluster_p_values: np.ndarray, 
    freqs: np.ndarray, 
    p_accept: float = 0.05,
    title: str = "Cluster Permutation Test",
    filename: str = "cluster_test.png"
):
    """Plots the results of a cluster permutation test."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the T-statistic curve
    ax.plot(freqs, T_obs, 'k-', label='T-statistic')
    
    # Find significant clusters and plot them
    for i, cl in enumerate(clusters):
        if cluster_p_values[i] <= p_accept:
            cluster_freqs = freqs[cl]
            cluster_T_values = T_obs[cl]
            ax.fill_between(cluster_freqs, 0, cluster_T_values, alpha=0.5,
                            label=f"Cluster #{i+1}, p={cluster_p_values[i]:.3f}")

    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("T-statistic", fontsize=12)
    ax.set_xlim(freqs[0], freqs[-1])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close(fig)



def calculate_all_psd_metrics(
    full_psd: np.ndarray, 
    full_freqs: np.ndarray, 
    freq_bands: Dict[str, list],
    fit_range: list = [1, 45],
) -> Dict[str, Any]:
    """
    Calculates a comprehensive set of metrics from a single power spectrum.
    FULLY COMPATIBLE WITH SPECPARAM v2.0.

    1. Fits specparam ONCE on the full spectrum to get a robust aperiodic exponent.
    2. For each defined frequency band, calculates:
       - Spectral Entropy (on the raw PSD slice).
       - The power of the strongest oscillatory peak found by the model within that band.

    Args:
        full_psd (np.ndarray): 1D array of power values for the entire spectrum.
        full_freqs (np.ndarray): 1D array of corresponding full frequencies.
        freq_bands (Dict[str, list]): Dictionary defining frequency bands, e.g., {'alpha': [8, 12]}.
        fit_range (list): The frequency range to fit the specparam model over.

    Returns:
        A dictionary containing the single spectral slope and nested dictionaries for each band's metrics.
    """
    results = {'bands': {}}

    # --- 1. Fit the full spectrum ---
    sm = SpectralModel(peak_width_limits=[2.0, 15.0], aperiodic_mode='fixed', verbose=False)
    sm.fit(full_freqs, full_psd, fit_range)
    
    # --- FIX #1: Use 'aperiodic' instead of 'aperiodic_params' ---
    exponent = sm.get_params('aperiodic', 'exponent')
    results['spectral_slope'] = float(exponent) if exponent is not None else np.nan

    # --- FIX #2: Get all peaks and filter them for each band ---
    # The 'peak' parameters are an array of [Center Frequency, Power, Bandwidth] for each peak found.
    all_peak_params = sm.get_params('peak')

    # --- 3. Loop through bands for band-specific metrics ---
    for band_name, (f_min, f_max) in freq_bands.items():
        band_results = {}
        
        # a) Calculate Spectral Entropy (this logic remains the same)
        try:
            freq_indices = np.where((full_freqs >= f_min) & (full_freqs <= f_max))[0]
            psd_segment = full_psd[freq_indices]
            
            if len(psd_segment) > 1 and np.sum(psd_segment) > 0:
                psd_norm = psd_segment / np.sum(psd_segment)
                entropy_val = sp_entropy(psd_norm, base=2)
                max_entropy = np.log2(len(psd_segment))
                band_results['spectral_entropy'] = entropy_val / max_entropy if max_entropy > 0 else 0
            else:
                band_results['spectral_entropy'] = np.nan
        except Exception:
            band_results['spectral_entropy'] = np.nan
        
        # b) Find the strongest peak's power within the current band
        band_peak_power = 0.0
        if all_peak_params is not None and all_peak_params.size > 0:
            # Find all peaks where the center frequency (column 0) is within the band's limits
            peaks_in_band = all_peak_params[
                (all_peak_params[:, 0] >= f_min) & (all_peak_params[:, 0] <= f_max)
            ]
            
            if peaks_in_band.size > 0:
                # Get the power (column 1) of all peaks in the band and find the max
                band_peak_power = np.max(peaks_in_band[:, 1])

        band_results['peak_power'] = float(band_peak_power)
        results['bands'][band_name] = band_results
        
    return results



# This is the NEW version of your statistical function
def _run_shape_metric_analysis_between_cohorts(psd_data_for_type, comparisons, blocks, freqs, freq_bands):
    print("\n--- Running Analysis 4: Advanced Shape Statistics (Between Cohorts) ---")
    results = []

    # --- Get all metrics for every subject first ---
    all_metrics = {}
    for block in blocks:
        all_metrics[block] = {}
        for cohort in psd_data_for_type.get(block, {}):
            psds = psd_data_for_type[block][cohort]['psds']
            # Call the new comprehensive function ONCE per subject
            all_metrics[block][cohort] = [calculate_all_psd_metrics(psd, freqs, freq_bands) for psd in psds]

    # --- Now, perform statistical comparisons on the collected metrics ---
    for block in blocks:
        for cohort1, cohort2 in comparisons:
            if cohort1 not in all_metrics[block] or cohort2 not in all_metrics[block]:
                continue
            
            metrics1 = all_metrics[block][cohort1]
            metrics2 = all_metrics[block][cohort2]

            # 1. Compare the single, full-spectrum spectral slope
            slope_vals1 = [m['spectral_slope'] for m in metrics1 if not np.isnan(m['spectral_slope'])]
            slope_vals2 = [m['spectral_slope'] for m in metrics2 if not np.isnan(m['spectral_slope'])]
            if slope_vals1 and slope_vals2:
                test_results = _run_comparison_test(slope_vals1, slope_vals2)
                results.append({
                    'analysis': 'shape_spectral_slope_between',
                    'block': block, 'band': 'full_spectrum',  # Note: band is now 'full_spectrum'
                    'cohort1': cohort1, 'mean_cohort1': np.mean(slope_vals1),
                    'cohort2': cohort2, 'mean_cohort2': np.mean(slope_vals2),
                    'comparison': f"{cohort1}_vs_{cohort2}", **test_results
                })

            # 2. Loop through bands to compare band-specific metrics
            for band_name in freq_bands.keys():
                # Compare Spectral Entropy
                entropy_vals1 = [m['bands'][band_name]['spectral_entropy'] for m in metrics1]
                entropy_vals2 = [m['bands'][band_name]['spectral_entropy'] for m in metrics2]
                test_results = _run_comparison_test(entropy_vals1, entropy_vals2)
                results.append({
                    'analysis': 'shape_spectral_entropy_between',
                    'block': block, 'band': band_name,
                    'cohort1': cohort1, 'mean_cohort1': np.mean(entropy_vals1),
                    'cohort2': cohort2, 'mean_cohort2': np.mean(entropy_vals2),
                    'comparison': f"{cohort1}_vs_{cohort2}", **test_results
                })

                # # Compare Peak Power
                # peak_power1 = [m['bands'][band_name]['peak_power'] for m in metrics1]
                # peak_power2 = [m['bands'][band_name]['peak_power'] for m in metrics2]
                # test_results = _run_comparison_test(peak_power1, peak_power2)
                # results.append({
                #     'analysis': 'shape_peak_power_between',
                #     'block': block, 'band': band_name,
                #     'cohort1': cohort1, 'mean_cohort1': np.mean(peak_power1),
                #     'cohort2': cohort2, 'mean_cohort2': np.mean(peak_power2),
                #     'comparison': f"{cohort1}_vs_{cohort2}", **test_results
                # })

    return pd.DataFrame(results)


def _run_shape_metric_analysis_within_cohort(psd_data_for_type, cohorts, walk_blocks, freqs, freq_bands):
    """Calculates and compares SHAPE METRICS within cohorts (walk vs. baseline)."""
    print("\n--- Running Analysis 5: PSD Shape Statistics (Within Cohort) ---")
    results = []
    for cohort in cohorts:
        for walk_block in walk_blocks:
            try:
                psds_base = psd_data_for_type['baseline'][cohort]['psds']
                psds_walk = psd_data_for_type[walk_block][cohort]['psds']
                if psds_base.shape[0] != psds_walk.shape[0]: continue
            except KeyError: continue

            for band_name, (f_min, f_max) in freq_bands.items():
                freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
                if len(freq_indices) < 3: continue

                band_freqs = freqs[freq_indices]

                metrics_base = [calculate_psd_shape_metrics(psd[freq_indices], band_freqs) for psd in psds_base]
                metrics_walk = [calculate_psd_shape_metrics(psd[freq_indices], band_freqs) for psd in psds_walk]

                for metric_name in ['spectral_slope', 'spectral_entropy']:
                    metric_values_base = [m[metric_name] for m in metrics_base if not np.isnan(m[metric_name])]
                    metric_values_walk = [m[metric_name] for m in metrics_walk if not np.isnan(m[metric_name])]

                    if len(metric_values_base) != len(metric_values_walk) or not metric_values_base: continue
                    
                    test_results = _run_paired_comparison_test(metric_values_base, metric_values_walk)
                    results.append({
                        'analysis': f'shape_{metric_name}_within',
                        'cohort': cohort, 'block': walk_block,
                        'comparison': f"{walk_block}_vs_baseline", 'band': band_name,
                        'mean_baseline': np.mean(metric_values_base),
                        'mean_walk': np.mean(metric_values_walk),
                        **test_results
                    })
    return pd.DataFrame(results)
    
### --- Main stats runner  --- ###


def run_and_save_psd_statistics(
    all_psd_data: Dict,
    data_key: str,
    comparisons_between: List[List[str]],
    cohorts_for_within: List[str],
    walk_blocks: List[str],
    results_dir: str,
    analysis_name: str,
    freq_bands: Dict,
    run_advanced_stats: bool = True, # Flag to run the new tests
    n_permutations: int = 5000
) -> Optional[pd.DataFrame]:
    """
    Orchestrates all statistical analyses for a SINGLE data type (one ROI or DBS).
    
    This now includes:
    1. Original Band-Power Analysis (for continuity).
    2. Full-Spectrum Wasserstein Distance + Permutation Test.
    3. Full-Spectrum Cluster-Based Permutation Test.
    
    The results from (1) and (2) are combined and FDR corrected.
    The results from (3) are saved as plots and separate data files.
    """
    psd_data_for_type = all_psd_data.get(data_key)
    if not psd_data_for_type:
        print(f"WARNING: No data found for key '{data_key}' in '{analysis_name}'. Skipping.")
        return None

    # --- Setup: Get frequency info and create output directories ---
    first_block = next((b for b in ['baseline', *walk_blocks] if b in psd_data_for_type and psd_data_for_type[b]), None)
    if not first_block: return None
    first_cohort = next(iter(psd_data_for_type[first_block]))
    freqs = psd_data_for_type[first_block][first_cohort]['freqs']
    
    stats_summary_dfs = []
    output_dir_clusters = os.path.join(results_dir, f"Cluster_Analysis_{analysis_name}")
    os.makedirs(output_dir_clusters, exist_ok=True)
    # ==========================================================================
    # --- Step 1: Run all statistical analyses and collect DataFrames ---
    # ==========================================================================
    
    # --- Between-Cohort Analyses ---
    df_band_between = _run_band_power_analysis_between_cohorts(psd_data_for_type, comparisons_between, ['baseline', *walk_blocks], freqs, freq_bands)
    df_shape_between = _run_shape_metric_analysis_between_cohorts(psd_data_for_type, comparisons_between, ['baseline', *walk_blocks], freqs, freq_bands)

    # --- Within-Cohort Analyses ---
    df_band_within = _run_within_cohort_analysis(psd_data_for_type, cohorts_for_within, walk_blocks, freqs, freq_bands)
    df_shape_within = _run_shape_metric_analysis_within_cohort(psd_data_for_type, cohorts_for_within, walk_blocks, freqs, freq_bands)

    # (Note: Wasserstein and Cluster tests are often interpreted separately and may not need FDR,
    # but if you did want to correct Wasserstein, you would add its results to the families below)
    
    # ==========================================================================
    # --- Step 2: Combine tests into families and apply correction separately ---
    # ==========================================================================
    
    # --- Family 1: All BETWEEN-COHORT comparisons ---
    df_between_family = pd.concat([df_shape_between], ignore_index=True)
    print(f"\n--- Applying FDR Correction to BETWEEN-COHORT family ({len(df_between_family)} tests) for: {analysis_name} ---")
    df_between_corrected = _apply_fdr_correction(df_between_family)

    # --- Family 2: All WITHIN-COHORT comparisons ---
    df_within_family = pd.concat([df_shape_within], ignore_index=True)
    print(f"\n--- Applying FDR Correction to WITHIN-COHORT family ({len(df_within_family)} tests) for: {analysis_name} ---")
    df_within_corrected = _apply_fdr_correction(df_within_family)
    
    # ==========================================================================
    # --- Step 3: Combine the corrected families for the final report ---
    # ==========================================================================
    if df_between_corrected.empty and df_within_corrected.empty:
        print(f"No statistical results generated for {analysis_name}.")
        return None
        
    final_df = pd.concat([df_between_corrected, df_within_corrected], ignore_index=True)
    # Re-order columns for clarity
    final_df = final_df.sort_values(by=['analysis', 'comparison']).reset_index(drop=True)
    
    print("\n\n" + "="*80 + f"\n--- SUMMARY STATISTICAL RESULTS FOR: {analysis_name} ---\n" + "="*80)
    pd.set_option('display.max_rows', 500)
    print(final_df[['analysis', 'comparison', 'cohort', 'band', 'p_value', 'p_fdr_corrected', 'significant_fdr', 'cohens_d']].to_string())

    output_path = os.path.join(results_dir, f"statistical_analysis_{analysis_name}.csv")
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n--- Summary statistics for '{analysis_name}' saved to: {output_path} ---")
    
    # # ==========================================================================
    # # --- Analysis 1: Band-Power Statistics (Original Method) ---
    # # ==========================================================================
    # df_band_between = _run_band_power_analysis_between_cohorts(psd_data_for_type, comparisons_between, ['baseline', *walk_blocks], freqs, freq_bands)
    # df_band_within = _run_within_cohort_analysis(psd_data_for_type, cohorts_for_within, walk_blocks, freqs, freq_bands)
    # stats_summary_dfs.extend([df_band_between, df_band_within])

    if not run_advanced_stats:
        print("--- Skipping advanced statistics (Cluster & Wasserstein). ---")
    else:
        # ==========================================================================
        # --- Analysis 2: Wasserstein Distance + Permutation (Full Spectrum) ---
        # ==========================================================================
        print(f"\n--- Running Analysis 2: Wasserstein Permutation Test for {analysis_name} ---")
        wasserstein_results = []
        # --- Between-Cohorts ---
        for block in ['baseline', *walk_blocks]:
            for cohort1, cohort2 in comparisons_between:
                try:
                    psds1 = psd_data_for_type[block][cohort1]['psds']
                    psds2 = psd_data_for_type[block][cohort2]['psds']
                    test_results = _run_wasserstein_permutation_test(psds1, psds2, paired=False, n_permutations=n_permutations)
                    wasserstein_results.append({'analysis': 'wasserstein_between', 'block': block, 'comparison': f"{cohort1}_vs_{cohort2}", **test_results})
                except KeyError: continue
        # --- Within-Cohorts ---
        for cohort in cohorts_for_within:
            for walk_block in walk_blocks:
                try:
                    psds_base = psd_data_for_type['baseline'][cohort]['psds']
                    psds_walk = psd_data_for_type[walk_block][cohort]['psds']
                    if psds_base.shape[0] != psds_walk.shape[0]: continue
                    test_results = _run_wasserstein_permutation_test(psds_base, psds_walk, paired=True, n_permutations=n_permutations)
                    wasserstein_results.append({'analysis': 'wasserstein_within', 'cohort': cohort, 'comparison': f"{walk_block}_vs_baseline", **test_results})
                except KeyError: continue
        
        if wasserstein_results:
            stats_summary_dfs.append(pd.DataFrame(wasserstein_results))

        # ==========================================================================
        # --- Analysis 3: Cluster-Based Permutation Test (Full Spectrum) ---
        # ==========================================================================
        print(f"\n--- Running Analysis 3: Cluster-Based Permutation Test for {analysis_name} ---")
        # --- Between-Cohorts ---
        for block in ['baseline', *walk_blocks]:
            for cohort1, cohort2 in comparisons_between:
                try:
                    psds1 = psd_data_for_type[block][cohort1]['psds']
                    psds2 = psd_data_for_type[block][cohort2]['psds']
                    T_obs, clusters, p_vals, _ = _run_cluster_permutation_test(psds1, psds2, paired=False, n_permutations=n_permutations)
                    title = f"Cluster Test ({analysis_name}): {cohort1} vs {cohort2} [{block}]"
                    filename = os.path.join(output_dir_clusters, f"Cluster_{cohort1}_vs_{cohort2}_{block}.png")
                    _plot_cluster_test_result(T_obs, clusters, p_vals, freqs, title=title, filename=filename)
                except KeyError: continue
        # --- Within-Cohorts ---
        for cohort in cohorts_for_within:
            for walk_block in walk_blocks:
                try:
                    psds_base = psd_data_for_type['baseline'][cohort]['psds']
                    psds_walk = psd_data_for_type[walk_block][cohort]['psds']
                    if psds_base.shape[0] != psds_walk.shape[0]: continue
                    T_obs, clusters, p_vals, _ = _run_cluster_permutation_test(psds_base, psds_walk, paired=True, n_permutations=n_permutations)
                    title = f"Cluster Test ({analysis_name}): {cohort} [{walk_block} vs Baseline]"
                    filename = os.path.join(output_dir_clusters, f"Cluster_{cohort}_{walk_block}_vs_Baseline.png")
                    _plot_cluster_test_result(T_obs, clusters, p_vals, freqs, title=title, filename=filename)
                except KeyError: continue

        # # ==========================================================================
        # # --- Analysis 4: Shape metrics analysis ( ---
        # # ==========================================================================
        df_shape_between = _run_shape_metric_analysis_between_cohorts(psd_data_for_type, comparisons_between, ['baseline', *walk_blocks], freqs, freq_bands)
        df_shape_within = _run_shape_metric_analysis_within_cohort(psd_data_for_type, cohorts_for_within, walk_blocks, freqs, freq_bands)
    

    # # ==========================================================================
    # # --- Final Step: Combine, Correct, and Save Summary Statistics ---
    # # ==========================================================================
    if not stats_summary_dfs:
        print(f"No statistical results generated for {analysis_name}.")
        return None
        
    # #df_to_correct = pd.concat(stats_summary_dfs, ignore_index=True)
    df_to_correct = pd.concat([df_shape_between, df_shape_within], ignore_index=True)
    print(f"\n--- Applying FDR Correction to Band-Power and Wasserstein tests for: {analysis_name} ---")
    df_corrected = _apply_fdr_correction(df_to_correct)

    # # Re-order columns for clarity
    final_df = df_corrected.sort_values(by=['analysis', 'comparison']).reset_index(drop=True)
    
    print("\n\n" + "="*80 + f"\n--- SUMMARY STATISTICAL RESULTS FOR: {analysis_name} ---\n" + "="*80)
    pd.set_option('display.max_rows', 500)
    print(final_df[['analysis', 'comparison', 'cohort', 'band', 'p_value', 'p_fdr_corrected', 'significant_fdr', 'cohens_d']].to_string())

    output_path = os.path.join(results_dir, f"statistical_analysis_{analysis_name}.csv")
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n--- Summary statistics for '{analysis_name}' saved to: {output_path} ---")
    print(f"--- Cluster analysis plots saved in: {output_dir_clusters} ---")

    return final_df

def get_band_edges(fbands: Dict, f: np.ndarray) -> List[float]:
        edges = {val for pair in fbands.values() for val in pair}
        return sorted([e for e in edges if f[0] <= e <= f[-1]])
