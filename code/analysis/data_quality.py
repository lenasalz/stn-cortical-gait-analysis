import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import minmax_scale
from IPython.display import display
import mne
from scipy.stats import pearsonr
from scipy.signal import welch

from data_loader import load_data_from_pickle

# ------------------------------------------------------------------------------------
# Participants' data quality check and ranking
# ------------------------------------------------------------------------------------


def get_psd_metrics(data, fs, bands):
    f, Pxx = welch(data, fs, nperseg=1024)
    res = {}
    mask = (f >= 1) & (f <= 45)
    res['PSD_total_power'] = np.mean(Pxx[mask])
    for band, (lo, hi) in bands.items():
        band_mask = (f >= lo) & (f <= hi)
        res[f'PSD_{band}'] = np.mean(Pxx[band_mask])
    res['PSD_peak_freq'] = f[mask][np.argmax(Pxx[mask])]
    log_psd = np.log10(Pxx[mask] + 1e-20)
    res['PSD_flat_flag'] = np.std(log_psd) < 0.1
    band_means = np.array([np.mean(Pxx[(f >= lo) & (f <= hi)]) for (lo, hi) in bands.values()])
    zscores = (band_means - np.mean(band_means)) / (np.std(band_means) + 1e-20)
    res['PSD_outlier_flag'] = np.any(np.abs(zscores) > 3)
    return res


def print_metric_definitions():
    """Prints a clear explanation of each data quality metric used in the ranking and describes output columns."""
    print("="*90)
    print("Data Quality Metric & Output Definitions")
    print("="*90)
    print("""
A 'Quality Score' is calculated for each participant. Higher is better.
It is a weighted sum of the following normalized metrics:

1.  Rejection Rate (Weight: 0.25):
    - What: Percentage of epochs discarded by the autoreject algorithm across all blocks.
    - Why: Measures the prevalence of transient artifacts (e.g., movement, blinks).
    - Good Value: LOWER is better.

2.  Valid Gait Cycles (Weight: 0.25):
    - What: The total number of complete, sequential gait cycles detected in walking blocks.
    - Why: Measures the quality and quantity of the behavioral task recording.
    - Good Value: HIGHER is better.

3.  Task Epoch Count (Weight: 1.0):
    - What: The total number of clean, usable epochs across all *walking* blocks.
    - Why: Directly reflects the amount of data available for task-related analysis.
    - Good Value: HIGHER is better.

4.  Bad Critical Channel Count (Weight: 1.0):
    - What: Average number of bad channels within key motor/frontal areas across all blocks.
    - Why: High number of bad channels in ROIs severely impacts analysis.
    - Good Value: LOWER is better.

5.  EEG Signal STD (Weight: 1.0):
    - What: Average standard deviation of the EEG signal across all channels and blocks.
    - Why: A proxy for the overall signal amplitude and noise level.
    - Good Value: LOWER is better.
    
6.  High-Frequency Power Ratio (Weight: 1.0):
    - What: Ratio of power in high (30–70Hz) vs. low (2–20Hz) EEG frequencies, averaged across blocks.
    - Why: High ratio is a strong indicator of EMG contamination.
    - Good Value: LOWER is better.

7.  DBS Signal STD (Weight: 1.0, Patients Only):
    - What: Standard deviation of the DBS signal, averaged across blocks.
    - Why: Measures overall DBS signal stability and noise.
    - Good Value: LOWER is better.

PENALTIES:
- Baseline Epochs: Participants with < 20 clean baseline epochs receive a 50% penalty.
- Baseline Consistency: Participants flagged for inconsistencies between baseline and walking
  data receive a multiplicative penalty based on the number of flags.
- Missing/Unusable Data: Participants with missing files or ZERO usable task epochs are ranked at the bottom.

------------------------------------------------------------------------------
OUTPUT DATAFRAME/CSV COLUMNS:
------------------------------------------------------------------------------

- Rank:             Integer participant ranking within each group (1 = best quality).
- ID:               Participant code.
- Group:            Group (e.g., 'Patient_NFB', 'AgeMatched_NFB').
- Missing_Data:     True if data was incomplete or unusable, otherwise False.
- Final_Score:      Overall quality score **after all penalties** (higher = better). Formatted to 2 decimals; 'N/A' for missing data.
- Rejection_Rate:   Percentage of rejected epochs (lower = better), formatted as percent (e.g., '18.2%'); 'N/A' for missing data.
- Baseline_Epoch_Count:  Number of clean epochs in 'baseline' block.
- Task_Epoch_Count:      Number of clean epochs in task (walking) blocks.
- Total_Valid_Cycles:    Number of valid gait cycles detected.
- Bad_Channel_Count:     Average # of bad critical channels; lower is better. (1 decimal).
- EEG_STD:          Mean EEG signal standard deviation (2 decimals).
- HF_Power_Ratio:   Mean high-frequency (30–70Hz) to low-frequency (2–20Hz) EEG power ratio (2 decimals).
- Consistency_Flags: Number of baseline-to-task inconsistency flags.
- DBS_STD:          Mean DBS signal standard deviation (patients only).

- For any participant with 'Missing_Data' == True, numeric fields are displayed as 'N/A', and their Final_Score is set to -1 (lowest rank).

INTERPRETATION:
- Higher Final_Score and lower Rank indicate better overall data quality for downstream analyses.
- Participants with incomplete, missing, or unusable data are always ranked last.

------------------------------------------------------------------------------
    """)
    print("="*90 + "\n")



def check_baseline_task_consistency(blocks_to_consider: list, subjects_dico: dict, data_dir: str):
    """
    Performs a diagnostic check for inconsistencies between baseline and walking blocks.
    Returns a DataFrame with results and flags for each subject/block pair.
    """
    try:
        all_data_epoched = load_data_from_pickle("all_data_epoched_dico", data_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find 'all_data_epoched_dico.pkl'. {e}")
        return pd.DataFrame()

    consistency_results = []
    critical_channels = {'Cz', 'C1', 'C3', 'C2', 'C4', 'Fz', 'AF3', 'F1', 'AF4', 'F2'}


    for group, subjects in subjects_dico.items():
        for subject_id in subjects:
            try:
                baseline_idx = all_data_epoched[group]['baseline']['ID'].index(subject_id)
                baseline_epochs = all_data_epoched[group]['baseline']['EEG'][baseline_idx]

                if baseline_epochs is None or len(baseline_epochs) == 0: continue
                
                picks = [ch for ch in critical_channels if ch in baseline_epochs.ch_names]
                if not picks: continue

                baseline_std = baseline_epochs.copy().pick(picks).get_data().std()
                psd_base, freqs = baseline_epochs.compute_psd(fmin=2, fmax=80, picks=picks, method='welch', n_jobs=-1, verbose=False).get_data(return_freqs=True)
                avg_psd_base = psd_base.mean(axis=0).mean(axis=0)
                hf_ratio_base = avg_psd_base[(freqs >= 30) & (freqs <= 70)].mean() / avg_psd_base[(freqs >= 2) & (freqs <= 20)].mean()

                bad_crit_pairs = set() 
                
                for block in blocks_to_consider:
                    task_idx = all_data_epoched[group][block]['ID'].index(subject_id)
                    task_epochs = all_data_epoched[group][block]['EEG'][task_idx]

                    if task_epochs is None or len(task_epochs) == 0: continue

                    task_std = task_epochs.copy().pick(picks).get_data().std()
                    psd_task, _ = task_epochs.compute_psd(fmin=2, fmax=80, picks=picks, method='welch', n_jobs=-1, verbose=False).get_data(return_freqs=True)
                    avg_psd_task = psd_task.mean(axis=0).mean(axis=0)
                    hf_ratio_task = avg_psd_task[(freqs >= 30) & (freqs <= 70)].mean() / avg_psd_task[(freqs >= 2) & (freqs <= 20)].mean()

                    std_ratio = task_std / baseline_std if baseline_std > 0 else np.inf
                    hf_ratio_of_ratios = hf_ratio_task / hf_ratio_base if hf_ratio_base > 0 else np.inf
                    spectral_corr, _ = pearsonr(np.log10(avg_psd_base), np.log10(avg_psd_task))
                    
                    flags = []
                    if std_ratio > 10: flags.append("High STD Ratio")
                    if hf_ratio_of_ratios > 5: flags.append("High HF-Ratio Increase")
                    if spectral_corr < 0.7: flags.append("Low Spectral Correlation")
                    
                    consistency_results.append({
                        'ID': subject_id, 'Block': block, 'Flag_Count': len(flags),
                        'STD_Ratio': std_ratio, 'HF_Ratio_of_Ratios': hf_ratio_of_ratios,
                        'Spectral_Correlation': spectral_corr, 'Flags': ", ".join(flags) if flags else "OK"
                    })
            except (ValueError, IndexError):
                continue
                
    return pd.DataFrame(consistency_results)


def rank_data_quality(
    blocks_to_consider: list, 
    subjects_dico: dict, 
    results_dir: str, 
    data_dir: str,
    consistency_df: pd.DataFrame
):
    """
    Ranks participants based on a comprehensive quality score and saves the result to a CSV file.
    """
    try:
        autoreject_path = os.path.join(results_dir, "Autoreject_epochs", "epochs_autoreject_summary_ALL.csv")
        gait_quality_path = os.path.join(results_dir, "gait_events_quality_summary.csv")
        ar_df = pd.read_csv(autoreject_path)
        gait_df = pd.read_csv(gait_quality_path)
        all_data_epoched = load_data_from_pickle("all_data_epoched_dico", data_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return

    all_participants_data = []
    critical_channels = {'Cz', 'C1', 'C3', 'C2', 'C4', 'Fz', 'AF3', 'F1', 'AF4', 'F2'}
    CRIT_PAIRS = [("C1", "C3"), ("C2", "C4"), ("AF3", "F1"), ("AF4", "F2")]

    all_required_blocks = ['baseline'] + blocks_to_consider

    for group, subjects in subjects_dico.items():
        for subject_id in subjects:
            ar_subj_df = ar_df[ar_df['ID'] == subject_id]
            ar_subj_all_blocks_df = ar_subj_df[ar_subj_df['Block'].isin(all_required_blocks)]
            gait_subj_df = gait_df[(gait_df['ID'] == subject_id) & (gait_df['block'].isin(blocks_to_consider))]

            has_missing_data = (len(ar_subj_all_blocks_df) != len(all_required_blocks) or len(gait_subj_df) != len(blocks_to_consider))
            task_epoch_count = 0 if has_missing_data else ar_subj_all_blocks_df[ar_subj_all_blocks_df['Block'].isin(blocks_to_consider)]['Cleaned_N_epochs'].sum()

            if not has_missing_data and task_epoch_count == 0:
                print(f"-> {subject_id} has 0 usable task epochs. Treating as missing data.")
                has_missing_data = True
            
            if has_missing_data:
                metrics = {m: np.nan for m in ['Rejection_Rate', 'EEG_STD', 'DBS_STD', 'Total_Valid_Cycles', 
                                              'Baseline_Epoch_Count', 'Task_Epoch_Count', 'Bad_Channel_Count', 'HF_Power_Ratio']}
            else:
                total_orig_epochs = ar_subj_all_blocks_df['Original_N_epochs'].sum()
                total_clean_epochs = ar_subj_all_blocks_df['Cleaned_N_epochs'].sum()
                rejection_rate = (1 - (total_clean_epochs / total_orig_epochs)) if total_orig_epochs > 0 else 1.0
                baseline_epoch_count = ar_subj_all_blocks_df[ar_subj_all_blocks_df['Block'] == 'baseline']['Cleaned_N_epochs'].sum()
                
                bad_channel_counts, hf_power_ratios = [], []
                bad_crit_pairs = set() 
                for block in all_required_blocks:
                    try:
                        idx = all_data_epoched[group][block]['ID'].index(subject_id)
                        eeg_epochs = all_data_epoched[group][block]['EEG'][idx]
                        if eeg_epochs is not None and len(eeg_epochs) > 0:
                            bad_chans = set(eeg_epochs.info['bads'])
                            bad_channel_counts.append(len(critical_channels.intersection(bad_chans)))
                            for pair in CRIT_PAIRS:
                                if pair[0] in bad_chans and pair[1] in bad_chans:
                                    bad_crit_pairs.add(pair)
                            picks = [ch for ch in critical_channels if ch in eeg_epochs.ch_names]
                            if picks:
                                psds, freqs = eeg_epochs.compute_psd(fmin=2, fmax=80, picks=picks, method='welch', n_jobs=-1, verbose=False).get_data(return_freqs=True)
                                avg_psd = psds.mean(axis=0).mean(axis=0)
                                hf_power_ratios.append(avg_psd[(freqs >= 30) & (freqs <= 70)].mean() / avg_psd[(freqs >= 2) & (freqs <= 20)].mean())
                            else: hf_power_ratios.append(np.nan)
                        else:
                             bad_channel_counts.append(len(critical_channels)); hf_power_ratios.append(np.nan)
                    except (ValueError, IndexError):
                        bad_channel_counts.append(len(critical_channels)); hf_power_ratios.append(np.nan)
                
                metrics = {
                    'Rejection_Rate': rejection_rate, 'EEG_STD': ar_subj_all_blocks_df['EEG_std'].mean(),
                    'DBS_STD': ar_subj_all_blocks_df['DBS_std'].mean() if group == 'Patient_NFB' else np.nan,
                    'Total_Valid_Cycles': gait_subj_df['valid gait cycles (HSR)'].sum(),
                    'Baseline_Epoch_Count': baseline_epoch_count, 'Task_Epoch_Count': task_epoch_count,
                    'Bad_Channel_Count': np.mean(bad_channel_counts), 'HF_Power_Ratio': np.nanmean(hf_power_ratios),
                    'Bad_Critical_Pairs': len(bad_crit_pairs)  # <-- ADD THIS
                }
            all_participants_data.append({'ID': subject_id, 'Group': group, 'Missing_Data': has_missing_data, **metrics})

    quality_df = pd.DataFrame(all_participants_data)
    
    consistency_summary = consistency_df.groupby('ID')['Flag_Count'].sum().reset_index(name='Consistency_Flags')
    quality_df = pd.merge(quality_df, consistency_summary, on='ID', how='left').fillna({'Consistency_Flags': 0})

    final_ranked_dfs = []
    for group_name in subjects_dico.keys():
        group_df = quality_df[quality_df['Group'] == group_name].copy()
        valid_df = group_df[~group_df['Missing_Data']].copy()
        missing_df = group_df[group_df['Missing_Data']].copy()

        if not valid_df.empty:
            weights = {'Rejection': 0.25, 'Cycles': 0.25, 'Task_Epochs': 1.0, 'Bad_Channels': 1.5, 'EEG_STD': 1.0, 'HF_Ratio': 1.0, 'DBS_STD': 1.0}
            
            valid_df['Norm_Rejection'] = 1 - minmax_scale(valid_df['Rejection_Rate'])
            valid_df['Norm_Cycles'] = minmax_scale(valid_df['Total_Valid_Cycles'])
            valid_df['Norm_Task_Epochs'] = minmax_scale(valid_df['Task_Epoch_Count'])
            valid_df['Norm_Bad_Channels'] = 1 - minmax_scale(valid_df['Bad_Channel_Count'])
            valid_df['Norm_EEG_STD'] = 1 - minmax_scale(valid_df['EEG_STD'])
            valid_df['Norm_HF_Ratio'] = 1 - minmax_scale(valid_df['HF_Power_Ratio'])

            quality_score = (weights['Rejection'] * valid_df['Norm_Rejection'] + weights['Cycles'] * valid_df['Norm_Cycles'] +
                             weights['Task_Epochs'] * valid_df['Norm_Task_Epochs'] + weights['Bad_Channels'] * valid_df['Norm_Bad_Channels'] +
                             weights['EEG_STD'] * valid_df['Norm_EEG_STD'] + weights['HF_Ratio'] * valid_df['Norm_HF_Ratio'])
            
            if group_name == 'Patient_NFB': valid_df['Norm_DBS_STD'] = 1 - minmax_scale(valid_df['DBS_STD']); quality_score += weights['DBS_STD'] * valid_df['Norm_DBS_STD']
            
            valid_df['Quality_Score'] = quality_score
            valid_df['Baseline_Penalty'] = valid_df['Baseline_Epoch_Count'].apply(lambda x: 0.5 if x < 20 else 1.0)
            penalty_map = {0: 1.0, 1: 0.75, 2: 0.5}; valid_df['Consistency_Penalty'] = valid_df['Consistency_Flags'].apply(lambda x: penalty_map.get(x, 0.25))
            valid_df['Final_Score'] = valid_df['Quality_Score'] * valid_df['Baseline_Penalty'] * valid_df['Consistency_Penalty']

            valid_df = valid_df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True); valid_df['Rank'] = valid_df.index + 1
        
        if not missing_df.empty: missing_df['Final_Score'] = -1; missing_df['Rank'] = len(valid_df) + 1 if not valid_df.empty else 1
        
        final_ranked_dfs.append(pd.concat([valid_df, missing_df], ignore_index=True))

    final_df = pd.concat(final_ranked_dfs, ignore_index=True)
    
    # Save the raw numerical data to CSV
    scenario_name = "_and_".join(blocks_to_consider)
    filename = f"data_quality_ranking_{scenario_name}.csv"
    filepath = os.path.join(results_dir, filename)
    final_df.to_csv(filepath, index=False, float_format='%.4f')
    print(f"\nSaved raw quality ranking data to: {filepath}")

    # Create a formatted version for display
    display_cols = ['Rank', 'ID', 'Group', 'Missing_Data', 'Final_Score', 'Rejection_Rate', 'Baseline_Epoch_Count', 
                    'Task_Epoch_Count', 'Total_Valid_Cycles', 'Bad_Channel_Count', 'EEG_STD', 'HF_Power_Ratio', 
                    'Consistency_Flags', 'DBS_STD']
    for col in display_cols:
        if col not in final_df.columns: final_df[col] = np.nan
            
    df_display = final_df[display_cols].copy()
    
    df_display['Final_Score'] = df_display['Final_Score'].map('{:,.2f}'.format)
    df_display['Rejection_Rate'] = (df_display['Rejection_Rate']*100).map('{:.1f}%'.format)
    df_display['Bad_Channel_Count'] = df_display['Bad_Channel_Count'].map('{:.1f}'.format)
    df_display['EEG_STD'] = df_display['EEG_STD'].map('{:.2f}'.format)
    df_display['HF_Power_Ratio'] = df_display['HF_Power_Ratio'].map('{:.2f}'.format)
    df_display['DBS_STD'] = df_display['DBS_STD'].map('{:.2f}'.format)
    df_display.loc[df_display['Missing_Data'] == True, ['Final_Score', 'Bad_Channel_Count', 'EEG_STD', 'HF_Power_Ratio', 'Consistency_Flags']] = 'N/A'

    with pd.option_context('display.max_rows', None, 'display.width', 1200):
        print("\n" + "="*80); print(f"Data Quality Ranking Considering Blocks: {blocks_to_consider}"); print("="*80)
        print("\n--- Patient (Patient_NFB) Ranking ---"); display(df_display[df_display['Group'] == 'Patient_NFB'])
        print("\n--- Control (AgeMatched_NFB) Ranking ---"); display(df_display[df_display['Group'] == 'AgeMatched_NFB'])

    return final_df
