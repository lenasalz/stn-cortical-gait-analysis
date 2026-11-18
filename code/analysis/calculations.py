import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import mannwhitneyu, shapiro, ttest_1samp, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

def z_score_normalize(data: np.ndarray, baseline_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalization of data based on baseline metrics.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data to be normalized.
    baseline_data : np.ndarray
        Baseline EEG data used for calculating mean and standard deviation.
        
    Returns
    -------
    normalized_data : np.ndarray
        Z-score normalized EEG data.
        
    Notes
    -----
    Z-score normalization transforms data to have zero mean and unit variance
    based on baseline statistics: (data - baseline_mean) / baseline_std.
    """
    mean_baseline = np.mean(baseline_data)
    std_baseline = np.std(baseline_data)
    normalized_data = (data - mean_baseline) / std_baseline
    return normalized_data

def calculate_band_power(data: np.ndarray, fs: float, freq_band: Tuple[float, float], nperseg: int = 1024) -> float:
    """
    Calculate the average power in a specific frequency band.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data to compute power for.
    fs : float
        Sampling frequency of the data in Hz.
    freq_band : tuple of (float, float)
        Frequency band limits (low, high) in Hz.
    nperseg : int, default 1024
        Length of each segment for welch method.
        
    Returns
    -------
    band_power : float
        Median power in the specified frequency band.
        
    Notes
    -----
    Uses Welch's method with Hanning window to compute power spectral density.
    Returns median instead of mean to reduce sensitivity to outliers.
    """
    f, Pxx = welch(data, fs, window='hann', nperseg=nperseg)
    band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
    band_power = np.median(Pxx[band_mask]) #Pxx[band_mask].mean() # median ? 
    return band_power

def process_band_power_data(
        subjects: list,
        all_data: dict,
        blocks: list,               # ← list of task blocks
        band_range: tuple,
        region: dict):
    """
    Compute band power for baseline and one or more task blocks.

    Args
    ----
    subjects       : list of subject IDs.
    all_data       : nested dict holding EEG/DBS data for 'baseline' and each task block.
    blocks         : list of block names, e.g. ['task', 'rest2'].
    band_range     : (low, high) tuple for band-power calculation.
    region         : brain regions to study -- as defined in variables.py.

    Returns
    -------
    results : list[dict]
        One dictionary per subject containing
        - 'EEG_baseline_{ch}'                    (baseline power per channel)
        - 'EEG_{block}_{ch}'                     (power per channel for every block)
        - 'baseline_DBS' and '{block}_DBS'       (DBS powers)
        - 'channels_to_pick', 'reference_channel'
    """

    # ------------------------------------------------------------------
    # 1. Region-specific channel lists
    # ------------------------------------------------------------------

    # channels_list = region_channels[region]
    channels_list = region['Both']
    results = []

    # ------------------------------------------------------------------
    # 2. Iterate over subjects
    # ------------------------------------------------------------------
    for subject_id in subjects:
        # ------------------------------------------
        # Locate subject index in baseline first
        # ------------------------------------------
        try:
            sub_index = all_data['baseline']['ID'].index(subject_id)
        except ValueError:
            print(f"[!] Subject {subject_id} missing in 'baseline'. Skipped.")
            continue

        eeg_data_baseline = all_data['baseline']['EEG'][sub_index]
        dbs_data_baseline = all_data['baseline']['DBS_proc'][sub_index]

        # --- container for this subject’s results ---
        subj_dict = {'channels_to_pick': channels_list,
                     'reference_channel': next((ch for ch in channels_list
                                                if ch.lower().endswith('z')), None)}

        # ==========================================================
        # 2a.  Baseline EEG band-power per channel
        # ==========================================================
        if eeg_data_baseline is not None:
            fs_eeg = eeg_data_baseline.info['sfreq']
        else: 
            fs_eeg = np.nan

        for ch in channels_list:
            try:
                ch_idx = eeg_data_baseline.ch_names.index(ch)
                signal_bl = eeg_data_baseline.get_data()[ch_idx]
                if ch in eeg_data_baseline.info.get('bads', []):
                    raise RuntimeError('bad channel')
                norm_bl = z_score_normalize(signal_bl, signal_bl)
                power_bl = calculate_band_power(norm_bl, fs_eeg, band_range, nperseg = 1024)
            except Exception:  # channel missing, bad, or other error
                power_bl = np.nan
            subj_dict[f'EEG_baseline_{ch}'] = power_bl

        # ==========================================================
        # 2b.  Baseline DBS band-power
        # ==========================================================
        try:
            fs_dbs = dbs_data_baseline.info['sfreq']
            dbs_sig_bl = dbs_data_baseline.get_data()[0]
            norm_dbs_bl = z_score_normalize(dbs_sig_bl, dbs_sig_bl)
            subj_dict['baseline_DBS'] = calculate_band_power(
                norm_dbs_bl, fs_dbs, band_range, nperseg = 256)
        except Exception:
            subj_dict['baseline_DBS'] = np.nan

        # ------------------------------------------------------------------
        # 3. Loop over requested blocks
        # ------------------------------------------------------------------
        for block in blocks:
            # ..............................................................
            # 3a. Retrieve block-specific data
            # ..............................................................
            try:
                blk_index = all_data[block]['ID'].index(subject_id)
                eeg_data_block = all_data[block]['EEG'][blk_index]
                dbs_data_block = all_data[block]['DBS_proc'][blk_index]
            except Exception:
                print(f"    └─ block '{block}' not found for {subject_id}")
                eeg_data_block = None
                dbs_data_block = None

            # ..............................................................
            # 3b. EEG powers per channel
            # ..............................................................
            for ch in channels_list:
                key = f'EEG_{block}_{ch}'
                if eeg_data_block is None:
                    subj_dict[key] = np.nan
                    continue
                try:
                    ch_idx = eeg_data_block.ch_names.index(ch)
                    signal_blk = eeg_data_block.get_data()[ch_idx]
                    signal_bl  = eeg_data_baseline.get_data()[ch_idx]
                    if ch in eeg_data_block.info.get('bads', []):
                        raise RuntimeError('bad channel')
                    norm_blk = z_score_normalize(signal_blk, signal_bl)
                    subj_dict[key] = calculate_band_power(
                        norm_blk, fs_eeg, band_range)
                except Exception:
                    subj_dict[key] = np.nan

            # ..............................................................
            # 3c. DBS power for this block
            # ..............................................................
            dbs_key = f'{block}_DBS'
            if dbs_data_block is None:
                subj_dict[dbs_key] = np.nan
            else:
                try:
                    fs_dbs_blk = dbs_data_block.info['sfreq']
                    norm_blk_dbs = z_score_normalize(
                        dbs_data_block.get_data()[0],
                        dbs_data_baseline.get_data()[0])
                    subj_dict[dbs_key] = calculate_band_power(
                        norm_blk_dbs, fs_dbs_blk, band_range)
                except Exception:
                    subj_dict[dbs_key] = np.nan

        # ..............................................................
        # 4.  Collect
        # ..............................................................
        results.append(subj_dict)

    return results


def process_band_power_data_epochs(
        subjects: list,
        all_data: dict,
        blocks: list,
        band_range: tuple,
        region: dict):
    """
    Compute band power for baseline and task blocks using mne.Epochs.
    Returns a list of dicts as before.
    """
    channels_list = region['Both']
    results = []

    def extract_epochs_channel(epochs, ch_name):
        ch_idx = epochs.ch_names.index(ch_name)
        data = epochs.get_data()[:, ch_idx, :]  # shape: (n_epochs, n_times)
        return data


    for subject_id in subjects:
        # Locate baseline index
        try:
            sub_index = all_data['baseline']['ID'].index(subject_id)
        except ValueError:
            print(f"[!] Subject {subject_id} missing in 'baseline'. Skipped.")
            continue

        eeg_data_baseline = all_data['baseline']['EEG'][sub_index]  # mne.Epochs
        dbs_data_baseline = all_data['baseline']['DBS'][sub_index]

        subj_dict = {'channels_to_pick': channels_list,
                     'reference_channel': next((ch for ch in channels_list
                                                if ch.lower().endswith('z')), None)}

        # =========== Baseline EEG band-power per channel =============
        if eeg_data_baseline is not None:
            fs_eeg = eeg_data_baseline.info['sfreq']
        else:
            fs_eeg = np.nan

        for ch in channels_list:
            try:
                # get (n_epochs, n_times)
                data_epochs = extract_epochs_channel(eeg_data_baseline, ch)
                if ch in eeg_data_baseline.info.get('bads', []):
                    raise RuntimeError('bad channel')
                # z-score each epoch by itself (since baseline)
                #normed = np.array([z_score_normalize(epoch, epoch) for epoch in data_epochs])
                # calculate band power per epoch, then mean
                baseline_powers = [calculate_band_power(epoch, fs_eeg, band_range, nperseg = 1024 ) for epoch in data_epochs] # 1.95 Hz resolution
                mean_bl_power = np.nanmean(baseline_powers)
                std_bl_power = np.nanstd(baseline_powers)
            except Exception:
                power_bl = np.nan
            subj_dict[f'EEG_baseline_{ch}'] = mean_bl_power

            for block in blocks:
                key_z = f'EEG_{block}_{ch}'
                dbs_key = f'{block}_DBS'
                
                try:
                    blk_index = all_data[block]['ID'].index(subject_id)
                    eeg_data_block = all_data[block]['EEG'][blk_index]
                    
                    # Calculate raw power for all task epochs
                    task_epochs = extract_epochs_channel(eeg_data_block, ch)
                    task_powers = [calculate_band_power(epoch, fs_eeg, band_range, nperseg=1024) for epoch in task_epochs]
                    # --- Step 4: Normalize task power using baseline power stats ---
                    
                    # Z-transform
                    if std_bl_power > 1e-20: # Avoid division by zero
                        task_powers_z = [(p - mean_bl_power) / std_bl_power for p in task_powers]
                        subj_dict[key_z] = np.nanmean(task_powers_z) #median ? 
                    else:
                        subj_dict[key_z] = np.nan

                except Exception as e:
                    print(f"Could not process block '{block}' for {subject_id}, channel {ch}: {e}")
                    subj_dict[dbs_key] = np.nan
                    subj_dict[key_z] = np.nan

        # =========== Baseline DBS band-power =============
        try:
            if dbs_data_baseline is not None:
                fs_dbs = dbs_data_baseline.info['sfreq']
                dbs_epochs = dbs_data_baseline.get_data()[:, 0, :]  # assuming channel 0 is DBS
                # normed = np.array([z_score_normalize(epoch, epoch) for epoch in dbs_epochs]) -> wrong ?! %AB ?
                #baseline_powers = [calculate_band_power(e, fs_dbs, band_range, nperseg = 128) for e in normed] #1,95 Hz resolution
                baseline_powers = [calculate_band_power(e, fs_dbs, band_range, nperseg = 128) for e in dbs_epochs]
                # 2. Compute the statistics (mean and std) FROM the baseline power values
                mean_baseline_power = np.nanmean(baseline_powers)
                # The textbook Z-transform also needs the standard deviation. Let's add it.
                std_baseline_power = np.nanstd(baseline_powers)
                subj_dict['baseline_DBS'] = mean_baseline_power
            else:
                subj_dict['baseline_DBS'] = np.nan
        except Exception:
            subj_dict['baseline_DBS'] = np.nan
            # for block in blocks:
            #     subj_dict[f'{block}_DBS'] = np.nan
            #     #continue

        # =========== Loop over requested blocks ===========
        for block in blocks:
            try:
            #     blk_index = all_data[block]['ID'].index(subject_id)
            #     eeg_data_block = all_data[block]['EEG'][blk_index]  # mne.Epochs
                 dbs_data_block = all_data[block]['DBS'][blk_index]
            except Exception:
            #     print(f"    └─ block '{block}' not found for {subject_id}")
            #     eeg_data_block = None
                 dbs_data_block = None

            # # ---- EEG band-power per channel for this block ----
            # for ch in channels_list:
            #     key = f'EEG_{block}_{ch}'
            #     if eeg_data_block is None:
            #         subj_dict[key] = np.nan
            #         continue
            #     try:
            #         data_epochs = extract_epochs_channel(eeg_data_block, ch)
            #         if ch in eeg_data_block.info.get('bads', []):
            #             raise RuntimeError('bad channel')
            #         # Baseline per-epoch for normalization
            #         baseline_epochs = extract_epochs_channel(eeg_data_baseline, ch)
            #         # Z-score per epoch (block wrt baseline epoch)
            #         if data_epochs.shape[0] != baseline_epochs.shape[0]:
            #             # fallback: use per-epoch normalization
            #             normed = np.array([z_score_normalize(e, e) for e in data_epochs])
            #         else:
            #             normed = np.array([z_score_normalize(e, b)
            #                                for e, b in zip(data_epochs, baseline_epochs)])
            #         powers = [calculate_band_power(e, fs_eeg, band_range, nperseg=1024) for e in normed]
            #         subj_dict[key] = np.nanmean(powers)
            #     except Exception:
            #         subj_dict[key] = np.nan

                

            # ---- DBS power for this block ----
            
            if dbs_data_block is None:
                subj_dict[dbs_key] = np.nan
            else:
                try:
                    fs_dbs_blk = dbs_data_block.info['sfreq']
                    dbs_epochs_blk = dbs_data_block.get_data()[:, 0, :]  # (n_epochs, n_times)
                    #dbs_epochs_bl = dbs_data_baseline.get_data()[:, 0, :]  # (n_epochs, n_times)
                    # z-score normalization (block wrt baseline epoch)
                    # if dbs_epochs_blk.shape[0] != dbs_epochs_bl.shape[0]: #wrong ? %AB
                    #     normed = np.array([z_score_normalize(e, e) for e in dbs_epochs_blk])
                    # else:
                    #     normed = np.array([z_score_normalize(e, b)
                    #                        for e, b in zip(dbs_epochs_blk, dbs_epochs_bl)])
                    task_powers = [calculate_band_power(e, fs_dbs_blk, band_range, nperseg=128) for e in dbs_epochs_blk]
                    if std_baseline_power > 0:
                        task_powers_z = [(p - mean_baseline_power) / std_baseline_power for p in task_powers]
                        #task_powers_z = np.array([z_score_normalize(power, ) for power in dbs_epochs_blk])
                        subj_dict[dbs_key] = np.nanmean(task_powers_z) #_z)
                        #print(np.nanmean(task_powers_z))
                        #print("appened dbs_key")
                    else:
                        subj_dict[dbs_key] = np.nan
                    #subj_dict[dbs_key] = np.nanmean(powers)
                except Exception:
                    subj_dict[dbs_key] = np.nan
            #print("res", subj_dict[dbs_key])
        results.append(subj_dict)

    return results

# ==============================================================================
# Relative power
# ==============================================================================

def get_averaged_psd(data_epochs, fs, nperseg=1024):
    """
    Calculates a single, noise-reduced Power Spectral Density (PSD) by averaging
    across all provided epochs using Welch's method.

    Args:
        data_epochs (numpy.ndarray): 2D array of time series data (n_epochs, n_samples).
        fs (float): Sampling frequency in Hz.
        nperseg (int): Length of each segment for Welch's method.

    Returns:
        f (numpy.ndarray): Array of sample frequencies.
        Pxx_avg (numpy.ndarray): A single, averaged power spectral density.
    """
    if data_epochs is None or data_epochs.ndim != 2 or data_epochs.shape[0] == 0:
        return None, None
    # Welch's method on a 2D array automatically averages the PSDs of each row (epoch).
    f, Pxx = welch(data_epochs, fs, window='hann', nperseg=nperseg, axis=-1)
    Pxx_avg = np.median(Pxx, axis=0)
    
    return f, Pxx_avg

def calculate_relative_band_power(f, Pxx_avg, freq_band, norm_freq_range=(10.0, 100.0)):
    """
    Calculates relative band power from an averaged PSD, as described in the paper.
    The PSD is normalized by the mean power in a specified frequency range.

    Args:
        f (numpy.ndarray): Array of sample frequencies from get_averaged_psd.
        Pxx_avg (numpy.ndarray): The single, averaged power spectral density.
        freq_band (tuple): The frequency band of interest (low, high) in Hz.
        norm_freq_range (tuple): The frequency range used for normalization.

    Returns:
        float: The final average relative power in the band of interest.
    """
    if f is None or Pxx_avg is None:
        return np.nan

    # 1. Define the mask for the normalization range and calculate the mean power
    norm_mask = (f >= norm_freq_range[0]) & (f <= norm_freq_range[1])
    if not np.any(norm_mask):
        return np.nan

    mean_power_for_norm = np.median(Pxx_avg[norm_mask])
    
    if mean_power_for_norm == 0 or np.isnan(mean_power_for_norm):
        return np.nan

    # 2. Create the relative power spectrum by dividing the entire PSD by the mean power
    Pxx_relative = Pxx_avg / mean_power_for_norm

    # 3. Define the mask for the band of interest and calculate power from the *relative* spectrum
    band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
    if not np.any(band_mask):
        return np.nan
        
    relative_power = np.median(Pxx_relative[band_mask])
    return relative_power

def run_bandpower_analysis_relative(
    subjects: list,
    all_data: dict,
    blocks: list,
    band_range: tuple,
    region: dict,
    norm_freq_range: tuple = (1.0, 45.0), # As per paper
    nperseg_ratio: float = 0.5
):
    """
    Computes relative band power, creating a data structure identical to the original script.
    """
    channels_list = region['Both']
    results = []
    def extract_channel_data_2d(epochs, ch_name):
        if epochs is None or ch_name not in epochs.ch_names or ch_name in epochs.info.get('bads', []):
            return None
        return epochs.get_data(picks=[ch_name])[:, 0, :]

    for subject_id in subjects:
        try:
            sub_idx_base = all_data['baseline']['ID'].index(subject_id)
            eeg_epochs_base = all_data['baseline']['EEG'][sub_idx_base]
            dbs_epochs_base = all_data['baseline'].get('DBS', [None]*len(all_data['baseline']['ID']))[sub_idx_base]
        except (ValueError, IndexError):
            print(f"[!] Subject {subject_id} or their baseline data missing. Skipped.")
            continue

        # ======================= THE FIX STARTS HERE =======================
        # Initialize the dictionary for this subject
        subj_results = {'ID': subject_id}
        
        # 1. Add the 'channels_to_pick' key, just like in the original script.
        subj_results['channels_to_pick'] = channels_list
        
        # 2. Add the 'reference_channel' key, using the same logic as the original script.
        subj_results['reference_channel'] = next((ch for ch in channels_list if ch.lower().endswith('z')), None)
        # ======================= THE FIX ENDS HERE =========================

        n_times = eeg_epochs_base.times.shape[0] if eeg_epochs_base else 2048
        nperseg = int(n_times * nperseg_ratio)

        for block in ['baseline'] + blocks:
            try:
                if block == 'baseline':
                    eeg_epochs, dbs_epochs = eeg_epochs_base, dbs_epochs_base
                else:
                    sub_idx_blk = all_data[block]['ID'].index(subject_id)
                    eeg_epochs = all_data[block]['EEG'][sub_idx_blk]
                    dbs_epochs = all_data[block].get('DBS', [None]*len(all_data[block]['ID']))[sub_idx_blk]
                    if eeg_epochs is None: 
                        print(f" no eeg epochs for subject {subject_id}")
                        for ch in channels_list: subj_results[f'EEG_{block}_{ch}'] = np.nan
                        subj_results[f'{block}_DBS'] = np.nan
                        continue
            except (ValueError, IndexError):
                for ch in channels_list: subj_results[f'EEG_{block}_{ch}'] = np.nan
                subj_results[f'{block}_DBS'] = np.nan
                continue

            # --- Process EEG Channels ---
            for ch in channels_list:
                eeg_data_2d = extract_channel_data_2d(eeg_epochs, ch)
                f_eeg, pxx_eeg_avg = get_averaged_psd(eeg_data_2d, eeg_epochs.info['sfreq'], nperseg)
                power = calculate_relative_band_power(f_eeg, pxx_eeg_avg, band_range, norm_freq_range)
                subj_results[f'EEG_{block}_{ch}'] = power

            # --- Process DBS Channel ---
            dbs_ch_name = dbs_epochs.ch_names[0] if dbs_epochs and dbs_epochs.ch_names else None
            dbs_data_2d = extract_channel_data_2d(dbs_epochs, dbs_ch_name)
            f_dbs, pxx_dbs_avg = get_averaged_psd(dbs_data_2d, dbs_epochs.info['sfreq'] if dbs_epochs else 200, nperseg)
            power = calculate_relative_band_power(f_dbs, pxx_dbs_avg, band_range, norm_freq_range)
            subj_results[f'{block}_DBS'] = power
        
        results.append(subj_results)

    return results
# def run_bandpower_analysis_relative(
#     subjects: list,
#     all_data: dict,
#     blocks: list,
#     band_range: tuple,
#     region: dict,
#     norm_freq_range: tuple = (10.0, 100.0), # As per paper
#     nperseg_ratio: float = 0.5
# ):
#     """
#     Computes relative band power using the paper's method (Average PSDs First).
#     This is a separate workflow from the original z-score method.

#     Args:
#         subjects (list): List of subject IDs.
#         all_data (dict): Nested dict holding MNE Epochs objects.
#         blocks (list): List of block names to process.
#         band_range (tuple): (low, high) tuple for the band of interest.
#         region (dict): Brain regions dictionary.
#         norm_freq_range (tuple): Frequency range for normalization.
#         nperseg_ratio (float): Ratio of epoch length to use for Welch's nperseg.

#     Returns:
#         list[dict]: A list of dictionaries, one per subject, with calculated power values.
#     """
#     channels_list = region['Both']
#     results = []

#     def extract_channel_data_2d(epochs, ch_name):
#         """Safely extracts a 2D numpy array (n_epochs, n_times) from MNE Epochs."""
#         if epochs is None or ch_name not in epochs.ch_names or ch_name in epochs.info.get('bads', []):
#             return None
#         return epochs.get_data(picks=[ch_name])[:, 0, :]

#     for subject_id in subjects:
#         try:
#             sub_idx_base = all_data['baseline']['ID'].index(subject_id)
#             eeg_epochs_base = all_data['baseline']['EEG'][sub_idx_base]
#             dbs_epochs_base = all_data['baseline'].get('DBS', [None]*len(all_data['baseline']['ID']))[sub_idx_base]
#         except (ValueError, IndexError):
#             print(f"[!] Subject {subject_id} or their baseline data missing. Skipped.")
#             continue

#         subj_results = {'ID': subject_id}
#         subj_results['channels_to_pick'] = channels_list
#         n_times = eeg_epochs_base.times.shape[0] if eeg_epochs_base else 2048
#         nperseg = int(n_times * nperseg_ratio)

#         for block in ['baseline'] + blocks:
#             try:
#                 if block == 'baseline':
#                     eeg_epochs, dbs_epochs = eeg_epochs_base, dbs_epochs_base
#                 else:
#                     sub_idx_blk = all_data[block]['ID'].index(subject_id)
#                     eeg_epochs = all_data[block]['EEG'][sub_idx_blk]
#                     dbs_epochs = all_data[block].get('DBS', [None]*len(all_data[block]['ID']))[sub_idx_blk]
#                     if eeg_epochs == None: 
#                         print(f" no eeg epochs for subject {subject_id}")
#                         for ch in channels_list: subj_results[f'EEG_{block}_{ch}'] = np.nan
#                         subj_results[f'{block}_DBS'] = np.nan
#                         continue
#             except (ValueError, IndexError):
#                 for ch in channels_list: subj_results[f'EEG_{block}_{ch}'] = np.nan
#                 subj_results[f'{block}_DBS'] = np.nan
#                 continue

#             # --- Process EEG Channels ---
#             for ch in channels_list:
#                 # 1. Get all epochs for this channel as a single 2D array
#                 eeg_data_2d = extract_channel_data_2d(eeg_epochs, ch)
                
#                 # 2. Get one averaged PSD from all epochs
#                 f_eeg, pxx_eeg_avg = get_averaged_psd(eeg_data_2d, eeg_epochs.info['sfreq'], 1024)

#                 # 3. Calculate relative power from the single averaged PSD
#                 power = calculate_relative_band_power(f_eeg, pxx_eeg_avg, band_range, norm_freq_range)
#                 subj_results[f'EEG_{block}_{ch}'] = power

#             # --- Process DBS Channel ---
#             dbs_ch_name = dbs_epochs.ch_names[0] if dbs_epochs and dbs_epochs.ch_names else None
#             dbs_data_2d = extract_channel_data_2d(dbs_epochs, dbs_ch_name)
            
#             f_dbs, pxx_dbs_avg = get_averaged_psd(dbs_data_2d, dbs_epochs.info['sfreq'] if dbs_epochs else 200, 128)
            
#             power = calculate_relative_band_power(f_dbs, pxx_dbs_avg, band_range, norm_freq_range)
#             subj_results[f'{block}_DBS'] = power
        
#         results.append(subj_results)

#     return results

# ==============================================================================
# Power percent change
# ==============================================================================

def process_band_power_percent_change_epochs(
        subjects: list,
        all_data: dict,  # Note: This is all_data_dico[group_var]
        blocks: list,
        band_range: tuple,
        region: dict):
    """
    Computes PERCENT CHANGE in band power for task blocks relative to baseline.
    Designed to be a drop-in replacement for `process_band_power_data`
    in your main loop.
    """
    channels_list = region['Both']
    results = []

    def extract_epochs_channel(epochs, ch_name):
        ch_idx = epochs.ch_names.index(ch_name)
        return epochs.get_data()[:, ch_idx, :]

    for subject_id in subjects:
        try:
            sub_index = all_data['baseline']['ID'].index(subject_id)
        except ValueError:
            print(f"[!] Subject {subject_id} missing in 'baseline'. Skipped.")
            continue

        eeg_data_baseline = all_data['baseline']['EEG'][sub_index]
        dbs_data_baseline = all_data['baseline']['DBS'][sub_index]
        
        subj_dict = {'channels_to_pick': channels_list}
        fs_eeg = eeg_data_baseline.info['sfreq'] if eeg_data_baseline else np.nan

        baseline_powers_eeg = {}
        baseline_power_dbs = np.nan

        # Calculate Raw Baseline Power (EEG)
        for ch in channels_list:
            try:
                data_epochs = extract_epochs_channel(eeg_data_baseline, ch)
                if ch in eeg_data_baseline.info.get('bads', []): raise RuntimeError('bad channel')
                powers = [calculate_band_power(e, fs_eeg, band_range) for e in data_epochs]
                baseline_powers_eeg[ch] = np.nanmean(powers)
            except Exception:
                baseline_powers_eeg[ch] = np.nan

        # Calculate Raw Baseline Power (DBS)
        try:
            if dbs_data_baseline:
                fs_dbs = dbs_data_baseline.info['sfreq']
                dbs_epochs = dbs_data_baseline.get_data()[:, 0, :]
                powers = [calculate_band_power(e, fs_dbs, band_range) for e in dbs_epochs]
                baseline_power_dbs = np.nanmean(powers)
        except Exception:
            baseline_power_dbs = np.nan

        # Loop over blocks to compute PERCENT CHANGE
        for block in blocks:
            try:
                blk_index = all_data[block]['ID'].index(subject_id)
                eeg_data_block = all_data[block]['EEG'][blk_index]
                dbs_data_block = all_data[block]['DBS'][blk_index]
            except Exception:
                eeg_data_block, dbs_data_block = None, None

            # EEG percent change
            for ch in channels_list:
                key = f'EEG_{block}_{ch}_pct_change'
                if not eeg_data_block or np.isnan(baseline_powers_eeg.get(ch, np.nan)):
                    subj_dict[key] = np.nan
                    continue
                try:
                    power_block = np.nanmean([calculate_band_power(e, fs_eeg, band_range)
                                             for e in extract_epochs_channel(eeg_data_block, ch)])
                    power_base = baseline_powers_eeg[ch]
                    subj_dict[key] = ((power_block - power_base) / power_base) * 100 if power_base > 0 else np.nan
                except Exception:
                    subj_dict[key] = np.nan

            # DBS percent change
            dbs_key = f'{block}_DBS_pct_change'
            if not dbs_data_block or np.isnan(baseline_power_dbs):
                subj_dict[dbs_key] = np.nan
            else:
                try:
                    fs_dbs_blk = dbs_data_block.info['sfreq']
                    dbs_epochs_blk = dbs_data_block.get_data()[:, 0, :]
                    power_block = np.nanmean([calculate_band_power(e, fs_dbs_blk, band_range) for e in dbs_epochs_blk])
                    subj_dict[dbs_key] = ((power_block - baseline_power_dbs) / baseline_power_dbs) * 100 if baseline_power_dbs > 0 else np.nan
                except Exception:
                    subj_dict[dbs_key] = np.nan
        
        results.append(subj_dict)
    return results

def bonferroni_correction(p_value, n_tests):
    """Apply Bonferroni correction to a p-value based on the number of tests.
    
    Args:
        p_value (float): The p-value to correct.
        n_tests (int): Number of tests being performed.
        
    Returns:
        corrected_p (float): The Bonferroni corrected p-value, capped at 1.0.
    """
    if np.isnan(p_value):
        return np.nan
    return min(p_value * n_tests, 1.0)



def stats_boxplot_data(
    data_to_plot: dict,
    groups       : list,
    blocks       : list,
    categories   : list,
    fdr_method   : str = "fdr_bh",
    alpha        : float = 0.1,
    paired_within: bool = True,
    verbose      : bool = False
):
    """
    Run statistics on boxplot data structure:
        data_to_plot[group][block][category] = list of values (subjects)
    Returns a DataFrame with all test results and FDR-corrected p-values.
    """


    results = []
    # 1. Across-group comparisons
    for block in blocks:
        for cat in categories:
            vals_g0 = data_to_plot.get(groups[0], {}).get(block, {}).get(cat, [])
            vals_g1 = data_to_plot.get(groups[1], {}).get(block, {}).get(cat, [])
            vals_g0 = np.array(vals_g0, dtype=float)
            vals_g1 = np.array(vals_g1, dtype=float)
            vals_g0 = vals_g0[np.isfinite(vals_g0)]
            vals_g1 = vals_g1[np.isfinite(vals_g1)]
            if len(vals_g0) < 3 or len(vals_g1) < 3:
                continue
            norm0 = shapiro(vals_g0)[1] > 0.05 if len(vals_g0) > 3 else False
            norm1 = shapiro(vals_g1)[1] > 0.05 if len(vals_g1) > 3 else False
            if norm0 and norm1:
                stat, p = ttest_ind(vals_g0, vals_g1, equal_var=False, nan_policy='omit')
                test = "t-test (unpaired)"
            else:
                stat, p = mannwhitneyu(vals_g0, vals_g1, alternative='two-sided')
                test = "Mann–Whitney"
            results.append(dict(
                test_type="across-group",
                category=cat, block1=block, group1=groups[0], block2=block, group2=groups[1],
                test=test, stat=stat, p=p, N1=len(vals_g0), N2=len(vals_g1)
            ))
            #print(dict(
            #    test_type="across-group",
            #    category=cat, block1=block, group1=groups[0], block2=block, group2=groups[1],
            #    test=test, stat=stat, p=p, N1=len(vals_g0), N2=len(vals_g1)))

    # 2. Within-group, between-block comparisons
    for group in groups:
        for cat in categories:
            for i, b1 in enumerate(blocks):
                for j, b2 in enumerate(blocks):
                    if j <= i:
                        continue
                    vals_1 = data_to_plot.get(group, {}).get(b1, {}).get(cat, [])
                    vals_2 = data_to_plot.get(group, {}).get(b2, {}).get(cat, [])
                    vals_1 = np.array(vals_1, dtype=float)
                    vals_2 = np.array(vals_2, dtype=float)
                    vals_1 = vals_1[np.isfinite(vals_1)]
                    vals_2 = vals_2[np.isfinite(vals_2)]
                    if len(vals_1) < 3 or len(vals_2) < 3:
                        if not(group == "AgeMatched_NFB" and cat == 'DBS'):
                            print(f"Not enough data for {group} {cat} {b1} vs {b2}")
                            print(f"{b1} has only {len(vals_1)} subjects, {b2} has {len(vals_2)}")
                        continue
                    if paired_within and len(vals_1) == len(vals_2):
                        norm1 = shapiro(vals_1)[1] > 0.05 if len(vals_1) > 3 else False
                        norm2 = shapiro(vals_2)[1] > 0.05 if len(vals_2) > 3 else False
                        if norm1 and norm2:
                            stat, p = ttest_rel(vals_1, vals_2, nan_policy='omit')
                            test = "t-test (paired)"
                        else:
                            try:
                                stat, p = wilcoxon(vals_1, vals_2)
                                test = "Wilcoxon (paired)"
                            except ValueError:
                                stat, p = np.nan, np.nan
                                test = "Wilcoxon (paired, failed)"
                    else:
                        norm1 = shapiro(vals_1)[1] > 0.05 if len(vals_1) > 3 else False
                        norm2 = shapiro(vals_2)[1] > 0.05 if len(vals_2) > 3 else False
                        if norm1 and norm2:
                            stat, p = ttest_ind(vals_1, vals_2, equal_var=False, nan_policy='omit')
                            test = "t-test (unpaired)"
                        else:
                            stat, p = mannwhitneyu(vals_1, vals_2, alternative='two-sided')
                            test = "Mann–Whitney"
                    results.append(dict(
                        test_type="within-group",
                        category=cat, block1=b1, group1=group, block2=b2, group2=group,
                        test=test, stat=stat, p=p, N1=len(vals_1), N2=len(vals_2)
                    ))
                    #print(dict(
                    #    test_type="within-group",
                    #    category=cat, block1=b1, group1=group, block2=b2, group2=group,
                    #    test=test, stat=stat, p=p, N1=len(vals_1), N2=len(vals_2)
                    #))

    # Correction for multiple testing
    df = pd.DataFrame(results)
    if not df.empty:
        reject, p_corr, _, _ = multipletests(df["p"].values, alpha=alpha, method=fdr_method)
        df["p_fdr"] = p_corr
        df["significant"] = reject
    else:
        df["p_fdr"] = []
        df["significant"] = []
    if verbose:
        print(df)
    return df

def summarize_significant_results(stats_df):
    """Summarizes significant within- and across-group results from a stats_df."""
    # Ensure columns exist
    if 'p_fdr' not in stats_df.columns or 'significant' not in stats_df.columns:
        raise ValueError("DataFrame must include 'p_fdr' and 'significant' columns")

    # Only significant results
    sig = stats_df[stats_df["significant"] == True].copy()

    # Within-group summary
    within = sig[sig["test_type"] == "within-group"]
    if not within.empty:
        within_summary = within[[
            "group1", "category", "freq_band", "block1", "block2", "test", "stat", "p", "p_fdr"
        ]].sort_values(by=["freq_band", "category", "group1"])
    else:
        within_summary = None

    # Across-group summary
    across = sig[sig["test_type"] == "across-group"]
    if not across.empty:
        across_summary = across[[
            "category", "block1", "freq_band", "group1", "group2", "test", "stat", "p", "p_fdr"
        ]].sort_values(by=["freq_band", "category", "block1"])
    else:
        across_summary = None

    return within_summary, across_summary
def compute_and_save_bandpower_stats(
    results_dir,
    prefix,
    data_per_freq_band,
    blocks_to_test,
    group_vars,
    categories
):
    """
    Compute statistics for band powers, save results as CSV files.
    
    Parameters
    ----------
    results_dir : str
        Directory to save the result CSV files.
    prefix : str
        Prefix for output file names.
    data_per_freq_band : dict
        Dictionary with frequency band as key, data for each band as value.
    blocks_to_test : list
        List of blocks/conditions to test.
    group_vars : list
        List of group variables for grouping data.
    categories : list
        List of category names for analysis.
    stats_boxplot_data : function
        Function to compute stats DataFrame from band data.
    summarize_significant_results : function
        Function to extract significance summary DataFrames.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_stats = []
    for band_name, data_to_plot in data_per_freq_band.items():
        df = stats_boxplot_data(data_to_plot, group_vars, blocks_to_test, categories)
        df['freq_band'] = band_name
        all_stats.append(df)

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df_path = os.path.join(results_dir, f"{prefix}_stats.csv")
    stats_df.to_csv(stats_df_path, index=False)

    within_sig, across_sig = summarize_significant_results(stats_df)

    if within_sig is not None:
        within_path = os.path.join(results_dir, f"{prefix}_within_group_significant.csv")
        within_sig.to_csv(within_path, index=False)
    else:
        print(f"No significant within-group differences for {prefix}.")

    if across_sig is not None:
        across_path = os.path.join(results_dir, f"{prefix}_across_group_significant.csv")
        across_sig.to_csv(across_path, index=False)
    else:
        print(f"No significant across-group differences for {prefix}.")

    # Optional: return DataFrames in case further in-memory analysis is needed
    return stats_df, within_sig, across_sig


def stats_percent_change_data(
    data_to_plot: dict,
    groups: list,
    blocks: list,
    categories: list,
    fdr_method: str = "fdr_bh",
    alpha: float = 0.05
):
    """
    Run statistics on percent change boxplot data.
    1. One-sample test vs. 0: Is the change significant within a group/block?
    2. Across-group test: Is the change different between groups?
    
    Based on the structure of `stats_boxplot_data`.
    """
    results = []
    
    # 1. One-sample tests (is the change significantly different from 0?)
    for group in groups:
        for block in blocks:
            for cat in categories:
                vals = data_to_plot.get(group, {}).get(block, {}).get(cat, [])
                vals = np.array(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 3: continue
                
                # Use Wilcoxon signed-rank test (non-parametric one-sample) if not normal
                is_normal = shapiro(vals)[1] > 0.05 if len(vals) > 3 else False
                if is_normal:
                    stat, p = ttest_1samp(vals, popmean=0)
                    test = "t-test (one-sample)"
                else:
                    # Wilcoxon tests if the distribution is centered around 0
                    stat, p = wilcoxon(vals)
                    test = "Wilcoxon (one-sample)"
                
                results.append(dict(
                    test_type="one-sample_vs_0",
                    category=cat, block1=block, group1=group,
                    test=test, stat=stat, p=p, N1=len(vals)
                ))

    # 2. Across-group comparisons (is the % change different between groups?)
    # This part is identical to your original function, just operating on % change data.
    for block in blocks:
        for cat in categories:
            vals_g0 = data_to_plot.get(groups[0], {}).get(block, {}).get(cat, [])
            vals_g1 = data_to_plot.get(groups[1], {}).get(block, {}).get(cat, [])
            # (The rest of this logic is copied from your stats_boxplot_data)
            vals_g0 = np.array(vals_g0, dtype=float)[np.isfinite(vals_g0)]
            vals_g1 = np.array(vals_g1, dtype=float)[np.isfinite(vals_g1)]
            if len(vals_g0) < 3 or len(vals_g1) < 3: continue
            norm0 = shapiro(vals_g0)[1] > 0.05 if len(vals_g0) > 3 else False
            norm1 = shapiro(vals_g1)[1] > 0.05 if len(vals_g1) > 3 else False
            if norm0 and norm1:
                stat, p = ttest_ind(vals_g0, vals_g1, equal_var=False)
                test = "t-test (unpaired)"
            else:
                stat, p = mannwhitneyu(vals_g0, vals_g1, alternative='two-sided')
                test = "Mann–Whitney"
            results.append(dict(
                test_type="across-group",
                category=cat, block1=block, group1=groups[0], block2=block, group2=groups[1],
                test=test, stat=stat, p=p, N1=len(vals_g0), N2=len(vals_g1)
            ))

    # Correction for multiple testing (identical to your original)
    df = pd.DataFrame(results)
    if not df.empty:
        reject, p_corr, _, _ = multipletests(df["p"].dropna().values, alpha=alpha, method=fdr_method)
        df.loc[df["p"].notna(), "p_fdr"] = p_corr
        df.loc[df["p"].notna(), "significant"] = reject
    return df