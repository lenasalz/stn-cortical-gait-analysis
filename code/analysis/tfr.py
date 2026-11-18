# imports
import os
import warnings
from typing import Dict, List, Any, Tuple, Optional

import mne
from mne.time_frequency import AverageTFR
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# External dependencies from project structure
from epoching import filter_gait_events, segment_gait_cycles, replace_events_in_mne
from data_loader import save_data_to_pickle


def extract_tfr_epochs_from_cycles(continuous_tfr, gait_cycles):
    """
    Fast path: given precomputed gait cycles (list of dicts with 'HSrightlocs' and 'HSrightlocs_end'),
    slice the provided continuous TFR into a list of epochs without re-running event detection.
    Returns a list of numpy arrays shaped (1, n_freqs, n_times).
    """
    tfr_epochs = []
    if continuous_tfr is None or not gait_cycles:
        return tfr_epochs
    for cycle in gait_cycles:
        if 'HSrightlocs' not in cycle or 'HSrightlocs_end' not in cycle:
            continue
        start_time = cycle['HSrightlocs']
        end_time = cycle['HSrightlocs_end']
        try:
            start_idx = continuous_tfr.time_as_index(start_time, use_rounding=True)[0]
            end_idx = continuous_tfr.time_as_index(end_time, use_rounding=True)[0]
        except IndexError:
            continue
        epoch_data = continuous_tfr.data[:, :, start_idx:end_idx]
        if epoch_data.shape[2] > 0:
            tfr_epochs.append(epoch_data)
    return tfr_epochs 

def compute_tfr_on_subset_and_normalize_optimized(
    all_data_dico, group_vars, subjects_dico, blocks, freqs, motor_areas, frontal_areas,
    target_sfreq=None, tfr_decim=1, n_jobs=1
):
    """
    Computes continuous TFR on a SUBSET of channels and normalizes by baseline.
    
    OPTIMIZED VERSION: This function calculates the TFR for all channels at once
    to minimize computational load, then normalizes each channel, and finally
    averages the data into the predefined regions.

    Performance knobs:
        - target_sfreq: If set, resamples the raw data before TFR computation.
        - tfr_decim: Decimation factor for TFR (higher = faster, lower time resolution).
        - n_jobs: Number of parallel jobs for TFR computation.
    """
    # --- 1. Define Analysis Parameters ---
    desired_fwhm_s = 0.200
    sigma = desired_fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    n_cycles_evolving = 2 * np.pi * np.array(freqs) * sigma
    print(f"--- TFR Analysis (Optimized) ---")
    print(f"Using evolving wavelets for ~{desired_fwhm_s*1000:.0f} ms temporal resolution.")
    print(f"Resample before TFR: {target_sfreq if target_sfreq else 'native sfreq'}, decim: {tfr_decim}, n_jobs: {n_jobs}")

    # --- Define all channels of interest to compute TFR on them only once ---
    all_eeg_channels = sorted(list(set(
        motor_areas['Left'] + motor_areas['Right'] + motor_areas['Central'] +
        frontal_areas['Left'] + frontal_areas['Right'] + frontal_areas['Central']
    )))
    # STN is just one channel, handled separately
    stn_channel = ['DBS'] 

    regions_to_process = {'motor': motor_areas, 'frontal': frontal_areas}

    # --- 2. Initialize Output Dictionary ---
    all_data_tfr_continuous_dico = {}
    for group_var in group_vars:
        all_data_tfr_continuous_dico[group_var] = {}
        for block in ['baseline'] + blocks:
            all_data_tfr_continuous_dico[group_var][block] = {
                "ID": [],
                "motor": {'Left': [], 'Right': [], 'Central': [], 'Both': []},
                "frontal": {'Left': [], 'Right': [], 'Central': [], 'Both': []},
                "STN": {'Left': [], 'Right': []}
            }

    # --- 3. Main Loop ---
    for group_var in group_vars:
        for subj in subjects_dico[group_var]:
            print(f"Processing TFRs for {group_var} {subj}...")

            # --- A. COMPUTE BASELINE TFRs (Once per subject) ---
            try:
                base_idx = all_data_dico[group_var]['baseline']['ID'].index(subj)
                eeg_base_raw_full = all_data_dico[group_var]['baseline']['EEG'][base_idx]
                dbs_base_raw_full = all_data_dico[group_var]['baseline'].get('DBS_proc', [None])[base_idx]
            except (ValueError, IndexError):
                warnings.warn(f"Baseline data not found for subject {subj}. Skipping.")
                continue
            
            # --- EEG BASELINE ---
            base_eeg_power_map = {}
            if eeg_base_raw_full:
                present_eeg_chs = [ch for ch in all_eeg_channels if ch in eeg_base_raw_full.ch_names]
                if present_eeg_chs:
                    eeg_base_raw_picked = eeg_base_raw_full.copy().pick(present_eeg_chs, verbose=False)
                    if target_sfreq is not None:
                        eeg_base_raw_picked.resample(target_sfreq, npad="auto")
                    # Compute TFR for all EEG channels at once, keeping individual channels
                    tfr_eeg_base = mne.time_frequency.tfr_morlet(
                        eeg_base_raw_picked, freqs=freqs, n_cycles=n_cycles_evolving,
                        return_itc=False, average=False, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                    )
                    # Create a map of channel_name -> baseline_power_vector
                    base_power_vectors = tfr_eeg_base.data.mean(axis=2) # Shape: (n_channels, n_freqs)
                    # or use median; less skewed by transients
                    # base_power_vectors = np.median(tfr_eeg_base.data, axis=2)
                    for i, ch_name in enumerate(tfr_eeg_base.ch_names):
                        base_eeg_power_map[ch_name] = base_power_vectors[i, :]

            # --- STN/DBS BASELINE ---
            base_dbs_power_vector = None
            if dbs_base_raw_full:
                if stn_channel[0] in dbs_base_raw_full.ch_names:
                    dbs_base_proc = dbs_base_raw_full.copy()
                    if target_sfreq is not None:
                         dbs_base_proc.resample(target_sfreq, npad="auto")
                    tfr_dbs_base = mne.time_frequency.tfr_morlet(
                        dbs_base_proc, freqs=freqs, n_cycles=n_cycles_evolving,
                        return_itc=False, average=False, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                    )
                    base_dbs_power_vector = tfr_dbs_base.data.mean(axis=2).squeeze()

            # --- B. PROCESS TASK BLOCKS ---
            for block in blocks:
                all_data_tfr_continuous_dico[group_var][block]['ID'].append(subj)
                
                try:
                    task_idx = all_data_dico[group_var][block]['ID'].index(subj)
                    eeg_task_raw_full = all_data_dico[group_var][block]['EEG'][task_idx]
                    dbs_task_raw_full = all_data_dico[group_var][block].get('DBS_proc', [None])[task_idx]
                except (ValueError, IndexError):
                    # Append None if task block is missing
                    for region_name, sides in regions_to_process.items():
                        for side_name in sides:
                            all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)
                    continue

                # --- NORMALIZE AND AVERAGE EEG REGIONS ---
                if eeg_task_raw_full and base_eeg_power_map:
                    present_task_chs = [ch for ch in all_eeg_channels if ch in eeg_task_raw_full.ch_names]
                    common_chs = [ch for ch in present_task_chs if ch in base_eeg_power_map]
                    
                    if common_chs:
                        eeg_task_raw_picked = eeg_task_raw_full.copy().pick(common_chs, verbose=False)
                        if target_sfreq is not None:
                            eeg_task_raw_picked.resample(target_sfreq, npad="auto")
                        # Compute TFR for all task channels at once
                        tfr_eeg_task = mne.time_frequency.tfr_morlet(
                            eeg_task_raw_picked, freqs=freqs, n_cycles=n_cycles_evolving,
                            return_itc=False, average=False, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                        )
                        
                        # Normalize each channel by its own baseline
                        normalized_data = np.zeros_like(tfr_eeg_task.data)
                        for i, ch_name in enumerate(tfr_eeg_task.ch_names):
                            if ch_name in base_eeg_power_map:
                                baseline = base_eeg_power_map[ch_name][:, np.newaxis]
                                normalized_data[i, :, :] = 10 * np.log10(tfr_eeg_task.data[i, :, :] / baseline)
                        
                        tfr_eeg_task.data = normalized_data

                        # Now, create the region-averaged TFRs
                        for region_name, sides in regions_to_process.items():
                            for side_name, side_chs in sides.items():
                                region_chs_present = [ch for ch in side_chs if ch in tfr_eeg_task.ch_names]
                                if region_chs_present:
                                    # 1. Pick the channels for the current region
                                    region_tfr = tfr_eeg_task.copy().pick(region_chs_present, verbose=False)
                                    
                                    # 2. Manually average the data across the channel axis (axis=0)
                                    #    keepdims=True ensures the output shape is (1, n_freqs, n_times)
                                    avg_data = region_tfr.data.mean(axis=0, keepdims=True)
                                    
                                    # 3. Create a new info object from scratch for our new "virtual channel".
                                    #    This is the robust way to avoid the RuntimeError.
                                    #    We create a more descriptive name, e.g., "motor_Left".
                                    new_ch_name = f"{region_name}_{side_name}"
                                    avg_tfr_info = mne.create_info(
                                        ch_names=[new_ch_name],
                                        sfreq=tfr_eeg_task.info['sfreq'],
                                        # Use 'misc' as the channel type for this virtual, averaged channel.
                                        ch_types='misc' 
                                    )
                                    
                                    # 4. Construct the final AverageTFR object using the new info and averaged data.
                                    avg_tfr_state = {
                                        "data": avg_data,
                                        "times": region_tfr.times,
                                        "freqs": region_tfr.freqs,
                                        "info": avg_tfr_info,
                                        "nave": int(getattr(region_tfr, "nave", 1)),
                                        "method": "morlet",
                                        "comment": new_ch_name,
                                    }
                                    avg_tfr = mne.time_frequency.AverageTFR(inst=avg_tfr_state)
                                    #
                                    all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(avg_tfr)
                                else:
                                    all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)
                    else: # No common channels
                         for region_name, sides in regions_to_process.items():
                            for side_name in sides:
                                all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)

                # --- NORMALIZE STN ---
                norm_dbs_tfr = None
                if dbs_task_raw_full and base_dbs_power_vector is not None:
                    if stn_channel[0] in dbs_task_raw_full.ch_names:
                        dbs_task_proc = dbs_task_raw_full.copy()
                        if target_sfreq is not None:
                            dbs_task_proc.resample(target_sfreq, npad="auto")
                        tfr_dbs_task = mne.time_frequency.tfr_morlet(
                            dbs_task_proc, freqs=freqs, n_cycles=n_cycles_evolving,
                            return_itc=False, average=True, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                        )
                        tfr_dbs_task.data = 10 * np.log10(tfr_dbs_task.data / base_dbs_power_vector[:, np.newaxis])
                        norm_dbs_tfr = tfr_dbs_task     

                # Store STN result based on side
                dbs_side_info = dbs_base_raw_full.info.get('description', '').lower() if dbs_base_raw_full else ''
                if 'dbs side: left' in dbs_side_info:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(norm_dbs_tfr)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)
                elif 'dbs side: right' in dbs_side_info:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(norm_dbs_tfr)
                else:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)

    return all_data_tfr_continuous_dico

def warp_tfr_cycles(tfr_epochs_data, gait_cycles_info, event_percentages, sfreq):
    """
    Applies linear time warping to a list of TFR epochs.

    FIXES:
    - Sort events by their target percentage to ensure monotonic order.
    - Enforce strictly increasing original and target indices (resolve rounding ties).
    - Always fill the whole epoch from 0% to 100% so there are no zeroed tails.

    Args:
        tfr_epochs_data (list of np.ndarray):
            List of TFR epochs, where each epoch is a (1, n_freqs, n_times) array.
        gait_cycles_info (list of dict):
            List of dictionaries containing the absolute event times for each cycle.
        event_percentages (dict):
            Maps event names to target percentages (0-100).
        sfreq (float):
            Sampling frequency of the TFR data (after any decimation).

    Returns:
        list: A list of warped TFR epochs (numpy arrays).
    """
    warped_epochs = []

    # Sort events by percentage (0 -> 100). Dict order can be arbitrary in inputs.
    ordered_events = [k for k, _ in sorted(event_percentages.items(), key=lambda kv: kv[1])]

    for i, tfr_epoch in enumerate(tfr_epochs_data):
        cycle_info = gait_cycles_info[i]
        n_freqs = tfr_epoch.shape[1]
        n_times = tfr_epoch.shape[2]

        # Relative time axis within this epoch
        epoch_times = np.linspace(0, n_times / sfreq, n_times, endpoint=True)
        cycle_start_time = cycle_info.get('HSrightlocs', next(iter(cycle_info.values())))

        # --- Build original and target event indices ---
        orig_idx = []
        for ev in ordered_events:
            # absolute -> relative (s)
            relative_time = float(cycle_info[ev] - cycle_start_time)
            idx = int(np.argmin(np.abs(epoch_times - relative_time)))
            orig_idx.append(idx)
        orig_idx = np.asarray(orig_idx, dtype=int)

        tgt_idx = np.asarray([int(round((event_percentages[ev] / 100.0) * (n_times - 1))) for ev in ordered_events], dtype=int)

        # Force endpoints and strict monotonicity to avoid empty segments
        tgt_idx[0] = 0
        tgt_idx[-1] = n_times - 1
        for arr in (orig_idx, tgt_idx):
            for j in range(1, len(arr)):
                if arr[j] <= arr[j - 1]:
                    arr[j] = min(arr[j - 1] + 1, n_times - 1)

        # --- Piecewise-linear warp between consecutive events ---
        warped_epoch_data = np.zeros((n_freqs, n_times), dtype=tfr_epoch.dtype)
        for seg in range(len(ordered_events) - 1):
            o0, o1 = orig_idx[seg], orig_idx[seg + 1]
            t0, t1 = tgt_idx[seg], tgt_idx[seg + 1]
            if o0 >= o1 or t0 >= t1:
                # nothing to map for this segment after guarding above
                continue
            segment = tfr_epoch[0, :, o0:o1 + 1]  # (n_freqs, L)
            orig_lin = np.linspace(0, 1, segment.shape[1])
            new_lin = np.linspace(0, 1, t1 - t0 + 1)
            f = interp1d(orig_lin, segment, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
            warped_epoch_data[:, t0:t1 + 1] = f(new_lin)

        warped_epochs.append(warped_epoch_data[np.newaxis, :, :])

    return warped_epochs

def resample_tfr_cycles(warped_tfr_epochs, desired_duration, sfreq):
    """
    Resamples a list of warped TFR epochs to a fixed length.

    Args:
        warped_tfr_epochs (list of np.ndarray):
            List of warped TFR epochs (1, n_freqs, n_times arrays).
        desired_duration (float):
            The target duration in seconds.
        sfreq (float):
            The sampling frequency of the TFR data.

    Returns:
        list: A list of resampled TFR epochs (numpy arrays).
    """
    desired_length = int(round(desired_duration * sfreq))
    resampled_epochs = []

    for warped_epoch in warped_tfr_epochs:

        original_length = warped_epoch.shape[2]
        original_x = np.linspace(0, 1, original_length)
        target_x = np.linspace(0, 1, desired_length)
        segment = warped_epoch[0, :, :]  # (n_freqs, T)
        f = interp1d(original_x, segment, kind='linear', axis=1, fill_value='extrapolate')
        resampled = f(target_x)  # (n_freqs, desired_length)
        resampled_epochs.append(resampled[np.newaxis, :, :])

    return resampled_epochs


def create_gait_warped_tfr_epochs(
    all_data_tfr_continuous_dico,
    all_data_dico,  # Original raw data for annotations
    rejection_mask_dico, # Placeholder for epoch rejection masks
    event_percentages,
    gait_events_correct_sequence,
    epoch_duration=1.1
):
    """
    Orchestrates the segmentation, warping, and averaging of continuous TFR data
    based on gait cycles.

    Args:
        all_data_tfr_continuous_dico (dict):
            The output from our previous step, containing continuous AverageTFRs.
        all_data_dico (dict):
            The original data dictionary containing mne.Raw objects with annotations.
        rejection_mask_dico (dict):
            A dictionary mapping (subject, block) -> boolean_mask. The mask
            indicates which epochs to KEEP (True) or reject (False).
        event_percentages (dict):
            Maps gait event names to target percentages (0-100).
        gait_events_correct_sequence (list):
            The ordered list of events defining a valid gait cycle.
        epoch_duration (float):
            The desired final duration of the warped gait cycle in seconds.

    Returns:
        dict: A new dictionary with the same structure, but each entry contains a
              single AverageTFR object representing the mean warped gait cycle.
    """
    all_data_tfr_epoched_dico = {} # Initialize the final output dict
    all_data_tfr_epoched_dico_all_epochs = {}
    # Deep copy the structure from the input dict
    for group, blocks_data in all_data_tfr_continuous_dico.items():
        all_data_tfr_epoched_dico[group] = {}
        all_data_tfr_epoched_dico_all_epochs[group] = {}
        for block, data in blocks_data.items():
            # Prepare the nested dictionary structure
            all_data_tfr_epoched_dico[group][block] = {
                "ID": [], "motor": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "frontal": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "STN": {'Left': [], 'Right': []}
            }
            all_data_tfr_epoched_dico_all_epochs[group][block] = {
                "ID": [], "motor": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "frontal": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "STN": {'Left': [], 'Right': []}
            }

    # --- Main Loop ---
    for group, group_data in all_data_tfr_continuous_dico.items():
        for block, block_data in group_data.items():
            # We don't process baseline as it has no gait events
            if block == 'baseline':
                continue

            for i_subj, subj_id in enumerate(block_data['ID']):
                print(f"--- Processing {group} {subj_id} - {block} ---")
                all_data_tfr_epoched_dico[group][block]['ID'].append(subj_id)
                all_data_tfr_epoched_dico_all_epochs[group][block]['ID'].append(subj_id)

                # Get the original Raw object to extract annotations
                try:
                    raw_idx = all_data_dico[group][block]['ID'].index(subj_id)
                    eeg_raw_for_events = all_data_dico[group][block]['EEG'][raw_idx]
                    dbs_raw_for_events = all_data_dico[group][block].get('DBS_proc', [None])[raw_idx]
                except (ValueError, IndexError):
                    print(f"  > Warning: Could not find original Raw data for {subj_id} in {block}. Skipping.")
                    continue

                                # Precompute gait cycles once (avoid recomputing for each region)
                gait_cycles_cache_eeg, gait_cycles_cache_dbs = [], []
                if 'EEG' in all_data_dico[group][block] and eeg_raw_for_events is not None:
                    _fe_eeg = filter_gait_events(eeg_raw_for_events, gait_events_correct_sequence)
                    _eeg_filt = replace_events_in_mne(eeg_raw_for_events, _fe_eeg, gait_events_correct_sequence)
                    gait_cycles_cache_eeg = segment_gait_cycles(_eeg_filt, gait_events_correct_sequence)
                if dbs_raw_for_events is not None:
                    _fe_dbs = filter_gait_events(dbs_raw_for_events, gait_events_correct_sequence)
                    _dbs_filt = replace_events_in_mne(dbs_raw_for_events, _fe_dbs, gait_events_correct_sequence)
                    gait_cycles_cache_dbs = segment_gait_cycles(_dbs_filt, gait_events_correct_sequence)


                # Process each region (motor, frontal, STN)
                for region_name in ['motor', 'frontal', 'STN']:
                            
                    for side_name in all_data_tfr_continuous_dico[group][block][region_name]:
                        continuous_tfr = block_data[region_name][side_name][i_subj]

                        if continuous_tfr is None:
                            all_data_tfr_epoched_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_epoched_dico_all_epochs[group][block][region_name][side_name].append(None)
                            continue

                        # Determine which raw object has the events for this region
                        raw_for_events = dbs_raw_for_events if region_name == 'STN' else eeg_raw_for_events

                        # 1. SEGMENT into gait cycles (use cached events)
                        if region_name == 'STN' and gait_cycles_cache_dbs:
                            gait_cycles_info = gait_cycles_cache_dbs
                        else:
                            gait_cycles_info = gait_cycles_cache_eeg
                        tfr_epochs_data = extract_tfr_epochs_from_cycles(continuous_tfr, gait_cycles_info)
                        if not tfr_epochs_data:
                            print(f"  > No valid gait cycles found for {region_name}-{side_name}.")
                            all_data_tfr_epoched_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_epoched_dico_all_epochs[group][block][region_name][side_name].append(None)
                            continue

                        # 2. WARP each cycle based on event percentages using the TFR's sampling rate (post-decimation)
                        tfr_sfreq = continuous_tfr.info['sfreq']
                        warped_tfr_epochs = warp_tfr_cycles(
                            tfr_epochs_data, gait_cycles_info, event_percentages, tfr_sfreq
                        )
                        # 3. RESAMPLE each warped cycle to a fixed duration (same TFR sampling rate)
                        resampled_tfr_epochs = resample_tfr_cycles(
                            warped_tfr_epochs, epoch_duration, tfr_sfreq
                        )

                        # 4. APPLY Rejection Mask
                        mask_key = (subj_id, block)
                        rejection_mask = rejection_mask_dico.get(mask_key)

                        if rejection_mask is None:
                            print(f"  > Warning: No rejection mask found for {mask_key}. Keeping all epochs.")
                            final_epochs_data = np.array(resampled_tfr_epochs)
                        else:
                            if len(rejection_mask) != len(resampled_tfr_epochs):
                                print(f"  > ERROR: Mismatch between TFR epochs ({len(resampled_tfr_epochs)}) and rejection mask ({len(rejection_mask)}) for {mask_key}. Keeping all.")
                                final_epochs_data = np.array(resampled_tfr_epochs)
                            else:
                                # Apply the mask to keep only good epochs
                                final_epochs_data = np.array(resampled_tfr_epochs)[rejection_mask]
                                print(f"  > Applied rejection mask: kept {final_epochs_data.shape[0]}/{len(rejection_mask)} epochs.")
                        
                        if final_epochs_data.shape[0] == 0:
                            print(f"  > All epochs rejected for {region_name}-{side_name}.")
                            all_data_tfr_epoched_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_epoched_dico_all_epochs[group][block][region_name][side_name].append(None)
                            continue

                        # 5. AVERAGE valid epochs and create final MNE object
                        mean_tfr_data = final_epochs_data.mean(axis=0, keepdims=True)
                        
                        # Create a new info object for the epoched data
                        tmin = 0.0
                        try:
                            stacked_data = np.stack(final_epochs_data, axis=0)
                        except ValueError as e:
                            print("ERROR: Could not stack the epochs. Check if they all have the same shape after warping.")
                            # You might want to add more debugging here, like printing the shape of each epoch in the list
                            for i, epoch in enumerate(final_epochs_data):
                                print(f"Shape of epoch {i}: {epoch.shape}")
                            raise e
                        
                        # 2. Now, calculate the mean across the 'epochs' dimension (axis=0).
                        # The shape goes from (80, channels, 45, 2200) to (channels, 45, 2200).
                        mean_tfr_data = stacked_data.mean(axis=0)
                        
                        # 3. Create the metadata for the final object.
                        info = mne.create_info(
                            ch_names=[f"{region_name}_{side_name}"],
                            sfreq=tfr_sfreq,
                            ch_types='misc'
                        )
                        
                        # 4. Create the corresponding time vector.
                        new_times = np.linspace(tmin, tmin + epoch_duration, mean_tfr_data.shape[1]) # Note: shape is now (freq, time) so axis is 1
                        
                        # Let's fix the shape of mean_tfr_data and new_times
                        # The output of mean_tfr_data from stacked_data.mean(axis=0) will be (1, 45, 2200)
                        # which is (n_channels, n_freqs, n_times). So the time dimension is axis 2.
                        mean_tfr_data = stacked_data.mean(axis=0, keepdims=False) # Shape will be (1, 45, 2200)
                        if mean_tfr_data.ndim == 2: # If for some reason the channel dimension was squeezed
                            mean_tfr_data = mean_tfr_data[np.newaxis, :, :]
                        
                        new_times = np.linspace(tmin, tmin + epoch_duration, mean_tfr_data.shape[2])
                        
                        # 5. Call the constructor with explicit keywords. It will now work because
                        #    'mean_tfr_data' is a single, correctly shaped array, and 'new_times' matches it.
                        # final_avg_tfr = mne.time_frequency.AverageTFR(
                        #     info=info,
                        #     data=mean_tfr_data,
                        #     times=new_times,
                        #     freqs=continuous_tfr.freqs,
                        #     nave=stacked_data.shape[0] # nave is the number of stacked epochs
                        # )

                        # 5. Build state dict and construct AverageTFR via `inst` (MNE >=1.6 API)
                        avg_tfr_state = {
                            "data": mean_tfr_data,
                            "times": new_times,
                            "freqs": continuous_tfr.freqs,
                            "info": info,
                            "nave": int(stacked_data.shape[0]),  # number of stacked epochs
                            "method": "morlet",
                            "comment": f"{region_name}_{side_name}",
                        }
                        final_avg_tfr = mne.time_frequency.AverageTFR(inst=avg_tfr_state)

                        # 6. Append the final, valid object.
                        all_data_tfr_epoched_dico[group][block][region_name][side_name].append(final_avg_tfr)
                        all_data_tfr_epoched_dico_all_epochs[group][block][region_name][side_name].append(final_epochs_data)
                        # plot_title_prefix = f"Subject: {subj_id} | Block: {block} | Region: {region_name}_{side_name}"

                        # # Call the plotting function with all the required pieces of data
                        # plot_tfr_processing_stages(
                        #     continuous_tfr=continuous_tfr,
                        #     gait_cycle_start_time=list_of_gait_start_times[0],
                        #     gait_cycle_end_time=list_of_gait_end_times[0],
                        #     unwarped_epoch_data=list_of_unwarped_epochs[0],
                        #     unwarped_epoch_times=list_of_unwarped_times[0],
                        #     warped_epoch_data=list_of_warped_epochs[0],
                        #     warped_epoch_times=warped_times_vector,
                        #     final_averaged_tfr=final_averaged_tfr,
                        #     title_prefix=plot_title_prefix
                        # )

    return all_data_tfr_epoched_dico, all_data_tfr_epoched_dico_all_epochs


def process_and_save_tfr_data(
    all_data_dico,
    group_vars,
    subjects_dico,
    blocks,
    freqs,
    motor_areas,
    frontal_areas,
    rejection_mask_dico,
    event_percentages,
    gait_events_correct_sequence,
    tfr_resultsDir,
    *,
    target_sfreq=None,
    tfr_decim=1,
    n_jobs=1,
):
    """
    Calculates TFR on continuous data, normalizes it, epochs it (gait-warped), 
    applies a rejection mask, restructures the data, and saves the results to disk.

    The function performs the following steps:
    1. Computes continuous TFR and normalizes by baseline.
    2. Epochs the TFR based on gait events.
    3. Saves intermediate epoched data (pickles).
    4. Restructures data into group/block/region format, separating raw arrays from TFR objects.
    5. Saves final aggregated results.

    Warning: This method assumes specific dictionary structures for areas. 
    It would need refinement if additional areas were to be added.
    
    Args:
        all_data_dico (dict): Dictionary containing all the raw data.
        group_vars (list): List of group variables (e.g., ['PD', 'Healthy']).
        subjects_dico (dict): Dictionary mapping groups to subject IDs.
        blocks (list): List of experimental blocks/conditions.
        freqs (array-like): Array of frequencies used for TFR computation.
        motor_areas (dict): Dictionary defining motor EEG channels/areas.
        frontal_areas (dict): Dictionary defining frontal EEG channels/areas.
        rejection_mask_dico (dict): Dictionary containing the boolean rejection mask for epochs.
        event_percentages (list): List of event percentages used for gait warping/epoching.
        gait_events_correct_sequence (list): The expected sequence of gait events for validation.
        tfr_resultsDir (str): Path to the directory where results will be saved.
        target_sfreq (float, optional): Target sampling frequency for resampling TFR data. Defaults to None.
        tfr_decim (int, optional): Decimation factor for TFR computation. Defaults to 1.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.

    Returns:
        tuple: A tuple containing three dictionaries:
        
        - **tfr_group_results** (dict): Nested dictionary containing the **raw numpy arrays** 
          of the TFR data, averaged on all epochs. Structure: `[group][block][region][side]`.
        
        - **avg_tfr_group_results** (dict): Nested dictionary containing the **TFR objects** 
          (e.g., MNE AverageTFR objects) containing metadata + data. 
          Structure: `[group][block][region][side]`.
        
        - **all_data_tfr_epoched_all_epochs** (dict): The comprehensive dictionary of the 
          epoched TFR data before the final restructuring.
    """

    # ==========================================================================
    # Calculate TFR on continuous data, epoch it, apply rejection mask, save epoched dicos to results directory
    # ==========================================================================

    print("--- Starting New Pipeline: Morlet -> Epoch ---")
    print(f"Performance settings → target_sfreq={target_sfreq}, tfr_decim={tfr_decim}, n_jobs={n_jobs}")

    # Step 1: Compute continuous TFR (already respects native sfreq)
    print("\nStep 1: Computing continuous TFR and normalizing by baseline...")
    all_data_tfr_continuous = compute_tfr_on_subset_and_normalize_optimized(
    # print("\nStep 1: Computing continuous TFR...")
    # all_data_tfr_continuous = compute_tfr_on_subset(
        all_data_dico, group_vars, subjects_dico, blocks, freqs, motor_areas, frontal_areas,
        target_sfreq=target_sfreq, tfr_decim=tfr_decim, n_jobs=n_jobs
    )

    # After generating the rejection mask dico
    print("\nStep 2: epoching the TFR...")
    all_data_tfr_epoched, all_data_tfr_epoched_all_epochs = create_gait_warped_tfr_epochs(
        all_data_tfr_continuous,
        all_data_dico,
        rejection_mask_dico,
        event_percentages,
        gait_events_correct_sequence,
    )
    save_data_to_pickle(
        all_data_tfr_epoched_all_epochs, "all_data_tfr_epoched_all_epochs", tfr_resultsDir
    )
    save_data_to_pickle(all_data_tfr_epoched, "all_data_tfr_epoched", tfr_resultsDir)

    print("\nStep 3: Restructuring epoched data into final result dictionaries...")

    # Use the epoched data as the source
    source_data = all_data_tfr_epoched

    # Initialize the final output dictionaries
    tfr_group_results = {}
    avg_tfr_group_results = {}

    for group_var in group_vars:
        tfr_group_results[group_var] = {}
        avg_tfr_group_results[group_var] = {}

        for block in blocks:
            # Initialize dictionaries for each block, with STN as a dictionary
            tfr_group_results[group_var][block] = {
                "ID": [],
                "EEG_motor": {},
                "EEG_frontal": {},
                "STN": {},
            }
            avg_tfr_group_results[group_var][block] = {
                "ID": [],
                "EEG_motor": {},
                "EEG_frontal": {},
                "STN": {},
            }

            if block not in source_data.get(group_var, {}):
                print(f"  - Block '{block}' not found for group '{group_var}'. Skipping.")
                continue

            subject_ids_in_block = source_data[group_var][block]["ID"]
            tfr_group_results[group_var][block]["ID"] = subject_ids_in_block
            avg_tfr_group_results[group_var][block]["ID"] = subject_ids_in_block

            for idx, subject_id in enumerate(subject_ids_in_block):

                # --- Process EEG Regions (Motor and Frontal) ---
                for region_key in ["motor", "frontal"]:
                    target_key = f"EEG_{region_key}"
                    source_region_dict = source_data[group_var][block][region_key]

                    for side_name, tfr_list in source_region_dict.items():
                        tfr_object = tfr_list[idx]
                        avg_tfr_group_results[group_var][block][target_key].setdefault(
                            side_name, []
                        ).append(tfr_object)

                        # Squeeze the data array to remove the channel dimension
                        data_array = (
                            tfr_object.data.squeeze() if tfr_object is not None else None
                        )
                        tfr_group_results[group_var][block][target_key].setdefault(
                            side_name, []
                        ).append(data_array)

                # --- Process STN Data ---
                source_stn_dict = source_data[group_var][block]["STN"]

                for side_name, tfr_list in source_stn_dict.items():
                    tfr_object = tfr_list[idx]
                    avg_tfr_group_results[group_var][block]["STN"].setdefault(
                        side_name, []
                    ).append(tfr_object)

                    # Squeeze the data array to remove the channel dimension
                    data_array = (
                        tfr_object.data.squeeze() if tfr_object is not None else None
                    )
                    tfr_group_results[group_var][block]["STN"].setdefault(
                        side_name, []
                    ).append(data_array)

    print("Step 3 complete.")

    # ==========================================================================
    # 4. SAVE THE FINAL AGGREGATED RESULTS
    # ==========================================================================
    print("\nStep 4: Saving final result dictionaries...")
    save_data_to_pickle(
        tfr_group_results, "tfr_group_results_new_method", tfr_resultsDir
    )
    save_data_to_pickle(
        avg_tfr_group_results, "avg_tfr_group_results_new_method", tfr_resultsDir
    )

    return tfr_group_results, avg_tfr_group_results, all_data_tfr_epoched_all_epochs

    print("\n--- Pipeline Finished Successfully! ---")



# TFR PLOTTING:

# --- Shared heatmap font policy -------------------------------------------
# A figure's *on-page* font size equals its point size multiplied by the
# LaTeX scale factor (print_width / figure_width). Because the cortical grid
# (28 in wide) is squeezed into \textwidth while the STN (13 in) and
# ipsi/contra (6.5 in at 0.48\textwidth) figures are not, identical point
# sizes render at very different sizes on the page. To keep axis labels
# visually consistent across every heatmap, each figure derives its font
# sizes from a single target *rendered* size, scaled by its own geometry.
TARGET_LABEL_PT_ON_PAGE = 8.0          # desired rendered axis-label size (pt)
PRINT_FULLWIDTH_IN = 6.9               # sn-jnl \textwidth (figure* span), inches
_BASE_HEATMAP_FONTS = {
    'figure_title': 28, 'title': 20, 'label': 20,
    'cbar_label': 24, 'ticks': 16, 'gait_ticks': 20,
}


def _scaled_heatmap_fonts(fig_width_in: float, print_width_in: float) -> dict:
    """Scale ``_BASE_HEATMAP_FONTS`` so axis labels render at
    ``TARGET_LABEL_PT_ON_PAGE`` once LaTeX scales a ``fig_width_in``-wide
    figure to ``print_width_in``. Within-figure ratios are preserved.

    All fonts, including the per-panel gait ticks, scale by the full factor
    ``k`` so every heatmap renders its axis text at the same on-page size.
    The x-axis is labelled only at 0/50/100%, so the ticks stay uncrowded
    even in the wide, many-panel cortical grid."""
    target_fig_pt = TARGET_LABEL_PT_ON_PAGE * fig_width_in / print_width_in
    k = target_fig_pt / _BASE_HEATMAP_FONTS['label']
    gait_k = k
    out = {}
    for key, val in _BASE_HEATMAP_FONTS.items():
        scale = gait_k if key == 'gait_ticks' else k
        out[key] = max(1, round(val * scale))
    return out


def _plot_single_tfr(ax, data, area_name, times, freqs, scale, font_settings, band_edges, is_outer_left, is_bottom_row, is_top_row, draw_xlabel=True, draw_ylabel=True):
    """Helper function to plot a single TFR subplot with final styling.

    ``draw_xlabel`` controls only the per-panel "Gait Cycle (%)" axis label;
    the rotated gait-event tick labels are always drawn on bottom-row panels.
    ``draw_ylabel`` controls the per-panel "Frequency (Hz)" axis label;
    the numeric y-ticks are always drawn. Multi-row callers set both to
    ``False`` and add a single centred label spanning the panel grid."""

    mesh = ax.pcolormesh(times, freqs, data, shading='gouraud', vmin=scale[0], vmax=scale[1], cmap='RdBu_r', zorder =-10, rasterized=True)

    if is_top_row:
        ax.set_title(area_name, fontsize=font_settings['title'])

    span = times[-1] - times[0]
    # Vertical event lines mark TOL (10%), HSL (50%), TOR (60%); heel strikes
    # bound the cycle at 0/100%. Event names are defined in the caption/Methods.
    event_times = [times[0] + 0.1*span, times[0] + 0.5*span, times[0] + 0.6*span]
    # Label only 0/50/100% so the narrow multi-panel columns stay uncrowded
    # (previously five rotated "10%:TOL"-style labels overlapped between columns).
    xtick_pos = [times[0], times[0] + 0.5*span, times[-1]]
    xtick_lbls = ['0', '50', '100']

    ax.set_xticks(xtick_pos)
    if is_bottom_row:
        lbls = ax.set_xticklabels(xtick_lbls, rotation=0, fontsize=font_settings['gait_ticks'])
        # Anchor the end labels inside the panel so the 0/100 of adjacent
        # columns do not collide across the narrow inter-column gap.
        lbls[0].set_ha('left'); lbls[1].set_ha('center'); lbls[-1].set_ha('right')
        if draw_xlabel:
            ax.set_xlabel('Gait Cycle (%)', fontsize=font_settings['label'])
    else:
        # Hide x-tick labels for non-bottom plots to keep it clean
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if is_outer_left and draw_ylabel:
        ax.set_ylabel('Frequency (Hz)', fontsize=font_settings['label'])

    # Y-axis tick labels (frequency: 4, 8, 12, 20, 30, 45) sized to match
    # the x-axis gait-event tick labels for within-figure consistency. The
    # colorbar still uses font_settings['ticks'] (set in the caller).
    ax.tick_params(axis='y', labelsize=font_settings['gait_ticks'])

    for et in event_times:
        ax.axvline(et, color='black', linestyle='-', linewidth=1.5)
    for be in band_edges:
        ax.axhline(be, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.set_yticks([be for be in band_edges if be != 8])  # drop the crowded 8 Hz label (band line at 8 stays)
    ax.tick_params(axis='y', labelsize=font_settings['gait_ticks'])

    ax.set_ylim(band_edges[0], band_edges[-1])
    return mesh


def _create_stn_figure(group_name, block_name, stn_data, freqs, time_axis, scale, band_edges, results_dir, save_format, font_settings, group_name_map, block_name_map, method):
    """Creates a standard 3-subplot figure for STN data with updated styling."""
    
    desired_order = ["Left", "Both", "Right"]
    all_items = [(area, arr) for area, arr in stn_data.items() if arr is not None]
    sorted_items = sorted(all_items, key=lambda item: desired_order.index(item[0]) if item[0] in desired_order else len(desired_order))

    if not sorted_items: return

    nplots = len(sorted_items)
    fig_width = 6.5 * nplots
    fig, axes = plt.subplots(1, nplots, figsize=(fig_width, 5.5), sharey=True, constrained_layout=True)
    if nplots == 1: axes = [axes]

    # Printed at \textwidth in figure*; scale fonts to the common on-page target.
    font_settings = _scaled_heatmap_fonts(fig_width, PRINT_FULLWIDTH_IN)

    group_short = group_name_map.get(group_name, group_name)
    block_short = block_name_map.get(block_name, block_name)
    main_title = f"Subcortical Morlet-TFR: {group_short} - {block_short} relative to rest"
    #fig.suptitle(main_title, fontsize=font_settings['figure_title'], y=1.06)

    mesh = None
    for i, (area, data) in enumerate(sorted_items):
        mesh = _plot_single_tfr(axes[i], data, area, time_axis, freqs, scale, font_settings, band_edges, is_outer_left=(i==0), is_bottom_row=True, is_top_row=True, draw_xlabel=False)

    # Single centred "Gait Cycle (%)" instead of one per hemisphere panel.
    fig.supxlabel('Gait Cycle (%)', fontsize=font_settings['label'])

    if mesh:
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('Power (dB)', size=font_settings['cbar_label'])
        cbar.ax.tick_params(labelsize=font_settings['ticks'])

    fname = f"STN_{group_name.replace(' ','_')}_{block_name}_{method}.{save_format}"
    full_path = os.path.join(results_dir, fname)
    print(f"  - Saving STN plot to: {full_path}")
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    

def compute_and_plot_grand_averages_flexible(
    tfr_group_results: Dict[str, Dict[str, Any]],
    blocks: List[str],
    freqs: np.ndarray,
    freq_bands: Dict[str, Tuple[float, float]],
    time_axis: np.ndarray,
    time_axis_stn: np.ndarray,
    results_dir: str,
    expected_time_samples_eeg: int,
    expected_time_samples_stn: int,
    comparisons: List[Dict[str, Any]],
    method: str = 'mean',
    save_format: str = "pdf",
    scale: Tuple[float, float] = (-1.25, 1.25)
) -> Dict[str, Any]:
    """
    MODIFIED and DEBUGGED version.
    
    This function computes grand averages and generates plots based on a flexible
    'comparisons' dictionary, allowing for main group and subgroup analyses.

    Args:
        tfr_group_results: The main data dictionary from your pipeline.
        comparisons: A list where each dictionary defines a plottable comparison. Example:
            [
                {
                    "name": "PD_vs_HC",
                    "cohorts": {
                        "PD": {"group": "Patient_NFB", "ids": None},
                        "HC": {"group": "AgeMatched_NFB", "ids": None}
                    }
                },
                {
                    "name": "PD_Subgroups",
                    "cohorts": {
                        "PD-HC-like": {"group": "Patient_NFB", "ids": cluster_pd_like},
                        "PD-HC-away": {"group": "Patient_NFB", "ids": cluster_pd_away}
                    }
                }
            ]
        ... (other arguments are the same)
        
    Returns:
        A dictionary containing all the computed grand averages for each cohort.
    """
    print("--- Starting Flexible Grand Average and Plotting (Corrected Version) ---")
    os.makedirs(results_dir, exist_ok=True)

    # --- Helper functions ---
    font_settings = {'figure_title': 28, 'title': 20, 'label': 20, 'cbar_label': 24, 'ticks': 16, 'gait_ticks': 14}
    block_name_map = {"pre8walk": "Walk", "post8walk": "Post-8walk", "baseline": "Rest"}

    def get_band_edges(fbands: Dict, f: np.ndarray) -> List[float]:
        edges = {val for pair in fbands.values() for val in pair}
        return sorted([e for e in edges if f[0] <= e <= f[-1]])

    def grand_average(arrays: List[Optional[np.ndarray]], expected_len: int, method: str = 'mean') -> Optional[np.ndarray]:
        valid_arrays = [np.squeeze(a) for a in arrays if a is not None]
        if not valid_arrays: return None

        def _to_len(arr: np.ndarray, L: int) -> Optional[np.ndarray]:
            if arr.ndim != 2:
                if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                else: return None
            if arr.shape[1] == L: return arr
            x_old, x_new = np.linspace(0.0, 1.0, arr.shape[1]), np.linspace(0.0, 1.0, L)
            return np.vstack([np.interp(x_new, x_old, row) for row in arr])

        resampled = [_to_len(a, expected_len) for a in valid_arrays]
        resampled = [r for r in resampled if r is not None]
        if not resampled: return None
        stacked_arrays = np.stack(resampled, axis=0)
        if method == 'median':
            print("    -> using MEDIAN for grand average.")
            return np.nanmedian(stacked_arrays, axis=0)
        elif method == 'mean':
            print("    -> using MEAN for grand average.")
            return np.nanmean(stacked_arrays, axis=0)
        else:
            raise ValueError("Method must be 'mean' or 'median'")
        #return np.nanmean(np.stack(resampled, axis=0), axis=0)

    # Step 1: Compute all grand averages for all defined cohorts
    all_averages: Dict[str, Dict[str, Any]] = {}
    print("\n--- Pre-computing all required Grand Averages ---")
    
    all_cohorts_to_compute = {
        cohort_name: cohort_info
        for comp in comparisons
        for cohort_name, cohort_info in comp["cohorts"].items()
    }

    for cohort_name, cohort_info in all_cohorts_to_compute.items():
        all_averages[cohort_name] = {}
        group_key = cohort_info["group"]
        id_filter = set(cohort_info["ids"]) if cohort_info["ids"] is not None else None

        for block in blocks:
            print(f"  - Averaging for cohort '{cohort_name}' in block '{block}'")
            all_averages[cohort_name][block] = {}
            
            block_data = tfr_group_results.get(group_key, {}).get(block, {})
            if not block_data: continue
            
            all_subject_ids = block_data.get('ID', [])
            
            # Determine which subjects/indices to keep
            if id_filter:
                indices_to_keep = [i for i, sid in enumerate(all_subject_ids) if sid in id_filter]
            else:
                indices_to_keep = list(range(len(all_subject_ids)))

            # --- BUG FIX: The original filtering logic was flawed. This is a robust replacement. ---
            # This new logic correctly filters the lists of arrays for each region and side.
            def filter_and_average(source_dict: Dict[str, Any], expected_len: int, method: str = 'mean') -> Dict[str, Any]:
                avg_data = {}
                for key, value in source_dict.items():
                    if isinstance(value, list): # STN case: {'Left': [arr1, ...], 'Right': [arr2, ...]}
                        arrays_to_avg = [value[i] for i in indices_to_keep if i < len(value)]
                        avg_data[key] = grand_average(arrays_to_avg, expected_len, method)
                    elif isinstance(value, dict): # EEG case: {'EEG_motor': {'Left': [...], ...}}
                         avg_data[key] = filter_and_average(value, expected_len, method)
                return avg_data

            avg_motor = filter_and_average(block_data.get("EEG_motor", {}), expected_time_samples_eeg, method)
            avg_frontal = filter_and_average(block_data.get("EEG_frontal", {}), expected_time_samples_eeg, method)
            avg_stn = filter_and_average(block_data.get("STN", {}), expected_time_samples_stn, method)
            
            all_averages[cohort_name][block]["avg_motor"] = avg_motor
            all_averages[cohort_name][block]["avg_frontal"] = avg_frontal
            all_averages[cohort_name][block]["avg_stn"] = avg_stn

    # Step 2: Generate plots for each comparison
    print("\n--- Generating Plots based on Comparisons ---")
    band_edges = get_band_edges(freq_bands, freqs)
    
    for comp in comparisons:
        comp_name = comp["name"]
        cohort_labels = list(comp["cohorts"].keys())
        print(f"\nGenerating plots for comparison: '{comp_name}'")

        for block in blocks:
            if block == 'baseline': continue
            
            block_data_for_plot = {label: all_averages.get(label, {}).get(block, {}) for label in cohort_labels}
            if not any(v for v in block_data_for_plot.values()): continue

            _create_combined_eeg_figure_flexible(
                block, block_data_for_plot, cohort_labels, freqs, time_axis, scale,
                band_edges, results_dir, save_format, font_settings, block_name_map, comp_name, method
            )

        for cohort_label in cohort_labels:
            group_name_map = {cohort_label: cohort_label} # Use label directly as title
            for block in blocks:
                if "avg_stn" in all_averages.get(cohort_label, {}).get(block, {}):
                    stn_data = all_averages[cohort_label][block]["avg_stn"]
                    if not stn_data or all(v is None for v in stn_data.values()): continue
                    
                    _create_stn_figure(
                        cohort_label, block, stn_data, freqs, time_axis_stn, scale,
                        band_edges, results_dir, save_format, font_settings, 
                        group_name_map, block_name_map, method
                    )

    print("\n--- Flexible Grand Average and Plotting Finished ---")
    return all_averages


def _create_combined_eeg_figure_flexible(
    block_name: str,
    cohort_data: Dict[str, Any],
    cohort_labels: List[str],
    freqs: np.ndarray,
    time_axis: np.ndarray,
    scale: Tuple[float, float],
    band_edges: List[float],
    results_dir: str,
    save_format: str,
    font_settings: Dict,
    block_name_map: Dict,
    filename_suffix: str,
    method: str
):
    """
    Generic version of _create_combined_eeg_figure that plots any two cohorts side-by-side.
    """
    if len(cohort_labels) != 2:
        print(f"WARNING: EEG figure generation expects exactly 2 cohorts, but got {len(cohort_labels)}. Skipping '{filename_suffix}'.")
        return
        
    fig_width = 28
    fig, axes = plt.subplots(
        2, 7, figsize=(fig_width, 8), sharex=True, sharey=True,
        gridspec_kw={'width_ratios': [1, 1, 1, 0.2, 1, 1, 1], 'wspace': 0.2, 'hspace': 0.15}
    )
    axes[0, 3].axis('off'); axes[1, 3].axis('off')

    # Printed at \textwidth in figure*; scale fonts to the common on-page target.
    font_settings = _scaled_heatmap_fonts(fig_width, PRINT_FULLWIDTH_IN)

    block_short = block_name_map.get(block_name, block_name)
    main_title = f"Cortical TFR: {block_short} ({cohort_labels[0]} vs {cohort_labels[1]})"
    #fig.suptitle(main_title, fontsize=font_settings['figure_title'], y=1.1, x=0.45)
    
    desired_order = ["Left", "Both", "Right"]
    region_map = {"Motor": "avg_motor", "Frontal": "avg_frontal"}
    mesh = None

    for row_idx, region_name in enumerate(["Motor", "Frontal"]):
        for col_offset, cohort_name in enumerate(cohort_labels):
            data_key = region_map[region_name]
            tfr_data = cohort_data.get(cohort_name, {}).get(data_key, {})
            
            sorted_items = sorted(
                [(area, arr) for area, arr in tfr_data.items() if arr is not None and area in desired_order],
                key=lambda item: desired_order.index(item[0])
            )

            for area_idx, (area, data) in enumerate(sorted_items):
                ax = axes[row_idx, area_idx + (4 * col_offset)]
                mesh = _plot_single_tfr(ax, data, area, time_axis, freqs, scale, font_settings, band_edges,
                                        is_outer_left=(area_idx == 0 and col_offset == 0),
                                        is_bottom_row=(row_idx == 1),
                                        is_top_row=(row_idx == 0),
                                        draw_xlabel=False, draw_ylabel=False)

    # Colour bar FIRST, so the label positions below are read from the final
    # (post-colour-bar) axes layout. The colour bar shrinks the axes to the
    # left; positioning labels beforehand made them drift right, worst for the
    # right-hand (HC) cohort block.
    if mesh:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
        cbar.set_label('Power (dB)', size=font_settings['cbar_label'])
        cbar.ax.tick_params(labelsize=font_settings['ticks'])
    fig.canvas.draw()  # finalise the colour-bar-adjusted geometry before reading positions

    # Cohort titles, centred over each cohort's full three-column block
    # (columns 0-2 for PD, 4-6 for HC).
    pd_t_l = axes[0, 0].get_position(); pd_t_r = axes[0, 2].get_position()
    hc_t_l = axes[0, 4].get_position(); hc_t_r = axes[0, 6].get_position()
    fig.text((pd_t_l.x0 + pd_t_r.x1) / 2, pd_t_r.y1 + 0.06, cohort_labels[0], va='bottom', ha='center', fontsize=font_settings['title']+2, weight='bold')
    fig.text((hc_t_l.x0 + hc_t_r.x1) / 2, hc_t_r.y1 + 0.06, cohort_labels[1], va='bottom', ha='center', fontsize=font_settings['title']+2, weight='bold')

    # "Gait Cycle (%)" centred under each cohort's full three-column block,
    # placed clearly below the x-tick labels.
    pd_l = axes[1, 0].get_position(); pd_r = axes[1, 2].get_position()
    hc_l = axes[1, 4].get_position(); hc_r = axes[1, 6].get_position()
    xlab_y = min(pd_l.y0, hc_l.y0) - 0.09
    fig.text((pd_l.x0 + pd_r.x1) / 2, xlab_y, 'Gait Cycle (%)', va='top', ha='center', fontsize=font_settings['label'])
    fig.text((hc_l.x0 + hc_r.x1) / 2, xlab_y, 'Gait Cycle (%)', va='top', ha='center', fontsize=font_settings['label'])

    # Row labels + a single centred "Frequency (Hz)" spanning both rows.
    pos_motor = axes[0, 0].get_position(); pos_frontal = axes[1, 0].get_position()
    fig.text(pos_motor.x0 - 0.04, (pos_motor.y0 + pos_motor.y1) / 2, 'Motor', va='center', ha='right', rotation='vertical', fontsize=font_settings['label']+2, weight='bold')
    fig.text(pos_frontal.x0 - 0.04, (pos_frontal.y0 + pos_frontal.y1) / 2, 'Frontal', va='center', ha='right', rotation='vertical', fontsize=font_settings['label']+2, weight='bold')
    fig.text(pos_motor.x0 - 0.06, (pos_motor.y1 + pos_frontal.y0) / 2, 'Frequency (Hz)',
             va='center', ha='right', rotation='vertical', fontsize=font_settings['label'])

    fname = f"Combined_EEG_{block_short}_{filename_suffix}_{method}.{save_format}"
    full_path = os.path.join(results_dir, fname)
    print(f"  - Saving combined EEG plot to: {full_path}")
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Plot STN ipsi-contra:

import numpy as np
import matplotlib.pyplot as plt
import os

# ========= Helper Functions =========

def _circ_shift_percent_time(arr_2d, shift_percent):
    """
    Circularly shift a (n_freqs, T) array along time by 'shift_percent' of T.
    Positive = shift forward in time (to the right); negative = left.
    """
    if arr_2d is None:
        return None
    a = np.squeeze(arr_2d)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        return None
    T = a.shape[1]
    if T == 0:
        return None
    k = int(round((shift_percent / 100.0) * T))
    return np.roll(a, shift=k, axis=1)

def _resample_to_len_2d(a, L):
    """
    Resample a (n_freqs, T) 2D array along time -> length L (linear interp per frequency).
    """
    if a is None:
        return None
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        return None
    nF, T = a.shape
    if T == L:
        return a
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, L)
    return np.vstack([np.interp(x_new, x_old, a[i, :]) for i in range(nF)])

def _plot_single_tfr_customticks(ax, data, area_name, times, freqs, scale, font_settings,
                                 band_edges, is_outer_left, is_bottom_row, is_top_row,
                                 tick_locs_pct=(0,10,50,60,100),
                                 tick_labels=('0%: IHS','10%: CTO','50%: CHS','60%: ITO','100%: IHS')):
    """Plots TFR with custom x-tick labels for ipsi/contra gait cycle."""
    mesh = ax.pcolormesh(times, freqs, data, shading='gouraud', vmin=scale[0], vmax=scale[1], cmap='RdBu_r', rasterized=True)

    if is_top_row:
        ax.set_title(area_name, fontsize=font_settings['title'])

    t0, t1 = times[0], times[-1]
    pts = [t0 + (p/100.0)*(t1 - t0) for p in tick_locs_pct]
    event_times = pts[1:-1]

    ax.set_xticks(pts)
    if is_bottom_row:
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', rotation_mode='anchor',
                           fontsize=font_settings['gait_ticks'])
        ax.set_xlabel('Gait Cycle (%)', fontsize=font_settings['label'])
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    if is_outer_left:
        ax.set_ylabel('Frequency (Hz)', fontsize=font_settings['label'])

    # Match y-axis tick label size to x-axis gait-tick size (same policy as
    # _plot_single_tfr); colorbar tick size is set by the caller.
    ax.tick_params(axis='y', labelsize=font_settings['gait_ticks'])

    for et in event_times:
        ax.axvline(et, color='black', linestyle='-', linewidth=1.5)
    for be in band_edges:
        ax.axhline(be, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.set_yticks([be for be in band_edges if be != 8])  # drop the crowded 8 Hz label (band line at 8 stays)
    ax.tick_params(axis='y', labelsize=font_settings['gait_ticks'])
    ax.set_ylim(freqs[0], freqs[-1])
    return mesh

def _create_stn_figure_both_ipsicontra(
    group_name, block_name, both_data, freqs, time_axis, scale, band_edges,
    results_dir, save_format, font_settings, group_name_map, block_name_map, method = 'mean'
):
    """Single-panel STN figure (Both) with ipsi/contra gait ticks."""
    if both_data is None:
        return
    fig_width = 6.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5.5), constrained_layout=True)

    # Printed at 0.48\textwidth (subfigure); scale fonts to the common on-page target.
    font_settings = _scaled_heatmap_fonts(fig_width, 0.48 * PRINT_FULLWIDTH_IN)

    group_short = group_name_map.get(group_name, group_name)
    block_short = block_name_map.get(block_name, block_name)
    title = f"Subcortical Morlet-TFR (ipsi/contra): {group_short} - {block_short}"
    #fig.suptitle(title, fontsize=font_settings['figure_title'], y=1.06)

    mesh = _plot_single_tfr_customticks(
        ax=ax, data=both_data, area_name="Both (Ipsi-Aligned)", times=time_axis, freqs=freqs, scale=scale,
        font_settings=font_settings, band_edges=band_edges,
        is_outer_left=True, is_bottom_row=True, is_top_row=True,
    )

    if mesh:
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Power (dB)', size=font_settings['cbar_label'])
        cbar.ax.tick_params(labelsize=font_settings['ticks'])

    fname = f"STN_{group_name.replace(' ','_')}_{block_name}_Both_ipsicontra_{method}.{save_format}"
    full_path = os.path.join(results_dir, fname)
    print(f"  - Saving STN (Both, ipsi/contra) plot to: {full_path}")
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ========= CORRECTED FLEXIBLE FUNCTION =========

def compute_and_plot_stn_ipsicontra_flexible(
    tfr_group_results,
    blocks,
    freqs,
    freq_bands,
    time_axis_stn,
    results_dir,
    expected_time_samples_stn,
    save_format="pdf",
    scale=(-1.25, 1.25),
    comparisons_to_run=None,
    method = 'mean'
):
    """
    Corrected flexible function to plot ipsi/contra aligned STN data.

    - Re-expresses STN time courses in an ipsilateral-centric gait cycle:
        - STN-Left arrays are circularly shifted by -50%.
        - STN-Right arrays are not shifted.
    - Grand-averages across all aligned Left and Right data together.
    - Plots single-panel figures for each main group.
    - Plots multi-panel comparison figures for specified subgroups.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ---- NEW Local Helper: The CORRECT averaging logic ----
    def _compute_ipsicontra_average(left_arrays, right_arrays, L_expected):
        """
        Performs ipsilateral alignment and then grand-averages all data.
        - Shifts left data by -50%.
        - Does not shift right data.
        - Resamples all and computes a single mean.
        """
        aligned = []

        # Process Left STN data with a -50% shift
        for a in left_arrays:
            if a is None: continue
            a_shift = _circ_shift_percent_time(a, shift_percent=-50.0)
            a_rs = _resample_to_len_2d(a_shift, L_expected)
            if a_rs is not None:
                aligned.append(a_rs)

        # Process Right STN data with no shift
        for a in right_arrays:
            if a is None: continue
            # No shift needed for right side as it's already ipsilateral
            a_rs = _resample_to_len_2d(a, L_expected)
            if a_rs is not None:
                aligned.append(a_rs)

        if not aligned:
            return None
        stacked_arrays = np.stack(aligned, axis=0)
        if method == 'median':
            print("    -> using MEDIAN for grand average.")
            return np.nanmedian(stacked_arrays, axis=0)
        elif method == 'mean':
            print("    -> using MEAN for grand average.")
            return np.nanmean(stacked_arrays, axis=0)
        else:
            raise ValueError("Method must be 'mean' or 'median'")
        #return np.nanmean(np.stack(aligned, axis=0), axis=0)

    # ---- Standard Setup ----
    def _band_edges_within(freq_bands_dict, freq_vec):
        edges = set()
        for lo, hi in freq_bands_dict.values():
            edges.add(lo); edges.add(hi)
        return sorted([f for f in edges if freq_vec[0] <= f <= freq_vec[-1]])

    font_settings = {
        'figure_title': 28, 'title': 22, 'label': 22,
        'cbar_label': 26, 'ticks': 18, 'gait_ticks': 18,
    }
    group_name_map = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}
    block_name_map = {"pre8walk": "Walk", "post8walk": "Post-8walk", "baseline": "Rest"}
    band_edges = _band_edges_within(freq_bands, freqs)

    # ---------- A) "All patients" plots with CORRECT logic ----------
    for group_var, group_data in tfr_group_results.items():
        for block in blocks:
            if block not in group_data: continue

            subj_ids = group_data[block].get('ID', [])
            stn_dict = group_data[block].get('STN', {})
            left_list  = stn_dict.get('Left',  []) or []
            right_list = stn_dict.get('Right', []) or []
            
            printable_group = group_name_map.get(group_var, group_var)
            printable_block = block_name_map.get(block, block)
            print(f"\n[STN Ipsi/Contra Avg] Group={printable_group} | Block={printable_block}")
            
            # Use the new helper to compute the correctly aligned average
            avg_both_ipsi = _compute_ipsicontra_average(left_list, right_list, expected_time_samples_stn)
            
            if avg_both_ipsi is None:
                print("  No STN data available to average for this block.")
                continue

            # Use the plotting function with custom ticks
            _create_stn_figure_both_ipsicontra(
                group_name=group_var, block_name=block, both_data=avg_both_ipsi,
                freqs=freqs, time_axis=time_axis_stn, scale=scale,
                band_edges=band_edges, results_dir=results_dir,
                save_format=save_format, font_settings=font_settings,
                group_name_map=group_name_map, block_name_map=block_name_map, method=method
            )

    # ---------- B) Subgroup comparison plots with CORRECT logic ----------
    if not comparisons_to_run:
        return

    # --- Local plotting helper for multi-panel figures ---
    # MODIFIED to use the custom tick plotter
    def _plot_multi_cohort_ipsicontra(comparison_name, block_name, cohort_data_dict, results_dir, method):
        labels = list(cohort_data_dict.keys())
        nplots = len(labels)
        if nplots == 0: return

        fig, axes = plt.subplots(1, nplots, figsize=(6.5 * nplots, 5.5),
                                 sharey=True, constrained_layout=True)
        if nplots == 1: axes = [axes]

        block_short = block_name_map.get(block_name, block_name)
        main_title = f"STN Morlet-TFR (Ipsi-Aligned): {comparison_name} – {block_short}"
        #fig.suptitle(main_title, fontsize=font_settings['figure_title'], y=1.06)

        mesh = None
        for i, label in enumerate(labels):
            data = cohort_data_dict[label]
            if data is None:
                data = np.full((len(freqs), len(time_axis_stn)), np.nan)
                axes[i].text(0.5, 0.5, "No data", ha='center', va='center',
                             fontsize=font_settings['title']-2, alpha=0.7, transform=axes[i].transAxes)
            
            # USE THE CUSTOM TICK PLOTTER
            mesh = _plot_single_tfr_customticks(
                axes[i], data, label, time_axis_stn, freqs, scale, font_settings,
                band_edges, is_outer_left=(i == 0), is_bottom_row=True, is_top_row=True
            )

        if mesh:
            cbar = fig.colorbar(mesh, ax=axes, shrink=0.80, pad=0.02)
            cbar.set_label('Power (dB)', size=font_settings['cbar_label'])
            cbar.ax.tick_params(labelsize=font_settings['ticks'])

        fname = f"STN_IpsiContra_{comparison_name}_{block_short}_{method}.{save_format}"
        full_path = os.path.join(results_dir, fname)
        print(f"  - Saving STN (Ipsi-Aligned) multi-cohort plot to: {full_path}")
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # --- Build and plot for each comparison spec ---
    for comp in comparisons_to_run:
        comp_name = comp.get("name", "comparison")
        cohorts = comp.get("cohorts", {})
        if not cohorts: continue

        for block in blocks:
            cohort_both_ipsi = {}
            for label, spec in cohorts.items():
                base_group = spec.get("group")
                id_filter = spec.get("ids", None)

                base = tfr_group_results.get(base_group, {})
                if block not in base:
                    cohort_both_ipsi[label] = None
                    continue

                subj_ids = base[block].get("ID", [])
                stn_dict = base[block].get("STN", {})

                if id_filter is None:
                    idxs = np.arange(len(subj_ids))
                else:
                    wanted = set(id_filter)
                    idxs = np.array([i for i, sid in enumerate(subj_ids) if sid in wanted], dtype=int)

                if idxs.size == 0:
                    cohort_both_ipsi[label] = None
                    continue

                left_list_full  = stn_dict.get("Left",  []) or []
                right_list_full = stn_dict.get("Right", []) or []
                left_subset  = [left_list_full[i]  for i in idxs if i < len(left_list_full)]
                right_subset = [right_list_full[i] for i in idxs if i < len(right_list_full)]

                # Use the new helper for the CORRECT aligned average
                cohort_both_ipsi[label] = _compute_ipsicontra_average(
                    left_subset, right_subset, expected_time_samples_stn
                )

            # Plot this comparison for the current block
            _plot_multi_cohort_ipsicontra(comp_name, block, cohort_both_ipsi, results_dir, method)



#AVGGAIT NORMALIZATION TFR COMPUTING: 

# ==============================================================================
# Helper Functions specific to the Average Gait Normalization (AvgGaitNorm) Method
# ==============================================================================

def _normalize_avg_tfr_whole_cycle_inplace(tfr_obj):
    """
    AvgGaitNorm-style: after averaging across cycles, normalize by subtracting
    the whole-gait-cycle mean (per frequency) on the *log-power/dB* TFR.
    
    This recenters each frequency band around zero across the stride.
    Formula: Data_dB(f, t) = Data_dB(f, t) - Mean_t(Data_dB(f, :))
    """
    if tfr_obj is None:
        return
    # shape: (1, n_freqs, n_times)
    # Calculate mean over the time dimension (axis 2) for each frequency
    mu_f = tfr_obj.data.mean(axis=2, keepdims=True)  
    tfr_obj.data = tfr_obj.data - mu_f  # in-place recenter

def _build_group_results_from_epoched(source_data, blocks, results_dir, suffix_label,
                                      save_fn_pickle):
    """
    Re-uses your Step 3 logic to:
      - (a) turn epoched AverageTFRs into numpy arrays for stats/plots
      - (b) also keep the AverageTFRs
      - (c) save both.

    Returns (tfr_group_results, avg_tfr_group_results).
    """
    tfr_group_results = {}
    avg_tfr_group_results = {}

    for group_var, group_blocks in source_data.items():
        tfr_group_results[group_var] = {}
        avg_tfr_group_results[group_var] = {}
        for block in blocks:
            if block not in group_blocks:
                continue
            tfr_group_results[group_var][block] = {
                "ID": group_blocks[block]["ID"][:],
                "EEG_motor": {},
                "EEG_frontal": {},
                "STN": {},
            }
            avg_tfr_group_results[group_var][block] = {
                "ID": group_blocks[block]["ID"][:],
                "EEG_motor": {},
                "EEG_frontal": {},
                "STN": {},
            }
            # EEG regions
            for region_key in ["motor", "frontal"]:
                target_key = f"EEG_{region_key}"
                for side_name, lst in group_blocks[block][region_key].items():
                    avg_tfr_group_results[group_var][block][target_key].setdefault(side_name, [])
                    tfr_group_results[group_var][block][target_key].setdefault(side_name, [])
                    for tfr_obj in lst:
                        avg_tfr_group_results[group_var][block][target_key][side_name].append(tfr_obj)
                        tfr_group_results[group_var][block][target_key][side_name].append(
                            None if tfr_obj is None else tfr_obj.data.squeeze()
                        )
            # STN
            for side_name, lst in group_blocks[block]["STN"].items():
                avg_tfr_group_results[group_var][block]["STN"].setdefault(side_name, [])
                tfr_group_results[group_var][block]["STN"].setdefault(side_name, [])
                for tfr_obj in lst:
                    avg_tfr_group_results[group_var][block]["STN"][side_name].append(tfr_obj)
                    tfr_group_results[group_var][block]["STN"][side_name].append(
                        None if tfr_obj is None else tfr_obj.data.squeeze()
                    )

    # save with clear suffix
    save_fn_pickle(tfr_group_results, f"tfr_group_results_{suffix_label}", results_dir)
    save_fn_pickle(avg_tfr_group_results, f"avg_tfr_group_results_{suffix_label}", results_dir)
    return tfr_group_results, avg_tfr_group_results


def create_gait_warped_tfr_epochs_avggaitnorm(
    all_data_tfr_continuous_dico,
    all_data_dico,  # Original raw data for annotations
    rejection_mask_dico, # Placeholder for epoch rejection masks
    event_percentages,
    gait_events_correct_sequence,
    epoch_duration=1.1
):
    """
    Orchestrates the segmentation and warping of continuous TFR data, then
    aggregates cycles using an element-wise MEDIAN for robustness to outliers. NOPE; BACK TO MEAN !!

    Args:
        all_data_tfr_continuous_dico (dict): TFRs for continuous data.
        all_data_dico (dict): Original data with mne.Raw objects and annotations.
        rejection_mask_dico (dict): Maps (subject, block) -> boolean_mask for epochs to KEEP.
        event_percentages (dict): Maps gait events to target percentages (0-100).
        gait_events_correct_sequence (list): Ordered list of events for a valid gait cycle.
        epoch_duration (float): Final duration of the warped gait cycle in seconds.

    Returns:
        tuple[dict, dict]:
            - A dictionary with mean-aggregated AverageTFR objects per subject/condition.
            - A dictionary containing all individual (kept) epochs before aggregation.
    """
    # Initialize the final output dicts with a new name for clarity
    all_data_tfr_mean_aggregated_dico = {}
    all_data_tfr_all_epochs_dico = {}
    
    # Deep copy the structure from the input dict
    for group, blocks_data in all_data_tfr_continuous_dico.items():
        all_data_tfr_mean_aggregated_dico[group] = {}
        all_data_tfr_all_epochs_dico[group] = {}
        for block in blocks_data:
            all_data_tfr_mean_aggregated_dico[group][block] = {
                "ID": [], "motor": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "frontal": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "STN": {'Left': [], 'Right': []}
            }
            all_data_tfr_all_epochs_dico[group][block] = {
                "ID": [], "motor": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "frontal": {'Left': [], 'Right': [], 'Central':[], 'Both': []},
                "STN": {'Left': [], 'Right': []}
            }

    # --- Main Loop ---
    for group, group_data in all_data_tfr_continuous_dico.items():
        for block, block_data in group_data.items():
            if block == 'baseline':
                continue

            for i_subj, subj_id in enumerate(block_data['ID']):
                print(f"--- Processing {group} {subj_id} - {block} ---")
                all_data_tfr_mean_aggregated_dico[group][block]['ID'].append(subj_id)
                all_data_tfr_all_epochs_dico[group][block]['ID'].append(subj_id)

                try:
                    raw_idx = all_data_dico[group][block]['ID'].index(subj_id)
                    eeg_raw_for_events = all_data_dico[group][block]['EEG'][raw_idx]
                    dbs_raw_for_events = all_data_dico[group][block].get('DBS_proc', [None])[raw_idx]
                except (ValueError, IndexError):
                    print(f"  > Warning: Could not find original Raw data for {subj_id} in {block}. Skipping.")
                    continue

                gait_cycles_cache_eeg, gait_cycles_cache_dbs = [], []
                if eeg_raw_for_events:
                    _fe_eeg = filter_gait_events(eeg_raw_for_events, gait_events_correct_sequence)
                    _eeg_filt = replace_events_in_mne(eeg_raw_for_events, _fe_eeg, gait_events_correct_sequence)
                    gait_cycles_cache_eeg = segment_gait_cycles(_eeg_filt, gait_events_correct_sequence)
                if dbs_raw_for_events:
                    _fe_dbs = filter_gait_events(dbs_raw_for_events, gait_events_correct_sequence)
                    _dbs_filt = replace_events_in_mne(dbs_raw_for_events, _fe_dbs, gait_events_correct_sequence)
                    gait_cycles_cache_dbs = segment_gait_cycles(_dbs_filt, gait_events_correct_sequence)

                for region_name in ['motor', 'frontal', 'STN']:
                    for side_name in block_data[region_name]:
                        continuous_tfr = block_data[region_name][side_name][i_subj]

                        if continuous_tfr is None:
                            all_data_tfr_mean_aggregated_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_all_epochs_dico[group][block][region_name][side_name].append(None)
                            continue

                        gait_cycles_info = gait_cycles_cache_dbs if region_name == 'STN' else gait_cycles_cache_eeg
                        tfr_epochs_data = extract_tfr_epochs_from_cycles(continuous_tfr, gait_cycles_info)
                        
                        if not tfr_epochs_data:
                            print(f"  > No valid gait cycles found for {region_name}-{side_name}.")
                            all_data_tfr_mean_aggregated_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_all_epochs_dico[group][block][region_name][side_name].append(None)
                            continue

                        tfr_sfreq = continuous_tfr.info['sfreq']
                        warped_tfr_epochs = warp_tfr_cycles(tfr_epochs_data, gait_cycles_info, event_percentages, tfr_sfreq)
                        resampled_tfr_epochs = resample_tfr_cycles(warped_tfr_epochs, epoch_duration, tfr_sfreq)

                        mask = rejection_mask_dico.get((subj_id, block))
                        if mask is not None and len(mask) == len(resampled_tfr_epochs):
                            final_epochs_data = np.array(resampled_tfr_epochs)[mask]
                        else:
                            final_epochs_data = np.array(resampled_tfr_epochs)

                        if final_epochs_data.shape[0] == 0:
                            print(f"  > All epochs rejected for {region_name}-{side_name}.")
                            all_data_tfr_mean_aggregated_dico[group][block][region_name][side_name].append(None)
                            all_data_tfr_all_epochs_dico[group][block][region_name][side_name].append(None)
                            continue
                        
                        # ========================== KEY MODIFICATION HERE ==========================
                        # STEP 5: Aggregate valid epochs using MEDIAN and create the final MNE object.
                        
                        # Stack all kept epochs into a single numpy array.
                        # Shape: (n_epochs, n_channels, n_freqs, n_times)
                        stacked_data = np.stack(final_epochs_data, axis=0)

                        # Calculate the MEDIAN across the epochs dimension (axis=0).
                        # The shape goes from (n_epochs, 1, 45, 2200) to (1, 45, 2200).
                        mean_tfr_data = np.mean(stacked_data, axis=0)
                        
                        # ========================== END OF MODIFICATION ==========================

                        # Create metadata for the final object.
                        info = mne.create_info(
                            ch_names=[f"{region_name}_{side_name}"],
                            sfreq=tfr_sfreq,
                            ch_types='misc'
                        )
                        new_times = np.linspace(0.0, epoch_duration, mean_tfr_data.shape[2])

                        # Build the state dictionary for the constructor
                        final_tfr_state = {
                            "data": mean_tfr_data, # Use the mean-aggregated data
                            "times": new_times,
                            "freqs": continuous_tfr.freqs,
                            "info": info,
                            "nave": int(stacked_data.shape[0]), # Number of epochs used for median
                            "method": "morlet",
                            "comment": f"{region_name}_{side_name}_mean_aggregated",
                        }
                        final_tfr = mne.time_frequency.AverageTFR(inst=final_tfr_state)

                        # Append the final object to the output dictionaries.
                        all_data_tfr_mean_aggregated_dico[group][block][region_name][side_name].append(final_tfr)
                        all_data_tfr_all_epochs_dico[group][block][region_name][side_name].append(final_epochs_data)

    return all_data_tfr_mean_aggregated_dico, all_data_tfr_all_epochs_dico

def compute_tfr_on_subset(
    all_data_dico, group_vars, subjects_dico, blocks, freqs, motor_areas, frontal_areas,
    target_sfreq=None, tfr_decim=1, n_jobs=1,
    *, power_to_db=True, eps=1e-12
):
    """
    Compute continuous Morlet TFRs for a subset of channels (EEG motor/frontal regions
    and STN) for the specified *task* blocks, **without** any baseline normalization.

    Output structure matches `compute_tfr_on_subset_and_normalize_optimized` so the
    downstream epoching/warping code can be reused as-is.

    Parameters
    ----------
    all_data_dico : dict
        Source dictionary with raw data per group/block/subject.
    group_vars : list[str]
        Group keys, e.g. ["Patient_NFB", "AgeMatched_NFB"].
    subjects_dico : dict[str, list[str]]
        Selected subjects per group.
    blocks : list[str]
        Task blocks to process (e.g., ['pre8walk', 'post8walk']).
    freqs : array-like
        Frequencies for Morlet TFR.
    motor_areas, frontal_areas : dict[str, list[str]]
        Channel lists per hemisphere/central for EEG areas.
    target_sfreq : float or None
        If set, resample raws before TFR.
    tfr_decim : int
        Decimation for TFR.
    n_jobs : int
        Parallel jobs for TFR.
    power_to_db : bool
        If True, convert absolute power to dB: 10*log10(power).
    eps : float
        Small value for numerical safety in log10.
    """
    import numpy as np
    import mne

    # Temporal resolution via evolving wavelets (same scheme as optimized fn)
    desired_fwhm_s = 0.200
    sigma = desired_fwhm_s / (2 * np.sqrt(2 * np.log(2)))
    n_cycles_evolving = 2 * np.pi * np.array(freqs) * sigma
    print(f"--- TFR Analysis (No Baseline) ---")
    print(f"Temporal FWHM ≈ {desired_fwhm_s*1000:.0f} ms | Resample: {target_sfreq if target_sfreq else 'native'} | decim: {tfr_decim} | n_jobs: {n_jobs}")
    print("Normalization mode: absolute power" + (" in dB" if power_to_db else " (linear)"))

    # Channels of interest
    all_eeg_channels = sorted(list(set(
        motor_areas['Left'] + motor_areas['Right'] + motor_areas['Central'] +
        frontal_areas['Left'] + frontal_areas['Right'] + frontal_areas['Central']
    )))
    stn_channel = ['DBS']
    regions_to_process = {'motor': motor_areas, 'frontal': frontal_areas}

    # Output skeleton (include 'baseline' key to keep downstream structure stable)
    all_data_tfr_continuous_dico = {}
    for group_var in group_vars:
        all_data_tfr_continuous_dico[group_var] = {}
        for block in ['baseline'] + blocks:
            all_data_tfr_continuous_dico[group_var][block] = {
                'ID': [],
                'motor':   {'Left': [], 'Right': [], 'Central': [], 'Both': []},
                'frontal': {'Left': [], 'Right': [], 'Central': [], 'Both': []},
                'STN':     {'Left': [], 'Right': []}
            }

    # Main loop over subjects/blocks
    for group_var in group_vars:
        for subj in subjects_dico[group_var]:
            print(f"Processing TFRs (no baseline) for {group_var} {subj}...")

            for block in blocks:
                all_data_tfr_continuous_dico[group_var][block]['ID'].append(subj)
                # Locate subject raws
                try:
                    task_idx = all_data_dico[group_var][block]['ID'].index(subj)
                    eeg_task_raw_full = all_data_dico[group_var][block]['EEG'][task_idx]
                    dbs_task_raw_full = all_data_dico[group_var][block].get('DBS_proc', [None])[task_idx]
                except (ValueError, IndexError):
                    # Append Nones if task block missing
                    for region_name, sides in regions_to_process.items():
                        for side_name in sides:
                            all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)
                    continue

                # EEG task → compute TFR per needed channels once, then region-average
                if eeg_task_raw_full:
                    present_task_chs = [ch for ch in all_eeg_channels if ch in eeg_task_raw_full.ch_names]
                    if present_task_chs:
                        eeg_task_raw_picked = eeg_task_raw_full.copy().pick(present_task_chs, verbose=False)
                        if target_sfreq is not None:
                            eeg_task_raw_picked.resample(target_sfreq, npad='auto')
                        tfr_eeg_task = mne.time_frequency.tfr_morlet(
                            eeg_task_raw_picked, freqs=freqs, n_cycles=n_cycles_evolving,
                            return_itc=False, average=False, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                        )
                        if power_to_db:
                            tfr_eeg_task.data = 10 * np.log10(np.maximum(tfr_eeg_task.data, eps))

                        # Region-averaged TFRs
                        for region_name, sides in regions_to_process.items():
                            for side_name, side_chs in sides.items():
                                region_chs_present = [ch for ch in side_chs if ch in tfr_eeg_task.ch_names]
                                if region_chs_present:
                                    region_tfr = tfr_eeg_task.copy().pick(region_chs_present, verbose=False)
                                    avg_data = region_tfr.data.mean(axis=0, keepdims=True)
                                    new_ch_name = f"{region_name}_{side_name}"
                                    avg_tfr_info = mne.create_info(
                                        ch_names=[new_ch_name], sfreq=tfr_eeg_task.info['sfreq'], ch_types='misc'
                                    )
                                    avg_tfr_state = {
                                        'data': avg_data,
                                        'times': region_tfr.times,
                                        'freqs': region_tfr.freqs,
                                        'info': avg_tfr_info,
                                        'nave': int(getattr(region_tfr, 'nave', 1)),
                                        'method': 'morlet',
                                        'comment': new_ch_name,
                                    }
                                    avg_tfr = mne.time_frequency.AverageTFR(inst=avg_tfr_state)
                                    all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(avg_tfr)
                                else:
                                    all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)
                    else:
                        for region_name, sides in regions_to_process.items():
                            for side_name in sides:
                                all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)
                else:
                    for region_name, sides in regions_to_process.items():
                        for side_name in sides:
                            all_data_tfr_continuous_dico[group_var][block][region_name][side_name].append(None)

                # STN task → compute single-channel TFR; store by DBS side
                stn_tfr_obj = None
                dbs_side_info = ''
                if dbs_task_raw_full is not None:
                    dbs_side_info = dbs_task_raw_full.info.get('description', '').lower()
                    if stn_channel[0] in dbs_task_raw_full.ch_names:
                        dbs_task_proc = dbs_task_raw_full.copy()
                        if target_sfreq is not None:
                            dbs_task_proc.resample(target_sfreq, npad='auto')
                        tfr_dbs_task = mne.time_frequency.tfr_morlet(
                            dbs_task_proc, freqs=freqs, n_cycles=n_cycles_evolving,
                            return_itc=False, average=True, n_jobs=n_jobs, decim=tfr_decim, use_fft=True, verbose=False
                        )
                        if power_to_db:
                            tfr_dbs_task.data = 10 * np.log10(np.maximum(tfr_dbs_task.data, eps))
                        stn_tfr_obj = tfr_dbs_task

                if 'dbs side: left' in dbs_side_info:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(stn_tfr_obj)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)
                elif 'dbs side: right' in dbs_side_info:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(stn_tfr_obj)
                else:
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Left'].append(None)
                    all_data_tfr_continuous_dico[group_var][block]['STN']['Right'].append(None)

    return all_data_tfr_continuous_dico
# ==============================================================================
# MAIN ORCHESTRATOR FOR AVG GAIT NORM METHOD (formerly Guo)
# ==============================================================================


def process_and_save_tfr_data_avggaitnorm(
    all_data_dico,
    group_vars,
    subjects_dico,
    blocks,
    freqs,
    motor_areas,
    frontal_areas,
    rejection_mask_dico,
    event_percentages,
    gait_events_correct_sequence,
    tfr_resultsDir,
    *,
    target_sfreq=None,
    tfr_decim=1,
    n_jobs=-1,
):
    os.makedirs(tfr_resultsDir, exist_ok=True)

    # 1) Continuous TFR (absolute dB)
    all_data_tfr_continuous = compute_tfr_on_subset(
        all_data_dico, group_vars, subjects_dico, blocks, freqs,
        motor_areas, frontal_areas,
        target_sfreq=target_sfreq, tfr_decim=tfr_decim, n_jobs=n_jobs,
        power_to_db=True 
    )

    # 2) Epoch + warp + average
    # all_data_tfr_epoched contains the AVERAGES
    # all_data_tfr_epoched_all_epochs contains the SINGLE TRIALS
    all_data_tfr_epoched, all_data_tfr_epoched_all_epochs = create_gait_warped_tfr_epochs_avggaitnorm(
        all_data_tfr_continuous,
        all_data_dico,
        rejection_mask_dico,
        event_percentages,
        gait_events_correct_sequence,
    )

    # =========================================================================
    # NEW STEP: Normalize Single Epochs using the Subject's Average "Baseline"
    # =========================================================================
    print("Normalizing single epochs based on subject average...")
    
    # Iterate through the Group structure
    for group in all_data_tfr_epoched:
        for block in blocks:
            if block not in all_data_tfr_epoched[group]:
                continue
            
            # Iterate through regions (Motor, Frontal, STN)
            for region in ["motor", "frontal", "STN"]:
                # Check if region exists in both Average and Epochs dicts
                if region not in all_data_tfr_epoched[group][block]: 
                    continue
                
                # Iterate through sides (Left, Right)
                for side, avg_tfr_list in all_data_tfr_epoched[group][block][region].items():
                    
                    # Get the corresponding list of Single Epochs
                    epochs_tfr_list = all_data_tfr_epoched_all_epochs[group][block][region][side]
                    
                    # Loop through subjects (by index)
                    for i, avg_tfr_obj in enumerate(avg_tfr_list):
                        epoch_tfr_obj = epochs_tfr_list[i]
                        
                        if avg_tfr_obj is None or epoch_tfr_obj is None:
                            continue

                        # ---------------------------------------------------
                        # CALCULATION:
                        # 1. Get the mean per frequency from the AVERAGE TFR
                        #    avg_tfr_obj.data shape: (n_channels, n_freqs, n_times)
                        #    We calculate mean across Time (last axis)
                        # ---------------------------------------------------
                        
                        # Shape: (n_channels, n_freqs, 1)
                        # Note: We keep dims to make broadcasting easy later
                         # 1. Get the mean from the AVERAGE object 
                        avg_data = avg_tfr_obj.data
                        
                        # Calculate mean across time (last axis)
                        # Shape: (n_channels, n_freqs, 1)
                        subject_mean_vector = np.mean(avg_data, axis=-1, keepdims=True)
                        
                        # 2. Subtract from the SINGLE EPOCHS object
                        # Expand dims for broadcasting: (1, n_channels, n_freqs, 1)
                        subject_mean_vector_expanded = subject_mean_vector[np.newaxis, ...]

                        # --- FIX IS HERE ---
                        # If epoch_tfr_obj is a numpy array, subtract directly.
                        if isinstance(epoch_tfr_obj, np.ndarray):
                            epoch_tfr_obj -= subject_mean_vector_expanded
                        else:
                            # If it is an MNE object (EpochsTFR)
                            epoch_tfr_obj.data -= subject_mean_vector_expanded

    # =========================================================================
    
    # NOW save the pickles (They are now Normalized!)
    save_data_to_pickle(
        all_data_tfr_epoched_all_epochs, "all_data_tfr_epoched_all_epochs_guo", tfr_resultsDir
    )

    # 3) Apply whole-gait-cycle normalization on the AVERAGED TFRs
    # (This ensures the group results are also normalized correctly)
    for group in all_data_tfr_epoched:
        for block in blocks:
            if block not in all_data_tfr_epoched[group]:
                continue
            for region in ["motor", "frontal", "STN"]:
                for side, lst in all_data_tfr_epoched[group][block][region].items():
                    for i, tfr_obj in enumerate(lst):
                        if tfr_obj is None:
                            continue
                        _normalize_avg_tfr_whole_cycle_inplace(tfr_obj)

    # 4) Build group dictionaries and save
    tfr_group_results, avg_tfr_group_results = _build_group_results_from_epoched(
        all_data_tfr_epoched, blocks, tfr_resultsDir, "guo_method", save_data_to_pickle
    )

    return tfr_group_results, avg_tfr_group_results, all_data_tfr_epoched_all_epochs
    