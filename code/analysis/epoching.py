import os
from typing import List, Dict

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from autoreject import AutoReject, get_rejection_threshold

from data_loader import save_data_to_pickle

# --- Helper Functions for create_epoched_data_dictionary ---

def filter_gait_events(eeg_or_dbs_mne, gait_events_correct_sequence):
    if eeg_or_dbs_mne is None:
        return []

    gait_events = [(time, duration, label) for time, duration, label in zip(
        eeg_or_dbs_mne.annotations.onset, 
        eeg_or_dbs_mne.annotations.duration, 
        eeg_or_dbs_mne.annotations.description) 
        if label in gait_events_correct_sequence]

    if not gait_events:
        return []

    filtered_events = []
    last_index = -1
    current_sequence = []
    seeking_new_start = True

    for i in range(len(gait_events) - 1):
        current_event = gait_events[i]
        next_event = gait_events[i + 1]
        time_diff = next_event[0] - current_event[0]

        if seeking_new_start:
            if len(current_sequence) < len(gait_events_correct_sequence):
                if not current_sequence:
                    current_sequence.append(current_event)
                    last_index = gait_events_correct_sequence.index(current_event[2])
                else:
                    expected_index = (last_index + 1) % len(gait_events_correct_sequence)
                    time_gap = current_event[0] - current_sequence[-1][0]
                    if current_event[2] == gait_events_correct_sequence[expected_index] and (time_gap < 1.5):
                        current_sequence.append(current_event)
                        last_index = expected_index
                    else:
                        if current_event[2] == gait_events_correct_sequence[0]:
                            current_sequence = [current_event]
                            last_index = 0
                        else:
                            current_sequence = []
                            last_index = -1
            if len(current_sequence) == len(gait_events_correct_sequence):
                filtered_events.extend(current_sequence)
                seeking_new_start = False
                last_index = gait_events_correct_sequence.index(current_sequence[-1][2])
                current_sequence = []
        else:
            if time_diff < 1.5:
                expected_index = (last_index + 1) % len(gait_events_correct_sequence)
                if current_event[2] == gait_events_correct_sequence[expected_index]:
                    filtered_events.append(current_event)
                    last_index = expected_index
                else:
                    seeking_new_start = True
                    current_sequence = []
            else:
                seeking_new_start = True
                current_sequence = []

    if not seeking_new_start and gait_events[-1][2] == gait_events_correct_sequence[(last_index + 1) % len(gait_events_correct_sequence)]:
        filtered_events.append(gait_events[-1])

    return filtered_events

def replace_events_in_mne(eeg_or_dbs_mne, filtered_gait_events, gait_events_correct_sequence):
    if eeg_or_dbs_mne is None:
        return None
    
    new_raw = mne.io.RawArray(eeg_or_dbs_mne.get_data(), eeg_or_dbs_mne.info)
    non_gait_events_tuples = [(event['onset'], event['duration'], event['description']) 
                            for event in eeg_or_dbs_mne.annotations 
                            if event['description'] not in gait_events_correct_sequence]

    combined_events = non_gait_events_tuples + filtered_gait_events
    combined_events.sort(key=lambda x: x[0])

    new_annotations = mne.Annotations(
        onset=[event[0] for event in combined_events],
        duration=[event[1] for event in combined_events],
        description=[event[2] for event in combined_events]
    )
    new_raw.set_annotations(new_annotations)
    return new_raw

def segment_gait_cycles(eeg_or_dbs_mne_filtered, valid_sequence, max_gap=1.5):
    if eeg_or_dbs_mne_filtered is None:
        return None

    annotations = [{'onset': ann[0], 'duration': ann[1], 'description': ann[2]} 
               for ann in zip(eeg_or_dbs_mne_filtered.annotations.onset,
                              eeg_or_dbs_mne_filtered.annotations.duration,
                              eeg_or_dbs_mne_filtered.annotations.description) 
               if ann[2] in valid_sequence]
    
    gait_cycles = []
    current_cycle = {}
    last_time = None

    for idx, annotation in enumerate(annotations):
        label, time = annotation['description'], annotation['onset']
        if label == valid_sequence[0]:
            if current_cycle and len(current_cycle) == len(valid_sequence):
                if last_time is not None and (time - last_time) <= max_gap:
                    current_cycle[valid_sequence[0] + '_end'] = time
                gait_cycles.append(current_cycle)
            current_cycle, last_time = {label: time}, time
        elif label in valid_sequence and label not in current_cycle:
            if last_time is not None and (time - last_time) <= max_gap and valid_sequence.index(label) == len(current_cycle):
                current_cycle[label], last_time = time, time
        
        if len(current_cycle) == len(valid_sequence):
            next_ann = annotations[idx + 1] if idx + 1 < len(annotations) else None
            if next_ann and next_ann['description'] == valid_sequence[0]:
                current_cycle[valid_sequence[0] + '_end'] = next_ann['onset']
                if (current_cycle[valid_sequence[0] + '_end'] - current_cycle[valid_sequence[0]]) < max_gap:
                    gait_cycles.append(current_cycle)
            current_cycle = {}
    return gait_cycles

def warp_dbs_or_eeg_data(eeg_data, gait_cycles, event_percentages):
    warped_eeg_data = []
    for cycle in gait_cycles:
        event_onsets = [cycle[event] for event in event_percentages.keys()]
        cycle_start, cycle_end = event_onsets[0], min(event_onsets[-1], eeg_data.times[-1])
        cycle_data = eeg_data.copy().crop(cycle_start, cycle_end)
        times, total_samples = cycle_data.times, len(cycle_data.times)
        target_indices = [int(round((event_percentages[event] / 100.0) * (total_samples - 1))) for event in event_percentages.keys()]
        warped_cycle = {ch: np.empty(total_samples) for ch in cycle_data.ch_names}
        original_event_indices = [np.argmin(np.abs(times - (cycle[event] - cycle_start))) for event in event_percentages.keys()]
        
        for seg in range(len(event_percentages.keys()) - 1):
            o_s, o_e = original_event_indices[seg], original_event_indices[seg + 1]
            t_s, t_e = target_indices[seg], target_indices[seg + 1]
            for ch in cycle_data.ch_names:
                data_seg = cycle_data.get_data(picks=ch).flatten()[o_s:o_e + 1]
                f = interp1d(np.arange(len(data_seg)), data_seg, kind='linear', fill_value="extrapolate")
                warped_cycle[ch][t_s:t_e + 1] = f(np.linspace(0, len(data_seg) - 1, num=t_e - t_s + 1))
        warped_eeg_data.append(warped_cycle)
    return warped_eeg_data

def resample_to_fixed_length(warped_cycles, desired_length=2400):
    resampled_cycles = []
    for warped_cycle in warped_cycles:
        resampled_cycle = {}
        for ch_name, ch_data in warped_cycle.items():
            f = interp1d(np.linspace(0, 1, len(ch_data)), ch_data, kind='linear')
            resampled_cycle[ch_name] = f(np.linspace(0, 1, desired_length))
        resampled_cycles.append(resampled_cycle)
    return resampled_cycles

def create_epochs_array(normalized_cycles, eeg_info, tmin=0):
    ch_names = list(normalized_cycles[0].keys())
    data = np.empty((len(normalized_cycles), len(ch_names), len(normalized_cycles[0][ch_names[0]])))
    for i, cycle in enumerate(normalized_cycles):
        for j, ch in enumerate(ch_names):
            data[i, j, :] = cycle[ch]
    return mne.EpochsArray(data, eeg_info, tmin=tmin)

# --- Helper Classes/Functions for clean_epochs_autoreject_paired ---

class SimpleRejectLog:
    def __init__(self, bad_epochs):
        self.bad_epochs = np.array(bad_epochs)
        self.labels = np.zeros((len(bad_epochs), 1), dtype=bool)

    def plot(self, show=False, title="Rejected epochs (DBS)"):
        fig, ax = plt.subplots()
        ax.imshow(self.labels.T, aspect='auto', cmap='gray_r', interpolation='nearest')
        for i, bad in enumerate(self.bad_epochs):
            if bad: ax.plot(i, 0, 'rx')
        if show: plt.show()
        return fig

def find_bad_dbs_epochs(dbs_epochs, threshold=500.):
    data = dbs_epochs.get_data()
    max_amplitudes = np.abs(data).max(axis=2).max(axis=1)
    return max_amplitudes > threshold

def auto_threshold_single_channel(epochs, percentile=99.5):
    data = epochs.get_data()
    max_amplitudes = np.abs(data).max(axis=2).flatten()
    return np.percentile(max_amplitudes, percentile)

def plot_evoked_before_after(epochs_before, epochs_after, ch_types, fname_prefix, resultsDir):
    for ch_type in ch_types:
        if epochs_before is None or epochs_after is None: continue
        picks = mne.pick_types(epochs_before.info, meg=(ch_type=='mag'), eeg=(ch_type=='eeg'), 
                               dbs=(ch_type=='dbs'), exclude=[])
        if len(picks) == 0: continue
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        epochs_before.copy().pick(picks).average().plot(axes=axes[0], show=False)
        epochs_after.copy().pick(picks).average().plot(axes=axes[1], show=False)
        fig.savefig(os.path.join(resultsDir, f"{fname_prefix}_{ch_type}_evoked_before_after.png"))
        plt.close(fig)

def plot_bad_epochs(epochs, bad_mask, resultsDir, subj, block, n_epochs_plot=5, picks=None):
    bad_idxs = np.where(bad_mask)[0][:n_epochs_plot]
    if len(bad_idxs) == 0: return
    if picks is None: picks = np.arange(len(epochs.ch_names))
    for idx in bad_idxs:
        data = epochs.get_data()[idx, picks, :]
        fig, ax = plt.subplots(figsize=(10, 3))
        for i, ch_idx in enumerate(picks): ax.plot(epochs.times, data[i] + i * 10)
        plt.savefig(os.path.join(resultsDir, f"bad_epoch_{subj}_{block}_{idx}.png"))
        plt.close(fig)

def plot_mean_sd_epochs(epochs_before, epochs_after, resultsDir, subj, block, picks=None, n_channels_plot=5, suptitle=None):
    if picks is None: picks = np.arange(min(n_channels_plot, len(epochs_before.ch_names)))
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 2 * len(picks)), sharex=True)
    if len(picks) == 1: axes = axes[np.newaxis, :]
    for i, pick in enumerate(picks):
        for j, ep in enumerate([epochs_before, epochs_after]):
            d = ep.get_data()[:, pick, :]
            m, s = d.mean(axis=0), d.std(axis=0)
            axes[i, j].plot(ep.times, m)
            axes[i, j].fill_between(ep.times, m - s, m + s, alpha=0.3)
    if suptitle: fig.suptitle(suptitle)
    fig.savefig(os.path.join(resultsDir, f'mean_sd_epochs_{subj}_{block}.png'))
    plt.close(fig)

# --- Main Functions ---

def create_epoched_data_dictionary(all_data_dico, event_percentages, gait_events_correct_sequence, blocks, group_vars, subjects_dico, dataDir, resultsDir, save_name="all_data_epoched_dico", resample_eeg=True, epoch_duration=1.1):
    all_data_epoched_dico, quality_summary = {}, []
    for group_var in group_vars:
        all_data_epoched_dico[group_var] = {"baseline": {"ID": [], "EEG": [], "DBS": []}, **{b: {"ID": [], "EEG": [], "DBS": []} for b in blocks}}
        for subj in subjects_dico[group_var]:
            try: idx_base = all_data_dico[group_var]['baseline']['ID'].index(subj)
            except ValueError: continue
            
            eeg_base = all_data_dico[group_var]['baseline']['EEG'][idx_base]
            eeg_base_ep = mne.Epochs(eeg_base, mne.make_fixed_length_events(eeg_base, duration=epoch_duration), tmin=0, tmax=epoch_duration, baseline=None, preload=True) if eeg_base else None
            dbs_base = all_data_dico[group_var]['baseline']['DBS'][idx_base]
            dbs_base_ep = mne.Epochs(dbs_base, mne.make_fixed_length_events(dbs_base, duration=epoch_duration), tmin=0, tmax=epoch_duration, baseline=None, preload=True) if group_var == "Patient_NFB" and dbs_base else None

            if eeg_base_ep and dbs_base_ep:
                min_n = min(len(eeg_base_ep), len(dbs_base_ep))
                eeg_base_ep, dbs_base_ep = eeg_base_ep[:min_n], dbs_base_ep[:min_n]

            all_data_epoched_dico[group_var]['baseline']['ID'].append(subj)
            all_data_epoched_dico[group_var]['baseline']['EEG'].append(eeg_base_ep)
            all_data_epoched_dico[group_var]['baseline']['DBS'].append(dbs_base_ep)

            for block in blocks:
                try: idx_blk = all_data_dico[group_var][block]['ID'].index(subj)
                except ValueError: continue
                eeg_blk, dbs_proc = all_data_dico[group_var][block]['EEG'][idx_blk], all_data_dico[group_var][block]['DBS_proc'][idx_blk]
                eeg_blk_ep = None
                if eeg_blk:
                    filt = replace_events_in_mne(eeg_blk, filter_gait_events(eeg_blk, gait_events_correct_sequence), gait_events_correct_sequence)
                    cycles = segment_gait_cycles(filt, gait_events_correct_sequence)
                    if cycles and resample_eeg:
                        warped = warp_dbs_or_eeg_data(filt, cycles, event_percentages)
                        eeg_blk_ep = create_epochs_array(resample_to_fixed_length(warped, desired_length=int(epoch_duration * eeg_blk.info['sfreq'])), filt.info)
                
                dbs_blk_ep = None
                if group_var == "Patient_NFB" and dbs_proc:
                    filt_d = replace_events_in_mne(dbs_proc, filter_gait_events(dbs_proc, gait_events_correct_sequence), gait_events_correct_sequence)
                    cycles_d = segment_gait_cycles(filt_d, gait_events_correct_sequence)
                    if cycles_d:
                        warped_d = warp_dbs_or_eeg_data(filt_d, cycles_d, event_percentages)
                        dbs_blk_ep = create_epochs_array(resample_to_fixed_length(warped_d, desired_length=int(epoch_duration * dbs_proc.info['sfreq'])), filt_d.info)
                
                if eeg_blk_ep and dbs_blk_ep:
                    min_n = min(len(eeg_blk_ep), len(dbs_blk_ep))
                    eeg_blk_ep, dbs_blk_ep = eeg_blk_ep[:min_n], dbs_blk_ep[:min_n]
                
                all_data_epoched_dico[group_var][block]['ID'].append(subj)
                all_data_epoched_dico[group_var][block]['EEG'].append(eeg_blk_ep)
                all_data_epoched_dico[group_var][block]['DBS'].append(dbs_blk_ep)

    if save_name: save_data_to_pickle(all_data_epoched_dico, save_name, dataDir)
    return all_data_epoched_dico

def clean_epochs_autoreject_paired(all_data_epoched_dico, resultsDir, save_csv_name="epochs_autoreject_summary.csv", plot_examples=True):
    summary_rows, all_data_epoched_cleaned_dico, rejection_mask_dico = [], {}, {}
    for group, group_dict in all_data_epoched_dico.items():
        all_data_epoched_cleaned_dico[group] = {}
        for block, block_dict in group_dict.items():
            all_data_epoched_cleaned_dico[group][block] = {"ID": block_dict['ID'].copy(), "EEG": [], "DBS": []}
            for i_subj, subj in enumerate(block_dict['ID']):
                eeg_ep, dbs_ep = block_dict['EEG'][i_subj], block_dict['DBS'][i_subj]
                if eeg_ep is None and dbs_ep is None:
                    all_data_epoched_cleaned_dico[group][block]['EEG'].append(None)
                    all_data_epoched_cleaned_dico[group][block]['DBS'].append(None)
                    continue

                n_orig = len(eeg_ep) if eeg_ep else len(dbs_ep)
                bad_eeg = AutoReject(random_state=42, n_jobs=-1, verbose=False).fit(eeg_ep).transform(eeg_ep, return_log=True)[1].bad_epochs if eeg_ep else np.zeros(n_orig, dtype=bool)
                bad_dbs = find_bad_dbs_epochs(dbs_ep, threshold=auto_threshold_single_channel(dbs_ep)) if dbs_ep else np.zeros(n_orig, dtype=bool)
                
                keep_mask = ~(bad_eeg | bad_dbs)
                rejection_mask_dico[(subj, block)] = keep_mask
                eeg_cl = eeg_ep[keep_mask] if eeg_ep else None
                dbs_cl = dbs_ep[keep_mask] if dbs_ep else None
                
                all_data_epoched_cleaned_dico[group][block]['EEG'].append(eeg_cl)
                all_data_epoched_cleaned_dico[group][block]['DBS'].append(dbs_cl)

                if plot_examples:
                    if eeg_cl: plot_evoked_before_after(eeg_ep, eeg_cl, ['eeg'], f"{subj}_{block}_EEG", resultsDir)
                    if dbs_cl: plot_evoked_before_after(dbs_ep, dbs_cl, ['dbs'], f"{subj}_{block}_DBS", resultsDir)

                summary_rows.append({"ID": subj, "Group": group, "Block": block, "Original_N": n_orig, "Cleaned_N": keep_mask.sum()})

    pd.DataFrame(summary_rows).to_csv(os.path.join(resultsDir, save_csv_name), index=False)
    return all_data_epoched_cleaned_dico, pd.DataFrame(summary_rows), rejection_mask_dico