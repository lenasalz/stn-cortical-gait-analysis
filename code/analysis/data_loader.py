import csv
import json
import os
import pickle
import yaml
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

# crop_to_walking_part_only filters annotations against this palette's keys;
# variables.py is import-side-effect-free since the migration.
from variables import gait_event_colors


def load_eeg_data(dataDir: str, group_var: str, subject: str, block: str) -> Optional[mne.io.Raw]:
    """
    Load preprocessed EEG data with gait events (before ICA).

    Parameters
    ----------
    dataDir : str
        Base directory path for the data.
    group_var : str
        Group name (e.g., 'Patient_NFB', 'AgeMatched_NFB').
    subject : str
        Subject identifier.
    block : str
        Block name (e.g., 'baseline', 'pre8walk', 'treadmill').

    Returns
    -------
    eeg_data : mne.io.Raw or None
        MNE Raw object containing EEG data converted to µV, or None if loading fails.
    """
    file_path = os.path.join(dataDir, group_var, subject, 'ProcessedData', 'EEG', block, f'{block}_proc.fif')

    if not os.path.exists(file_path):
        print(f"EEG data not found at {file_path}")
        return None
    
    try:
        eeg_data = mne.io.read_raw_fif(file_path, preload=True)
        #eeg_data.apply_function(lambda x: x * 1e6, channel_wise=True)
        print(f"Loaded EEG data: {eeg_data.n_times} time points, {len(eeg_data.ch_names)} channels, converted to µV")
        return eeg_data
    except Exception as e:
        print(f"Error loading EEG data: {str(e)}")
        return None
    
def _apply_dbs_side_override(subject: str, side: Optional[str]) -> Optional[str]:
    """
    Replace the acquisition-config DBS side with a documented correction
    from config.yaml (`dbs_side_overrides`), if one exists for this subject.

    The server data root is read-only by project convention, so verified
    errors in `config_<subject>.yml` are corrected here rather than by
    editing the server file. Provenance for each override is documented in
    config.yaml next to the entry.
    """
    from _common import load_config
    overrides = load_config().get("dbs_side_overrides") or {}
    if subject in overrides:
        corrected = overrides[subject]
        print(f"WARNING {subject}: DBS side override applied — acquisition "
              f"config says {side!r}, using {corrected!r} "
              f"(see dbs_side_overrides in config.yaml)")
        return corrected
    return side


def load_dbs_data(dataDir: str, subject: str, block: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load preprocessed DBS data for a subject and extract the data for the requested block.
    
    The function also loads the configuration file to determine the side of the recording.

    Parameters
    ----------
    dataDir : str
        Base directory path for the data.
    subject : str
        Subject identifier.
    block : str
        Block name to filter the data.

    Returns
    -------
    dbs_block_data : pandas.DataFrame or None
        DataFrame containing the DBS time domain data for the requested block.
    side : str or None
        The side of the recording ('left', 'right', or None if unknown).
    """
    # Default file pattern: one CSV per subject with a 'block' column.
    # Per-block fallback (e.g. P4-2010 baseline): a separate recording
    # produced `<subject>_dbs_df_td_<block>.csv` + `config_<subject>_<block>.yml`.
    # Try the per-block variant first; if absent, fall back to the shared file.
    dbs_dir = os.path.join(dataDir, "Patient_NFB", subject, 'RawData', 'DBS')
    td_per_block     = os.path.join(dbs_dir, f'{subject}_dbs_df_td_{block}.csv')
    config_per_block = os.path.join(dbs_dir, f'config_{subject}_{block}.yml')
    td_shared        = os.path.join(dbs_dir, f'{subject}_dbs_df_td.csv')
    config_shared    = os.path.join(dbs_dir, f'config_{subject}.yml')

    if os.path.exists(td_per_block):
        td_file = td_per_block
        config_file_path = config_per_block if os.path.exists(config_per_block) else config_shared
    else:
        td_file = td_shared
        config_file_path = config_shared

    if os.path.exists(td_file):
        try:
            dbs_data = pd.read_csv(td_file)
            print(f"Loaded DBS time domain data: {dbs_data.shape[0]} time points")
            # Per-block CSV may omit the 'block' column entirely (single
            # block per file); shared CSV always has it.
            if 'block' in dbs_data.columns:
                dbs_block_data = dbs_data.loc[dbs_data['block'] == block].copy()
            else:
                dbs_block_data = dbs_data.copy()
                dbs_block_data['block'] = block
            dbs_block_data['TimeDomainData'] = dbs_block_data['TimeDomainData'] * 1e-6
            print(f" Found {dbs_block_data.shape[0]} time points for block {block}")
            if os.path.exists(config_file_path):
                try:
                    with open(config_file_path, 'r') as file:
                        config_file = yaml.safe_load(file)
                        side = config_file["patient"].get("sens_lead_loc", "Unknown")
                except Exception as e:
                    print(f"Error loading config file: {str(e)}")
            else:
                print("No config file found")
                side = None
            side = _apply_dbs_side_override(subject, side)
            if len(dbs_block_data) == 0:
                print(f"No DBS data found for block {block}")
                return None, None

            return dbs_block_data, side
        except Exception as e:
            print(f"Error loading DBS data: {str(e)}")
            return None, None  
    else:
        print(f"DBS data not found at {td_file}")
        return None, None
    
def align_dbs_with_eeg(eeg_data: mne.io.Raw, dbs_data: pd.DataFrame, block_name: str) -> pd.DataFrame:
    """
    Align DBS data with EEG data using the synchronization pulse.

    Parameters
    ----------
    eeg_data : mne.io.Raw
        MNE Raw object containing EEG data.
    dbs_data : pandas.DataFrame
        DataFrame containing DBS data.
    block_name : str
        Block name for reporting.

    Returns
    -------
    aligned_dbs_data : pandas.DataFrame
        DataFrame with DBS data aligned to EEG timing.
    """
    sync_events = [event for event in eeg_data.annotations if event['description'] == 'dbsSync']
    print(sync_events)
    
    if not sync_events:
        print("No synchronization pulse found in EEG data, /!\ cannot align DBS with EEG.")
        return eeg_data, None
    
    # Get the time of the synchronization pulse
    sync_time_eeg = sync_events[0]['onset']


    if dbs_data.empty:
        print(f"No DBS data found for block {block_name}")
        return None
    all_events_before = [event for event in eeg_data.annotations]
    print(f"first event onset: {all_events_before[0]['onset']}")

    eeg_data.crop(tmin=sync_time_eeg)

    eeg_data_crop = mne.io.RawArray(eeg_data.get_data(), eeg_data.info) # bug correction of MNE (annotations not shifted with "crop" function)

    eeg_cropped_ann = mne.Annotations(onset=[], duration=[], description=[]) # onset is in seconds
    
    # Add the gait events to the DBS annotations
    for event in eeg_data.annotations: # onset is in seconds
        eeg_cropped_ann.append(onset=(event['onset']-sync_time_eeg), duration=event['duration'], description=event['description']) 
    
    eeg_data_crop.set_annotations(eeg_cropped_ann)

    eeg_data_crop.crop(tmin=0)  #set the start time to 0

    print(f"first event onset after sync: {eeg_data_crop.annotations[0]['onset']}")
    print("first event after sync:", eeg_data_crop.annotations[0]['onset'])
    print(f"Cropped EEG data using synchronization pulse at {sync_time_eeg:.2f}s")

    return eeg_data_crop, sync_time_eeg


def convert_dbs_to_mne(dbs_data, dbs_side):
    """Convert DBS data from dataframe to MNE Raw object

    Args:
        dbs_data (pd.DataFrame): Dataframe extracted from CSV and cropped to block of interest
        dbs_side (str): The side of the recording ('left', 'right', or None if unknown)

    Returns:
        mne.io.Raw or None: MNE Raw object containing the DBS data
    """
    if dbs_data is None:
        return None
    data = np.array(dbs_data['TimeDomainData']).reshape(1, -1)
    sfreq_dbs = dbs_data['fs'].values[0]

    info = mne.create_info(
        ch_names=['DBS'],
        sfreq=sfreq_dbs,
        ch_types=['dbs']
    )
    raw_dbs = mne.io.RawArray(data, info)
    raw_dbs.info['description'] = f'DBS side: {dbs_side}'

    for ch in raw_dbs.info['chs']: # by default, MNE fills only the first 3 locs with zeros, but for Autoreject, we need all 12 locs to be non nan.
        ch['loc'] = np.zeros(12, dtype=float)

    return raw_dbs    


def add_gait_events_to_dbs_mne_file(dbs_data, eeg_data, gait_event_colors_palette, resample_dbs = False):
    """Add gait events from EEG data to DBS data as annotations. Be careful to set resample_dbs to True only if you want to upsample the DBS data to match the EEG sampling frequency.

    Args:
        dbs_data (mne.io.Raw): DBS data in MNE format
        eeg_data (mne.io.Raw): EEG data in MNE format, containing gait events
        gait_event_colors_palette (dict): Dictionary mapping event types to colors for visualization
        resample_dbs (bool, optional): If True, upsample DBS data to match EEG sampling frequency. Defaults to False.

    Returns:
        dbs_data (mne.io.Raw or None): DBS data with gait events as annotations, or None if input dbs_data is None
    """
    if dbs_data is None: 
        return None
#Get sampling frequencies to adjust gait events
    dbs_fs = dbs_data.info['sfreq']
    eeg_fs = eeg_data.info['sfreq']
    
    # Get the gait event annotations from the EEG data
    gait_events = [event for event in eeg_data.annotations if event['description'] in gait_event_colors_palette.keys()]
    print(f"Found {len(gait_events)} gait events in EEG data)") #, first event at {gait_events[0]['onset']:.2f}s, last event at {gait_events[-1]['onset']}")
    dbs_annotations = mne.Annotations(onset=[], duration=[], description=[]) # onset is in seconds
    
# Add the gait events to the DBS annotations
    for event in gait_events: # onset is in seconds
        dbs_annotations.append(onset=event['onset'], duration=event['duration'], description=event['description'])
    
    dbs_data.set_annotations(dbs_annotations)
    print(f"Added {len(dbs_annotations)} gait events to DBS data")

    if resample_dbs:
        dbs_data.resample(eeg_fs)
        print(f"Upsampled DBS data to match EEG sampling frequency: {eeg_fs} Hz")
        print(f"New dbs frequency check: {dbs_data.info['sfreq']}")

    return dbs_data

def preprocess_dbs_data(dbs_data):
    """Preprocess DBS data by applying a bandpass filter

    Args:
        dbs_data (mne.io.Raw): DBS data in MNE format

    Returns:
        dbs_data_proc (mne.io.Raw or None): Filtered DBS data (bandpass 1-100 Hz), or None if input is None
    """
    if dbs_data is None: 
        return None
    
    dbs_data_proc = dbs_data.copy().filter(l_freq=4, h_freq=100, method = 'fir', n_jobs = -1) #can be cuda if cupy
    print("Applied bandpass filter to DBS data (1-100 Hz)")

    return dbs_data_proc

def crop_to_walking_part_only(eeg, raw_dbs, dbs):
    """Crop the EEG and DBS data to the walking part only. /!\ this function might introduce a small shift of 0.0035 s. 
    
    Args:
        eeg (mne.io.Raw): MNE Raw object containing EEG data
        raw_dbs (mne.io.Raw): MNE Raw object containing raw DBS data
        dbs (mne.io.Raw): MNE Raw object containing processed DBS data

    Returns:
        eeg (mne.io.Raw): Cropped EEG data
        dbs (mne.io.Raw): Cropped DBS data
    """
    if dbs is None: 
        one_frame = 1 / 250 # default sampling frequency for DBS
    else:
        one_frame = 1 / dbs.info['sfreq']
    if eeg is None: 
        eeg_data_crop = None
    else: 
        eeg_copy = eeg.copy()
        # Get the gait events from the EEG data
        gait_events = [event for event in eeg_copy.annotations if event['description'] in gait_event_colors.keys()]
        #print(f" first gait event eeg onset: {gait_events[0]['onset']}, description: {gait_events[0]['description']}")
        # Get the start and end times of the walking part
        start_time = gait_events[0]['onset'] - one_frame # 1 frame before to be sure to include the first event
        #print("start time eeg:", start_time)
        end_time = gait_events[-1]['onset'] + gait_events[-1]['duration'] + 2* one_frame #duration should be 0
        #print(f" First gait event at {start_time:.2f}s, last gait event at {end_time:.2f}s")

        # Crop both datasets to the walking part
        eeg_copy.crop(tmin=start_time, tmax=end_time)
        gait_events = [event for event in eeg_copy.annotations if event['description'] in gait_event_colors.keys()]
        #print(f" first gait event eeg onset after copying: {gait_events[0]['onset']}, description: {gait_events[0]['description']}")
        eeg_data_crop = mne.io.RawArray(eeg_copy.get_data(), eeg_copy.info)
        eeg_cropped_ann = mne.Annotations(onset=[], duration=[], description=[]) # onset is in seconds
        
        # Add the gait events to the DBS annotations
        for event in eeg_copy.annotations: # onset is in seconds
            eeg_cropped_ann.append(onset=event['onset']-start_time, duration=event['duration'], description=event['description'])
        
        eeg_data_crop.set_annotations(eeg_cropped_ann)
        eeg_data_crop.crop(tmin=0)  #set the start time to 0
        gait_events = [event for event in eeg_data_crop.annotations if event['description'] in gait_event_colors.keys()]
        #print(f" first gait event eeg onset after cropping: {gait_events[0]['onset']}, description: {gait_events[0]['description']}")
        

    if dbs is None: 
        dbs_data_crop = None
    else:
        dbs_copy = dbs.copy()
        raw_dbs_copy = raw_dbs.copy()
        gait_events = [event for event in dbs_copy.annotations if event['description'] in gait_event_colors.keys()]
        # Get the start and end times of the walking part
        start_time = gait_events[0]['onset'] - one_frame # 1 frame before to be sure to include the first event
        #print("start time dbs:", start_time)
        end_time = gait_events[-1]['onset'] + gait_events[-1]['duration'] + 2*one_frame #duraion should be 0
        #print("DBS gait event 0 onset minus start_time:", gait_events[0]['onset'] - start_time)

        dbs_copy.crop(tmin=start_time, tmax=end_time)
        raw_dbs_copy.crop(tmin=start_time, tmax=end_time)

        gait_events = [event for event in dbs_copy.annotations if event['description'] in gait_event_colors.keys()]
        #print(f" first gait event dbs_copy onset crop: {gait_events[0]['onset']}, description: {gait_events[0]['description']}")

        dbs_data_crop = mne.io.RawArray(dbs_copy.get_data(), dbs_copy.info)
        dbs_cropped_ann = mne.Annotations(onset=[], duration=[], description=[]) # onset is in seconds
        #print("DBS gait event 0 onset minus start_time:", gait_events[0]['onset'] - start_time)
        raw_dbs_data_crop = mne.io.RawArray(raw_dbs_copy.get_data(), dbs_copy.info)
        raw_dbs_cropped_ann = mne.Annotations(onset=[], duration=[], description=[])

        for event in dbs_copy.annotations: # onset is in seconds
            #print(f" loop dbs event onset: {event['onset']}, description: {event['description']}")
            dbs_cropped_ann.append(onset=event['onset']-start_time, duration=event['duration'], description=event['description'])
            raw_dbs_cropped_ann.append(onset=event['onset']-start_time, duration=event['duration'], description=event['description'])
        
        dbs_data_crop.set_annotations(dbs_cropped_ann)
        raw_dbs_data_crop.set_annotations(raw_dbs_cropped_ann)

        dbs_data_crop.crop(tmin=0)
        raw_dbs_data_crop.crop(tmin=0)
        #gait_events = [event for event in dbs_data_crop.annotations if event['description'] in gait_event_colors.keys()]
        #print(f" first gait event dbs onset after cropping: {gait_events[0]['onset']}, description: {gait_events[0]['description']}")

        # print("DEBUG data_loader 311:")
        # print("EEG gait events after crop:")
        # for e in eeg_data_crop.annotations:
        #     if e['description'] in gait_event_colors.keys():
        #         print(e['onset'], e['description'])

        # print("DBS gait events after crop:")
        # for e in dbs_data_crop.annotations:
        #     if e['description'] in gait_event_colors.keys():
        #         print(e['onset'], e['description'])

    return eeg_data_crop, raw_dbs_data_crop, dbs_data_crop

def create_data_dictionary(
    dataDir: str,
    blocks: list,
    group_var: str,
    subjects: list,
    gait_event_colors_palette: dict,
    resample_eeg: bool = True,
    crop_to_walking: bool = True,
):
    """Create a dictionary containing the EEG and DBS data for all subjects in the group
    for the baseline and each block of interest.

    Args:
        dataDir (str): Base directory path for the data.
        blocks (list[str]): List of block names (e.g., ['pre8walk', 'treadmill']).
        group_var (str): Group name (e.g., 'Patient_NFB', 'AgeMatched_NFB').
        subjects (list[str]): List of subject identifiers.
        gait_event_colors_palette (dict): Mapping event types → colors for visualisation.
        resample_eeg (bool, optional): If True, resample EEG to match DBS sampling frequency. Defaults to True.

    Returns:
        dict[str, dict]: Nested dictionary with one entry per block (including 'baseline'),
            where each entry is a dict with keys 'ID', 'EEG', 'DBS', 'DBS_proc'.
    """

    # Prepare an output container for every requested block plus the mandatory baseline
    all_data = {
        "baseline": {"ID": [], "EEG": [], "DBS": [], "DBS_proc": []},
        **{blk: {"ID": [], "EEG": [], "DBS": [], "DBS_proc": []} for blk in blocks},
    }
    
    # Iterate over baseline + experimental blocks in a single loop
    for actual_block in ["baseline", *blocks]:
        for sub in subjects:
            eeg_data, dbs_data, dbs_data_proc = None, None, None
            eeg_data = load_eeg_data(dataDir, group_var, sub, actual_block)
            if eeg_data is not None:
                if group_var == "Patient_NFB":
                    print(dataDir, sub, actual_block)
                    dbs_data_df, dbs_side = load_dbs_data(dataDir, sub, actual_block)
                    if dbs_data_df is not None:
                        # Align, convert, and preprocess DBS data
                        eeg_data, dbs_sync_pulse = align_dbs_with_eeg(eeg_data, dbs_data_df, actual_block)
                        
                        dbs_data = convert_dbs_to_mne(dbs_data_df, dbs_side)
                        
                        if dbs_sync_pulse is not None:
                            dbs_data = add_gait_events_to_dbs_mne_file(
                                dbs_data,
                                eeg_data,
                                gait_event_colors_palette,
                                resample_dbs=False,
                            )
                            dbs_data_proc = preprocess_dbs_data(dbs_data)
                        else:
                            print(
                                "No synchronization pulse found in EEG data, cannot align DBS with EEG. DBS data can not be analyzed, so processing skipped, dbs data not saved."
                            )
                            dbs_data = None
                            dbs_data_proc = None

                        # Optionally resample EEG to DBS frequency
                        if resample_eeg:
                            dbs_fs = dbs_data.info["sfreq"] if dbs_data is not None else 250
                            eeg_data = eeg_data.resample(dbs_fs)
                            print(
                                f"EEG data resampled to {dbs_fs} Hz to match DBS sampling frequency"
                            )
                    else:
                        dbs_data = None
                        dbs_data_proc = None
                    if crop_to_walking and actual_block != "baseline":
                            eeg_data, dbs_data, dbs_data_proc = crop_to_walking_part_only(
                                eeg_data, dbs_data, dbs_data_proc
                            )
                            print(f"Cropped EEG and DBS data to walking part only")
                else:
                    dbs_data = None
                    dbs_data_proc = None
            else:
                dbs_data = None
                dbs_data_proc = None

            # Store data in the corresponding block container
            all_data[actual_block]["ID"].append(sub)
            all_data[actual_block]["EEG"].append(eeg_data)
            all_data[actual_block]["DBS"].append(dbs_data)
            all_data[actual_block]["DBS_proc"].append(dbs_data_proc)

    return all_data

def resample_eeg_to_dbs(eeg_data, dbs_data, default_fs=250):
    """Resample EEG data to match the sampling frequency of DBS data

    Args:
        eeg_data (mne.io.Raw): MNE Raw object containing EEG data
        dbs_data (mne.io.Raw): MNE Raw object containing DBS data
        default_fs (int, optional): Default sampling frequency to use if dbs_data is None. Defaults to 250.

    Returns:
        eeg_data (mne.io.Raw): Resampled EEG data
    """
    if dbs_data is None:
        print("DBS data is None, using default sampling frequency")
        dbs_fs = default_fs
    else:
        dbs_fs = dbs_data.info['sfreq']
    if eeg_data is None:
        print("EEG data is None, cannot resample")
        return None
    else:
        if eeg_data.info['sfreq'] != dbs_fs:
            print(f"Resampling EEG data from {eeg_data.info['sfreq']} Hz to {dbs_fs} Hz")
            eeg_data.resample(dbs_fs)
        else:
            print("EEG data already at the correct sampling frequency")
        return eeg_data


def save_data_to_pickle(data: any, filename: str, dir_path: str) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    data : any
        Any picklable Python object.
    filename : str
        Name of the file (without extension).
    dir_path : str
        Directory path where the file will be saved.
        
    Returns
    -------
    None
        Saves the data to disk as a pickle file.
    """
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename + ".pkl")
    # Save the data to a pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data successfully saved to '{filepath}'")


def load_data_from_pickle(filename, dir_path):
    """Load a Python object from a pickle file

    Args:
        filename (str): Name of the file (without extension)
        dir_path (str): Directory path where the pickle is located

    Returns:
        data (any): The Python object that was pickled

    Raises:
        FileNotFoundError: If the file does not exist
    """
    filepath = os.path.join(dir_path, filename + ".pkl")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from '{filepath}'")
    return data


def export_band_powers_dico_regions_to_csv(band_powers_dico_regions, csv_file_path):
    """Export a nested dictionary of band powers data to a CSV file

    Args:
        band_powers_dico_regions (dict): Nested dictionary with structure:
            {region: {group: {freq_band: [band power dictionaries]}}}
        csv_file_path (str): Path to the CSV file to be written
    """
    rows = []
    for region, region_data in band_powers_dico_regions.items():
        for group, group_data in region_data.items():
            for freq_band, bp_list in group_data.items():
                for idx, bp in enumerate(bp_list):
                # Create a row with the extra metadata.
                    row = {
                        'region': region,
                        'group': group,
                        'freq_band': freq_band,
                        'subject_index': idx
                    }
                    # Add the band power dictionary keys.
                    row.update(bp)
                    
                    # If any value is a list (e.g., channels_to_pick), convert it to a JSON string.
                    for key, value in row.items():
                        if isinstance(value, list):
                            row[key] = json.dumps(value)
                    
                    rows.append(row)
    
# Gather all column names across rows.
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys = list(all_keys)
    
# Write rows to CSV.
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Exported band power data to {csv_file_path}")


def import_band_powers_dico_regions_from_csv(csv_file_path):
    """Import band powers data from a CSV file and reconstruct the nested dictionary structure

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        nested_dic (dict): Reconstructed nested dictionary with structure:
            {region: {group: {freq_band: [band power dictionaries]}}}
    """
    nested_dict = {}
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row.pop('region')
            group = row.pop('group')
            freq_band = row.pop('freq_band')
            row.pop('subject_index')
            
            for key, value in row.items():
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        row[key] = json.loads(value)
                    except Exception:
                        pass  # if conversion fails, leave as string
                # Try to convert numeric values to float.
                try:
                    row[key] = float(row[key])
                except Exception:
                    pass
            
# Rebuild the nested structure.
            if region not in nested_dict:
                nested_dict[region] = {}
            if group not in nested_dict[region]:
                nested_dict[region][group] = {}
            if freq_band not in nested_dict[region][group]:
                nested_dict[region][group][freq_band] = []
            nested_dict[region][group][freq_band].append(row)
    print(f"Imported band power data from {csv_file_path}")
    return nested_dict

