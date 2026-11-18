import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
#############################################################
#                 v13, final
#############################################################

class ERDSDataExporter:
    """
    Exports gait-related TFR data to CSV with flexible aggregation levels.
    
    Dimensions:
    1. Subject Aggregation (data_type): 'epochs', 'mean', 'median'
    2. Time Aggregation (time_aggregation): 'mean', 'median', or None (Full Time Series Array)
    """

    def __init__(self, tfr_group_results: Dict, tfr_group_results_epochs: Dict, freqs: np.ndarray,
                 freq_bands: Dict, time_axis_eeg: np.ndarray, time_axis_dbs: np.ndarray,
                 gait_phases: Dict, block_to_analyze: str = 'pre8walk'):
        
        print(f"--- Initializing GaitDataExporter for block: '{block_to_analyze}' ---")
        self.tfr_results = tfr_group_results               # Source for 'mean'
        self.tfr_results_epochs = tfr_group_results_epochs # Source for 'epochs' and 'median'
        self.freqs = freqs
        self.freq_bands = freq_bands
        self.time_axis_eeg = time_axis_eeg
        self.time_axis_dbs = time_axis_dbs
        self.block = block_to_analyze
        # Convert phases to percentage 0-100
        self.gait_phases_perc = {name: (start * 100, end * 100) for name, (start, end) in gait_phases.items()}

    # ==========================================================================
    # HELPERS
    # ==========================================================================
    def _circ_shift_percent_time(self, arr_2d: np.ndarray, shift_percent: float) -> np.ndarray:
        if arr_2d is None: return None
        a = np.squeeze(arr_2d)
        if a.ndim != 2: return None
        T = a.shape[1]
        k = int(round((shift_percent / 100.0) * T))
        return np.roll(a, shift=k, axis=1)

    def _resample_to_len_2d(self, arr_2d: np.ndarray, L: int) -> np.ndarray:
        if arr_2d is None: return None
        a = np.squeeze(arr_2d)
        if a.ndim != 2: return None
        nF, T = a.shape
        if T == L: return a
        x_old = np.linspace(0.0, 1.0, T)
        x_new = np.linspace(0.0, 1.0, L)
        return np.vstack([np.interp(x_new, x_old, a[i, :]) for i in range(nF)])

    def _get_freq_indices(self, freq_band_lim: tuple) -> np.ndarray:
        return np.where((self.freqs >= freq_band_lim[0]) & (self.freqs <= freq_band_lim[1]))[0]

    def _get_time_indices(self, time_perc_lim: tuple, time_axis: np.ndarray) -> np.ndarray:
        """Returns indices corresponding to a percentage range."""
        start_perc, end_perc = time_perc_lim
        total_len = len(time_axis)
        idx_start = int((start_perc / 100.0) * total_len)
        idx_end = int((end_perc / 100.0) * total_len)
        
        # Clamp
        idx_start = max(0, idx_start)
        idx_end = min(total_len - 1, idx_end)
        
        return np.arange(idx_start, idx_end + 1)

    def _get_source_and_prepare(self, data_type: str):
        """Selects the correct dictionary based on aggregation type."""
        if data_type in ['epochs', 'median']:
            return self.tfr_results_epochs
        elif data_type == 'mean':
            return self.tfr_results
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def _process_subject_data(self, tfr_data, data_type):
        """
        Handles the dimensionality of the raw data.
        Returns a list of tuples: (iteration_index, 2D_Array[freq, time])
        """
        tfr_data = np.squeeze(tfr_data)
        
        # 1. Subject Median: Collapse epochs -> 1 Matrix
        if data_type == 'median':
            if tfr_data.ndim == 3: # [epochs, freq, time]
                return [(0, np.median(tfr_data, axis=0))] #AB Here it is to change ! 
            elif tfr_data.ndim == 2: # Already averaged/single
                return [(0, tfr_data)]

        # 2. Subject Mean: Use pre-calc or collapse
        elif data_type == 'mean':
            if tfr_data.ndim == 2:
                return [(0, tfr_data)]
            elif tfr_data.ndim == 3:
                return [(0, np.mean(tfr_data, axis=0))] # Fallback if using epoch dict for mean

        # 3. Epochs: Yield every epoch
        elif data_type == 'epochs':
            if tfr_data.ndim == 3:
                return [(i, tfr_data[i, :, :]) for i in range(tfr_data.shape[0])]
            elif tfr_data.ndim == 2:
                return [(0, tfr_data)]

        return []

    # ==========================================================================
    # DATA FRAME CREATORS
    # ==========================================================================

    def _create_eeg_df(self, cohort_definitions: Dict, data_type: str, time_aggregation: Optional[str]) -> pd.DataFrame:
        all_rows = []
        data_source = self._get_source_and_prepare(data_type)
        # Mapping for Epoch dict vs Mean dict keys
        epoch_area_map = {'EEG_motor': 'motor', 'EEG_frontal': 'frontal'}

        for group, info in cohort_definitions.items():
            src_grp, ids = info["group"], set(info.get("ids") or [])
            if src_grp not in data_source or self.block not in data_source[src_grp]: continue
            
            # IDs might be in the dict or we rely on enumeration if list matches
            sub_ids_list = data_source[src_grp][self.block].get('ID', [])

            for area in ['EEG_motor', 'EEG_frontal']:
                # Handle key differences
                lookup = epoch_area_map.get(area, area) if data_type != 'mean' else area
                
                for hemi in ['Left', 'Both', 'Right']:
                    if lookup not in data_source[src_grp][self.block]: continue
                    if hemi not in data_source[src_grp][self.block][lookup]: continue
                    
                    subj_data_list = data_source[src_grp][self.block][lookup][hemi]

                    for s_idx, raw_tfr in enumerate(subj_data_list):
                        # Safety check for ID
                        if s_idx < len(sub_ids_list):
                            s_id = sub_ids_list[s_idx]
                        else:
                            s_id = f"Unknown_{s_idx}"
                            
                        if (ids and s_id not in ids) or raw_tfr is None: continue

                        # 1. Subject Aggregation Step
                        processed = self._process_subject_data(raw_tfr, data_type)

                        for idx, tfr_2d in processed:
                            if tfr_2d.ndim != 2: continue
                            
                            base_row = {'SubjectID': s_id, 'Group': group, 'Block': self.block, 'Area': area, 'Hemisphere': hemi}
                            if data_type == 'epochs': base_row['Epoch'] = idx + 1

                            # 2. Time Aggregation Step
                            self._append_rows(all_rows, base_row, tfr_2d, time_aggregation, self.time_axis_eeg)

        return pd.DataFrame(all_rows)

    def _create_dbs_df(self, cohort_definitions: Dict, data_type: str, time_aggregation: Optional[str]) -> pd.DataFrame:
        all_rows = []
        data_source = self._get_source_and_prepare(data_type)

        for group, info in cohort_definitions.items():
            src_grp, ids = info["group"], set(info.get("ids") or [])
            if src_grp not in data_source or self.block not in data_source[src_grp]: continue
            sub_ids_list = data_source[src_grp][self.block].get('ID', [])

            for area in ['STN']:
                for hemi in ['Left', 'Right']:
                    if area not in data_source[src_grp][self.block]: continue
                    if hemi not in data_source[src_grp][self.block][area]: continue
                    
                    subj_data_list = data_source[src_grp][self.block][area][hemi]

                    for s_idx, raw_tfr in enumerate(subj_data_list):
                        if s_idx < len(sub_ids_list): s_id = sub_ids_list[s_idx]
                        else: s_id = f"Unknown_{s_idx}"

                        if (ids and s_id not in ids) or raw_tfr is None: continue

                        processed = self._process_subject_data(raw_tfr, data_type)

                        for idx, tfr_2d in processed:
                            if tfr_2d.ndim != 2: continue
                            
                            base_row = {'SubjectID': s_id, 'Group': group, 'Block': self.block, 'Area': area, 'Hemisphere': hemi}
                            if data_type == 'epochs': base_row['Epoch'] = idx + 1
                            
                            self._append_rows(all_rows, base_row, tfr_2d, time_aggregation, self.time_axis_dbs)

        return pd.DataFrame(all_rows)

    def _create_dbs_ipsicontra_df(self, cohort_definitions: Dict, data_type: str, time_aggregation: Optional[str]) -> pd.DataFrame:
        all_rows = []
        data_source = self._get_source_and_prepare(data_type)
        target_len = len(self.time_axis_dbs)

        for group, info in cohort_definitions.items():
            src_grp, ids = info["group"], set(info.get("ids") or [])
            if src_grp not in data_source or self.block not in data_source[src_grp]: continue
            sub_ids_list = data_source[src_grp][self.block].get('ID', [])

            for area in ['STN']:
                for hemi in ['Left', 'Right']:
                    if area not in data_source[src_grp][self.block]: continue
                    if hemi not in data_source[src_grp][self.block][area]: continue
                    
                    subj_data_list = data_source[src_grp][self.block][area][hemi]

                    for s_idx, raw_tfr in enumerate(subj_data_list):
                        if s_idx < len(sub_ids_list): s_id = sub_ids_list[s_idx]
                        else: s_id = f"Unknown_{s_idx}"

                        if (ids and s_id not in ids) or raw_tfr is None: continue

                        processed = self._process_subject_data(raw_tfr, data_type)

                        for idx, tfr_2d in processed:
                            if tfr_2d.ndim != 2: continue

                            # --- IPSI/CONTRA ALIGNMENT ---
                            # Left -> Shift -50% (becomes Ipsi-like)
                            # Right -> No Shift (already Ipsi-like)
                            if hemi == 'Left':
                                tfr_final = self._circ_shift_percent_time(tfr_2d, shift_percent=-50.0)
                            else:
                                tfr_final = tfr_2d
                            
                            tfr_final = self._resample_to_len_2d(tfr_final, target_len)
                            if tfr_final is None: continue

                            base_row = {'SubjectID': s_id, 'Group': group, 'Block': self.block, 'Area': area, 'Hemisphere': hemi}
                            if data_type == 'epochs': base_row['Epoch'] = idx + 1

                            self._append_rows(all_rows, base_row, tfr_final, time_aggregation, self.time_axis_dbs, is_ipsicontra=True)

        return pd.DataFrame(all_rows)

    def _append_rows(self, rows_list, base_row, tfr_2d, time_aggregation, time_axis, is_ipsicontra=False):
        """
        Internal logic to extract frequency bands and either:
        A) Return full array (time_aggregation=None)
        B) Aggregate by phase (time_aggregation='mean'/'median')
        """
        
        # --- OPTION A: Time Series Export (Array) ---
        if time_aggregation is None:
            row = base_row.copy()
            for band_name, band_lims in self.freq_bands.items():
                f_idx = self._get_freq_indices(band_lims)
                sanitized = band_name.replace(' ', '_')
                
                # Average over Frequency Band, Keep Time Dimension
                # Result: 1D array of length T
                time_series = np.mean(tfr_2d[f_idx, :], axis=0)
                row[f'ERD_ERS_{sanitized}'] = list(time_series)
            rows_list.append(row)

        # --- OPTION B: Phase Aggregation ---
        else:
            for phase_name, phase_perc in self.gait_phases_perc.items():
                
                # Handle Ipsi/Contra Phase Renaming
                current_phase_name = phase_name
                if is_ipsicontra:
                    midpoint = np.mean(phase_perc)
                    # Mapping based on standard cycle percentages
                    if 10 <= midpoint < 50:
                        current_phase_name = "Contra Swing"
                    elif 60 <= midpoint <= 100:
                        current_phase_name = "Ipsi Swing"
                
                row = base_row.copy()
                row['GaitPhase'] = current_phase_name
                
                # Get Indices for this phase
                t_idx = self._get_time_indices(phase_perc, time_axis)
                
                for band_name, band_lims in self.freq_bands.items():
                    f_idx = self._get_freq_indices(band_lims)
                    sanitized = band_name.replace(' ', '_')
                    
                    # Extract subset: [Freqs, TimePoints]
                    subset = tfr_2d[f_idx[:, None], t_idx]
                    
                    # 1. Average over Frequency (Standard practice)
                    # 2. Aggregate over Time (Mean or Median as requested)
                    if time_aggregation == 'mean':
                        val = np.mean(subset)
                    elif time_aggregation == 'median':
                        val = np.median(subset)
                    
                    row[f'ERD_ERS_{sanitized}'] = val
                
                rows_list.append(row)

    # ==========================================================================
    # EXPORT MANAGER
    # ==========================================================================
    def export_data(self, output_path: str, cohort_definitions: Dict, 
                    data_type: str,            # 'epochs', 'mean', 'median'
                    export_mode: str,          # 'eeg', 'dbs', 'dbs_ipsicontra'
                    time_aggregation: Optional[str] = 'median'): # 'mean', 'median', None
        
        # Validation
        if data_type not in ['epochs', 'mean', 'median']:
            raise ValueError(f"Invalid data_type: {data_type}")
        if time_aggregation not in ['mean', 'median', None]:
            raise ValueError(f"Invalid time_aggregation: {time_aggregation}")

        print(f"Generating: Mode={export_mode} | Subj={data_type} | Time={time_aggregation}")

        # Dispatch
        if export_mode == 'eeg':
            df = self._create_eeg_df(cohort_definitions, data_type, time_aggregation)
        elif export_mode == 'dbs':
            df = self._create_dbs_df(cohort_definitions, data_type, time_aggregation)
        elif export_mode == 'dbs_ipsicontra':
            df = self._create_dbs_ipsicontra_df(cohort_definitions, data_type, time_aggregation)
        else:
            raise ValueError("export_mode must be 'eeg', 'dbs', or 'dbs_ipsicontra'")

        # Save
        if not df.empty:
            # Clean up Categoricals
            cat_cols = ['Group', 'Area', 'Hemisphere', 'SubjectID', 'Block']
            
            # Phase Ordering (Only exists if time_aggregation is not None)
            if 'GaitPhase' in df.columns:
                cat_cols.append('GaitPhase')
                if export_mode == 'dbs_ipsicontra':
                    # Custom Order
                    order = ['IDS', 'Contra Swing', 'FDS', 'Ipsi Swing']
                else:
                    # Standard Order
                    order = list(self.gait_phases_perc.keys())

                # Apply ordering safely
                available = df['GaitPhase'].unique().tolist()
                final_order = [x for x in order if x in available] + [x for x in available if x not in order]
                df['GaitPhase'] = df['GaitPhase'].astype('category').cat.set_categories(final_order, ordered=True)

            for col in cat_cols:
                if col in df.columns: df[col] = df[col].astype('category')

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"  -> Saved to {os.path.basename(output_path)}")
        else:
            print("  -> Warning: DataFrame empty. No file saved.")
