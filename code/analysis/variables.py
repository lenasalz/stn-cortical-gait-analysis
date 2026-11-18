"""Shared analysis constants ported from ngb_study/notebooks/final_analysis/scripts/variables.py.

Inputs:  none
Outputs: none (constants only — module has no side effects on import)
Description: gait-event labels, channel-region groups (motor, frontal),
freq-band edges, color palettes. Migrated verbatim except for two
deletions: (1) the hardcoded `serverPath` / `projectDir_local_machine`
strings that re-identified co-author home paths, and (2) the
`os.makedirs(...)` calls that mutated disk on import. Path resolution
now lives in `code/analysis/_common.py:resolve_paths` driven by
`config.yaml`. The `*_resultsDir` symbols imported by `analysis_paper.ipynb`
(dataDir_server, dataDir_local_machine, resultsDir_local_machine,
band_power_resultsDir, tfr_resultsDir, tfr_stats_manually_resultsDir,
basenorm_tfr_resultsDir, avggait_tfr_resultsDir, ersds_resultsDir,
psd_resultsDir, fooof_resultsDir, connectivity_resultsDir,
evoked_resultsDir) must be re-bound from `_common.resolve_paths(...)`
in any notebook or script that previously read them from this module.
"""
import numpy as np

# we don't take FF as gait events
gait_event_colors = {
    'HSleftlocs': 'r',
    'TOleftlocs' : 'c',
    'HSrightlocs': 'b',
    'TOrightlocs': 'm',
}

gait_event_colors_all = {
    'HSleftlocs': 'r',
    'TOleftlocs' : 'c',
    'FFleftlocs': 'g',
    'HSrightlocs': 'b',
    'TOrightlocs': 'm',
    'FFrightlocs' : 'y'    
}

channel_colors = {
    # Left side channels (warm: red, orange, magenta, pink)
    'FC1': '#D7263D',  # deep red (crimson)
    'C1':  '#FF8C00',  # vivid orange
    'CP1': '#C3073F',  # strong magenta-red
    'FC3': '#FF61A6',  # bright pink
    'C3':  '#FFB300',  # strong gold (deeper than yellow)
    'CP3': '#E65100',  # dark orange

    # Right side channels (cool: blue, cyan, teal, green, purple, turquoise)
    'FC2': '#1338BE',  # deep blue
    'C2':  '#00CFFF',  # cyan
    'CP2': '#6A4FB6',  # violet
    'FC4': '#00A676',  # teal green
    'C4':  '#39FF14',  # neon green
    'CP4': '#005377',  # navy blue

    # Reference and DBS
    'Cz':  '#000000',  # black
    'DBS': '#808080'   # grey
}

freqs = np.arange(1, 46, 1) # freqs to analyze in time-freq analysis !

# Frequency bands definitions
freq_bands = {
    #'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Low Beta': (12, 20),
    'High Beta': (20, 30),
    'Gamma': (30, 45)
}

freq_bands_single_beta = {
    #'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)
}

gait_events_correct_sequence = [
    'HSrightlocs', 
    'TOleftlocs', 
    'HSleftlocs', 
    'TOrightlocs'
]

gait_events_correct_sequence_all = [
    'HSrightlocs', 
    'TOleftlocs', 
    'FFrightlocs',
    'HSleftlocs', 
    'TOrightlocs',
    'FFleftlocs'
]

event_percentages = {
    'HSrightlocs': 0,
    'TOleftlocs': 10, 
    'HSleftlocs': 50,
    'TOrightlocs': 60,
    'HSrightlocs_end' : 100
}

gait_phases = {
    'IDS': (event_percentages['HSrightlocs']/100, event_percentages['TOleftlocs']/100),  # Final Double Support
    'LLS': (event_percentages['TOleftlocs']/100,  event_percentages['HSleftlocs']/100),  # Left Leg Swing
    'FDS': (event_percentages['HSleftlocs']/100, event_percentages['TOrightlocs']/100),  # Initial Double Support
    'RLS': (event_percentages['TOrightlocs']/100, event_percentages['HSrightlocs_end']/100)   # Right Leg Swing
}

# Define areas

motor_areas = {
    'Left':  ['C1','C3'],
    'Right': ['C2','C4'],
    'Central':['Cz']
}
motor_areas['Both'] = (
    motor_areas['Left'] + motor_areas['Right'] + motor_areas['Central'] #, 'FCz', 'CPz']
)
frontal_areas = {
    'Left':  ['AF3','F1'],
    'Right': ['AF4','F2'],
    'Central': ['Fz']
}
frontal_areas['Both'] = (
    frontal_areas['Left'] + frontal_areas['Right'] +  frontal_areas['Central']
)
# Define categories to analyze for band powers
categories = ['EEG_motor', 'EEG_frontal', 'DBS']
categories_sides = ['EEG_motor_ipsi', 'EEG_motor_contra', 'EEG_frontal_ipsi', 'EEG_frontal_contra','DBS']
