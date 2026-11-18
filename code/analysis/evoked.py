import mne 
import matplotlib.pyplot as plt
import numpy as np 
import os 

def plot_joint_for_subject(epochs: mne.Epochs,
                           baseline_epochs: mne.Epochs,
                           subject_id: str,
                           event_percentages: dict,
                           out_dir: str = None):
    """
    Baseline-correct, average epochs, and make a `plot_joint`
    that shows topomaps at the requested gait-cycle landmarks.
    If you don't want to apply baseline correction, set baseline_epochs=None.
    """
    if epochs is None:
        print(f"Skipping subject {subject_id} (no epochs).")
        return
    # --- 1. Baseline-correct 
    # --- 2.  Average epochs → Evoked

    epochs = epochs.copy()
    

    if baseline_epochs is not None:
        baseline_epochs = baseline_epochs.copy()

        union_bads = list(set(epochs.info['bads']) | set(baseline_epochs.info['bads']))
        epochs = epochs.drop_channels(union_bads)
        if 'STI 014' in union_bads:
            union_bads.remove('STI 014')
        baseline_epochs = baseline_epochs.drop_channels(union_bads)
    else:
        print(f" Removed following bad channels: {epochs.info['bads']}")
        epochs = epochs.drop_channels(epochs.info['bads'])
    
    evoked = epochs.average()
    if baseline_epochs is not None:                    
        baseline_evoked = baseline_epochs.average()
        dc_offset = baseline_evoked.data.mean(axis=1, keepdims=True)
        evoked_corrected = evoked.copy()
        evoked_corrected.data -= dc_offset
    else:
        evoked_corrected = evoked


    
    # --- 3.  Convert % landmarks → times (skip 0 and 100 % for topomaps if desired)
    times_to_plot = [
        percentage_to_time(evoked_corrected, pct)
        for name, pct in event_percentages.items()
        #if 0 < pct < 100
    ]
    
    # --- 4.  Plot
    fig = evoked_corrected.plot_joint(
        times=times_to_plot,
        title=f"{subject_id}  |  EEG  (n={len(epochs)} epochs)",
        ts_args=dict(gfp=True),              # draws the black “global field power” trace
        topomap_args=dict(outlines='head')   # cleaner heads in the insets
    )
    
    # --- 5.  Annotate vertical lines & labels
    ax_butterfly = fig.axes[-1]             # last axis = butterfly plot
    for name, pct in event_percentages.items():
        t = percentage_to_time(evoked_corrected, pct)
        ax_butterfly.axvline(t, ls='--', lw=0.8, color='k')
        # tiny text label above the line
        ax_butterfly.text(t, ax_butterfly.get_ylim()[1],
                          f"{name}\n({pct}%)", ha='center', va='bottom', fontsize=8)
    
    # --- 6.  Save
    if out_dir is not None:
        out_file_path = os.path.join(out_dir,f"{subject_id}_ERP_joint.png")
        fig.savefig(out_file_path, dpi=300)
        print(f"Figure for {subject_id} saved to {out_file_path}")

    return fig



def percentage_to_time(evoked, pct):
    """Return the absolute time (in seconds) corresponding to `pct` % of epoch."""
    tmin, tmax = evoked.times[0], evoked.times[-1]
    return tmin + (tmax - tmin) * (pct / 100)


def run_joint_evoked_plots(
    resultsDir,
    group_vars,
    subjects_dico,
    all_data_epoched_dico,
    event_percentages,
    blocks=('pre8walk',),   # list or tuple of blocks to process (besides 'baseline')
    plot_baseline=True,
    plot_nocorrection=True,
    plot_with_baseline_correction=True,
    verbose=True
):
    """
    Generate joint ERP/evoked plots for all subjects/groups/blocks, baseline-corrected if desired.

    Parameters
    ----------
    blocks : iterable
        List of blocks (besides 'baseline') to plot.
    plot_baseline : bool
        Plot baseline ERP for each subject.
    plot_nocorrection : bool
        Plot ERP for each block, no baseline correction.
    plot_with_baseline_correction : bool
        Plot ERP for each block, with baseline correction.
    event_percentages : dict
        Mapping from event label to percent-in-epoch (as before).
    """
    event_percentages_clean = {
    k.replace("locs", ""): v for k, v in event_percentages.items()
    }
    for group_var in group_vars:
        for subject_id in subjects_dico[group_var]:
            if verbose:
                print(f"\n--- {group_var} / {subject_id} ---")
            try:
                idx_baseline = all_data_epoched_dico[group_var]['baseline']['ID'].index(subject_id)
            except ValueError:
                print(f"Subject {subject_id} not found in baseline ― skipping.")
                continue

            # Baseline epochs
            epochs_eeg_baseline = all_data_epoched_dico[group_var]['baseline']['EEG'][idx_baseline]

            if plot_baseline:
                if verbose:
                    print("Baseline")
                try:
                    plot_joint_for_subject(
                        epochs_eeg_baseline, None,
                        subject_id=f"{subject_id}_baseline",
                        event_percentages=event_percentages_clean,
                        out_dir=resultsDir
                    )
                except Exception as e:
                    print(f"Failed for {subject_id} baseline: {e}")

            for block in blocks:
                try:
                    idx = all_data_epoched_dico[group_var][block]['ID'].index(subject_id)
                except (ValueError, KeyError):
                    print(f"Subject {subject_id} not found in {block} ― skipping.")
                    continue

                # EEG epochs for this block
                epochs_eeg = all_data_epoched_dico[group_var][block]['EEG'][idx]

                # 1. No baseline correction
                if plot_nocorrection:
                    if verbose:
                        print(f"{block}, no baseline correction")
                    try:
                        plot_joint_for_subject(
                            epochs_eeg, None,
                            subject_id=f"{subject_id}_{block}_nobasecorr",
                            event_percentages=event_percentages_clean,
                            out_dir=resultsDir
                        )
                    except Exception as e:
                        print(f"Failed for {subject_id} {block} (no baseline): {e}")

                # 2. With baseline correction
                if plot_with_baseline_correction:
                    if verbose:
                        print(f"{block}, with baseline correction")
                    try:
                        plot_joint_for_subject(
                            epochs_eeg, epochs_eeg_baseline,
                            subject_id=f"{subject_id}_{block}_withbasecorr",
                            event_percentages=event_percentages_clean,
                            out_dir=resultsDir
                        )
                    except Exception as e:
                        print(f"Failed for {subject_id} {block} (with baseline): {e}")

