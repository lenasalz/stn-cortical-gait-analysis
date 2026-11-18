"""Gait-warp a basenorm continuous TFR into per-cycle epochs and average.

Inputs:
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-cont-basenorm.h5  (compute_tfr_basenorm)
  - data/derivatives/raw_clean/<subj>_<task>-{eeg,dbs-proc}-raw.fif    (build_raw_from_pickle; for gait annotations)
  - $RESIDUAL_DATA_ROOT/Final_filtered/rejection_mask_dico.pkl         (legacy autoreject masks)
  - code/analysis/config.yaml (gait_warp.* parameters; falls back to variables.py defaults)
Outputs:
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-gait.h5          (AverageTFR, gait-warped)
  - data/derivatives/tfr/<subj>_<task>-{eeg,dbs}-tfr-gait.h5.meta.json
Description:
  Replicates `create_gait_warped_tfr_epochs` from final_analysis/scripts/tfr.py
  at the channel level (the legacy operates on already-region-averaged
  TFRs; keeping channel granularity here lets downstream code aggregate
  into any region without recomputation):
    1. Pull `filter_gait_events` / `replace_events_in_mne` /
       `segment_gait_cycles` from epoching.py against the Raw object's
       annotations.
    2. For each gait cycle, slice the basenorm continuous TFR between
       HSrightlocs and HSrightlocs_end (`extract_tfr_epochs_from_cycles`).
    3. Piecewise-linear time-warp each cycle so the four canonical events
       land at their target percentages (0 / 10 / 50 / 60 / 100).
    4. Resample warped epochs to a fixed duration (default 1.1 s at the
       TFR's post-decim sfreq).
    5. Filter via `rejection_mask_dico[(subj, block)]` from the legacy
       autoreject output.
    6. Mean across kept epochs → AverageTFR with shape
       `(n_chans, n_freqs, n_times)`.

  Sidecar records the number of cycles, the number kept after rejection,
  and the channels carried through (matches `compute_tfr_basenorm.py`
  output minus any dropped during pick).

Usage:
  python compute_tfr_gait.py --subject P4-2002 --block pre8walk --modality eeg
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from _common import (
    atomic_write,
    cohort_of,
    load_config,
    output_path,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    validate_subject,
    write_sidecar,
)
from epoching import (
    filter_gait_events,
    replace_events_in_mne,
    segment_gait_cycles,
)
from tfr import (
    extract_tfr_epochs_from_cycles,
    resample_tfr_cycles,
    warp_tfr_cycles,
)
from variables import (
    event_percentages as DEFAULT_EVENT_PERCENTAGES,
    gait_events_correct_sequence as DEFAULT_GAIT_SEQ,
)

EPOCH_DURATION_S = 1.1   # legacy default in create_gait_warped_tfr_epochs


def _load_rejection_mask(per_subject_path: Path, pkl_path: Path,
                         key: tuple[str, str]) -> tuple[np.ndarray | None, str]:
    """Prefer per-subject mask from `compute_autoreject_mask.py`; fall back
    to the legacy `rejection_mask_dico.pkl` cache. Returns (mask, source)."""
    if per_subject_path.exists():
        return np.load(per_subject_path)["keep_mask"], "per_subject_npz"
    if not pkl_path.exists():
        return None, "missing"
    with pkl_path.open("rb") as f:
        masks = pickle.load(f)
    return masks.get(key), "legacy_pkl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True,
                        help="Task block (gait-warping is undefined for baseline).")
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.block == "baseline":
        parser.error("baseline has no gait events; pick a task block")

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    tfr_in = output_path(paths.tfr_dir, subject, block,
                         f"-{args.modality}-tfr-cont-basenorm.h5")
    raw_suffix = "-eeg-raw.fif" if args.modality == "eeg" else "-dbs-proc-raw.fif"
    raw_in = output_path(paths.raw_clean_dir, subject, block, raw_suffix)
    out_path = output_path(paths.tfr_dir, subject, block,
                           f"-{args.modality}-tfr-gait.h5")
    mask_pkl = paths.data_root / "Final_filtered" / "rejection_mask_dico.pkl"
    mask_npz = (paths.derivatives_root / "autoreject" /
                f"{subject}_{block}-mask.npz")

    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    for p, name in ((tfr_in, "basenorm TFR"), (raw_in, "raw")):
        if not p.exists():
            print(f"missing {name}: {p}", file=sys.stderr)
            return 2

    meta = setup_run("compute_tfr_gait", config)

    import mne
    from _common import patch_legacy_annotations

    tfr = mne.time_frequency.read_tfrs(tfr_in)
    tfr = tfr[0] if isinstance(tfr, list) else tfr
    raw = mne.io.read_raw_fif(raw_in, preload=False, verbose="WARNING")
    patch_legacy_annotations(raw)

    # 1) gait cycles
    gait_seq = DEFAULT_GAIT_SEQ
    ev_pct = DEFAULT_EVENT_PERCENTAGES
    fe = filter_gait_events(raw, gait_seq)
    raw_filt = replace_events_in_mne(raw, fe, gait_seq)
    gait_cycles = segment_gait_cycles(raw_filt, gait_seq)
    if not gait_cycles:
        print(f"no gait cycles found for {subject} {block}", file=sys.stderr)
        return 3

    # 2) extract per-cycle slices once (channel-wise data preserved)
    sfreq = tfr.info["sfreq"]
    n_chans, n_freqs = tfr.data.shape[0], tfr.data.shape[1]
    cycle_slices = []   # list of (cycle_info_dict, start_idx, end_idx)
    for cycle in gait_cycles:
        if "HSrightlocs" not in cycle or "HSrightlocs_end" not in cycle:
            continue
        try:
            start_idx = tfr.time_as_index(cycle["HSrightlocs"], use_rounding=True)[0]
            end_idx   = tfr.time_as_index(cycle["HSrightlocs_end"], use_rounding=True)[0]
        except IndexError:
            continue
        if end_idx > start_idx:
            cycle_slices.append((cycle, start_idx, end_idx))
    if not cycle_slices:
        print(f"no usable cycles for {subject} {block}", file=sys.stderr)
        return 4

    n_cycles = len(cycle_slices)
    n_times_fixed = int(round(EPOCH_DURATION_S * sfreq))
    # Complete cycles carry HSrightlocs_end; cycle_slices only drops
    # complete cycles that fall outside the TFR's time range. When no
    # such drop happened, slice index i == epoch index i in
    # build_epochs_from_raw's cycle order.
    n_complete_cycles = sum(
        1 for c in gait_cycles
        if "HSrightlocs" in c and "HSrightlocs_end" in c
    )

    # 5) rejection mask (per-subject npz from compute_autoreject_mask.py
    # preferred; fall back to legacy rejection_mask_dico.pkl). Applied
    # once across every channel's cycles.
    mask, mask_source = _load_rejection_mask(mask_npz, mask_pkl, (subject, block))
    if mask is None:
        kept_bool = np.ones(n_cycles, dtype=bool)
        rejection_status = "no_mask_found"
    elif len(mask) < n_cycles and n_cycles == n_complete_cycles:
        # Paired-epoch truncation: build_epochs_from_raw keeps the first
        # min(n_eeg, n_dbs) cycles when one modality's recording ends
        # early (e.g. P4-2009 post8walk, DBS stream shorter than EEG), so
        # the mask indexes the first len(mask) cycles in temporal order.
        # Drop the uncovered remainder to match the paired methodology.
        cycle_slices = cycle_slices[:len(mask)]
        n_cycles = len(cycle_slices)
        kept_bool = mask.astype(bool)
        rejection_status = (
            f"applied_truncated (mask={len(mask)} "
            f"complete_cycles={n_complete_cycles}; source={mask_source})"
        )
    elif len(mask) != n_cycles:
        kept_bool = np.ones(n_cycles, dtype=bool)
        rejection_status = f"length_mismatch (mask={len(mask)} cycles={n_cycles}; source={mask_source})"
    else:
        kept_bool = mask.astype(bool)
        rejection_status = f"applied (source={mask_source})"
    n_kept = int(kept_bool.sum())
    if n_kept == 0:
        print(f"all cycles rejected for {subject} {block}", file=sys.stderr)
        return 5

    # 3 + 4) warp + resample per channel using the legacy helpers (single-channel
    # contract). Each cycle becomes a (1, n_freqs, n_times) array; warp/resample
    # return the same shape with a fixed time dimension. Stack channels at the
    # end into a single (n_chans, n_freqs, n_times_fixed) AverageTFR.
    cycle_dicts = [c for c, _, _ in cycle_slices]
    mean_tfr = np.empty((n_chans, n_freqs, n_times_fixed), dtype=tfr.data.dtype)
    for ch_i in range(n_chans):
        per_cycle = [
            tfr.data[ch_i:ch_i + 1, :, s:e]  # (1, n_freqs, cycle_len)
            for _, s, e in cycle_slices
        ]
        warped = warp_tfr_cycles(per_cycle, cycle_dicts, ev_pct, sfreq)
        resampled = resample_tfr_cycles(warped, EPOCH_DURATION_S, sfreq)
        # resampled is a list of (1, n_freqs, n_times_fixed); stack to (n_cycles, ...).
        stacked = np.concatenate(resampled, axis=0)            # (n_cycles, n_freqs, n_times_fixed)
        mean_tfr[ch_i] = stacked[kept_bool].mean(axis=0)
    stacked_shape0 = n_kept   # for sidecar / nave

    new_times = np.linspace(0.0, EPOCH_DURATION_S, mean_tfr.shape[-1])
    info = mne.create_info(ch_names=list(tfr.ch_names),
                           sfreq=sfreq, ch_types="misc")
    # Carry the input TFR's description forward — downstream code reads
    # this to recover DBS side ('DBS side: left' / 'right') since the
    # single-channel STN can't be inferred from ch_names alone.
    input_description = tfr.info.get("description")
    if input_description:
        with info._unlock():
            info["description"] = input_description
    avg_state = {
        "data": mean_tfr,
        "times": new_times,
        "freqs": tfr.freqs,
        "info": info,
        "nave": int(stacked_shape0),
        "method": "morlet",
        "comment": f"gait-warped {block}",
    }
    out_tfr = mne.time_frequency.AverageTFR(inst=avg_state)

    with atomic_write(out_path) as tmp:
        mne.time_frequency.write_tfrs(tmp, out_tfr, overwrite=True)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality,
        "basenorm_input": str(tfr_in),
        "raw_input": str(raw_in),
        "rejection_mask_pkl": str(mask_pkl),
        "rejection_status": rejection_status,
        "n_cycles_segmented": len(gait_cycles),
        "n_cycles_after_warp": n_cycles,
        "n_cycles_kept": stacked_shape0,
        "epoch_duration_s": EPOCH_DURATION_S,
        "n_channels": mean_tfr.shape[0],
        "output": str(out_path),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  "
          f"shape={mean_tfr.shape}  kept={n_kept}/{n_cycles}  "
          f"({rejection_status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
