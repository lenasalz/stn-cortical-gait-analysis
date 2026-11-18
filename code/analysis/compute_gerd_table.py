"""Per-subject extraction of gait-event-related ERD/ERS values for LMM analysis.

Inputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-cont-basenorm.h5  (compute_tfr_basenorm)
  - data/derivatives/raw_clean/<subj>_<block>-{eeg,dbs-proc}-raw.fif    (build_raw_from_pickle)
  - $RESIDUAL_DATA_ROOT/Final_filtered/rejection_mask_dico.pkl
  - code/analysis/config.yaml (freq_bands, channels_of_interest)
  - code/analysis/variables.py (gait_phases, event_percentages, gait sequence)
Outputs:
  - data/derivatives/gerd/<subj>_<block>-<modality>-gerd.csv
      Long-format rows; one row per (cycle, area, hemisphere, gait_phase):
        SubjectID, Cohort, Group(PD/HC), Modality, Block,
        Area (motor|frontal|STN), Hemisphere (Left|Right|Central|Both),
        GaitPhase (IDS|LLS|FDS|RLS), Epoch (cycle index),
        ERD_ERS_Theta, ERD_ERS_Alpha, ERD_ERS_Low_Beta,
        ERD_ERS_High_Beta, ERD_ERS_Gamma
      Each ERD_ERS_<band> is the MEDIAN dB power within
      (band freq window, phase time window) for that cycle.
Description:
  Replaces the in-memory ERDS extraction in
  ngb_study/notebooks/final_analysis/scripts/ERDS_exporter.py.
  Per cycle, re-runs the same warp+resample chain that produced the
  cycle-averaged `*-tfr-gait.h5`, but does NOT average across cycles;
  instead, integrates each cycle independently into a single row per
  (area, hemisphere, gait_phase). The resulting long-format CSV is the
  input that `code/analysis/R/stats_lmm_*.R` expects.

  Aggregation method: median (matches the legacy filename suffix
  `_phase_median_<datatype>.csv`).

Usage:
  python compute_gerd_table.py --subject P4-2002 --block pre8walk --modality eeg
  python compute_gerd_table.py --subject P4-2002 --block pre8walk --modality dbs
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
from tfr import resample_tfr_cycles, warp_tfr_cycles
from variables import (
    event_percentages as DEFAULT_EVENT_PERCENTAGES,
    gait_events_correct_sequence as DEFAULT_GAIT_SEQ,
    gait_phases as DEFAULT_GAIT_PHASES,
)

EPOCH_DURATION_S = 1.1
COHORT_TO_GROUP = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}


def _band_col(name: str) -> str:
    return f"ERD_ERS_{name.replace(' ', '_')}"


def _per_cycle_per_channel_tfrs(tfr, raw, gait_seq, ev_pct):
    """Return (cycle_dicts, per_cycle_data_4d) — (n_kept_cycles, n_chans, n_freqs, n_times_fixed)."""
    from _common import patch_legacy_annotations
    patch_legacy_annotations(raw)

    fe = filter_gait_events(raw, gait_seq)
    raw_filt = replace_events_in_mne(raw, fe, gait_seq)
    gait_cycles = segment_gait_cycles(raw_filt, gait_seq)

    sfreq = tfr.info["sfreq"]
    cycle_slices: list[tuple] = []
    for c in gait_cycles:
        if "HSrightlocs" not in c or "HSrightlocs_end" not in c:
            continue
        try:
            s = tfr.time_as_index(c["HSrightlocs"], use_rounding=True)[0]
            e = tfr.time_as_index(c["HSrightlocs_end"], use_rounding=True)[0]
        except IndexError:
            continue
        if e > s:
            cycle_slices.append((c, s, e))
    if not cycle_slices:
        return [], np.zeros((0, 0, 0, 0))

    n_chans, n_freqs = tfr.data.shape[0], tfr.data.shape[1]
    n_times_fixed = int(round(EPOCH_DURATION_S * sfreq))
    out = np.empty((len(cycle_slices), n_chans, n_freqs, n_times_fixed),
                   dtype=tfr.data.dtype)
    cycle_dicts = [c for c, _, _ in cycle_slices]
    for ch_i in range(n_chans):
        per_cycle = [tfr.data[ch_i:ch_i + 1, :, s:e] for _, s, e in cycle_slices]
        warped = warp_tfr_cycles(per_cycle, cycle_dicts, ev_pct, sfreq)
        resampled = resample_tfr_cycles(warped, EPOCH_DURATION_S, sfreq)
        stacked = np.concatenate(resampled, axis=0)        # (n_cycles, n_freqs, T)
        out[:, ch_i, :, :] = stacked
    return cycle_dicts, out


def _region_channel_indices(ch_names: list[str], region_chans: list[str]
                            ) -> list[int]:
    return [ch_names.index(c) for c in region_chans if c in ch_names]


def _phase_time_slice(phase_range_pct: tuple[float, float], n_times: int) -> slice:
    """Inclusive-endpoint phase window — matches the final_analysis
    `ERDSDataExporter._get_time_indices` (v13) so per-cycle medians are
    bit-identical to the manuscript-canonical export. Each phase has one
    extra sample at its upper boundary, and adjacent phases overlap by
    one sample at the boundaries (e.g. IDS = idx 0..22, LLS = idx 22..110)."""
    lo, hi = phase_range_pct
    idx_start = max(0, int(lo * n_times))
    idx_end = min(n_times - 1, int(hi * n_times))
    if idx_end < idx_start:
        idx_end = idx_start
    return slice(idx_start, idx_end + 1)


def _band_freq_slice(band_range_hz: tuple[float, float], freqs: np.ndarray) -> slice:
    lo, hi = band_range_hz
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _emit_rows_eeg(per_cycle, kept_bool, freqs, ch_names, freq_bands,
                   gait_phases, config, subject, cohort, block):
    """For EEG: one row per (cycle_kept, area ∈ {motor,frontal}, side ∈ Sides, phase)."""
    rows = []
    n_times = per_cycle.shape[-1]
    band_slices = {b: _band_freq_slice(tuple(r), freqs) for b, r in freq_bands.items()}
    phase_slices = {p: _phase_time_slice(r, n_times) for p, r in gait_phases.items()}
    coi = config["connectivity"]["channels_of_interest"]
    sides = ("Left", "Right", "Central", "Both")
    group = COHORT_TO_GROUP[cohort]

    for area in ("motor", "frontal"):
        for side in sides:
            if side == "Both":
                target_chs = list(dict.fromkeys(
                    coi[area].get("Left", []) + coi[area].get("Right", [])
                    + coi[area].get("Central", [])
                ))
            else:
                target_chs = list(coi[area].get(side, []))
            ch_idx = _region_channel_indices(ch_names, target_chs)
            if not ch_idx:
                continue
            # (n_cycles, n_ch_in_area, n_freqs, n_times)
            slab = per_cycle[:, ch_idx, :, :]
            # Average across the channels in this area.side first.
            slab = slab.mean(axis=1)                       # (n_cycles, n_freqs, n_times)
            for cycle_i, keep in enumerate(kept_bool):
                if not keep:
                    continue
                for phase, t_sl in phase_slices.items():
                    row = {
                        "SubjectID": subject,
                        "Cohort": cohort,
                        "Group": group,
                        "Modality": "eeg",
                        "Block": block,
                        "Area": area,
                        "Hemisphere": side,
                        "GaitPhase": phase,
                        "Epoch": cycle_i,
                    }
                    for band, f_sl in band_slices.items():
                        window = slab[cycle_i, f_sl, t_sl]
                        row[_band_col(band)] = (
                            float(np.median(window)) if window.size else np.nan
                        )
                    rows.append(row)
    return rows


def _emit_rows_dbs(per_cycle, kept_bool, freqs, freq_bands, gait_phases,
                   subject, cohort, block, hemisphere):
    """For DBS: one row per (cycle_kept, phase). Single channel = STN, one hemisphere per subject."""
    rows = []
    n_times = per_cycle.shape[-1]
    band_slices = {b: _band_freq_slice(tuple(r), freqs) for b, r in freq_bands.items()}
    phase_slices = {p: _phase_time_slice(r, n_times) for p, r in gait_phases.items()}
    group = COHORT_TO_GROUP[cohort]
    slab = per_cycle[:, 0, :, :]                          # (n_cycles, n_freqs, n_times)
    for cycle_i, keep in enumerate(kept_bool):
        if not keep:
            continue
        for phase, t_sl in phase_slices.items():
            row = {
                "SubjectID": subject,
                "Cohort": cohort,
                "Group": group,
                "Modality": "dbs",
                "Block": block,
                "Area": "STN",
                "Hemisphere": hemisphere,
                "GaitPhase": phase,
                "Epoch": cycle_i,
            }
            for band, f_sl in band_slices.items():
                window = slab[cycle_i, f_sl, t_sl]
                row[_band_col(band)] = (
                    float(np.median(window)) if window.size else np.nan
                )
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--norm", choices=["basenorm", "avggait"], default="basenorm",
                        help="basenorm = 10*log10(task/baseline_avg) per cycle (default); "
                             "avggait = basenorm minus the subject's cycle-averaged "
                             "time-mean per (channel, freq).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.block == "baseline":
        parser.error("baseline has no gait events")

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    out_dir = paths.derivatives_root / "gerd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{block}-{args.modality}-{args.norm}-gerd.csv"
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0

    tfr_in = output_path(paths.tfr_dir, subject, block,
                         f"-{args.modality}-tfr-cont-basenorm.h5")
    raw_suffix = "-eeg-raw.fif" if args.modality == "eeg" else "-dbs-proc-raw.fif"
    raw_in = output_path(paths.raw_clean_dir, subject, block, raw_suffix)
    for p, name in ((tfr_in, "basenorm TFR"), (raw_in, "raw")):
        if not p.exists():
            print(f"missing {name}: {p}", file=sys.stderr)
            return 2

    meta = setup_run("compute_gerd_table", config)

    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        tfr = mne.time_frequency.read_tfrs(tfr_in)
    tfr = tfr[0] if isinstance(tfr, list) else tfr
    # Legacy basenorm TFRs (pre-2026-05-08) may contain ±Inf cells from
    # log10(0/baseline); newer ones write NaN there directly. Force the
    # NaN convention here so downstream nan-aware reductions behave
    # consistently regardless of which version produced the cache.
    tfr.data[~np.isfinite(tfr.data)] = np.nan
    raw = mne.io.read_raw_fif(raw_in, preload=False, verbose="WARNING")

    cycle_dicts, per_cycle = _per_cycle_per_channel_tfrs(
        tfr, raw, DEFAULT_GAIT_SEQ, DEFAULT_EVENT_PERCENTAGES
    )
    if per_cycle.shape[0] == 0:
        print(f"no usable cycles for {subject} {block}", file=sys.stderr)
        return 3
    n_cycles = per_cycle.shape[0]

    # Prefer per-subject npz mask from compute_autoreject_mask.py; fall
    # back to the legacy rejection_mask_dico.pkl cache.
    mask_npz = paths.derivatives_root / "autoreject" / f"{subject}_{block}-mask.npz"
    mask_pkl = paths.data_root / "Final_filtered" / "rejection_mask_dico.pkl"
    mask = None
    mask_source = "missing"
    if mask_npz.exists():
        mask = np.load(mask_npz)["keep_mask"]
        mask_source = "per_subject_npz"
    elif mask_pkl.exists():
        with mask_pkl.open("rb") as f:
            masks = pickle.load(f)
        mask = masks.get((subject, block))
        if mask is not None:
            mask_source = "legacy_pkl"
    if mask is None or len(mask) != n_cycles:
        kept_bool = np.ones(n_cycles, dtype=bool)
        rejection_status = ("no_mask_found" if mask is None
                            else f"length_mismatch (source={mask_source})")
    else:
        kept_bool = mask.astype(bool)
        rejection_status = f"applied (source={mask_source})"

    if args.norm == "avggait":
        # Match Aline's two-step reduction: nanmean over cycles (line 908 in
        # final_analysis/scripts/tfr.py) yields the per-subject averaged TFR,
        # then regular mean over time gives the (channel, freq) reference
        # that gets subtracted from every cycle. Using a single
        # `.mean(axis=(0,3))` here would propagate any basenorm NaN cell to
        # the entire (channel, freq) plane for every cycle/time.
        kept_slab = per_cycle[kept_bool]                       # (n_kept, C, F, T)
        avg_tfr = np.nanmean(kept_slab, axis=0)                # (C, F, T)
        subject_mean = avg_tfr.mean(axis=-1, keepdims=True)[np.newaxis, ...]  # (1, C, F, 1)
        per_cycle = per_cycle - subject_mean

    freq_bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    if args.modality == "eeg":
        rows = _emit_rows_eeg(per_cycle, kept_bool, tfr.freqs, list(tfr.ch_names),
                              freq_bands, DEFAULT_GAIT_PHASES, config,
                              subject, cohort, block)
    else:
        # Hemisphere = implant side from raw.info description.
        desc = (raw.info.get("description") or "").lower()
        hemi = "Left" if "left" in desc else "Right" if "right" in desc else "Unknown"
        rows = _emit_rows_dbs(per_cycle, kept_bool, tfr.freqs,
                              freq_bands, DEFAULT_GAIT_PHASES,
                              subject, cohort, block, hemi)

    df = pd.DataFrame(rows)
    with atomic_write(out_path) as tmp:
        df.to_csv(tmp, index=False)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "modality": args.modality, "norm": args.norm,
        "output": str(out_path),
        "n_cycles_total": int(n_cycles),
        "n_cycles_kept": int(kept_bool.sum()),
        "rejection_status": rejection_status,
        "n_rows": int(len(df)),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  n_rows={len(df)}  kept={int(kept_bool.sum())}/{n_cycles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
