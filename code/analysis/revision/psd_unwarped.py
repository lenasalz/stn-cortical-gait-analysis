"""Walking and rest power spectra from UNWARPED data (revision, R1.2/R1.3/R1.11).

Purpose:
  The submitted walking PSDs were computed on gait cycles that had been
  piecewise-linearly time-warped and resampled to 1.1 s
  (build_epochs_from_raw.py -> compute_fooof.py), which rescales every
  oscillation frequency by 1.1 s / stride time. This script recomputes the
  walking (pre8walk) and rest (baseline) spectra from the unwarped clean
  continuous data, applies the same AutoReject masks, and fits the same
  FOOOF model so that the submitted and unwarped spectra differ only in
  the warping/epoching step.

  Two spectrum variants are written per ROI:
    A  "continuous straight-walking": consecutive kept gait cycles are
       merged into continuous runs (a run breaks at a rejected cycle or at
       a gap between cycles, i.e. a turn); Welch with 2 s Hann windows and
       50 % overlap over each run; all windows pooled across runs (mean,
       i.e. runs are weighted by their number of windows). Rest: the same
       on runs of consecutive kept 1.1 s epochs.
    B  "per-cycle unwarped": Welch on every kept cycle at its native
       length (single Hann window of the cycle length, zero-padded to the
       2 s FFT grid), median across cycles (config psd.average). Rest:
       the same on the kept 1.1 s epochs.
  Spectra are restricted to the FOOOF fit range (config fooof.freq_range,
  4-45 Hz), channels in raw.info['bads'] are excluded, channels are averaged
  within ROI (motor, frontal; side "Both") before the fit, as in
  compute_fooof.py. DBS uses the single bipolar channel (region STN).

Inputs:
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif
  - data/derivatives/raw_clean/<subj>_<block>-dbs-proc-raw.fif (Patient_NFB)
  - data/derivatives/autoreject/<subj>_<block>-mask.npz  (keep_mask)
  - code/analysis/config.yaml (cohorts, blocks, freq_bands, fooof,
    connectivity.channels_of_interest, psd.average, repro.seed)
  - code/analysis/{epoching,variables,compute_fooof,_common}.py (imported)
Outputs:
  - data/derivatives/psd_unwarped/<subj>_<block>-<modality>-psd.csv
      long table: subject, cohort, block, modality, spectrum_variant,
      region, side, n_channels, freq, power  (+ .meta.json)
  - data/derivatives/psd_unwarped/<subj>_<block>-<modality>-segments.csv
      one row: cycle/run/stride-time summary  (+ .meta.json)
  - data/derivatives/psd_unwarped/<subj>_<block>-<modality>-fooof.csv
      FOOOF fit with the config parameters, one row per variant x ROI
      (same columns as compute_fooof.py)  (+ .meta.json)
  - with --all additionally:
    outputs/stats/revision_2026-10/psd_unwarped_segments_2026-10-13.csv
      concatenation of all *-segments.csv  (+ .meta.json)
Author: Lena
Date: 2026-10-13
Seed: config.yaml:repro.seed (42) via _common.setup_run; the computation
  is deterministic, the seed is set for completeness.

Usage:
  python psd_unwarped.py --subject P4-2005 --block pre8walk --modality eeg
  python psd_unwarped.py --all [--force]
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _common import (  # noqa: E402
    PROJECT_ROOT,
    atomic_write,
    cohort_of,
    load_config,
    output_path,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    validate_subject,
    write_sidecar,
)
from compute_fooof import (  # noqa: E402
    _assign_peaks_to_bands,
    _fit_fooof_one,
    _region_channels,
)
from epoching import (  # noqa: E402
    filter_gait_events,
    replace_events_in_mne,
    segment_gait_cycles,
)
from variables import gait_events_correct_sequence as GAIT_SEQ  # noqa: E402

# Output date tag; override with the environment variable REVISION_DATE for a rerun.
DATE = __import__("os").environ.get("REVISION_DATE", "2026-10-13")
WINDOW_S = 2.0            # Welch window for variant A (seconds)
REST_EPOCH_S = 1.1        # fixed-length rest epochs, as build_epochs_from_raw.py
STAGE = "psd_unwarped"
OUT_DIR = PROJECT_ROOT / "data" / "derivatives" / "psd_unwarped"
STATS_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
REGIONS = ("motor", "frontal")
SIDE = "Both"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _roi_channel_map(config: dict, modality: str, ch_names: list[str],
                     bads: list[str]) -> dict[str, list[str]]:
    """ROI -> channels present and not bad (compute_fooof._psd_for_eeg_region)."""
    if modality == "dbs":
        return {"STN": [c for c in ch_names if c not in bads]}
    out = {}
    for region in REGIONS:
        chans = _region_channels(config, region, SIDE)
        out[region] = [c for c in chans if c in ch_names and c not in bads]
    return out


def _load_raw_roi(path: Path, roi_map: dict[str, list[str]]):
    """Read the Raw with only the ROI channels loaded into memory."""
    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        raw = mne.io.read_raw_fif(path, preload=False, verbose="WARNING")
    picks = list(dict.fromkeys(c for chans in roi_map.values() for c in chans))
    raw.pick(picks, verbose="WARNING")
    raw.load_data(verbose="WARNING")
    patch_legacy_annotations(raw)
    return raw


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)["keep_mask"].astype(bool)


# --------------------------------------------------------------------------
# Segmentation into cycles / epochs and rejection-mask alignment
# --------------------------------------------------------------------------
def _walking_cycles(raw) -> list[dict]:
    """Gait cycles exactly as build_epochs_from_raw._build_task_epochs
    (filter_gait_events -> replace_events_in_mne -> segment_gait_cycles)."""
    filt = replace_events_in_mne(raw, filter_gait_events(raw, GAIT_SEQ), GAIT_SEQ)
    cycles = segment_gait_cycles(filt, GAIT_SEQ) or []
    end_key = GAIT_SEQ[0] + "_end"
    # warp_dbs_or_eeg_data (submitted pipeline) requires HSrightlocs_end on
    # every cycle; segment_gait_cycles only appends cycles that have it.
    missing = [i for i, c in enumerate(cycles) if end_key not in c]
    if missing:
        raise RuntimeError(f"cycles without {end_key}: {missing}")
    tmax = raw.times[-1]
    out = []
    for c in cycles:
        start, end = float(c[GAIT_SEQ[0]]), float(c[end_key])
        # warp_dbs_or_eeg_data clips the cycle end at the last sample.
        end = min(end, tmax)
        out.append({"start": start, "end": end})
    return out


def _rest_epochs(raw) -> list[dict]:
    """Consecutive fixed-length epochs as build_epochs_from_raw._build_baseline_epochs."""
    import mne
    events = mne.make_fixed_length_events(raw, duration=REST_EPOCH_S)
    ep = mne.Epochs(raw, events, tmin=0, tmax=REST_EPOCH_S, baseline=None,
                    preload=False, verbose="WARNING")
    ep.drop_bad(verbose="WARNING")
    kept_events = ep.events
    n_samp = len(ep.times)
    sfreq = raw.info["sfreq"]
    out = []
    for ev in kept_events:
        s0 = int(ev[0] - raw.first_samp)
        out.append({"start": s0 / sfreq, "end": (s0 + n_samp - 1) / sfreq,
                    "start_samp": s0, "n_samp": n_samp})
    return out


def _align_mask(segments: list[dict], mask: np.ndarray | None
                ) -> tuple[list[dict], np.ndarray, str]:
    """Replicates compute_tfr_gait.py mask handling.

    - mask shorter than the segment list -> paired-epoch truncation
      (build_epochs_from_raw keeps the first min(n_eeg, n_dbs) cycles/epochs),
      keep the first len(mask) segments;
    - same length -> applied;
    - longer -> length mismatch, no rejection (flagged);
    - no mask -> no rejection (flagged).
    """
    n = len(segments)
    if mask is None:
        return segments, np.ones(n, dtype=bool), "no_mask_found"
    if len(mask) < n:
        return (segments[: len(mask)], mask.copy(),
                f"applied_truncated (mask={len(mask)} segments={n})")
    if len(mask) > n:
        return segments, np.ones(n, dtype=bool), \
            f"length_mismatch (mask={len(mask)} segments={n})"
    return segments, mask.copy(), "applied"


def _runs(segments: list[dict], keep: np.ndarray, sfreq: float,
          tol_s: float | None = None) -> list[dict]:
    """Merge consecutive kept segments that are contiguous in time into runs.

    Walking: cycle i+1 is contiguous if its start equals the end of cycle i
    (both come from the same heel-strike annotation). Rest: epochs of
    1.1 s start every 1.1 s and are contiguous by construction.
    """
    if tol_s is None:
        tol_s = 1.5 / sfreq
    runs: list[dict] = []
    cur = None
    for i, seg in enumerate(segments):
        if not keep[i]:
            cur = None
            continue
        if cur is not None and abs(seg["start"] - cur["end"]) <= tol_s:
            cur["end"] = seg["end"]
            cur["n_segments"] += 1
            cur["idx"].append(i)
        else:
            cur = {"start": seg["start"], "end": seg["end"],
                   "n_segments": 1, "idx": [i]}
            runs.append(cur)
    return runs


# --------------------------------------------------------------------------
# Welch
# --------------------------------------------------------------------------
def _welch_segments(x: np.ndarray, sfreq: float, fmin: float, fmax: float,
                    n_fft: int, n_per_seg: int, n_overlap: int):
    """Hann-window Welch; returns (freqs, psd[n_ch, n_freq, n_windows])."""
    from mne.time_frequency import psd_array_welch
    psd, freqs = psd_array_welch(
        x, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
        n_overlap=n_overlap, window="hann", average=None, remove_dc=True,
        n_jobs=1, verbose="WARNING",
    )
    return freqs, psd


def _variant_a(raw, runs: list[dict], fmin: float, fmax: float):
    """Pool all 2 s / 50 % windows over every run that is >= 2 s long."""
    sfreq = raw.info["sfreq"]
    n_win = int(round(WINDOW_S * sfreq))
    pooled = []
    freqs = None
    n_runs_used = 0
    for run in runs:
        s0 = int(round(run["start"] * sfreq))
        s1 = int(round(run["end"] * sfreq)) + 1
        x = raw._data[:, s0:s1]
        if x.shape[-1] < n_win:
            continue
        freqs, psd = _welch_segments(x, sfreq, fmin, fmax, n_win, n_win, n_win // 2)
        pooled.append(psd)
        n_runs_used += 1
    if not pooled:
        return None, None, 0, 0
    allw = np.concatenate(pooled, axis=-1)             # (n_ch, n_freq, n_windows)
    return freqs, allw.mean(axis=-1), n_runs_used, int(allw.shape[-1])


def _variant_b(raw, segments: list[dict], keep: np.ndarray,
               fmin: float, fmax: float, average: str):
    """One Hann window per kept segment at its native length, zero-padded to
    the 2 s FFT grid; `average` ('median'/'mean') across segments."""
    sfreq = raw.info["sfreq"]
    n_win = int(round(WINDOW_S * sfreq))
    per = []
    freqs = None
    for i, seg in enumerate(segments):
        if not keep[i]:
            continue
        if "start_samp" in seg:
            s0, s1 = seg["start_samp"], seg["start_samp"] + seg["n_samp"]
        else:
            s0 = int(round(seg["start"] * sfreq))
            s1 = int(round(seg["end"] * sfreq)) + 1
        x = raw._data[:, s0:s1]
        n = x.shape[-1]
        if n < 2 or n > n_win:
            raise RuntimeError(f"segment length {n} outside (2, {n_win}]")
        freqs, psd = _welch_segments(x, sfreq, fmin, fmax, n_win, n, 0)
        per.append(psd[..., 0])
    if not per:
        return None, None, 0
    stack = np.stack(per, axis=0)                      # (n_seg, n_ch, n_freq)
    agg = np.median(stack, axis=0) if average == "median" else stack.mean(axis=0)
    return freqs, agg, int(stack.shape[0])


# --------------------------------------------------------------------------
# Per-(subject, block, modality) driver
# --------------------------------------------------------------------------
def process_one(subject: str, block: str, modality: str, config: dict,
                paths, force: bool) -> int:
    cohort = cohort_of(subject, config)
    if modality == "dbs" and cohort != "Patient_NFB":
        return 0
    suffix = "-eeg-raw.fif" if modality == "eeg" else "-dbs-proc-raw.fif"
    raw_in = output_path(paths.raw_clean_dir, subject, block, suffix)
    mask_in = paths.derivatives_root / "autoreject" / f"{subject}_{block}-mask.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    psd_out = output_path(OUT_DIR, subject, block, f"-{modality}-psd.csv")
    seg_out = output_path(OUT_DIR, subject, block, f"-{modality}-segments.csv")
    fooof_out = output_path(OUT_DIR, subject, block, f"-{modality}-fooof.csv")
    if should_skip(psd_out, force) and should_skip(fooof_out, force):
        print(f"skip {subject} {block} {modality} (exists)")
        return 0
    if not raw_in.exists():
        print(f"missing {raw_in}", file=sys.stderr)
        return 2

    meta = setup_run(STAGE, config)
    fcfg = config["fooof"]
    fmin, fmax = float(fcfg["freq_range"][0]), float(fcfg["freq_range"][1])
    freq_bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    average_b = str(config["psd"].get("average", "median"))

    import mne
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        hdr = mne.io.read_raw_fif(raw_in, preload=False, verbose="WARNING")
    roi_map = _roi_channel_map(config, modality, hdr.ch_names, hdr.info["bads"])
    roi_map = {k: v for k, v in roi_map.items() if v}
    desc = (hdr.info.get("description") or "")
    del hdr
    if not roi_map:
        print(f"no ROI channels for {subject} {block} {modality}", file=sys.stderr)
        return 3
    raw = _load_raw_roi(raw_in, roi_map)
    sfreq = float(raw.info["sfreq"])

    if block == "baseline":
        segments = _rest_epochs(raw)
        seg_kind = "rest_epoch_1.1s"
    else:
        segments = _walking_cycles(raw)
        seg_kind = "gait_cycle"
    n_segmented = len(segments)
    mask = _load_mask(mask_in)
    segments, keep, status = _align_mask(segments, mask)
    n_kept = int(keep.sum())
    if n_kept == 0:
        print(f"all segments rejected for {subject} {block} {modality}",
              file=sys.stderr)
        return 5
    runs = _runs(segments, keep, sfreq)
    durations_all = np.array([s["end"] - s["start"] for s in segments])
    durations_kept = durations_all[keep]

    freqs_a, psd_a, n_runs_used, n_windows = _variant_a(raw, runs, fmin, fmax)
    freqs_b, psd_b, n_b = _variant_b(raw, segments, keep, fmin, fmax, average_b)
    if psd_a is not None and psd_b is not None:
        assert np.allclose(freqs_a, freqs_b)
    freqs = freqs_b if freqs_a is None else freqs_a

    # ROI averages (mean across channels) + FOOOF with config parameters
    ch_index = {c: i for i, c in enumerate(raw.ch_names)}
    psd_rows, fooof_rows = [], []
    for variant, psd, n_ep in (("A", psd_a, n_windows), ("B", psd_b, n_b)):
        if psd is None:
            continue
        for region, chans in roi_map.items():
            idx = [ch_index[c] for c in chans]
            roi_psd = psd[idx].mean(axis=0)
            side = SIDE if modality == "eeg" else (
                "Left" if "left" in desc.lower()
                else "Right" if "right" in desc.lower() else "Unknown")
            for f, p in zip(freqs, roi_psd):
                psd_rows.append({
                    "subject": subject, "cohort": cohort, "block": block,
                    "modality": modality, "spectrum_variant": variant,
                    "region": region, "side": side, "n_channels": len(chans),
                    "freq": float(f), "power": float(p),
                })
            ap, peaks = _fit_fooof_one(freqs, roi_psd, fcfg)
            band_peaks = _assign_peaks_to_bands(peaks, freq_bands)
            row = {"subject": subject, "cohort": cohort, "block": block,
                   "modality": modality, "spectrum_variant": variant,
                   "region": region, "side": side,
                   "n_channels": len(chans), "n_epochs": n_ep, **ap}
            for band, vals in band_peaks.items():
                row[f"{band}_peak_present"] = vals["present"]
                row[f"{band}_cf"] = vals["cf"]
                row[f"{band}_pw"] = vals["pw"]
                row[f"{band}_bw"] = vals["bw"]
            fooof_rows.append(row)

    seg_row = {
        "subject": subject, "cohort": cohort, "block": block,
        "modality": modality, "segment_kind": seg_kind, "sfreq": sfreq,
        "n_segmented": n_segmented, "n_after_truncation": len(segments),
        "n_kept": n_kept, "rejection_status": status,
        "n_runs": len(runs),
        "n_runs_ge_2s": n_runs_used, "n_windows_A": n_windows,
        "n_segments_B": n_b,
        "run_len_segments_mean": float(np.mean([r["n_segments"] for r in runs])),
        "run_len_segments_max": int(max(r["n_segments"] for r in runs)),
        "total_kept_s": float(durations_kept.sum()),
        "total_runs_ge_2s_s": float(sum(
            r["end"] - r["start"] for r in runs
            if (r["end"] - r["start"]) * sfreq >= round(WINDOW_S * sfreq))),
        "duration_mean_s": float(durations_kept.mean()),
        "duration_sd_s": float(durations_kept.std(ddof=1)) if n_kept > 1 else np.nan,
        "duration_min_s": float(durations_kept.min()),
        "duration_max_s": float(durations_kept.max()),
        "duration_all_mean_s": float(durations_all.mean()),
        # Factor by which the submitted warping (resample every cycle to
        # 1.1 s) multiplied the frequencies of this subject's cycles:
        # f_warped = f_true * stride_time / 1.1 s (mean over kept cycles).
        "warp_freq_factor_mean": float(durations_kept.mean() / REST_EPOCH_S),
        "roi_channels": ";".join(f"{k}={','.join(v)}" for k, v in roi_map.items()),
    }

    psd_df = pd.DataFrame(psd_rows)
    with atomic_write(psd_out) as tmp:
        psd_df.to_csv(tmp, index=False)
    seg_df = pd.DataFrame([seg_row])
    with atomic_write(seg_out) as tmp:
        seg_df.to_csv(tmp, index=False)
    fooof_df = pd.DataFrame(fooof_rows)
    with atomic_write(fooof_out) as tmp:
        fooof_df.to_csv(tmp, index=False)

    meta.update({
        "author": "Lena", "date": DATE,
        "subject": subject, "cohort": cohort, "block": block,
        "modality": modality, "raw_input": str(raw_in),
        "mask_input": str(mask_in) if mask is not None else None,
        "window_s": WINDOW_S, "window": "hann", "overlap": 0.5,
        "n_fft": int(round(WINDOW_S * sfreq)), "fmin": fmin, "fmax": fmax,
        "variant_B_average": average_b, "fooof_params": dict(fcfg),
        "outputs": [str(psd_out), str(seg_out), str(fooof_out)],
        **{k: v for k, v in seg_row.items() if k not in ("subject", "cohort", "block", "modality")},
    })
    for p in (psd_out, seg_out, fooof_out):
        write_sidecar(p, meta)
    print(f"wrote {psd_out.name}: kept={n_kept}/{len(segments)} "
          f"runs={len(runs)} (>=2 s: {n_runs_used}, windows={n_windows}) "
          f"dur={seg_row['duration_mean_s']:.3f}+-{seg_row['duration_sd_s']:.3f} s "
          f"[{status}]")
    del raw
    gc.collect()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject")
    parser.add_argument("--block")
    parser.add_argument("--modality", choices=["eeg", "dbs"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)

    if args.all:
        rc = 0
        for cohort, members in config["cohorts"].items():
            for subject in members:
                for block in config["blocks"]:
                    for modality in ("eeg", "dbs"):
                        if modality == "dbs" and cohort != "Patient_NFB":
                            continue
                        r = process_one(subject, block, modality, config,
                                        paths, args.force)
                        rc = rc or r
        # aggregate segment summaries
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        segs = sorted(OUT_DIR.glob("*-segments.csv"))
        agg = pd.concat([pd.read_csv(p) for p in segs], ignore_index=True)
        out = STATS_DIR / f"psd_unwarped_segments_{DATE}.csv"
        with atomic_write(out) as tmp:
            agg.to_csv(tmp, index=False)
        meta = setup_run(STAGE + "_aggregate", config)
        meta.update({"author": "Lena", "date": DATE,
                     "inputs": [str(p) for p in segs], "output": str(out),
                     "n_rows": int(len(agg))})
        write_sidecar(out, meta)
        print(f"wrote {out}  rows={len(agg)}")
        return rc

    if not (args.subject and args.block and args.modality):
        parser.error("--subject, --block and --modality are required without --all")
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    return process_one(subject, block, args.modality, config, paths, args.force)


if __name__ == "__main__":
    sys.exit(main())
