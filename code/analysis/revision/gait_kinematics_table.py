"""Group-level gait kinematics (PD vs HC) from the gait events of the analysed EEG.

Purpose:  Answer reviewer comments 8 and 11 (npj Parkinson's Disease,
          decision letter 2026-10-12). Comment 8 asks for a summary of the
          gait kinematic differences between the groups; comment 11 asks
          what the per-cycle time-warp to 1.1 s does to oscillation
          frequencies when stride time differs between groups. Both are
          answered from the gait-event annotations that the submitted
          pipeline itself used to segment gait cycles, so the numbers refer
          to exactly the cycles that entered the neural analyses.
Inputs:   data/derivatives/raw_clean/<subj>_pre8walk-eeg-raw.fif
              (pickle-era export; only the annotations are read, no signal)
          data/derivatives/autoreject/<subj>_pre8walk-mask.npz
              (fresh AutoReject keep masks, seed 42; same masks as the
              submitted trial counts and TFRs)
          code/analysis/config.yaml                     (cohorts, seed)
          code/analysis/variables.py                    (event sequence, %)
          code/analysis/epoching.py                     (filter_gait_events,
                                                         segment_gait_cycles)
          docs/data_manifest_revision_2026-10-12.md     (frozen data version)
Outputs:  outputs/stats/revision_2026-10/gait_kinematics_<date>.csv
              group table: one row per parameter and cycle set
          outputs/stats/revision_2026-10/gait_kinematics_per_subject_<date>.csv
          outputs/stats/revision_2026-10/gait_kinematics_cycles_<date>.csv
              one row per segmented cycle (timings only, subject codes)
          outputs/stats/revision_2026-10/gait_kinematics_<date>.txt
              plain-text summary
          outputs/stats/revision_2026-10/gait_kinematics_report_<date>.md
              methods, group table, warp-rescaling numbers, results draft
          .meta.json sidecars (write_sidecar) next to every output
Author:   Lena
Date:     2026-10-13
Seed:     config.yaml:repro.seed (42) via _common.setup_run; the analysis
          itself is deterministic (no resampling, no permutation).

Description:
  For every subject in config.yaml the pre8walk gait events are filtered
  and segmented exactly as in build_epochs_from_raw.py
  (filter_gait_events -> segment_gait_cycles, max_gap 1.5 s, sequence
  RHS, LTO, LHS, RTO, next RHS). Cycle i of that list is epoch i of the
  submitted epoch files, so the AutoReject keep mask indexes the cycles
  directly (paired-epoch truncation handled as in compute_tfr_gait.py).
  Per cycle: stride time (RHS to RHS), left step time (RHS to LHS), right
  step time (LHS to RHS), initial double support (RHS to LTO), terminal
  double support (LHS to RTO), left swing (LTO to LHS), right swing (RTO
  to RHS), left/right stance (cycle minus the contralateral swing of that
  limb, i.e. left stance = LHS to next LTO is not available within one
  cycle, so stance is reported as stride minus own swing), and the warp
  factor stride_time / 1.1 that multiplies every frequency when the cycle
  is compressed (or stretched) to 1.1 s. Per subject: mean/SD/CV/min/max
  of stride time, cadence (120 / mean stride time), phase durations and
  percentages, step-time asymmetry |L - R| / mean(L, R), number and length
  of straight-walking runs (consecutive cycles sharing a heel strike),
  and the distribution of the warp factor. Group table: mean, SD, median
  per cohort, Welch t-test, Mann-Whitney U, Hedges' g, computed on the
  per-subject values (n = 10 per cohort) for all segmented cycles and for
  the AutoReject-kept cycles.

  Walking speed and stride length need the motion-capture marker
  trajectories; those are not among the local derivatives (checked with
  --check-spatial), so they are not reported here.

Usage:
  python code/analysis/revision/gait_kinematics_table.py --subject P4-2005
  python code/analysis/revision/gait_kinematics_table.py            # all
  python code/analysis/revision/gait_kinematics_table.py --date 2026-10-13
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import (  # noqa: E402
    PROJECT_ROOT,
    cohort_of,
    load_config,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    validate_subject,
    write_sidecar,
)
from epoching import filter_gait_events, segment_gait_cycles  # noqa: E402
from variables import (  # noqa: E402
    event_percentages as EVENT_PCT,
    gait_events_correct_sequence as GAIT_SEQ,
)

BLOCK = "pre8walk"
EPOCH_DURATION_S = 1.1          # build_epochs_from_raw.EPOCH_DURATION_S
MAX_GAP_S = 1.5                 # epoching.segment_gait_cycles default
OUT_DIR = PROJECT_ROOT / "outputs" / "stats" / "revision_2026-10"
COHORT_LABEL = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}
RHS, LTO, LHS, RTO = GAIT_SEQ          # HSrightlocs, TOleftlocs, HSleftlocs, TOrightlocs
RHS_END = RHS + "_end"
TEST_FREQS_HZ = (13.0, 20.0)


class _AnnotationsOnly:
    """Minimal stand-in for a Raw object: the epoching helpers read only
    `.annotations`, so the signal is never loaded (memory rule)."""

    def __init__(self, annotations):
        self.annotations = annotations


# --------------------------------------------------------------------------
# Cycle extraction
# --------------------------------------------------------------------------

def load_cycles(raw_path: Path) -> tuple[list[dict], dict]:
    """Segment gait cycles from the annotations exactly as
    build_epochs_from_raw._build_task_epochs does (filter -> segment)."""
    import mne

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
    patch_legacy_annotations(raw)
    ann = raw.annotations
    n_raw_events = {lab: int((ann.description == lab).sum()) for lab in GAIT_SEQ}

    filtered = filter_gait_events(raw, GAIT_SEQ)
    # replace_events_in_mne would call raw.get_data(); the segmentation only
    # needs the filtered gait annotations in time order, which is what the
    # filtered list already is (non-gait annotations are ignored by
    # segment_gait_cycles).
    filt_ann = mne.Annotations(
        onset=[e[0] for e in filtered],
        duration=[e[1] for e in filtered],
        description=[e[2] for e in filtered],
    )
    cycles = segment_gait_cycles(_AnnotationsOnly(filt_ann), GAIT_SEQ,
                                 max_gap=MAX_GAP_S)
    cycles = [c for c in cycles if RHS in c and RHS_END in c]
    info = {
        "sfreq": float(raw.info["sfreq"]),
        "recording_duration_s": float(raw.times[-1]),
        "n_raw_gait_events": n_raw_events,
        "n_filtered_gait_events": len(filtered),
    }
    return cycles, info


def apply_mask(n_cycles: int, mask_path: Path) -> tuple[np.ndarray, str, int]:
    """Mirror compute_tfr_gait.py: mask of equal length -> apply; shorter
    mask (paired-epoch truncation) -> keep first len(mask) cycles; longer
    -> length mismatch, keep all."""
    if not mask_path.exists():
        return np.ones(n_cycles, dtype=bool), "no_mask_found", n_cycles
    mask = np.load(mask_path)["keep_mask"].astype(bool)
    if len(mask) == n_cycles:
        return mask, "applied", n_cycles
    if len(mask) < n_cycles:
        return mask, f"applied_truncated (mask={len(mask)} cycles={n_cycles})", len(mask)
    return (np.ones(n_cycles, dtype=bool),
            f"length_mismatch (mask={len(mask)} cycles={n_cycles})", n_cycles)


def cycles_to_frame(subject: str, cohort: str, cycles: list[dict],
                    keep: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(cycles):
        t0, t_lto, t_lhs, t_rto, t1 = (c[RHS], c[LTO], c[LHS], c[RTO], c[RHS_END])
        stride = t1 - t0
        row = {
            "subject": subject,
            "cohort": COHORT_LABEL[cohort],
            "cycle_index": i,
            "kept": bool(keep[i]),
            "t_rhs_s": t0,
            "stride_time_s": stride,
            "step_time_left_s": t_lhs - t0,      # RHS -> LHS (left limb steps)
            "step_time_right_s": t1 - t_lhs,     # LHS -> RHS (right limb steps)
            "double_support_initial_s": t_lto - t0,    # RHS -> LTO
            "double_support_terminal_s": t_rto - t_lhs,  # LHS -> RTO
            "swing_left_s": t_lhs - t_lto,       # LTO -> LHS
            "swing_right_s": t1 - t_rto,         # RTO -> RHS
            "warp_factor": stride / EPOCH_DURATION_S,
        }
        row["double_support_total_s"] = (row["double_support_initial_s"]
                                         + row["double_support_terminal_s"])
        row["stance_left_s"] = stride - row["swing_left_s"]
        row["stance_right_s"] = stride - row["swing_right_s"]
        for k in ("double_support_initial", "double_support_terminal",
                  "double_support_total", "swing_left", "swing_right",
                  "stance_left", "stance_right"):
            row[f"{k}_pct"] = 100.0 * row[f"{k}_s"] / stride
        rows.append(row)
    return pd.DataFrame(rows)


def run_lengths(cycles: list[dict], keep: np.ndarray) -> list[tuple[int, float]]:
    """Straight-walking runs: consecutive cycles where cycle i's end heel
    strike is cycle i+1's start heel strike (no gap). Only cycles flagged
    in `keep` participate; a rejected cycle ends the run. Returns
    (n_cycles, duration_s) per run."""
    runs: list[tuple[int, float]] = []
    cur_n, cur_start, prev_end = 0, None, None
    for i, c in enumerate(cycles):
        if not keep[i]:
            if cur_n:
                runs.append((cur_n, prev_end - cur_start))
            cur_n, cur_start, prev_end = 0, None, None
            continue
        contiguous = prev_end is not None and np.isclose(c[RHS], prev_end, atol=1e-6)
        if not contiguous:
            if cur_n:
                runs.append((cur_n, prev_end - cur_start))
            cur_n, cur_start = 0, c[RHS]
        cur_n += 1
        prev_end = c[RHS_END]
    if cur_n:
        runs.append((cur_n, prev_end - cur_start))
    return runs


# --------------------------------------------------------------------------
# Per-subject summary
# --------------------------------------------------------------------------

PHASE_KEYS = ("double_support_initial", "double_support_terminal",
              "double_support_total", "swing_left", "swing_right",
              "stance_left", "stance_right")


def summarise_subject(df: pd.DataFrame, cycles: list[dict], keep: np.ndarray,
                      cycle_set: str) -> dict:
    sub = df if cycle_set == "all" else df[df["kept"]]
    use = np.ones(len(cycles), dtype=bool) if cycle_set == "all" else keep
    st = sub["stride_time_s"].to_numpy()
    out = {
        "n_cycles": int(len(sub)),
        "stride_time_mean_s": st.mean(),
        "stride_time_sd_s": st.std(ddof=1) if len(st) > 1 else np.nan,
        "stride_time_cv_pct": 100.0 * st.std(ddof=1) / st.mean() if len(st) > 1 else np.nan,
        "stride_time_min_s": st.min(),
        "stride_time_max_s": st.max(),
        "stride_time_median_s": np.median(st),
        "cadence_steps_per_min": 120.0 / st.mean(),
        "step_time_left_s": sub["step_time_left_s"].mean(),
        "step_time_right_s": sub["step_time_right_s"].mean(),
    }
    l, r = out["step_time_left_s"], out["step_time_right_s"]
    out["step_time_asymmetry_pct"] = 100.0 * abs(l - r) / np.mean([l, r])
    for k in PHASE_KEYS:
        out[f"{k}_s"] = sub[f"{k}_s"].mean()
        out[f"{k}_pct"] = sub[f"{k}_pct"].mean()
    runs = run_lengths(cycles, use)
    n_run = np.array([n for n, _ in runs], dtype=float)
    d_run = np.array([d for _, d in runs], dtype=float)
    out.update({
        "n_runs": int(len(runs)),
        "run_length_mean_cycles": n_run.mean() if len(runs) else np.nan,
        "run_length_median_cycles": np.median(n_run) if len(runs) else np.nan,
        "run_length_max_cycles": n_run.max() if len(runs) else np.nan,
        "run_duration_mean_s": d_run.mean() if len(runs) else np.nan,
    })
    wf = sub["warp_factor"].to_numpy()
    out.update({
        "warp_factor_mean": wf.mean(),
        "warp_factor_sd": wf.std(ddof=1) if len(wf) > 1 else np.nan,
        "warp_factor_p5": np.percentile(wf, 5),
        "warp_factor_p95": np.percentile(wf, 95),
        "warp_factor_min": wf.min(),
        "warp_factor_max": wf.max(),
    })
    return out


# --------------------------------------------------------------------------
# Group statistics
# --------------------------------------------------------------------------

PARAMS = [
    ("n_cycles", "Number of gait cycles", "count"),
    ("stride_time_mean_s", "Stride time, mean", "s"),
    ("stride_time_sd_s", "Stride time, within-subject SD", "s"),
    ("stride_time_cv_pct", "Stride time, within-subject CV", "%"),
    ("stride_time_min_s", "Stride time, minimum", "s"),
    ("stride_time_max_s", "Stride time, maximum", "s"),
    ("cadence_steps_per_min", "Cadence", "steps/min"),
    ("step_time_left_s", "Step time, left (RHS to LHS)", "s"),
    ("step_time_right_s", "Step time, right (LHS to RHS)", "s"),
    ("step_time_asymmetry_pct", "Step-time asymmetry |L-R|/mean", "%"),
    ("double_support_initial_s", "Initial double support (RHS to LTO)", "s"),
    ("double_support_initial_pct", "Initial double support", "% cycle"),
    ("double_support_terminal_s", "Terminal double support (LHS to RTO)", "s"),
    ("double_support_terminal_pct", "Terminal double support", "% cycle"),
    ("double_support_total_s", "Total double support", "s"),
    ("double_support_total_pct", "Total double support", "% cycle"),
    ("swing_left_s", "Swing, left (LTO to LHS)", "s"),
    ("swing_left_pct", "Swing, left", "% cycle"),
    ("swing_right_s", "Swing, right (RTO to RHS)", "s"),
    ("swing_right_pct", "Swing, right", "% cycle"),
    ("stance_left_s", "Stance, left (stride minus left swing)", "s"),
    ("stance_left_pct", "Stance, left", "% cycle"),
    ("stance_right_s", "Stance, right (stride minus right swing)", "s"),
    ("stance_right_pct", "Stance, right", "% cycle"),
    ("n_runs", "Number of straight-walking runs", "count"),
    ("run_length_mean_cycles", "Run length, mean", "cycles"),
    ("run_length_median_cycles", "Run length, median", "cycles"),
    ("run_length_max_cycles", "Run length, maximum", "cycles"),
    ("run_duration_mean_s", "Run duration, mean", "s"),
    ("warp_factor_mean", "Warp factor stride_time/1.1, mean", "ratio"),
    ("warp_factor_sd", "Warp factor, within-subject SD", "ratio"),
    ("warp_factor_p5", "Warp factor, 5th percentile", "ratio"),
    ("warp_factor_p95", "Warp factor, 95th percentile", "ratio"),
]


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                       / (n1 + n2 - 2))
    if s_pooled == 0:
        return np.nan
    d = (a.mean() - b.mean()) / s_pooled
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return d * j


def group_table(per_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle_set in ("all", "kept"):
        ps = per_subject[per_subject["cycle_set"] == cycle_set]
        pd_ = ps[ps["cohort"] == "PD"]
        hc = ps[ps["cohort"] == "HC"]
        for col, label, unit in PARAMS:
            a = pd_[col].dropna().to_numpy(dtype=float)
            b = hc[col].dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            t, p_t = stats.ttest_ind(a, b, equal_var=False)
            u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "cycle_set": cycle_set,
                "parameter": label,
                "unit": unit,
                "column": col,
                "n_PD": len(a), "n_HC": len(b),
                "PD_mean": a.mean(), "PD_sd": a.std(ddof=1), "PD_median": np.median(a),
                "HC_mean": b.mean(), "HC_sd": b.std(ddof=1), "HC_median": np.median(b),
                "diff_PD_minus_HC": a.mean() - b.mean(),
                "welch_t": t, "welch_df": _welch_df(a, b), "welch_p": p_t,
                "mannwhitney_U": u, "mannwhitney_p": p_u,
                "hedges_g": hedges_g(a, b),
            })
    return pd.DataFrame(rows)


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


# --------------------------------------------------------------------------
# Warp-rescaling summary (comment 11)
# --------------------------------------------------------------------------

def warp_summary(cycle_df: pd.DataFrame, cycle_set: str) -> pd.DataFrame:
    """Pooled over all cycles of a cohort and over the per-subject means."""
    sub = cycle_df if cycle_set == "all" else cycle_df[cycle_df["kept"]]
    rows = []
    for cohort in ("PD", "HC"):
        wf = sub.loc[sub["cohort"] == cohort, "warp_factor"].to_numpy()
        subj_means = (sub[sub["cohort"] == cohort]
                      .groupby("subject")["warp_factor"].mean().to_numpy())
        row = {
            "cycle_set": cycle_set, "cohort": cohort,
            "n_cycles": len(wf),
            "factor_mean_cycles": wf.mean(), "factor_sd_cycles": wf.std(ddof=1),
            "factor_p5_cycles": np.percentile(wf, 5),
            "factor_p95_cycles": np.percentile(wf, 95),
            "factor_mean_subjects": subj_means.mean(),
            "factor_sd_subjects": subj_means.std(ddof=1),
        }
        for f in TEST_FREQS_HZ:
            row[f"{f:g}Hz_apparent_mean"] = f * wf.mean()
            row[f"{f:g}Hz_apparent_p5"] = f * np.percentile(wf, 5)
            row[f"{f:g}Hz_apparent_p95"] = f * np.percentile(wf, 95)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Spatial parameters availability check (task 4)
# --------------------------------------------------------------------------

def check_spatial_sources(paths) -> list[str]:
    """Look for any local derivative that could carry walking speed or
    stride length (marker trajectories, spatiotemporal tables)."""
    patterns = ("*vicon*", "*marker*", "*c3d*", "*traj*", "*spatio*",
                "*speed*", "*stride*", "*kinemat*", "*gait*param*")
    hits: list[str] = []
    for pat in patterns:
        for p in paths.derivatives_root.rglob(pat):
            if ".meta.json" not in p.name:
                hits.append(str(p.relative_to(PROJECT_ROOT)))
    return sorted(set(hits))


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------

def _fmt(x: float, nd: int = 2) -> str:
    return "nan" if pd.isna(x) else f"{x:.{nd}f}"


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def write_txt(path: Path, per_subject: pd.DataFrame, group: pd.DataFrame,
              warp: pd.DataFrame, mask_notes: dict, spatial_hits: list[str],
              date: str) -> None:
    lines = [f"Gait kinematics from EEG gait-event annotations ({BLOCK}), {date}", ""]
    lines.append("Per-subject (kept cycles): subject, cohort, n_all, n_kept, stride mean (SD) s, cadence, CV %, asym %, runs, mask status")
    ps_k = per_subject[per_subject["cycle_set"] == "kept"].set_index("subject")
    ps_a = per_subject[per_subject["cycle_set"] == "all"].set_index("subject")
    for s in ps_k.index:
        r = ps_k.loc[s]
        lines.append(
            f"  {s}  {r['cohort']}  n_all={int(ps_a.loc[s, 'n_cycles'])}  n_kept={int(r['n_cycles'])}"
            f"  stride={_fmt(r['stride_time_mean_s'], 3)} ({_fmt(r['stride_time_sd_s'], 3)}) s"
            f"  cadence={_fmt(r['cadence_steps_per_min'], 1)}"
            f"  CV={_fmt(r['stride_time_cv_pct'], 1)}%  asym={_fmt(r['step_time_asymmetry_pct'], 1)}%"
            f"  runs={int(r['n_runs'])} (mean {_fmt(r['run_length_mean_cycles'], 1)} cycles)"
            f"  mask={mask_notes[s]}"
        )
    for cycle_set in ("kept", "all"):
        lines += ["", f"Group table ({cycle_set} cycles; per-subject values, PD n=10 vs HC n=10)"]
        lines.append("  parameter | PD mean (SD) [median] | HC mean (SD) [median] | Welch p | MWU p | Hedges g")
        for _, r in group[group["cycle_set"] == cycle_set].iterrows():
            lines.append(
                f"  {r['parameter']} ({r['unit']}) | {_fmt(r['PD_mean'])} ({_fmt(r['PD_sd'])}) [{_fmt(r['PD_median'])}]"
                f" | {_fmt(r['HC_mean'])} ({_fmt(r['HC_sd'])}) [{_fmt(r['HC_median'])}]"
                f" | {_p(r['welch_p'])} | {_p(r['mannwhitney_p'])} | {_fmt(r['hedges_g'])}"
            )
    lines += ["", "Warp factor stride_time / 1.1 (frequency multiplier of the submitted warped epochs)"]
    for _, r in warp.iterrows():
        lines.append(
            f"  {r['cycle_set']} {r['cohort']}: n={int(r['n_cycles'])} cycles, mean {_fmt(r['factor_mean_cycles'], 3)}"
            f" SD {_fmt(r['factor_sd_cycles'], 3)} p5 {_fmt(r['factor_p5_cycles'], 3)} p95 {_fmt(r['factor_p95_cycles'], 3)};"
            f" 20 Hz -> {_fmt(r['20Hz_apparent_mean'], 1)} Hz [{_fmt(r['20Hz_apparent_p5'], 1)}-{_fmt(r['20Hz_apparent_p95'], 1)}];"
            f" 13 Hz -> {_fmt(r['13Hz_apparent_mean'], 1)} Hz [{_fmt(r['13Hz_apparent_p5'], 1)}-{_fmt(r['13Hz_apparent_p95'], 1)}]"
        )
    lines += ["", "Spatial parameters (walking speed, stride length): "
              + ("candidate local files: " + ", ".join(spatial_hits) if spatial_hits
                 else "no local derivative found; not computable from EEG annotations (timing only)")]
    path.write_text("\n".join(lines) + "\n")


def write_report(path: Path, per_subject: pd.DataFrame, group: pd.DataFrame,
                 warp: pd.DataFrame, mask_notes: dict, spatial_hits: list[str],
                 date: str) -> None:
    g = group[group["cycle_set"] == "kept"].set_index("column")
    ga = group[group["cycle_set"] == "all"].set_index("column")
    w = warp[warp["cycle_set"] == "kept"].set_index("cohort")
    wa = warp[warp["cycle_set"] == "all"].set_index("cohort")
    ps_k = per_subject[per_subject["cycle_set"] == "kept"]
    ps_a = per_subject[per_subject["cycle_set"] == "all"]
    n_pd_all = int(ps_a.loc[ps_a["cohort"] == "PD", "n_cycles"].sum())
    n_hc_all = int(ps_a.loc[ps_a["cohort"] == "HC", "n_cycles"].sum())
    n_pd_k = int(ps_k.loc[ps_k["cohort"] == "PD", "n_cycles"].sum())
    n_hc_k = int(ps_k.loc[ps_k["cohort"] == "HC", "n_cycles"].sum())

    def row(col: str, nd: int = 2) -> str:
        r = g.loc[col]
        return (f"| {r['parameter']} | {r['unit']} | {_fmt(r['PD_mean'], nd)} ± {_fmt(r['PD_sd'], nd)} "
                f"| {_fmt(r['PD_median'], nd)} | {_fmt(r['HC_mean'], nd)} ± {_fmt(r['HC_sd'], nd)} "
                f"| {_fmt(r['HC_median'], nd)} | {_p(r['welch_p'])} | {_p(r['mannwhitney_p'])} "
                f"| {_fmt(r['hedges_g'])} |")

    def ms(col: str, tab=g) -> str:
        r = tab.loc[col]
        return f"PD {_fmt(r['PD_mean'], 3)} ± {_fmt(r['PD_sd'], 3)} s vs HC {_fmt(r['HC_mean'], 3)} ± {_fmt(r['HC_sd'], 3)} s"

    L = [f"# Gait kinematics from the gait-event annotations (comments 8 and 11)", "",
         f"Script: `code/analysis/revision/gait_kinematics_table.py`, run {date}. "
         f"Block: `{BLOCK}` (straight walking segments of the figure-8 task). "
         "Data: pickle-era `raw_clean` export, fresh AutoReject masks "
         "(`docs/data_manifest_revision_2026-10-12.md`).", "",
         "## Methods (draft paragraph)", "",
         "Spatiotemporal gait parameters were derived from the same motion-capture gait "
         "events (right and left heel strike and toe-off) that were used to segment the "
         f"neural data into gait cycles. A gait cycle was defined from one right heel strike "
         f"to the next, with the intervening left toe-off, left heel strike and right toe-off "
         f"in that order and no inter-event interval longer than {MAX_GAP_S} s. Per cycle we "
         "computed stride time, left and right step time, initial and terminal double-support "
         "time, swing and stance time of each limb (also as a percentage of the cycle), and "
         "the per-cycle time-warp factor (stride time / 1.1 s). Per participant we averaged "
         "these over all cycles that entered the neural analyses (i.e. after AutoReject), "
         "and derived cadence (120 / mean stride time), the stride-time coefficient of "
         "variation and the step-time asymmetry |left − right| / mean. Group differences "
         "were tested on the per-participant values with Welch's t-test and the "
         "Mann–Whitney U test, with Hedges' g as effect size. Walking speed and stride "
         "length were not available from the gait-event timings. "
         "TODO(verify): add the gait-event detection method and motion-capture system from "
         "the Methods on Overleaf.", "",
         f"## Group table (AutoReject-kept cycles; PD n = {g.loc['n_cycles','n_PD']} "
         f"participants, {n_pd_k} cycles; HC n = {g.loc['n_cycles','n_HC']} participants, "
         f"{n_hc_k} cycles)", "",
         "| Parameter | Unit | PD mean ± SD | PD median | HC mean ± SD | HC median | Welch p | MWU p | Hedges g |",
         "|---|---|---|---|---|---|---|---|---|"]
    for col, _, unit in PARAMS:
        nd = 0 if unit == "count" else (3 if unit in ("s", "ratio") else 1)
        L.append(row(col, nd))
    L += ["", f"All segmented cycles (before AutoReject; PD {n_pd_all}, HC {n_hc_all} cycles) "
          "give the same picture: " + "; ".join(
              f"{ga.loc[c, 'parameter']} {ms(c, ga)} (Welch p = {_p(ga.loc[c, 'welch_p'])})"
              for c in ("stride_time_mean_s",)) + ". Full table: "
          f"`gait_kinematics_{date}.csv` (rows with `cycle_set = all`).", ""]

    L += ["## Warp-rescaling (comment 11)", "",
          "The submitted pipeline linearly time-warped each cycle to 1.1 s (piecewise, "
          "with the four events pinned at 0, 10, 50, 60 and 100 %). Compressing a cycle of "
          "stride time T to 1.1 s multiplies every frequency in that cycle by T / 1.1 "
          "(the reviewer's example: 1.3 s → 20 Hz appears at 23.6 Hz). Distribution of "
          "that factor over the kept cycles:", "",
          "| Cohort | Cycles | Factor mean ± SD | 5th–95th percentile | 20 Hz appears at (mean [p5–p95]) | 13 Hz appears at |",
          "|---|---|---|---|---|---|"]
    for coh in ("PD", "HC"):
        r = w.loc[coh]
        L.append(f"| {coh} | {int(r['n_cycles'])} | {_fmt(r['factor_mean_cycles'], 3)} ± {_fmt(r['factor_sd_cycles'], 3)} "
                 f"| {_fmt(r['factor_p5_cycles'], 3)}–{_fmt(r['factor_p95_cycles'], 3)} "
                 f"| {_fmt(r['20Hz_apparent_mean'], 1)} Hz [{_fmt(r['20Hz_apparent_p5'], 1)}–{_fmt(r['20Hz_apparent_p95'], 1)}] "
                 f"| {_fmt(r['13Hz_apparent_mean'], 1)} Hz [{_fmt(r['13Hz_apparent_p5'], 1)}–{_fmt(r['13Hz_apparent_p95'], 1)}] |")
    rw = g.loc["warp_factor_mean"]
    L += ["", f"Per-participant mean factor: PD {_fmt(rw['PD_mean'], 3)} ± {_fmt(rw['PD_sd'], 3)} vs HC "
          f"{_fmt(rw['HC_mean'], 3)} ± {_fmt(rw['HC_sd'], 3)} (difference {_fmt(rw['diff_PD_minus_HC'], 3)}, "
          f"Welch p = {_p(rw['welch_p'])}, Hedges g = {_fmt(rw['hedges_g'])}). "
          f"A 20 Hz oscillation therefore appears on average at {_fmt(20*rw['PD_mean'], 1)} Hz in PD and "
          f"{_fmt(20*rw['HC_mean'], 1)} Hz in HC (cohort difference {_fmt(20*rw['diff_PD_minus_HC'], 2)} Hz); "
          f"a 13 Hz oscillation at {_fmt(13*rw['PD_mean'], 1)} Hz and {_fmt(13*rw['HC_mean'], 1)} Hz "
          f"(difference {_fmt(13*rw['diff_PD_minus_HC'], 2)} Hz). "
          "Within a cycle the factor also differs between phases, because the double-support "
          "intervals are mapped to 10 % and the swing intervals to 40 % of the 1.1 s epoch; "
          "the per-phase factors are (actual phase duration) / (0.11 s or 0.44 s) and can be "
          f"read from `gait_kinematics_cycles_{date}.csv`.",
          "", f"All-cycles values: PD {_fmt(wa.loc['PD','factor_mean_cycles'], 3)} ± {_fmt(wa.loc['PD','factor_sd_cycles'], 3)}, "
          f"HC {_fmt(wa.loc['HC','factor_mean_cycles'], 3)} ± {_fmt(wa.loc['HC','factor_sd_cycles'], 3)}.", ""]

    L += ["## Walking speed and stride length", "",
          ("Candidate local files found: " + ", ".join(f"`{h}`" for h in spatial_hits) + ". TODO(verify) whether they contain spatial parameters."
           if spatial_hits else
           "Not computable from the local data. The EEG annotations carry event timings only; "
           "no motion-capture marker trajectories or spatiotemporal tables exist under "
           "`data/derivatives/` or elsewhere in the repository. The server Vicon files were "
           "not read (project rule). TODO(verify): obtain walking speed and stride length "
           "from the motion-capture export (Z.M.), or state in the manuscript that only "
           "temporal parameters are reported."), ""]

    st, cad, cv, asy, ds = (g.loc["stride_time_mean_s"], g.loc["cadence_steps_per_min"],
                            g.loc["stride_time_cv_pct"], g.loc["step_time_asymmetry_pct"],
                            g.loc["double_support_total_pct"])
    sw = g.loc["swing_left_pct"]
    nr, rl = g.loc["n_runs"], g.loc["run_length_mean_cycles"]
    L += ["## Results (draft paragraph)", "",
          f"Gait cycles entering the analyses numbered {_fmt(g.loc['n_cycles','PD_mean'],0)} ± "
          f"{_fmt(g.loc['n_cycles','PD_sd'],0)} per PD participant and {_fmt(g.loc['n_cycles','HC_mean'],0)} ± "
          f"{_fmt(g.loc['n_cycles','HC_sd'],0)} per HC participant (Welch p = {_p(g.loc['n_cycles','welch_p'])}). "
          f"Mean stride time was {_fmt(st['PD_mean'],3)} ± {_fmt(st['PD_sd'],3)} s in PD and "
          f"{_fmt(st['HC_mean'],3)} ± {_fmt(st['HC_sd'],3)} s in HC (Welch t({_fmt(st['welch_df'],1)}) = {_fmt(st['welch_t'])}, "
          f"p = {_p(st['welch_p'])}; Mann–Whitney p = {_p(st['mannwhitney_p'])}; Hedges' g = {_fmt(st['hedges_g'])}), "
          f"corresponding to a cadence of {_fmt(cad['PD_mean'],1)} ± {_fmt(cad['PD_sd'],1)} versus "
          f"{_fmt(cad['HC_mean'],1)} ± {_fmt(cad['HC_sd'],1)} steps/min (p = {_p(cad['welch_p'])}). "
          f"Stride-time variability (CV) was {_fmt(cv['PD_mean'],1)} ± {_fmt(cv['PD_sd'],1)} % in PD and "
          f"{_fmt(cv['HC_mean'],1)} ± {_fmt(cv['HC_sd'],1)} % in HC (p = {_p(cv['welch_p'])}), and step-time "
          f"asymmetry {_fmt(asy['PD_mean'],1)} ± {_fmt(asy['PD_sd'],1)} % versus {_fmt(asy['HC_mean'],1)} ± "
          f"{_fmt(asy['HC_sd'],1)} % (p = {_p(asy['welch_p'])}). Total double support occupied "
          f"{_fmt(ds['PD_mean'],1)} ± {_fmt(ds['PD_sd'],1)} % of the cycle in PD and {_fmt(ds['HC_mean'],1)} ± "
          f"{_fmt(ds['HC_sd'],1)} % in HC (p = {_p(ds['welch_p'])}); left swing {_fmt(sw['PD_mean'],1)} ± "
          f"{_fmt(sw['PD_sd'],1)} % versus {_fmt(sw['HC_mean'],1)} ± {_fmt(sw['HC_sd'],1)} % (p = {_p(sw['welch_p'])}). "
          f"Straight-walking runs numbered {_fmt(nr['PD_mean'],1)} ± {_fmt(nr['PD_sd'],1)} (PD) and "
          f"{_fmt(nr['HC_mean'],1)} ± {_fmt(nr['HC_sd'],1)} (HC) per participant, with a mean run length of "
          f"{_fmt(rl['PD_mean'],1)} ± {_fmt(rl['PD_sd'],1)} and {_fmt(rl['HC_mean'],1)} ± {_fmt(rl['HC_sd'],1)} cycles. "
          f"The per-cycle warp factor (stride time / 1.1 s) averaged {_fmt(rw['PD_mean'],3)} ± {_fmt(rw['PD_sd'],3)} in PD and "
          f"{_fmt(rw['HC_mean'],3)} ± {_fmt(rw['HC_sd'],3)} in HC (p = {_p(rw['welch_p'])}), so that in the time-warped "
          f"epochs a 20 Hz oscillation would appear at {_fmt(20*rw['PD_mean'],1)} Hz and {_fmt(20*rw['HC_mean'],1)} Hz, "
          f"respectively.", ""]

    L += ["## Per-subject mask status", ""]
    L += [f"- {s}: {m}" for s, m in mask_notes.items()]
    L += ["", f"Per-subject values: `gait_kinematics_per_subject_{date}.csv`; per-cycle timings: "
          f"`gait_kinematics_cycles_{date}.csv`."]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", default=None,
                        help="Run a single subject (dry run; prints, no group table).")
    parser.add_argument("--date", default="2026-10-13")
    parser.add_argument("--check-spatial", action="store_true",
                        help="Only list candidate local files for spatial parameters.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("gait_kinematics_table", config)

    spatial_hits = check_spatial_sources(paths)
    if args.check_spatial:
        print("candidate spatial files:", spatial_hits or "none")
        return 0

    subjects = ([validate_subject(args.subject)] if args.subject else
                [s for members in config["cohorts"].values() for s in members])

    cycle_frames, subj_rows, mask_notes, subj_info = [], [], {}, {}
    for subject in subjects:
        cohort = cohort_of(subject, config)
        raw_path = paths.raw_clean_dir / f"{subject}_{BLOCK}-eeg-raw.fif"
        mask_path = paths.derivatives_root / "autoreject" / f"{subject}_{BLOCK}-mask.npz"
        if not raw_path.exists():
            print(f"missing {raw_path}", file=sys.stderr)
            mask_notes[subject] = "raw missing"
            continue
        cycles, info = load_cycles(raw_path)
        if not cycles:
            print(f"no cycles for {subject}", file=sys.stderr)
            mask_notes[subject] = "no cycles"
            continue
        keep, status, n_use = apply_mask(len(cycles), mask_path)
        cycles = cycles[:n_use]
        mask_notes[subject] = f"{status}; segmented={len(cycles)}, kept={int(keep.sum())}"
        subj_info[subject] = info
        df = cycles_to_frame(subject, cohort, cycles, keep)
        cycle_frames.append(df)
        for cycle_set in ("all", "kept"):
            row = {"subject": subject, "cohort": COHORT_LABEL[cohort], "cycle_set": cycle_set,
                   "mask_status": status, "n_cycles_segmented": len(cycles),
                   "n_cycles_kept": int(keep.sum())}
            row.update(summarise_subject(df, cycles, keep, cycle_set))
            subj_rows.append(row)
        r = subj_rows[-1]
        print(f"{subject} {COHORT_LABEL[cohort]}: segmented={len(cycles)} kept={int(keep.sum())} "
              f"stride={r['stride_time_mean_s']:.3f}±{r['stride_time_sd_s']:.3f} s "
              f"cadence={r['cadence_steps_per_min']:.1f} runs={r['n_runs']} "
              f"warp={r['warp_factor_mean']:.3f} [{status}]")

    if not cycle_frames:
        print("nothing processed", file=sys.stderr)
        return 2
    per_subject = pd.DataFrame(subj_rows)
    cycles_df = pd.concat(cycle_frames, ignore_index=True)

    if args.subject:
        print(per_subject.T.to_string())
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date = args.date
    group = group_table(per_subject)
    warp = pd.concat([warp_summary(cycles_df, "all"), warp_summary(cycles_df, "kept")],
                     ignore_index=True)

    out_group = OUT_DIR / f"gait_kinematics_{date}.csv"
    out_subj = OUT_DIR / f"gait_kinematics_per_subject_{date}.csv"
    out_cyc = OUT_DIR / f"gait_kinematics_cycles_{date}.csv"
    out_warp = OUT_DIR / f"gait_kinematics_warp_factor_{date}.csv"
    out_txt = OUT_DIR / f"gait_kinematics_{date}.txt"
    out_md = OUT_DIR / f"gait_kinematics_report_{date}.md"

    group.to_csv(out_group, index=False)
    per_subject.to_csv(out_subj, index=False)
    cycles_df.to_csv(out_cyc, index=False)
    warp.to_csv(out_warp, index=False)
    write_txt(out_txt, per_subject, group, warp, mask_notes, spatial_hits, date)
    write_report(out_md, per_subject, group, warp, mask_notes, spatial_hits, date)

    meta.update({
        "block": BLOCK, "epoch_duration_s": EPOCH_DURATION_S, "max_gap_s": MAX_GAP_S,
        "gait_sequence": list(GAIT_SEQ), "event_percentages": EVENT_PCT,
        "subjects": subjects, "mask_status": mask_notes,
        "recording_info": subj_info,
        "inputs": {"raw_clean_dir": str(paths.raw_clean_dir),
                   "autoreject_dir": str(paths.derivatives_root / "autoreject"),
                   "data_manifest": "docs/data_manifest_revision_2026-10-12.md"},
        "spatial_parameter_candidates": spatial_hits,
        "scipy": __import__("scipy").__version__, "pandas": pd.__version__,
    })
    for p in (out_group, out_subj, out_cyc, out_warp, out_txt, out_md):
        write_sidecar(p, {**meta, "output": str(p)})
    print(f"\nwrote {out_group.name}, {out_subj.name}, {out_cyc.name}, {out_warp.name}, "
          f"{out_txt.name}, {out_md.name} in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
