"""Data-quality check of the post8walk block for all manuscript participants.

Purpose:  Assess whether the (so far unused) post8walk block is usable for
          analysis: EEG signal quality, bad critical channels, gait-event
          integrity and cycle counts, and (patients) DBS availability + sync.
Inputs:   $RESIDUAL_DATA_ROOT/<group>/<subj>/ProcessedData/EEG/post8walk/
              post8walk_proc.fif                      (read-only server mount)
          $RESIDUAL_DATA_ROOT/Patient_NFB/<subj>/RawData/DBS/
              <subj>_dbs_df_td.csv + config_<subj>.yml (via data_loader)
          data/derivatives/gerd/<subj>_pre8walk-eeg-basenorm-gerd.csv
              (pre8walk cycle-count reference, local)
          code/analysis/config.yaml (cohorts, freq bands, seed)
Outputs:  outputs/stats/post8walk_quality_<YYYY-MM-DD>.csv  (+ .meta.json)
          outputs/stats/post8walk_quality_<YYYY-MM-DD>.txt  (+ .meta.json)
Author:   Lena
Date:     2026-08-14
Seed:     42 (config repro.seed, applied via _common.setup_run)

Description:
  post8walk is not part of the manuscript pipeline (config blocks are
  baseline + pre8walk), so no local derivatives exist for it. This QA
  reads the server `post8walk_proc.fif` per subject (post-ICA; fine for
  new analyses per docs — always respecting info['bads']) and reports,
  per subject:
    1. EEG: duration, bad-channel counts overall and within the critical
       motor/frontal set used by data_quality.py, good-channel signal SD,
       high-frequency power ratio (30-70 Hz / 2-20 Hz, EMG proxy), and
       per-channel PSD flat/band-outlier flags (data_quality.get_psd_metrics).
    2. Gait: raw gait-event annotation count, events surviving
       filter_gait_events, cycles from segment_gait_cycles (the exact
       chain compute_tfr_gait.py uses), complete-cycle count, cycle
       duration stats, inter-cycle gaps (turns/breaks), and the subject's
       pre8walk cycle count as reference (from local gERD derivatives).
    3. DBS (Patient_NFB only): time-domain CSV rows for post8walk,
       recording side (with config.yaml dbs_side_overrides applied),
       signal SD, and whether the EEG carries a dbsSync annotation
       (without sync, task-block DBS cannot be aligned to gait events).
  Within-cohort z-scores flag outliers (|z| > 2) on signal SD, HF ratio,
  and complete-cycle count. Nothing is copied from the server; outputs
  contain anonymized codes only.

Usage:
  python code/analysis/check_post8walk_quality.py            # full cohorts
  python code/analysis/check_post8walk_quality.py --subject P4-2002
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from _common import (
    PROJECT_ROOT,
    cohort_of,
    load_config,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    validate_subject,
    write_sidecar,
)

# QA-only block: deliberately NOT validated against config['blocks']
# (those list the manuscript pipeline blocks baseline + pre8walk).
BLOCK = "post8walk"

# Critical channel set from data_quality.py (motor + frontal ROIs).
CRITICAL_CHANNELS = {"Cz", "C1", "C3", "C2", "C4", "Fz", "AF3", "F1", "AF4", "F2"}

# Same threshold as segment_gait_cycles' default; gaps above it are
# turns/breaks between straight-walking bouts (see turn-analysis notes).
MAX_GAP_S = 1.5

Z_FLAG = 2.0  # within-cohort outlier threshold


def eeg_metrics(raw, bands: dict) -> dict:
    """EEG-quality metrics on good channels only (respect info['bads'])."""
    from data_quality import get_psd_metrics

    sfreq = raw.info["sfreq"]
    bads = set(raw.info["bads"])
    eeg_names = [
        name for name, kind in zip(raw.ch_names, raw.get_channel_types())
        if kind == "eeg"
    ]
    bad_eeg = [ch for ch in eeg_names if ch in bads]
    bad_crit = sorted(CRITICAL_CHANNELS & bads)
    good_crit = [ch for ch in eeg_names
                 if ch in CRITICAL_CHANNELS and ch not in bads]

    good_data = raw.get_data(picks="eeg")  # picks='eeg' excludes bads
    eeg_std_uv = float(good_data.std() * 1e6)

    hf_ratio = np.nan
    n_flat = n_outlier = 0
    peak_freqs = []
    if good_crit:
        crit_data = raw.get_data(picks=good_crit)
        f, pxx = welch(crit_data, sfreq, nperseg=1024)
        mean_pxx = pxx.mean(axis=0)
        lo = mean_pxx[(f >= 2) & (f <= 20)].mean()
        hi = mean_pxx[(f >= 30) & (f <= 70)].mean()
        hf_ratio = float(hi / lo) if lo > 0 else np.inf
        for ch_row in crit_data:
            m = get_psd_metrics(ch_row, sfreq, bands)
            n_flat += int(m["PSD_flat_flag"])
            n_outlier += int(m["PSD_outlier_flag"])
            peak_freqs.append(m["PSD_peak_freq"])
        del crit_data
    del good_data

    return {
        "duration_s": float(raw.times[-1]),
        "sfreq": sfreq,
        "n_bad_eeg": len(bad_eeg),
        "n_bad_critical": len(bad_crit),
        "bad_critical": ";".join(bad_crit),
        "eeg_std_uV": eeg_std_uv,
        "hf_ratio": hf_ratio,
        "n_flat_crit_ch": n_flat,
        "n_outlier_crit_ch": n_outlier,
        "psd_peak_freq_med": float(np.median(peak_freqs)) if peak_freqs else np.nan,
        "dbs_sync_in_eeg": bool(
            any(d == "dbsSync" for d in raw.annotations.description)
        ),
    }


def gait_metrics(raw, gait_seq: list[str]) -> dict:
    """Gait-cycle metrics via the exact chain compute_tfr_gait.py uses."""
    from epoching import (
        filter_gait_events,
        replace_events_in_mne,
        segment_gait_cycles,
    )

    n_ann = sum(1 for d in raw.annotations.description if d in gait_seq)
    fe = filter_gait_events(raw, gait_seq)
    raw_filt = replace_events_in_mne(raw, fe, gait_seq)
    cycles = segment_gait_cycles(raw_filt, gait_seq, max_gap=MAX_GAP_S) or []
    del raw_filt
    gc.collect()

    complete = sorted(
        (c for c in cycles if "HSrightlocs_end" in c),
        key=lambda c: c["HSrightlocs"],
    )
    out = {
        "n_gait_ann": n_ann,
        "n_filt_events": len(fe),
        "n_cycles": len(cycles),
        "n_complete_cycles": len(complete),
        "cycle_dur_med_s": np.nan,
        "cycle_dur_iqr_s": np.nan,
        "n_gaps_gt_maxgap": np.nan,
        "max_intercycle_gap_s": np.nan,
    }
    if complete:
        start = np.array([c["HSrightlocs"] for c in complete])
        end = np.array([c["HSrightlocs_end"] for c in complete])
        dur = end - start
        out["cycle_dur_med_s"] = float(np.median(dur))
        q25, q75 = np.percentile(dur, [25, 75])
        out["cycle_dur_iqr_s"] = float(q75 - q25)
        if len(complete) > 1:
            gaps = start[1:] - end[:-1]
            out["n_gaps_gt_maxgap"] = int((gaps > MAX_GAP_S).sum())
            out["max_intercycle_gap_s"] = float(gaps.max())
    return out


def dbs_metrics(data_root: Path, subject: str) -> dict:
    """DBS availability + signal SD for one patient (post8walk rows)."""
    from data_loader import load_dbs_data

    dbs_df, side = load_dbs_data(str(data_root), subject, BLOCK)
    if dbs_df is None:
        return {"dbs_present": False, "dbs_side": None,
                "dbs_n_samples": 0, "dbs_duration_s": np.nan,
                "dbs_std_uV": np.nan}
    # Percept time-domain stream is 250 Hz (config.yaml tfr comment);
    # load_dbs_data scales TimeDomainData to volts.
    n = len(dbs_df)
    return {
        "dbs_present": True,
        "dbs_side": side,
        "dbs_n_samples": n,
        "dbs_duration_s": n / 250.0,
        "dbs_std_uV": float(dbs_df["TimeDomainData"].std() * 1e6),
    }


def qa_one_subject(subject: str, cohort: str, data_root: Path,
                   bands: dict, gait_seq: list[str]) -> dict:
    from data_loader import load_eeg_data

    row: dict = {"ID": subject, "Cohort": cohort}
    raw = load_eeg_data(str(data_root), cohort, subject, BLOCK)
    if raw is None:
        row["eeg_present"] = False
        return row
    row["eeg_present"] = True
    patch_legacy_annotations(raw)
    row.update(eeg_metrics(raw, bands))
    row.update(gait_metrics(raw, gait_seq))
    del raw
    gc.collect()

    if cohort == "Patient_NFB":
        row.update(dbs_metrics(data_root, subject))

    # pre8walk complete-cycle reference from local gERD derivative.
    gerd = (PROJECT_ROOT / "data" / "derivatives" / "gerd" /
            f"{subject}_pre8walk-eeg-basenorm-gerd.csv")
    row["pre8walk_cycles_ref"] = (
        int(pd.read_csv(gerd, usecols=["Epoch"])["Epoch"].nunique())
        if gerd.exists() else np.nan
    )
    return row


def add_cohort_zscores_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col, zcol in [("eeg_std_uV", "z_eeg_std"),
                      ("hf_ratio", "z_hf_ratio"),
                      ("n_complete_cycles", "z_cycles")]:
        df[zcol] = np.nan
        for cohort, grp in df.groupby("Cohort"):
            vals = grp[col].astype(float)
            if vals.notna().sum() >= 2 and vals.std(ddof=1) > 0:
                df.loc[grp.index, zcol] = (vals - vals.mean()) / vals.std(ddof=1)

    def flags_for(r) -> str:
        flags = []
        if not r.get("eeg_present", False):
            return "MISSING_EEG"
        if r["n_complete_cycles"] == 0:
            flags.append("NO_GAIT_CYCLES")
        if r["n_bad_critical"] > 0:
            flags.append(f"BAD_CRITICAL_CH({r['bad_critical']})")
        if r["n_flat_crit_ch"] > 0:
            flags.append("FLAT_PSD")
        if abs(r.get("z_eeg_std", 0) or 0) > Z_FLAG:
            flags.append("EEG_STD_OUTLIER")
        if abs(r.get("z_hf_ratio", 0) or 0) > Z_FLAG:
            flags.append("HF_RATIO_OUTLIER")
        if (r.get("z_cycles") or 0) < -Z_FLAG:
            flags.append("LOW_CYCLE_COUNT")
        if r["Cohort"] == "Patient_NFB":
            if not r.get("dbs_present", False):
                flags.append("NO_DBS")
            elif not r.get("dbs_sync_in_eeg", False):
                flags.append("NO_DBS_SYNC")
        return ";".join(flags) if flags else "OK"

    df["flags"] = df.apply(flags_for, axis=1)
    return df


def write_report(df: pd.DataFrame, txt_path: Path) -> None:
    lines = [
        f"post8walk data-quality check — {date.today().isoformat()}",
        f"Source: <data_root>/<group>/<subj>/ProcessedData/EEG/{BLOCK}/"
        f"{BLOCK}_proc.fif (server, post-ICA; bads respected)",
        f"Gait chain: filter_gait_events -> replace_events_in_mne -> "
        f"segment_gait_cycles (max_gap={MAX_GAP_S}s), as in compute_tfr_gait.py",
        f"Flags: within-cohort |z| > {Z_FLAG} on EEG SD / HF ratio / cycles;"
        " bad critical channels; flat PSD; missing DBS or sync (PD).",
        "CAVEAT eeg_std_uV: the AgeMatched_NFB proc fifs store uV-magnitude"
        " numbers in nominally-volt channels (verified 2026-08-14 on"
        " P4-1005 Cz vs P4-2002/P4-2005), so their eeg_std_uV is inflated"
        " 1e6x vs Patient_NFB. Compare EEG SD within cohort only.",
        "",
    ]
    show = ["ID", "duration_s", "n_bad_eeg", "n_bad_critical", "eeg_std_uV",
            "hf_ratio", "n_complete_cycles", "pre8walk_cycles_ref",
            "cycle_dur_med_s", "n_gaps_gt_maxgap", "flags"]
    pd_cols = show[:-1] + ["dbs_present", "dbs_side", "dbs_std_uV",
                           "dbs_sync_in_eeg", "flags"]
    for cohort, cols in [("Patient_NFB", pd_cols), ("AgeMatched_NFB", show)]:
        sub = df[df["Cohort"] == cohort]
        lines.append(f"=== {cohort} (n={len(sub)}) ===")
        cols_present = [c for c in cols if c in sub.columns]
        lines.append(sub[cols_present].to_string(index=False,
                                                 float_format="%.2f"))
        lines.append("")
    n_flagged = int((df["flags"] != "OK").sum())
    lines.append(f"Subjects flagged: {n_flagged}/{len(df)}")
    for _, r in df[df["flags"] != "OK"].iterrows():
        lines.append(f"  {r['ID']}: {r['flags']}")
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", help="Run a single subject only.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("check_post8walk_quality", config)
    bands = {k: tuple(v) for k, v in config["freq_bands"].items()}

    from variables import gait_events_correct_sequence as gait_seq

    if args.subject:
        s = validate_subject(args.subject)
        subjects = [(s, cohort_of(s, config))]
    else:
        subjects = [(s, cohort) for cohort, members in config["cohorts"].items()
                    for s in members]

    stats_dir = PROJECT_ROOT / "outputs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = stats_dir / f"post8walk_quality_{stamp}.csv"
    txt_path = stats_dir / f"post8walk_quality_{stamp}.txt"

    rows = []
    for i, (subject, cohort) in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject} ({cohort})", flush=True)
        try:
            rows.append(qa_one_subject(subject, cohort, paths.data_root,
                                       bands, gait_seq))
        except Exception as exc:  # keep going; report the failure per subject
            print(f"  FAILED: {exc!r}", file=sys.stderr, flush=True)
            rows.append({"ID": subject, "Cohort": cohort,
                         "eeg_present": False, "error": repr(exc)})
        # Progressive dump so a crash keeps completed subjects.
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")

    df = add_cohort_zscores_and_flags(pd.DataFrame(rows))
    df.to_csv(csv_path, index=False, float_format="%.4f")
    write_report(df, txt_path)

    sidecar = {**meta, "block": BLOCK, "max_gap_s": MAX_GAP_S,
               "z_flag": Z_FLAG, "n_subjects": len(df),
               "outputs": [str(csv_path), str(txt_path)]}
    write_sidecar(csv_path, sidecar)
    write_sidecar(txt_path, sidecar)
    print(f"\nSaved: {csv_path}\nSaved: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
