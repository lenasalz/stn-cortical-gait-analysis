"""Spatiotemporal gait parameters from the motion-capture export (PD vs HC).

Purpose:  Add walking speed, stride/step length and step width (reviewer
          Comment 8) from the Vicon gait summaries, complementing the
          event-timing table in gait_kinematics_table.py.
Inputs:   $RESIDUAL_DATA_ROOT/<cohort>/<subj>/ProcessedData/Vicon/matfile/pre8walk/GaitSummary.mat
              (read-only server share; one struct element per straight-walking
              trial, TimeSeriesGait.* per-cycle arrays, GaitParameters.* trial scalars)
          $RESIDUAL_DATA_ROOT/AgeMatched_NFB/GaitParameters_summary_mergedLR.xlsx
              (existing HC summary; used only to validate the aggregation)
          code/analysis/config.yaml (cohorts, seed)
Outputs:  outputs/stats/revision_2026-10/gait_kinematics_mocap_per_subject_2026-10-21.csv
          outputs/stats/revision_2026-10/gait_kinematics_mocap_2026-10-21.csv  (group table)
          outputs/stats/revision_2026-10/gait_kinematics_mocap_2026-10-21.txt
Author:   Lena
Date:     2026-10-21
Seed:     42 (config repro.seed; no randomness)

Description:
  Per subject, the per-trial GaitParameters (one straight-walking trial per
  struct element) are averaged over all trials of the pre8walk block with
  equal weight; left and right fields are merged. This reproduces the
  existing HC summary workbook (validation printed). Lengths in cm, speed in cm/s, times in s, cadence in steps/min.
  Group comparison: Welch t, Mann-Whitney U, Hedges g. Nothing from the
  share is copied; only subject codes and derived numbers are written.

Usage:
  RESIDUAL_DATA_ROOT=/path/to/data \
    python code/analysis/revision/gait_kinematics_mocap.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"
PARAMS = {  # output name: (TimeSeriesGait field(s), unit)
    "walking_speed_cm_s": (("walkingSpeed",), "cm/s"),
    "stride_length_cm": (("strideLengthL", "strideLengthR"), "cm"),
    "step_length_cm": (("stepLengthL", "stepLengthR"), "cm"),
    "step_width_cm": (("stepWidthL", "stepWidthR"), "cm"),
    "cadence_steps_min": (("Cadence",), "steps/min"),
    "gait_cycle_s": (("durationGaitCycleL", "durationGaitCycleR"), "s"),
    "stance_s": (("durationStancePhL", "durationStancePhR"), "s"),
    "swing_s": (("durationSwingPhL", "durationSwingPhR"), "s"),
    "double_support_s": (("dlsL", "dlsR"), "s"),
    "step_length_asymmetry": (("GaitAsymmetry_StepLength",), "index"),
    "step_time_asymmetry": (("GaitAsymmetry_StepTime",), "index"),
}


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def subject_values(mat_path: Path) -> dict:
    """Per-subject means of the per-trial GaitParameters (one straight-walking
    trial per struct element; L/R fields merged; trials weighted equally).
    TimeSeriesGait rows are [start_frame, end_frame, value] and are not used."""
    gs = np.atleast_1d(sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["GaitSummary"])
    out = {"n_trials": 0}
    per_trial = {name: [] for name in PARAMS}
    for el in gs:
        gp = getattr(el, "GaitParameters", None)
        if gp is None or not hasattr(gp, "_fieldnames"):        # empty trial
            continue
        out["n_trials"] += 1
        for name, (fields, _) in PARAMS.items():
            vals = []
            for fld in fields:
                if fld in gp._fieldnames:
                    v = np.atleast_1d(getattr(gp, fld)).astype(float).ravel()
                    vals.append(v[np.isfinite(v)])
            v = np.concatenate(vals) if vals else np.array([])
            if v.size:
                per_trial[name].append(v.mean())
    for name in PARAMS:
        out[name] = float(np.mean(per_trial[name])) if per_trial[name] else np.nan
        out[name + "_n"] = len(per_trial[name])
    return out


def main() -> int:
    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("gait_kinematics_mocap", config)
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    rows = []
    for cohort in (PD, HC):
        for subj in config["cohorts"][cohort]:
            f = paths.data_root / cohort / subj / "ProcessedData/Vicon/matfile/pre8walk/GaitSummary.mat"
            if not f.exists():
                rows.append({"subject": subj, "cohort": cohort, "status": "missing"}); continue
            rows.append({"subject": subj, "cohort": cohort, "status": "ok", **subject_values(f)})
    ps = out_dir / "gait_kinematics_mocap_per_subject_2026-10-21.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("subject", "cohort", "status"), k))
    with ps.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write_sidecar(ps, meta)

    lines = ["Spatiotemporal gait parameters from the Vicon GaitSummary (pre8walk, straight-walking trials pooled, L/R merged)", ""]
    # validation against the existing HC workbook
    try:
        import pandas as pd
        ref = pd.read_excel(paths.data_root / HC / "GaitParameters_summary_mergedLR.xlsx").set_index("Subject")
        for name, col in (("stride_length_cm", "strideLength"), ("walking_speed_cm_s", "walkingSpeed"), ("cadence_steps_min", "Cadence")):
            d = [r[name] - float(ref.loc[r["subject"], col]) for r in rows if r["cohort"] == HC and r.get("status") == "ok" and r["subject"] in ref.index]
            lines.append(f"validation vs HC workbook, {col}: mean abs diff {np.mean(np.abs(d)):.3f} (n={len(d)})")
    except Exception as e:  # pandas/openpyxl may be absent in this environment
        lines.append(f"validation vs HC workbook skipped ({type(e).__name__})")
    lines.append("")
    group = []
    for name, (_, unit) in PARAMS.items():
        a = np.array([r[name] for r in rows if r["cohort"] == PD and r.get("status") == "ok" and np.isfinite(r[name])])
        b = np.array([r[name] for r in rows if r["cohort"] == HC and r.get("status") == "ok" and np.isfinite(r[name])])
        if len(a) < 2 or len(b) < 2:
            continue
        t, pt = stats.ttest_ind(a, b, equal_var=False); pu = stats.mannwhitneyu(a, b).pvalue; g = hedges_g(a, b)
        group.append(dict(parameter=name, unit=unit, n_pd=len(a), n_hc=len(b), pd_mean=a.mean(), pd_sd=a.std(ddof=1),
                          hc_mean=b.mean(), hc_sd=b.std(ddof=1), welch_t=t, welch_p=pt, mwu_p=pu, hedges_g=g))
        lines.append(f"{name:26s} PD {a.mean():8.2f} ± {a.std(ddof=1):6.2f}  HC {b.mean():8.2f} ± {b.std(ddof=1):6.2f}  {unit:9s} Welch p={pt:.3f}  MWU p={pu:.3f}  g={g:+.2f}  (n={len(a)}/{len(b)})")
    gp = out_dir / "gait_kinematics_mocap_2026-10-21.csv"
    with gp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(group[0])); w.writeheader(); w.writerows(group)
    write_sidecar(gp, meta)
    missing = [r["subject"] for r in rows if r.get("status") != "ok"]
    lines.append(""); lines.append("missing GaitSummary: " + (", ".join(missing) or "none"))
    (out_dir / "gait_kinematics_mocap_2026-10-21.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
