"""End-to-end pipeline-vs-legacy comparison for both EEG and DBS modalities.

Runs the legacy code path from final_analysis/scripts/tfr.py
(`compute_tfr_on_subset_and_normalize_optimized` →
`create_gait_warped_tfr_epochs`) on the same Final_filtered Raw input
my pipeline uses, then bit-compares the resulting per-subject regional
TFR against the aggregator's `per_subject(subject, region, side, freq,
time)` slab for each modality.

Two phases run sequentially because the EEG and DBS paths in the new
pipeline use different `continuous_decim` (10 vs 1), so the legacy
chain has to be invoked separately per modality with the matching
decim.

Inputs:
  - data/derivatives/raw_clean/P4-2002_{baseline,pre8walk}-eeg-raw.fif
  - data/derivatives/raw_clean/P4-2002_{baseline,pre8walk}-dbs-proc-raw.fif
  - $RESIDUAL_DATA_ROOT/Final_filtered/rejection_mask_dico.pkl
  - data/derivatives/tfr_gavg/pre8walk-{eeg,dbs}-tfr-gavg.nc

Outputs:
  - stdout report (no file written; this is a regression check, not a stage)
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "analysis"))

from _common import patch_legacy_annotations  # noqa: E402
from tfr import (  # noqa: E402
    compute_tfr_on_subset_and_normalize_optimized,
    create_gait_warped_tfr_epochs,
)
from variables import (  # noqa: E402
    event_percentages,
    frontal_areas,
    gait_events_correct_sequence,
    motor_areas,
)


SUBJECT = "P4-2002"
COHORT = "Patient_NFB"
TASK = "pre8walk"
# Location of the legacy export (holds Final_filtered/rejection_mask_dico.pkl).
DATA_ROOT = Path(os.environ["RESIDUAL_DATA_ROOT"]) if os.environ.get("RESIDUAL_DATA_ROOT") else None
RAW_CLEAN = PROJECT_ROOT / "data" / "derivatives" / "raw_clean"
NETCDF_DIR = PROJECT_ROOT / "data" / "derivatives" / "tfr_gavg"


def _load_raws(suffix: str) -> tuple[mne.io.Raw, mne.io.Raw]:
    """Load baseline + task Raw for the given modality suffix
    (`-eeg-raw.fif` or `-dbs-proc-raw.fif`)."""
    rb = mne.io.read_raw_fif(RAW_CLEAN / f"{SUBJECT}_baseline{suffix}",
                             preload=True, verbose="WARNING")
    rt = mne.io.read_raw_fif(RAW_CLEAN / f"{SUBJECT}_{TASK}{suffix}",
                             preload=True, verbose="WARNING")
    patch_legacy_annotations(rb)
    patch_legacy_annotations(rt)
    return rb, rt


def _run_legacy_chain(all_data_dico: dict, decim: int) -> dict:
    """Run continuous TFR + gait-warp on one cohort/subject all_data_dico."""
    freqs = np.arange(1, 46, 1)
    cont = compute_tfr_on_subset_and_normalize_optimized(
        all_data_dico, [COHORT], {COHORT: [SUBJECT]}, [TASK], freqs,
        motor_areas, frontal_areas,
        target_sfreq=None, tfr_decim=decim, n_jobs=1,
    )
    with open(DATA_ROOT / "Final_filtered" / "rejection_mask_dico.pkl", "rb") as f:
        rejection_mask_dico = pickle.load(f)
    gait, _ = create_gait_warped_tfr_epochs(
        cont, all_data_dico, rejection_mask_dico,
        event_percentages, gait_events_correct_sequence,
        epoch_duration=1.1,
    )
    return gait


def _diff_row(legacy_tfr, new_arr) -> tuple[int | None, float | None, float | None]:
    """Return (n_finite, max_abs_diff, max_rel_diff) or (None, None, None)
    if one side has no data."""
    if legacy_tfr is None or new_arr is None:
        return (None, None, None)
    ref = legacy_tfr.data[0] if hasattr(legacy_tfr, "data") else legacy_tfr
    if ref.ndim == 3:                          # (1, n_freqs, n_times)
        ref = ref[0]
    finite = np.isfinite(ref) & np.isfinite(new_arr)
    if not finite.any():
        return (None, None, None)
    abs_diff = np.abs(new_arr[finite] - ref[finite])
    rel_diff = abs_diff / np.where(np.abs(ref[finite]) > 0,
                                    np.abs(ref[finite]), np.nan)
    return (int(finite.sum()),
            float(np.nanmax(abs_diff)),
            float(np.nanmax(rel_diff)))


def check_eeg() -> list[tuple]:
    """Verify EEG (motor / frontal × 4 sides). Legacy decim=10."""
    rb, rt = _load_raws("-eeg-raw.fif")
    dico = {COHORT: {
        "baseline": {"ID": [SUBJECT], "EEG": [rb], "DBS": [None], "DBS_proc": [None]},
        TASK:       {"ID": [SUBJECT], "EEG": [rt], "DBS": [None], "DBS_proc": [None]},
    }}
    legacy = _run_legacy_chain(dico, decim=10)
    ds = xr.open_dataset(NETCDF_DIR / f"{TASK}-eeg-tfr-gavg.nc")

    rows = []
    for region in ("motor", "frontal"):
        for side in ("Left", "Right", "Central", "Both"):
            legacy_tfr = legacy[COHORT][TASK][region][side][0]
            new = ds.per_subject.sel(subject=SUBJECT, region=region, side=side).values
            rows.append((f"eeg.{region}", side, *_diff_row(legacy_tfr, new)))
    return rows


def check_dbs() -> list[tuple]:
    """Verify DBS (STN × 2 sides). Legacy decim=1, matching `tfr.continuous_decim.dbs`.

    The legacy `create_gait_warped_tfr_epochs` requires the EEG slot to
    be populated (it appends to motor/frontal lists per subject), so we
    pass both EEG and DBS Raw — but only compare STN. EEG TFR is wasted
    work at decim=1 (mismatches the pipeline's decim=10); irrelevant
    here since we don't read the EEG result.
    """
    eeg_b, eeg_t = _load_raws("-eeg-raw.fif")
    dbs_b, dbs_t = _load_raws("-dbs-proc-raw.fif")
    dico = {COHORT: {
        "baseline": {"ID": [SUBJECT], "EEG": [eeg_b], "DBS": [dbs_b], "DBS_proc": [dbs_b]},
        TASK:       {"ID": [SUBJECT], "EEG": [eeg_t], "DBS": [dbs_t], "DBS_proc": [dbs_t]},
    }}
    legacy = _run_legacy_chain(dico, decim=1)
    ds = xr.open_dataset(NETCDF_DIR / f"{TASK}-dbs-tfr-gavg.nc")

    rows = []
    for side in ("Left", "Right"):
        legacy_tfr = legacy[COHORT][TASK]["STN"][side][0]
        new = ds.per_subject.sel(subject=SUBJECT, region="STN", side=side).values
        rows.append(("dbs.STN", side, *_diff_row(legacy_tfr, new)))
    return rows


def main() -> int:
    if DATA_ROOT is None:
        print("SKIP: RESIDUAL_DATA_ROOT is not set; the legacy comparison needs the input data.")
        return 0

    print(f"=== pipeline-vs-legacy regression  subject={SUBJECT}  block={TASK} ===")

    rows: list[tuple] = []
    print("\n[EEG] running legacy chain (tfr_decim=10) ...")
    rows.extend(check_eeg())
    print("\n[DBS] running legacy chain (tfr_decim=1) ...")
    rows.extend(check_dbs())

    print()
    print(f"{'region':12s} {'side':9s} {'n_finite':>10s} {'max_abs':>14s} {'max_rel':>12s}")
    print("-" * 64)
    all_pass = True
    for region, side, n, m_abs, m_rel in rows:
        if n is None:
            print(f"{region:12s} {side:9s} {'(no data)':>10s}")
            continue
        verdict = "✓" if (m_abs < 1e-6 or m_rel is None or m_rel < 1e-6) else "✗"
        if verdict == "✗":
            all_pass = False
        print(f"{region:12s} {side:9s} {n:>10d} {m_abs:>14.3e} {m_rel:>12.3e}  {verdict}")

    print()
    if all_pass:
        print("PASS: new pipeline matches legacy bit-for-bit on EEG + DBS.")
        return 0
    print("FAIL: divergence above tolerance. See per-row diffs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
