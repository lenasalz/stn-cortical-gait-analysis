"""Per-(subject, block) per-cycle gERD/ERS table in *absolute dB* — Guo-style.

Inputs:
  - data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-cont.h5     (raw power)
  - data/derivatives/raw_clean/<subj>_<block>-{eeg,dbs-proc}-raw.fif
  - data/derivatives/autoreject/<subj>_<block>-mask.npz
  - code/analysis/config.yaml (gait_phases, freq_bands, channels_of_interest)
Outputs:
  - data/derivatives/gerd/<subj>_<block>-{eeg,dbs}-guo-gerd.csv
  - data/derivatives/gerd/<subj>_<block>-{eeg,dbs}-guo-gerd.csv.meta.json
Description:
  Mirror of `process_and_save_tfr_data_gait_cycle_guo` from
  ngb_study/notebooks/analysis/scripts/continuous_time_freq_epoching_whole_gait_cycle.py:
    1. Read the continuous task TFR (raw power, NOT baseline-normalized).
    2. Convert to absolute dB: `data = 10 * log10(power + eps)`.
    3. Gait-warp per cycle (HSR → HSR_end), resample to fixed 1.1s length.
    4. Apply the per-subject autoreject mask.
    5. Per (cycle, area, hemisphere, gait_phase, band): emit the median.

  The per-cycle ABSOLUTE dB is what Aline's GaitDataExporter writes for
  avggait analysis in the legacy notebook (cell 49 of analysis.ipynb,
  consuming all_epochs_guo). Compared to compute_gerd_table.py --norm
  avggait, this path skips the subject-mean subtraction; the resulting
  cycle-level variance is what the legacy LMM was fit on.

  This script exists purely to reproduce Aline's pasted Tab. 10/11
  numbers. It is NOT the manuscript-correct pipeline (the manuscript
  says "expressed as decibel change"), so its output should not feed
  into the canonical revision; it is a diagnostic.

Usage:
  python code/analysis/compute_gerd_table_guo.py --subject P4-2002 --block pre8walk --modality eeg
"""
from __future__ import annotations

import argparse
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
# Reuse the helpers — same gait-warp, same per-cycle extraction.
from compute_gerd_table import (
    _emit_rows_dbs,
    _emit_rows_eeg,
    _per_cycle_per_channel_tfrs,
)
from variables import (
    event_percentages as DEFAULT_EVENT_PERCENTAGES,
    gait_events_correct_sequence as DEFAULT_GAIT_SEQ,
)

EPS = 1e-12   # legacy compute_tfr_on_subset eps (uses np.maximum(power, eps))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--modality", choices=["eeg", "dbs"], default="eeg")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.block == "baseline":
        parser.error("baseline has no gait events; this script is for task blocks")

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    # Inputs
    tfr_in = output_path(paths.tfr_dir, subject, block,
                         f"-{args.modality}-tfr-cont.h5")
    raw_suffix = "-eeg-raw.fif" if args.modality == "eeg" else "-dbs-proc-raw.fif"
    raw_in = output_path(paths.raw_clean_dir, subject, block, raw_suffix)
    mask_npz = (paths.derivatives_root / "autoreject"
                / f"{subject}_{block}-mask.npz")

    out_path = paths.derivatives_root / "gerd" / f"{subject}_{block}-{args.modality}-guo-gerd.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} {args.modality} (exists)")
        return 0
    for p, name in ((tfr_in, "continuous TFR"), (raw_in, "raw")):
        if not p.exists():
            print(f"missing {name}: {p}", file=sys.stderr)
            return 2

    meta = setup_run("compute_gerd_table_guo", config)

    import mne
    import pandas as pd
    from _common import patch_legacy_annotations

    tfr = mne.time_frequency.read_tfrs(tfr_in)
    tfr = tfr[0] if isinstance(tfr, list) else tfr
    # Convert raw power → absolute dB IN PLACE on tfr.data so the downstream
    # helpers operate on dB the same way they would for basenorm. Use the
    # *same* clamp-floor as the legacy compute_tfr_on_subset (np.maximum,
    # eps=1e-12) rather than an additive epsilon, otherwise small-power
    # points compress differently and the LMM diverges.
    tfr.data = 10.0 * np.log10(np.maximum(tfr.data, EPS))

    raw = mne.io.read_raw_fif(raw_in, preload=False, verbose="WARNING")
    patch_legacy_annotations(raw)

    cycle_dicts, per_cycle = _per_cycle_per_channel_tfrs(
        tfr, raw, DEFAULT_GAIT_SEQ, DEFAULT_EVENT_PERCENTAGES
    )
    if per_cycle.shape[0] == 0:
        print(f"no usable cycles for {subject} {block}", file=sys.stderr)
        return 3
    n_cycles = per_cycle.shape[0]

    if mask_npz.exists():
        mask = np.load(mask_npz)["keep_mask"]
        mask_source = "per_subject_npz"
    else:
        mask = None
        mask_source = "missing"
    if mask is None or len(mask) != n_cycles:
        kept_bool = np.ones(n_cycles, dtype=bool)
        rejection_status = ("no_mask_found" if mask is None
                            else f"length_mismatch (source={mask_source})")
    else:
        kept_bool = mask.astype(bool)
        rejection_status = f"applied (source={mask_source})"

    freq_bands = {k: tuple(v) for k, v in config["freq_bands"].items()}
    if args.modality == "eeg":
        from variables import gait_phases as DEFAULT_GAIT_PHASES
        rows = _emit_rows_eeg(per_cycle, kept_bool, tfr.freqs, list(tfr.ch_names),
                              freq_bands, DEFAULT_GAIT_PHASES, config,
                              subject, cohort, block)
    else:
        from variables import gait_phases as DEFAULT_GAIT_PHASES
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
        "modality": args.modality,
        "tfr_input": str(tfr_in),
        "raw_input": str(raw_in),
        "mask_source": mask_source,
        "rejection_status": rejection_status,
        "n_cycles_segmented": int(n_cycles),
        "n_cycles_kept": int(kept_bool.sum()),
        "output": str(out_path),
        "n_rows": int(len(df)),
        "note": "absolute dB per cycle (10*log10(power)); legacy Guo-style; not subject-mean-centered",
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.name}  rows={len(df)}  "
          f"kept={int(kept_bool.sum())}/{n_cycles}  ({rejection_status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
