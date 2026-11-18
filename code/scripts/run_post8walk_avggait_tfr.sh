#!/usr/bin/env bash
# Build gait-warped avggait TFR grand-average figures for the post8walk block.
#
# Inputs:  server post8walk_proc.fif per subject ($RESIDUAL_DATA_ROOT mount;
#          P4-2012 has none — skipped), local baseline raw_clean fifs
# Outputs: data/derivatives/{raw_clean,epochs,autoreject,tfr}/  (post8walk)
#          data/derivatives/tfr_gavg/post8walk-{eeg,dbs}-tfr-gavg-avggait.nc
#          outputs/figures/legacy_style/  post8walk avggait gERSP figures
#
# Chain per subject (mirrors the documented pre8walk chain in
# docs/pipeline.md; post8walk is an `extra_block`, so every call is
# per-subject — no --all sweeps):
#   build_raw_per_subject (--no-resample-eeg to match the 2000 Hz
#   pickle-era convention of the existing pre8walk derivatives)
#   -> compute_tfr_continuous (baseline + post8walk; baseline cont TFRs
#      were deleted after the May run, so regenerate)
#   -> compute_tfr_basenorm -> build_epochs_from_raw
#   -> compute_autoreject_mask -> compute_tfr_gait -> apply_avggait_norm
# The multi-GB cont/basenorm intermediates are deleted per subject after
# the gait-warp (disk would otherwise reach ~300 GB across the cohort).
#
# Usage: bash code/scripts/run_post8walk_avggait_tfr.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Python interpreter: activate the project environment, or set PYTHON.
PY="${PYTHON:-python}"
# Headless-safe plotting: plot_tfr_grandavg_legacy.py ends in plt.show(),
# which blocks forever on the interactive macosx backend when run
# unattended. Agg makes show() a no-op; files are savefig'd regardless.
export MPLBACKEND=Agg
phase() { echo ""; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 ==="; }
substep() { echo "--- $(date -u +%H:%M:%S) $1 ---"; }

# P4-2012 excluded: no post8walk_proc.fif on the server (QA 2026-08-14).
SUBJECTS_PD="P4-2002 P4-2003 P4-2004 P4-2005 P4-2007 P4-2008 P4-2009 P4-2010 P4-2011"
SUBJECTS_HC="P4-1005 P4-1010 P4-1011 P4-1012 P4-1013 P4-1014 P4-1015 P4-1017 P4-1018 P4-1019"

BLOCK=post8walk
TFRDIR="$PROJECT_ROOT/data/derivatives/tfr"
FAILED=""

run_subject() {
  local S=$1 COHORT=$2
  local MODALITIES="eeg"
  [ "$COHORT" = "PD" ] && MODALITIES="eeg dbs"

  substep "ingest $S $BLOCK (server)"
  "$PY" code/preprocessing/build_raw_per_subject.py \
      --subject "$S" --block "$BLOCK" --no-resample-eeg || return 1

  for M in $MODALITIES; do
    substep "tfr_continuous $S baseline $M"
    "$PY" code/analysis/compute_tfr_continuous.py \
        --subject "$S" --block baseline --modality "$M" || return 1
    substep "tfr_continuous $S $BLOCK $M"
    "$PY" code/analysis/compute_tfr_continuous.py \
        --subject "$S" --block "$BLOCK" --modality "$M" || return 1
    substep "tfr_basenorm $S $BLOCK $M"
    "$PY" code/analysis/compute_tfr_basenorm.py \
        --subject "$S" --block "$BLOCK" --modality "$M" || return 1
  done

  substep "epochs $S $BLOCK"
  "$PY" code/analysis/build_epochs_from_raw.py \
      --subject "$S" --block "$BLOCK" || return 1
  substep "autoreject mask $S $BLOCK"
  "$PY" code/analysis/compute_autoreject_mask.py \
      --subject "$S" --block "$BLOCK" || return 1

  for M in $MODALITIES; do
    substep "tfr_gait $S $BLOCK $M"
    "$PY" code/analysis/compute_tfr_gait.py \
        --subject "$S" --block "$BLOCK" --modality "$M" || return 1
    substep "avggait $S $BLOCK $M"
    "$PY" code/analysis/apply_avggait_norm.py \
        --subject "$S" --block "$BLOCK" --modality "$M" || return 1
  done
  return 0
}

cleanup_subject() {
  # Drop only this run's multi-GB intermediates (baseline cont TFRs did
  # not exist before this run; the small -tfr-gait* outputs are kept).
  local S=$1
  rm -f "$TFRDIR/${S}_baseline-"*"-tfr-cont.h5" \
        "$TFRDIR/${S}_baseline-"*"-tfr-cont.h5.meta.json" \
        "$TFRDIR/${S}_${BLOCK}-"*"-tfr-cont.h5" \
        "$TFRDIR/${S}_${BLOCK}-"*"-tfr-cont.h5.meta.json" \
        "$TFRDIR/${S}_${BLOCK}-"*"-tfr-cont-basenorm.h5" \
        "$TFRDIR/${S}_${BLOCK}-"*"-tfr-cont-basenorm.h5.meta.json"
}

phase "Phase 1: per-subject chain (PD: eeg+dbs, HC: eeg)"
for S in $SUBJECTS_PD; do
  phase "subject $S (PD)"
  if run_subject "$S" PD; then cleanup_subject "$S"; else
    echo "!!! $S FAILED — intermediates kept for debugging"; FAILED="$FAILED $S"
  fi
done
for S in $SUBJECTS_HC; do
  phase "subject $S (HC)"
  if run_subject "$S" HC; then cleanup_subject "$S"; else
    echo "!!! $S FAILED — intermediates kept for debugging"; FAILED="$FAILED $S"
  fi
done

phase "Phase 2: cohort grand averages (avggait)"
"$PY" code/analysis/aggregate_tfr_gavg.py --block "$BLOCK" --modality eeg --norm avggait
"$PY" code/analysis/aggregate_tfr_gavg.py --block "$BLOCK" --modality dbs --norm avggait

phase "Phase 3: figures (manuscript avggait style, +/-2 dB, median)"
"$PY" code/figures/plot_tfr_grandavg_legacy.py \
    --block "$BLOCK" --norm avggait --method median --scale -2 2 --ipsicontra

phase "DONE"
[ -n "$FAILED" ] && echo "Failed subjects:$FAILED"
df -h "$PROJECT_ROOT" | tail -1
