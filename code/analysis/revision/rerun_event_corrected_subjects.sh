#!/usr/bin/env bash
# Purpose:  Re-derive the per-subject cortical chain for the subjects listed in
#           config.yaml:gait_event_corrections after their gait events were
#           corrected, then rebuild the cohort-level EEG aggregates.
# Inputs:   data/derivatives/raw_clean/<subj>_{baseline,pre8walk}-eeg-raw.fif
#           (pre8walk with corrected annotations, see correct_gait_events.py)
#           code/analysis/config.yaml
# Outputs:  data/derivatives/{tfr,epochs,autoreject,gerd}/<subj>_pre8walk-*
#           data/derivatives/gerd/master_eeg_{basenorm,avggait}.csv
#           data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg{,-avggait}.nc
#           (all overwritten; the previous versions are kept under
#           data/derivatives/_before_eventfix_2026-11-16/)
# Author:   Lena
# Date:     2026-11-16
# Seed:     config.yaml:repro.seed (42), set inside every called script.
#
# Usage (from the project root, in the `residual_activity` conda env):
#   python code/analysis/revision/correct_gait_events.py
#   bash code/analysis/revision/rerun_event_corrected_subjects.sh P4-1018 P4-1019
#
# The chain is the one in docs/pipeline.md ("Per-subject derivative chain"),
# with --force on every stage that depends on the gait events. The continuous
# TFR does not depend on the events; it is recomputed only because the
# intermediate files are not kept on disk.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"
A=code/analysis
PY="${PYTHON:-python}"

if [[ $# -lt 1 ]]; then echo "usage: $0 <subject> [<subject> ...]" >&2; exit 2; fi

for S in "$@"; do
  echo "=== $S  $(date '+%F %T')"
  $PY $A/compute_tfr_continuous.py --subject "$S" --block baseline --modality eeg --n-jobs 1
  $PY $A/compute_tfr_continuous.py --subject "$S" --block pre8walk --modality eeg --n-jobs 1
  $PY $A/compute_tfr_basenorm.py   --subject "$S" --block pre8walk --modality eeg --force
  $PY $A/build_epochs_from_raw.py  --subject "$S" --block pre8walk --force
  $PY $A/compute_autoreject_mask.py --subject "$S" --block pre8walk --force
  $PY $A/compute_tfr_gait.py       --subject "$S" --block pre8walk --modality eeg --force
  $PY $A/apply_avggait_norm.py     --subject "$S" --block pre8walk --modality eeg --force
  for NORM in basenorm avggait; do
    $PY $A/compute_gerd_table.py   --subject "$S" --block pre8walk --modality eeg --norm "$NORM" --force
  done
done

echo "=== aggregates  $(date '+%F %T')"
$PY $A/aggregate_gerd.py --force
for NORM in basenorm avggait; do
  $PY $A/aggregate_tfr_gavg.py --block pre8walk --modality eeg --norm "$NORM" --force
done
echo "=== done  $(date '+%F %T')"
