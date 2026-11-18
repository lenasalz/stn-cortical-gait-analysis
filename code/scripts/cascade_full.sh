#!/usr/bin/env bash
# Full decim=2 cascade with streaming cleanup of intermediates.
#
# Inputs:  data/derivatives/raw_clean/*-raw.fif  (build_raw_from_pickle.py)
# Outputs:
#   Kept:    data/derivatives/tfr/*-tfr-gait*.h5  (small, input to figures)
#            data/derivatives/gerd/*.csv         (input to R LMM)
#            data/derivatives/tfr_gavg/*.nc      (input to figures)
#            outputs/stats/lmm_*_<date>.txt      (manuscript tables)
#   Deleted: tfr-cont, tfr-cont-basenorm (per subject, after consumption)
#
# Usage (from project root):
#   caffeinate -i bash code/scripts/cascade_full.sh
#
# Memory note: EEG TFR-cont at decim=2 produces ~3-4 GB per file; basenorm
# loads both baseline+task (~14 GB peak). Subject-by-subject loop keeps
# disk peak <10 GB and lets the next subject start with a clean slate.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${RESIDUAL_DATA_ROOT:?set RESIDUAL_DATA_ROOT to the input data location}"
export RESIDUAL_DATA_ROOT

# Python interpreter: activate the project environment, or set PYTHON.
PY="${PYTHON:-python}"
TFR_DIR=data/derivatives/tfr

phase() { echo ""; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 ==="; }
substep() { echo "--- $(date -u +%H:%M:%S) $1 ---"; }

# Roster: anonymized codes only (project rule).
SUBJECTS_PD="P4-2002 P4-2003 P4-2004 P4-2005 P4-2007 P4-2008 P4-2009 P4-2010 P4-2011 P4-2012"
SUBJECTS_HC="P4-1005 P4-1010 P4-1011 P4-1012 P4-1013 P4-1014 P4-1015 P4-1017 P4-1018 P4-1019"
ALL_SUBJECTS="$SUBJECTS_PD $SUBJECTS_HC"

phase "Cascade start (config: decim_eeg=2, decim_dbs=1)"
df -h "$PROJECT_ROOT" | tail -1

# ---------- Per-subject loop: TFR-cont → basenorm → gait-warp → gERD → cleanup ----------
for S in $ALL_SUBJECTS; do
  phase "Subject $S"

  # ----- 1a. TFR-cont EEG (baseline + pre8walk) -----
  for B in baseline pre8walk; do
    substep "TFR-cont EEG $S $B"
    "$PY" code/analysis/compute_tfr_continuous.py \
        --subject "$S" --block "$B" --modality eeg --force
  done

  # ----- 1b. TFR-cont DBS (baseline + pre8walk) -----
  for B in baseline pre8walk; do
    substep "TFR-cont DBS $S $B"
    "$PY" code/analysis/compute_tfr_continuous.py \
        --subject "$S" --block "$B" --modality dbs --force || \
        echo "  (skipped: no DBS for $S $B)"
  done

  # ----- 2. basenorm (pre8walk only) -----
  for M in eeg dbs; do
    substep "basenorm $M $S pre8walk"
    "$PY" code/analysis/compute_tfr_basenorm.py \
        --subject "$S" --block pre8walk --modality "$M" --force || \
        echo "  (skipped: missing input for $S $M)"
  done

  # ----- CLEANUP A: delete tfr-cont, keep basenorm -----
  substep "delete tfr-cont for $S"
  rm -f "$TFR_DIR/${S}_baseline-eeg-tfr-cont.h5"{,.meta.json}
  rm -f "$TFR_DIR/${S}_pre8walk-eeg-tfr-cont.h5"{,.meta.json}
  rm -f "$TFR_DIR/${S}_baseline-dbs-tfr-cont.h5"{,.meta.json}
  rm -f "$TFR_DIR/${S}_pre8walk-dbs-tfr-cont.h5"{,.meta.json}

  # ----- 3. gait-warp (compute_tfr_gait → tfr-gait.h5) -----
  for M in eeg dbs; do
    substep "gait-warp $M $S pre8walk"
    "$PY" code/analysis/compute_tfr_gait.py \
        --subject "$S" --block pre8walk --modality "$M" --force || \
        echo "  (skipped: missing basenorm for $S $M)"
  done

  # ----- 4. avggait normalization -----
  for M in eeg dbs; do
    substep "avggait $M $S pre8walk"
    "$PY" code/analysis/apply_avggait_norm.py \
        --subject "$S" --block pre8walk --modality "$M" --force || \
        echo "  (skipped: missing tfr-gait for $S $M)"
  done

  # ----- 5. gERD CSV -----
  for M in eeg dbs; do
    substep "gERD $M $S pre8walk"
    "$PY" code/analysis/compute_gerd_table.py \
        --subject "$S" --block pre8walk --modality "$M" --force || \
        echo "  (skipped: missing basenorm for $S $M)"
  done

  # ----- CLEANUP B: delete basenorm (all downstream done) -----
  substep "delete basenorm for $S"
  rm -f "$TFR_DIR/${S}_pre8walk-eeg-tfr-cont-basenorm.h5"{,.meta.json}
  rm -f "$TFR_DIR/${S}_pre8walk-dbs-tfr-cont-basenorm.h5"{,.meta.json}

  df -h "$PROJECT_ROOT" | tail -1
done

# ---------- Cohort-level steps ----------
phase "Aggregate gERD masters"
"$PY" code/analysis/aggregate_gerd.py

phase "Aggregate TFR grand averages (uses tfr-gait, which we kept)"
"$PY" code/analysis/aggregate_tfr_gavg.py || true

phase "R LMMs (cortical + STN)"
bash code/analysis/R/run_lmm.sh
mkdir -p outputs/stats
RDATE=$(date +%Y-%m-%d)
( cd code/analysis/R && Rscript stats_lmm_cortical_complex_final.R ) \
  > "outputs/stats/lmm_cortical_${RDATE}.txt" 2>&1 || true
( cd code/analysis/R && Rscript stats_lmm_STN_complex_final.R ) \
  > "outputs/stats/lmm_subcortical_${RDATE}.txt" 2>&1 || true

phase "Figures"
"$PY" code/figures/plot_tfr_grandavg_legacy.py || true
"$PY" code/figures/plot_gerd_bars.py || true

phase "=== DONE ==="
df -h "$PROJECT_ROOT" | tail -1
du -sh data/derivatives/tfr/ data/derivatives/gerd/ data/derivatives/tfr_gavg/ 2>/dev/null
