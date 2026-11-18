#!/usr/bin/env bash
# Rebuild PSD + FOOOF pipeline from the post-C2/FC2-fix raw_clean.
#
# Inputs: data/derivatives/raw_clean/*-raw.fif (post-C2/FC2-fix)
# Outputs: data/derivatives/{epochs,psd,fooof}/  +  outputs/stats/psd_fooof_<date>.{txt,csv}
#
# Usage: bash code/scripts/rerun_psd_fooof.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Python interpreter: activate the project environment, or set PYTHON.
PY="${PYTHON:-python}"
phase() { echo ""; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 ==="; }
substep() { echo "--- $(date -u +%H:%M:%S) $1 ---"; }

SUBJECTS_PD="P4-2002 P4-2003 P4-2004 P4-2005 P4-2007 P4-2008 P4-2009 P4-2010 P4-2011 P4-2012"
SUBJECTS_HC="P4-1005 P4-1010 P4-1011 P4-1012 P4-1013 P4-1014 P4-1015 P4-1017 P4-1018 P4-1019"
ALL="$SUBJECTS_PD $SUBJECTS_HC"

phase "Phase 1: rebuild epochs from raw_clean (force)"
for S in $ALL; do
  for B in baseline pre8walk; do
    substep "epochs $S $B"
    "$PY" code/analysis/build_epochs_from_raw.py \
        --subject "$S" --block "$B" --force || \
        echo "  (skipped: $S $B)"
  done
done

phase "Phase 2: compute PSD (force, all subjects via run_all.py)"
"$PY" code/analysis/run_all.py --stage psd --n-jobs 4 --force

phase "Phase 3: compute FOOOF (force, all subjects via run_all.py)"
"$PY" code/analysis/run_all.py --stage fooof --n-jobs 4 --force

phase "Phase 4: aggregate FOOOF master"
"$PY" code/analysis/aggregate_fooof.py --force

phase "Phase 5: PSD + FOOOF statistics"
RDATE=$(date +%Y-%m-%d)
"$PY" code/analysis/run_psd_stats.py \
  > "outputs/stats/psd_fooof_${RDATE}.txt" 2>&1 || true
"$PY" code/analysis/_run_fooof_statistics.py \
  > "outputs/stats/psd_fooof_${RDATE}_fooof.txt" 2>&1 || true

phase "=== DONE ==="
ls -la "outputs/stats/psd_fooof_${RDATE}"*
