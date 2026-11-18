#!/usr/bin/env bash
# Purpose:  Rerun the group-level statistics that read the gait events, cycle
#           masks, gERD tables or gait TFRs of the event-corrected subjects
#           (config.yaml:gait_event_corrections), with a new date tag so the
#           October 2026 outputs stay on disk.
# Inputs:   data/derivatives/gerd/master_eeg_{basenorm,avggait}.csv,
#           data/derivatives/tfr_gavg/pre8walk-eeg-tfr-gavg{,-avggait}.nc,
#           data/derivatives/raw_clean/*, data/derivatives/autoreject/*
#           (all after rerun_event_corrected_subjects.sh)
# Outputs:  outputs/stats/lmm_cortical_<date>.txt (submitted models)
#           outputs/stats/revision_2026-10/*_<date>.* (revision analyses)
#           outputs/tables/trial_counts.csv (previous version kept as
#           trial_counts_before_eventfix.csv), log in revision_2026-10/
# Author:   Lena
# Date:     2026-11-16
# Seed:     42, set inside every called script.
#
# Usage (project root, `residual_activity` conda env, R on PATH):
#   REVISION_DATE=2026-11-16 bash code/analysis/revision/rerun_after_eventfix.sh P4-1018 P4-1019

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"
export REVISION_DATE="${REVISION_DATE:?set REVISION_DATE, e.g. 2026-11-16}"
PY="${PYTHON:-python}"
R=code/analysis/revision
OUT=outputs/stats/revision_2026-10
step() { echo; echo "=== $1  $(date '+%F %T')"; }

step "submitted cortical LMMs (basenorm + avggait)"
bash code/analysis/R/run_lmm.sh > /dev/null
( cd code/analysis/R && Rscript stats_lmm_cortical_complex_final.R ) \
  > "outputs/stats/lmm_cortical_${REVISION_DATE}.txt" 2>&1 || echo "FAILED: submitted cortical LMM"

for NORM in basenorm avggait; do
  step "random-slope and aggregated models ($NORM)"
  Rscript $R/lmm_random_slopes.R $NORM || echo "FAILED: lmm_random_slopes $NORM"
  Rscript $R/lmm_aggregated.R    $NORM || echo "FAILED: lmm_aggregated $NORM"
  Rscript $R/lmm_refit_tables.R  $NORM || echo "FAILED: lmm_refit_tables $NORM"
done
step "classic repeated-measures ANOVA"
Rscript $R/rm_anova_classic.R || echo "FAILED: rm_anova_classic"

step "unwarped walking PSDs of the corrected subjects, FOOOF, band power"
for S in "$@"; do
  $PY $R/psd_unwarped.py --subject "$S" --block pre8walk --modality eeg --force || echo "FAILED: psd_unwarped $S"
done
$PY $R/psd_unwarped.py --all            || echo "FAILED: psd_unwarped --all"
$PY $R/fooof_refit.py                   || echo "FAILED: fooof_refit"
$PY $R/fooof_detection_rates.py         || echo "FAILED: fooof_detection_rates"
$PY $R/band_power_above_aperiodic.py    || echo "FAILED: band_power_above_aperiodic"
Rscript $R/psd_mixed_anova.R            || echo "FAILED: psd_mixed_anova"

step "gait kinematics from the gait events"
$PY $R/gait_kinematics_table.py --date "$REVISION_DATE" || echo "FAILED: gait_kinematics_table"

step "trial counts"
[[ -f outputs/tables/trial_counts_before_eventfix.csv ]] || cp -p outputs/tables/trial_counts.csv outputs/tables/trial_counts_before_eventfix.csv
$PY code/analysis/build_trial_counts_table.py --force > "$OUT/trial_counts_${REVISION_DATE}.txt" 2>&1 || echo "FAILED: trial counts"

step "cluster-based permutation tests"
$PY $R/cluster_perm_revision.py --date "$REVISION_DATE" || echo "FAILED: cluster_perm_revision"

step "done"
