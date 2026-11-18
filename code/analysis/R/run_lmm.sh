#!/usr/bin/env bash
# Wires the master gERD/ERS CSVs and the UPDRS scores into the
# `Data/` subdirectory the R LMM scripts hardcode, then invokes Rscript.
#
# Run from the project root — the script cd's into code/analysis/R/
# since the legacy R scripts use relative paths like
# `Data/STN_Staandard_PD_subj_epochs_phase_median_<datatype>.csv`.
#
# Usage:
#   export RESIDUAL_DATA_ROOT=/path/to/data
#   bash code/analysis/R/run_lmm.sh
#
# Required inputs (the script verifies these and exits cleanly if missing):
#   - data/derivatives/gerd/master_eeg_basenorm.csv  (aggregate_gerd.py)
#   - data/derivatives/gerd/master_dbs_basenorm.csv  (aggregate_gerd.py)
#   - $RESIDUAL_DATA_ROOT/Patient_NFB/UPDRS_scores_USZ.xlsx
#
# The avggait variants are referenced by the R script's datatype loop;
# if absent, the loop skips them with a warning.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT/code/analysis/R"

DATA="Data"
mkdir -p "$DATA"

GERD="$PROJECT_ROOT/data/derivatives/gerd"

# Symlink masters into the `Data/` names the R scripts expect.
for norm in basenorm avggait; do
  ln -sf "$GERD/master_eeg_${norm}.csv" \
         "$DATA/for_R_EEG_epochs_PD-vs-HC_phase_median_${norm}.csv"
  ln -sf "$GERD/master_dbs_${norm}.csv" \
         "$DATA/STN_Staandard_PD_subj_epochs_phase_median_${norm}.csv"
done
for f in "$GERD/master_eeg_basenorm.csv" "$GERD/master_dbs_basenorm.csv" \
         "$GERD/master_eeg_avggait.csv" "$GERD/master_dbs_avggait.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f — run aggregate_gerd.py first" >&2
    exit 2
  fi
done

# UPDRS. The R scripts only need a (SubjectID, UPDRS) sheet to satisfy the
# merge — the final model does not use UPDRS — so the local stub is the
# canonical input (the server UPDRS_scores_USZ.xlsx has an incompatible
# multi-header layout that breaks read_excel/merge; verified 2026-08-14).
UPDRS_STUB="$DATA/UPDRS_scores_USZ_finalscoreonly_stub.xlsx"
if [[ -f "$UPDRS_STUB" ]]; then
  ln -sf "$(basename "$UPDRS_STUB")" "$DATA/UPDRS_scores_USZ_finalscoreonly.xlsx"
  echo "UPDRS source: $UPDRS_STUB (local stub)"
else
  echo "WARN: UPDRS stub not at $UPDRS_STUB — STN LMM will fail at read_excel." >&2
fi

echo "---"
echo "Data/ wiring complete:"
ls -la "$DATA"
echo "---"
echo "Run the R LMMs (output is verbose):"
echo "  Rscript stats_lmm_cortical_complex_final.R"
echo "  Rscript stats_lmm_STN_complex_final.R"
echo "  Rscript stats_LMM_STN_ipsicontra_final.R"
