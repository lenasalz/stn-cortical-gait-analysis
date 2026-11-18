# Revision analyses (post npj Parkinson's Disease review, October 2026)

Scripts in this folder answer the reviewer comments listed in
`manuscript/revisions/2026-10_npjPD/analysis_log.md`. They are kept apart
from the submitted pipeline so that the submitted numbers stay reproducible
from the scripts in `code/analysis/` (tag `npjPD-submission-2026-09`).

Rules for this folder:

1. Never edit a submitted script to produce a revision result. Write a new
   script here; if it reuses functions, import them from `code/analysis/`.
2. Every script carries the standard header (Purpose, Inputs, Outputs,
   Author, Date, Seed) and sets the seed from `config.yaml:repro.seed`.
3. Read only the frozen data version listed in
   `docs/data_manifest_revision_2026-10.md` (the pickle-era `raw_clean`
   export and the derivatives built from it). Do not read the server
   `_proc.fif` files for revision results; they are a later preprocessing
   export and do not reproduce the submitted numbers.
4. Outputs go to `outputs/stats/revision_2026-10/` (tables, logs) or
   `outputs/figures/revision_2026-10/` (figures), with `.meta.json`
   sidecars written by `_common.write_sidecar`.
5. After every run, add or update the row in `analysis_log.md`: comment
   number, script, inputs, output file, one-line result, manuscript
   location.

Planned scripts (file names are fixed so the log can reference them now):

| Script | Comment | Purpose |
|---|---|---|
| `build_data_manifest.py` | all | Freeze the analysed data version (checksums). |
| `lmm_random_slopes.R` | 4, 5, 15 | Cortical and STN models with by-subject random slopes; three-way term; effect sizes; variance components. |
| `lmm_aggregated.R` | 4 | One value per subject x phase x region; repeated-measures model. |
| `psd_unwarped.py` | 11 | Walking PSDs from unwarped straight-walking data. |
| `fooof_refit.py` | 2, 3 | FOOOF on unwarped PSDs; fit-quality table; peak-height sensitivity. |
| `fooof_detection_rates.py` | 2 | Peak-detection counts per group, condition, band. |
| `band_power_above_aperiodic.py` | 2 | Band power above the aperiodic fit (no peak required). |
| `psd_mixed_anova.R` | 10, 14 | Cohort x Condition model per parameter and ROI. |
| `gait_kinematics_table.py` | 8, 11 | Stride time, cadence, speed, stride length per group. |
