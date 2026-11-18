# ===================================================================
# lmm_random_slopes.R
# -------------------------------------------------------------------
# Purpose: Reviewer 1, comments 4, 5 and 15 (npj Parkinson's Disease,
#          2026-10-12). Part A reproduces the submitted random-intercept
#          models (cortical: Cohort*Area + Cohort*GaitPhase +
#          (1|SubjectID) + (1|GaitCycle:SubjectID), ML; STN:
#          Hemisphere*GaitPhase + (1|SubjectID), REML as submitted) and
#          prints the unrounded Satterthwaite denominator df of the
#          Cohort / Hemisphere main effect. Part B refits with by-subject
#          random slopes and the full Cohort*Area*GaitPhase fixed
#          structure, simplifying the random-effects structure stepwise
#          when a fit is singular or does not converge. Reports Type III
#          ANOVA (Satterthwaite), partial eta squared (approximate),
#          random-effect variances, convergence/singularity flags, and
#          emmeans PD-HC (STN: Left-Right) contrasts with 95% CIs and
#          FDR-adjusted p. Benjamini-Hochberg across the five bands per
#          term is added in the tidy ANOVA table.
# Inputs:  code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_<datatype>.csv
#          code/analysis/R/Data/STN_Staandard_PD_subj_epochs_phase_median_<datatype>.csv
#          code/analysis/revision/lmm_revision_common.R
# Outputs: outputs/stats/revision_2026-10/lmm_random_slopes_2026-10-13[_<datatype>].txt
#          outputs/stats/revision_2026-10/lmm_submitted_reproduction_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_varcorr_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_contrasts_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_ladder_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_ladder_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_ladder_varcorr_2026-10-13[_<datatype>].csv
#          (suffix omitted for basenorm, the primary normalisation)
# Author:  Lena
# Date:    2026-10-13
# Seed:    set.seed(42)
# Usage:   Rscript code/analysis/revision/lmm_random_slopes.R [basenorm|avggait]
# ===================================================================

set.seed(42)
source("code/analysis/revision/lmm_revision_common.R")

args     <- commandArgs(trailingOnly = TRUE)
datatype <- if (length(args) >= 1) args[1] else "basenorm"
suffix   <- if (datatype == "basenorm") "" else paste0("_", datatype)
outfile  <- function(stem) file.path(OUT_DIR, paste0(stem, "_", REVISION_DATE, suffix))

log_con <- open_log(paste0(outfile("lmm_random_slopes"), ".txt"))
session_header("code/analysis/revision/lmm_random_slopes.R")
cat("datatype:", datatype, "\n\n")

anova_rows <- list(); vc_rows <- list(); ctr_rows <- list(); ladder_rows <- list()
ladder_anova_rows <- list(); ladder_vc_rows <- list()
repro_rows <- list()

# ===================================================================
# Part A: reproduce the submitted models
# ===================================================================
cat("###################################################################\n")
cat("# Part A: reproduction of the submitted random-intercept models\n")
cat("###################################################################\n")

df_c <- load_cortical(datatype)
df_s <- load_stn(datatype)
cat(sprintf("cortical: %d rows, %d subjects (%s)\n", nrow(df_c), nlevels(df_c$SubjectID),
            paste(names(table(unique(df_c[, c("SubjectID", "Cohort")])$Cohort)),
                  table(unique(df_c[, c("SubjectID", "Cohort")])$Cohort), collapse = ", ")))
cat(sprintf("STN: %d rows, %d subjects (%s)\n\n", nrow(df_s), nlevels(df_s$SubjectID),
            paste(names(table(unique(df_s[, c("SubjectID", "Hemisphere")])$Hemisphere)),
                  table(unique(df_s[, c("SubjectID", "Hemisphere")])$Hemisphere), collapse = ", ")))

for (band in BANDS) {
  y <- band_col(band)
  cat("\n--- Submitted cortical model |", band, "|", datatype, "---\n")
  fml <- as.formula(paste(y, "~ Cohort * Area + Cohort * GaitPhase + (1 | SubjectID) + (1 | GaitCycle:SubjectID)"))
  res <- fit_lmer(fml, "df_c", REML = FALSE, control_name = "LMM_CONTROL_SUBMITTED")  # default optimizer as submitted
  m <- res$fit
  a <- anova(m, type = 3, ddf = "Satterthwaite")
  print(a)
  cat(sprintf("Unrounded Cohort DenDF: %.6f   (singular: %s)\n", a["Cohort", "DenDF"], isSingular(m)))
  if (length(res$warnings)) cat("warnings:", paste(res$warnings, collapse = " | "), "\n")
  print(as.data.frame(VarCorr(m))[, c("grp", "vcov", "sdcor")])
  repro_rows[[length(repro_rows) + 1]] <- tidy_anova(
    m, model = "submitted", region = "cortical", datatype = datatype, band = band,
    structure = "(1|SubjectID) + (1|GaitCycle:SubjectID)", REML = FALSE,
    singular = isSingular(m), warnings = paste(res$warnings, collapse = " | "))
}

for (band in BANDS) {
  y <- band_col(band)
  cat("\n--- Submitted STN model |", band, "|", datatype, "---\n")
  fml <- as.formula(paste(y, "~ Hemisphere * GaitPhase + (1 | SubjectID)"))
  res <- fit_lmer(fml, "df_s", REML = TRUE, control_name = "LMM_CONTROL_SUBMITTED")  # REML default as submitted
  m <- res$fit
  a <- anova(m, type = 3, ddf = "Satterthwaite")
  print(a)
  cat(sprintf("Unrounded Hemisphere DenDF: %.6f   (singular: %s)\n", a["Hemisphere", "DenDF"], isSingular(m)))
  if (length(res$warnings)) cat("warnings:", paste(res$warnings, collapse = " | "), "\n")
  print(as.data.frame(VarCorr(m))[, c("grp", "vcov", "sdcor")])
  repro_rows[[length(repro_rows) + 1]] <- tidy_anova(
    m, model = "submitted", region = "STN", datatype = datatype, band = band,
    structure = "(1|SubjectID)", REML = TRUE,
    singular = isSingular(m), warnings = paste(res$warnings, collapse = " | "))
}
repro_tbl <- add_fdr_bands(do.call(rbind, repro_rows))
fwrite(repro_tbl, paste0(outfile("lmm_submitted_reproduction_anova"), ".csv"))

# ===================================================================
# Part B: random-slope models
# ===================================================================
cat("\n\n###################################################################\n")
cat("# Part B: by-subject random slopes, three-way fixed structure\n")
cat("###################################################################\n")

# Ladder of random-effects structures, from maximal to the submitted
# intercept-only structure. The first that converges without a singular
# fit is used. Steps 1-2 carry a by-subject Area x GaitPhase slope, which
# is the structure under which the Cohort x Area x GaitPhase term is
# tested against between-subject variability in modulation (comment 15);
# step 3 is the structure named in the plan, (1 + Area + GaitPhase |
# SubjectID). "Uncorrelated" steps use the explicit 0/1 indicators (see
# lmm_revision_common.R for why `||` with a factor is not used).
ladder_cortical <- list(
  list(label = "max3: (1+Area*GaitPhase|Subj) + (1|Cycle:Subj)",
       re = "(1 + Area * GaitPhase | SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "max3u: uncorrelated Area*GaitPhase slopes + cycle",
       re = "(1 + A_motor + G_IDS + G_LLS + G_RLS + AG_IDS + AG_LLS + AG_RLS || SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "max2: (1+Area+GaitPhase|Subj) + (1|Cycle:Subj)",
       re = "(1 + Area + GaitPhase | SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "a: uncorrelated Area+GaitPhase slopes + cycle",
       re = "(1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "b: uncorrelated Area+GaitPhase slopes, no cycle",
       re = "(1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID)"),
  list(label = "c1: GaitPhase slopes only + cycle",
       re = "(1 + G_IDS + G_LLS + G_RLS || SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "c2: Area slope only + cycle",
       re = "(1 + A_motor || SubjectID) + (1 | GaitCycle:SubjectID)"),
  list(label = "d: intercepts only (submitted structure)",
       re = "(1 | SubjectID) + (1 | GaitCycle:SubjectID)")
)
ladder_stn <- list(
  list(label = "max: (1+GaitPhase|Subj)", re = "(1 + GaitPhase | SubjectID)"),
  list(label = "a: uncorrelated GaitPhase slopes",
       re = "(1 + G_IDS + G_LLS + G_RLS || SubjectID)"),
  list(label = "d: (1|Subj) (submitted structure)", re = "(1 | SubjectID)")
)

for (band in BANDS) {
  y <- band_col(band)
  cat("\n======================================================\n")
  cat("Cortical random-slope model |", band, "|", datatype, "\n")
  cat("fixed: Cohort * Area * GaitPhase   REML = FALSE\n")
  cat("======================================================\n")
  acc <- fit_ladder(y, "Cohort * Area * GaitPhase", ladder_cortical, "df_c", REML = FALSE,
                    labels = list(model = "random_slopes", region = "cortical", datatype = datatype, band = band))
  m <- acc$fit
  cat("\nStructure used:", acc$re, "(step", acc$step, ")\n")
  cat("isSingular:", isSingular(m), " fallback exhausted:", acc$fallback_exhausted, "\n")
  if (length(acc$warnings)) cat("warnings:", paste(acc$warnings, collapse = " | "), "\n")
  cat("\nType III ANOVA (Satterthwaite):\n"); print(anova(m, type = 3, ddf = "Satterthwaite"))
  cat("\nRandom-effect variances:\n"); print(VarCorr(m), comp = c("Variance", "Std.Dev."))
  cat("\nemmeans PD - HC per GaitPhase (averaged over Area), FDR within family:\n")
  print_contrasts(m, ~ Cohort | GaitPhase)
  cat("\nemmeans PD - HC per Area (averaged over GaitPhase):\n")
  print_contrasts(m, ~ Cohort | Area)
  cat("\nemmeans PD - HC per Area x GaitPhase:\n")
  print_contrasts(m, ~ Cohort | Area * GaitPhase)

  lab <- list(model = "random_slopes", region = "cortical", datatype = datatype, band = band,
              structure = acc$re, ladder_step = acc$step, REML = FALSE,
              singular = isSingular(m), fallback_exhausted = acc$fallback_exhausted,
              warnings = paste(acc$warnings, collapse = " | "))
  anova_rows[[length(anova_rows) + 1]] <- do.call(tidy_anova, c(list(m), lab))
  vc_rows[[length(vc_rows) + 1]] <- do.call(tidy_varcorr, c(list(m), lab[1:5]))
  ctr_rows[[length(ctr_rows) + 1]] <- rbind(
    do.call(tidy_contrasts, c(list(m, ~ Cohort | GaitPhase, "Cohort:GaitPhase"), lab[1:5])) %>% mutate(Area = NA_character_),
    do.call(tidy_contrasts, c(list(m, ~ Cohort | Area, "Cohort:Area"), lab[1:5])) %>% mutate(GaitPhase = NA_character_),
    do.call(tidy_contrasts, c(list(m, ~ Cohort | Area * GaitPhase, "Cohort:Area:GaitPhase"), lab[1:5])))
  ladder_rows[[length(ladder_rows) + 1]] <- cbind(region = "cortical", datatype = datatype, band = band, acc$steps)
  ladder_anova_rows[[length(ladder_anova_rows) + 1]] <- acc$step_anova
  ladder_vc_rows[[length(ladder_vc_rows) + 1]] <- acc$step_vc
  rm(m, acc); invisible(gc())
}

for (band in BANDS) {
  y <- band_col(band)
  cat("\n======================================================\n")
  cat("STN random-slope model |", band, "|", datatype, "\n")
  cat("fixed: Hemisphere * GaitPhase   REML = FALSE\n")
  cat("======================================================\n")
  acc <- fit_ladder(y, "Hemisphere * GaitPhase", ladder_stn, "df_s", REML = FALSE,
                    labels = list(model = "random_slopes", region = "STN", datatype = datatype, band = band))
  m <- acc$fit
  cat("\nStructure used:", acc$re, "(step", acc$step, ")\n")
  cat("isSingular:", isSingular(m), " fallback exhausted:", acc$fallback_exhausted, "\n")
  if (length(acc$warnings)) cat("warnings:", paste(acc$warnings, collapse = " | "), "\n")
  cat("\nType III ANOVA (Satterthwaite):\n"); print(anova(m, type = 3, ddf = "Satterthwaite"))
  cat("\nRandom-effect variances:\n"); print(VarCorr(m), comp = c("Variance", "Std.Dev."))
  cat("\nemmeans Left - Right per GaitPhase, FDR within family:\n")
  print_contrasts(m, ~ Hemisphere | GaitPhase, reverse = FALSE)

  lab <- list(model = "random_slopes", region = "STN", datatype = datatype, band = band,
              structure = acc$re, ladder_step = acc$step, REML = FALSE,
              singular = isSingular(m), fallback_exhausted = acc$fallback_exhausted,
              warnings = paste(acc$warnings, collapse = " | "))
  anova_rows[[length(anova_rows) + 1]] <- do.call(tidy_anova, c(list(m), lab))
  vc_rows[[length(vc_rows) + 1]] <- do.call(tidy_varcorr, c(list(m), lab[1:5]))
  ctr_rows[[length(ctr_rows) + 1]] <- do.call(tidy_contrasts,
    c(list(m, ~ Hemisphere | GaitPhase, "Hemisphere:GaitPhase", reverse = FALSE), lab[1:5]))
  ladder_rows[[length(ladder_rows) + 1]] <- cbind(region = "STN", datatype = datatype, band = band, acc$steps)
  ladder_anova_rows[[length(ladder_anova_rows) + 1]] <- acc$step_anova
  ladder_vc_rows[[length(ladder_vc_rows) + 1]] <- acc$step_vc
}

anova_tbl <- add_fdr_bands(do.call(rbind, anova_rows))
fwrite(anova_tbl, paste0(outfile("lmm_random_slopes_anova"), ".csv"))
fwrite(do.call(rbind, vc_rows), paste0(outfile("lmm_random_slopes_varcorr"), ".csv"))
fwrite(rbindlist(ctr_rows, fill = TRUE), paste0(outfile("lmm_random_slopes_contrasts"), ".csv"))
fwrite(do.call(rbind, ladder_rows), paste0(outfile("lmm_random_slopes_ladder"), ".csv"))
fwrite(rbindlist(ladder_anova_rows, fill = TRUE), paste0(outfile("lmm_random_slopes_ladder_anova"), ".csv"))
fwrite(rbindlist(ladder_vc_rows, fill = TRUE), paste0(outfile("lmm_random_slopes_ladder_varcorr"), ".csv"))

cat("\n\n=== Summary: Type III ANOVA, random-slope models, with BH across bands per term ===\n")
print(anova_tbl %>% select(region, band, term, NumDF, DenDF, F, p, p_fdr_bands, eta2p, singular, ladder_step),
      digits = 4, row.names = FALSE)
cat("\nDone:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
close_log(log_con)
