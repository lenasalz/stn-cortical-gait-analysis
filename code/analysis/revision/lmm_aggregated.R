# ===================================================================
# lmm_aggregated.R
# -------------------------------------------------------------------
# Purpose: Reviewer 1, comment 4 (npj Parkinson's Disease, 2026-10-12):
#          sensitivity analysis that collapses the cycle-level data to
#          one value per subject and cell before modelling, so that the
#          Cohort interactions are tested against between-subject
#          variability only (a conventional repeated-measures mixed
#          ANOVA). Cortical: one value per SubjectID x Area x GaitPhase
#          (20 x 2 x 4 = 160 rows), model
#          y ~ Cohort * Area * GaitPhase + (1 | SubjectID). STN: one value
#          per SubjectID x GaitPhase (10 x 4 = 40 rows), model
#          y ~ Hemisphere * GaitPhase + (1 | SubjectID). The per-cell
#          summary is the median across gait cycles (matching the
#          submitted "median" convention); the mean is run as well and
#          both are reported. Fits use REML (the conventional choice for
#          a repeated-measures ANOVA; ML would bias the small-sample
#          variance components downwards). Type III ANOVA with
#          Satterthwaite df, approximate partial eta squared, emmeans
#          PD - HC (STN: Left - Right) contrasts with 95% CIs and FDR
#          within family, Benjamini-Hochberg across the five bands per
#          term.
# Inputs:  code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_<datatype>.csv
#          code/analysis/R/Data/STN_Staandard_PD_subj_epochs_phase_median_<datatype>.csv
#          code/analysis/revision/lmm_revision_common.R
# Outputs: outputs/stats/revision_2026-10/lmm_aggregated_2026-10-13[_<datatype>].txt
#          outputs/stats/revision_2026-10/lmm_aggregated_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_aggregated_varcorr_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_aggregated_contrasts_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_aggregated_cells_2026-10-13[_<datatype>].csv
#          (suffix omitted for basenorm, the primary normalisation)
# Author:  Lena
# Date:    2026-10-13
# Seed:    set.seed(42)
# Usage:   Rscript code/analysis/revision/lmm_aggregated.R [basenorm|avggait]
# ===================================================================

set.seed(42)
source("code/analysis/revision/lmm_revision_common.R")

args     <- commandArgs(trailingOnly = TRUE)
datatype <- if (length(args) >= 1) args[1] else "basenorm"
suffix   <- if (datatype == "basenorm") "" else paste0("_", datatype)
outfile  <- function(stem) file.path(OUT_DIR, paste0(stem, "_", REVISION_DATE, suffix))

log_con <- open_log(paste0(outfile("lmm_aggregated"), ".txt"))
session_header("code/analysis/revision/lmm_aggregated.R")
cat("datatype:", datatype, "\n\n")

df_c <- load_cortical(datatype)
df_s <- load_stn(datatype)
ycols <- band_col(BANDS)

# --- Aggregation ----------------------------------------------------
agg_c <- as.data.table(df_c)[, c(lapply(.SD, median), list(n_cycles = .N)),
                             by = .(SubjectID, Cohort, Area, GaitPhase), .SDcols = ycols]
agg_c_mean <- as.data.table(df_c)[, c(lapply(.SD, mean), list(n_cycles = .N)),
                                  by = .(SubjectID, Cohort, Area, GaitPhase), .SDcols = ycols]
agg_s <- as.data.table(df_s)[, c(lapply(.SD, median), list(n_cycles = .N)),
                             by = .(SubjectID, Hemisphere, GaitPhase), .SDcols = ycols]
agg_s_mean <- as.data.table(df_s)[, c(lapply(.SD, mean), list(n_cycles = .N)),
                                  by = .(SubjectID, Hemisphere, GaitPhase), .SDcols = ycols]
cat(sprintf("cortical cells: %d rows (median) / %d (mean); cycles per cell: min %d, median %g, max %d\n",
            nrow(agg_c), nrow(agg_c_mean), min(agg_c$n_cycles), median(agg_c$n_cycles), max(agg_c$n_cycles)))
cat(sprintf("STN cells: %d rows; cycles per cell: min %d, median %g, max %d\n\n",
            nrow(agg_s), min(agg_s$n_cycles), median(agg_s$n_cycles), max(agg_s$n_cycles)))

cells <- rbind(cbind(region = "cortical", summary = "median", as.data.frame(agg_c), Hemisphere = NA),
               cbind(region = "cortical", summary = "mean",   as.data.frame(agg_c_mean), Hemisphere = NA),
               cbind(region = "STN", summary = "median", as.data.frame(agg_s), Cohort = "PD", Area = "STN"),
               cbind(region = "STN", summary = "mean",   as.data.frame(agg_s_mean), Cohort = "PD", Area = "STN"),
               fill = TRUE)
fwrite(cells, paste0(outfile("lmm_aggregated_cells"), ".csv"))

# Data frames must be global for emmeans to re-evaluate the model call.
agg_c <- as.data.frame(agg_c); agg_c_mean <- as.data.frame(agg_c_mean)
agg_s <- as.data.frame(agg_s); agg_s_mean <- as.data.frame(agg_s_mean)

anova_rows <- list(); vc_rows <- list(); ctr_rows <- list()

run_cortical <- function(data_name, summary_label) {
  for (band in BANDS) {
    y <- band_col(band)
    cat("\n======================================================\n")
    cat("Aggregated cortical model |", band, "|", datatype, "| per-cell", summary_label, "\n")
    cat("fixed: Cohort * Area * GaitPhase   random: (1 | SubjectID)   REML = TRUE\n")
    cat("======================================================\n")
    fml <- as.formula(paste(y, "~ Cohort * Area * GaitPhase + (1 | SubjectID)"))
    res <- fit_lmer(fml, data_name, REML = TRUE)
    m <- res$fit
    cat("isSingular:", isSingular(m), "\n")
    if (length(res$warnings)) cat("warnings:", paste(res$warnings, collapse = " | "), "\n")
    cat("\nType III ANOVA (Satterthwaite):\n"); print(anova(m, type = 3, ddf = "Satterthwaite"))
    cat("\nRandom-effect variances:\n"); print(VarCorr(m), comp = c("Variance", "Std.Dev."))
    cat("\nemmeans PD - HC per GaitPhase (averaged over Area), FDR within family:\n")
    print_contrasts(m, ~ Cohort | GaitPhase)
    cat("\nemmeans PD - HC per Area (averaged over GaitPhase):\n")
    print_contrasts(m, ~ Cohort | Area)
    cat("\nemmeans PD - HC per Area x GaitPhase:\n")
    print_contrasts(m, ~ Cohort | Area * GaitPhase)

    model_name <- paste0("aggregated_", summary_label)
    lab <- list(model = model_name, region = "cortical", datatype = datatype, band = band,
                structure = "(1|SubjectID) on per-cell values", REML = TRUE,
                singular = isSingular(m), warnings = paste(res$warnings, collapse = " | "))
    anova_rows[[length(anova_rows) + 1]] <<- do.call(tidy_anova, c(list(m), lab))
    vc_rows[[length(vc_rows) + 1]] <<- do.call(tidy_varcorr, c(list(m), lab[1:5]))
    ctr_rows[[length(ctr_rows) + 1]] <<- rbind(
      do.call(tidy_contrasts, c(list(m, ~ Cohort | GaitPhase, "Cohort:GaitPhase"), lab[1:5])) %>% mutate(Area = NA_character_),
      do.call(tidy_contrasts, c(list(m, ~ Cohort | Area, "Cohort:Area"), lab[1:5])) %>% mutate(GaitPhase = NA_character_),
      do.call(tidy_contrasts, c(list(m, ~ Cohort | Area * GaitPhase, "Cohort:Area:GaitPhase"), lab[1:5])))
  }
}

run_stn <- function(data_name, summary_label) {
  for (band in BANDS) {
    y <- band_col(band)
    cat("\n======================================================\n")
    cat("Aggregated STN model |", band, "|", datatype, "| per-cell", summary_label, "\n")
    cat("fixed: Hemisphere * GaitPhase   random: (1 | SubjectID)   REML = TRUE\n")
    cat("======================================================\n")
    fml <- as.formula(paste(y, "~ Hemisphere * GaitPhase + (1 | SubjectID)"))
    res <- fit_lmer(fml, data_name, REML = TRUE)
    m <- res$fit
    cat("isSingular:", isSingular(m), "\n")
    if (length(res$warnings)) cat("warnings:", paste(res$warnings, collapse = " | "), "\n")
    cat("\nType III ANOVA (Satterthwaite):\n"); print(anova(m, type = 3, ddf = "Satterthwaite"))
    cat("\nRandom-effect variances:\n"); print(VarCorr(m), comp = c("Variance", "Std.Dev."))
    cat("\nemmeans Left - Right per GaitPhase, FDR within family:\n")
    print_contrasts(m, ~ Hemisphere | GaitPhase, reverse = FALSE)

    model_name <- paste0("aggregated_", summary_label)
    lab <- list(model = model_name, region = "STN", datatype = datatype, band = band,
                structure = "(1|SubjectID) on per-cell values", REML = TRUE,
                singular = isSingular(m), warnings = paste(res$warnings, collapse = " | "))
    anova_rows[[length(anova_rows) + 1]] <<- do.call(tidy_anova, c(list(m), lab))
    vc_rows[[length(vc_rows) + 1]] <<- do.call(tidy_varcorr, c(list(m), lab[1:5]))
    ctr_rows[[length(ctr_rows) + 1]] <<- do.call(tidy_contrasts,
      c(list(m, ~ Hemisphere | GaitPhase, "Hemisphere:GaitPhase", reverse = FALSE), lab[1:5]))
  }
}

cat("\n###################################################################\n")
cat("# Per-cell MEDIAN across gait cycles (primary, matches submitted convention)\n")
cat("###################################################################\n")
run_cortical("agg_c", "median")
run_stn("agg_s", "median")

cat("\n\n###################################################################\n")
cat("# Per-cell MEAN across gait cycles (sensitivity)\n")
cat("###################################################################\n")
run_cortical("agg_c_mean", "mean")
run_stn("agg_s_mean", "mean")

anova_tbl <- add_fdr_bands(do.call(rbind, anova_rows))
fwrite(anova_tbl, paste0(outfile("lmm_aggregated_anova"), ".csv"))
fwrite(do.call(rbind, vc_rows), paste0(outfile("lmm_aggregated_varcorr"), ".csv"))
fwrite(rbindlist(ctr_rows, fill = TRUE), paste0(outfile("lmm_aggregated_contrasts"), ".csv"))

cat("\n\n=== Summary: Type III ANOVA, aggregated models, with BH across bands per term ===\n")
print(anova_tbl %>% select(model, region, band, term, NumDF, DenDF, F, p, p_fdr_bands, eta2p, singular),
      digits = 4, row.names = FALSE)

cat("\n=== Median vs mean aggregation: terms whose significance (p < 0.05, raw) differs ===\n")
cmp <- anova_tbl %>% select(model, region, band, term, p) %>%
  tidyr::pivot_wider(names_from = model, values_from = p) %>%
  mutate(sig_median = aggregated_median < 0.05, sig_mean = aggregated_mean < 0.05) %>%
  filter(sig_median != sig_mean)
if (nrow(cmp) == 0) cat("none\n") else print(as.data.frame(cmp), digits = 4, row.names = FALSE)
cat("\nDone:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
close_log(log_con)
