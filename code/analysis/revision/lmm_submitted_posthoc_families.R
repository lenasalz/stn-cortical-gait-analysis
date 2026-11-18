# ===================================================================
# lmm_submitted_posthoc_families.R
# -------------------------------------------------------------------
# Purpose: Reviewer comments 5 and 16 (npj Parkinson's Disease, decision
#          letter 2026-10-12). Refit the submitted (retained) mixed models
#            cortical: gERD/ERS ~ Cohort*Area + Cohort*GaitPhase
#                      + (1|SubjectID) + (1|GaitCycle:SubjectID), ML
#            STN:      gERD/ERS ~ Hemisphere*GaitPhase + (1|SubjectID), REML
#          and report (i) the post-hoc contrasts with a multiplicity
#          correction over defined families, (ii) fixed-effect estimates
#          with 95% Wald confidence intervals, (iii) random-effect and
#          residual variances.
#          The submitted scripts called pairs(emm, adjust = "fdr") inside
#          each `by` level, i.e. on one comparison at a time, so no
#          correction was applied. Families used here (Benjamini-Hochberg):
#            cortical PD - HC per Area:       2 areas  x 5 bands = 10 tests
#            cortical PD - HC per GaitPhase:  4 phases x 5 bands = 20 tests
#            STN Left - Right per GaitPhase:  4 phases x 5 bands = 20 tests
#          each within datatype (basenorm or avggait). The raw p-values and
#          the Type III p of the parent interaction are kept alongside.
# Inputs:  code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_<datatype>.csv
#          code/analysis/R/Data/STN_Staandard_PD_subj_epochs_phase_median_<datatype>.csv
#          (symlinks to data/derivatives/gerd/master_{eeg,dbs}_<datatype>.csv)
# Outputs: outputs/stats/revision_2026-10/lmm_submitted_posthoc_<date>[_avggait].csv
#          outputs/stats/revision_2026-10/lmm_submitted_fixef_<date>[_avggait].csv
#          outputs/stats/revision_2026-10/lmm_submitted_varcorr_<date>[_avggait].csv
#          outputs/stats/revision_2026-10/lmm_submitted_posthoc_<date>[_avggait].txt
# Author:  Lena
# Date:    2026-11-16
# Seed:    42 (set.seed; the fits are deterministic)
# Usage:   REVISION_DATE=2026-11-16 Rscript code/analysis/revision/lmm_submitted_posthoc_families.R [basenorm|avggait] [anatomical|ipsicontra]
#          "ipsicontra" fits only the STN model on the stepping-limb-aligned
#          table (STN_IpsiContra_PD_subj_epochs_phase_median_<datatype>.csv;
#          gait phases IDS, CLS, FDS, ILS) and writes *_ipsicontra outputs.
# ===================================================================

set.seed(42)
source("code/analysis/revision/lmm_revision_common.R")

args     <- commandArgs(trailingOnly = TRUE)
datatype <- if (length(args) >= 1) args[1] else "basenorm"
stopifnot(datatype %in% c("basenorm", "avggait"))
alignment <- if (length(args) >= 2) args[2] else "anatomical"
stopifnot(alignment %in% c("anatomical", "ipsicontra"))
suffix   <- paste0(if (datatype == "basenorm") "" else paste0("_", datatype),
                   if (alignment == "ipsicontra") "_ipsicontra" else "")
outfile  <- function(stem, ext) file.path(OUT_DIR, paste0(stem, "_", REVISION_DATE, suffix, ext))

con <- open_log(outfile("lmm_submitted_posthoc", ".txt"))
session_header("lmm_submitted_posthoc_families.R")
cat("datatype:", datatype, " STN alignment:", alignment, "\n\n")

if (alignment == "anatomical") {
  df_c <- load_cortical(datatype)
  df_s <- load_stn(datatype)
} else {
  f <- file.path(DATA_DIR, paste0("STN_IpsiContra_PD_subj_epochs_phase_median_", datatype, ".csv"))
  stopifnot(file.exists(f))
  df_s <- as.data.frame(fread(f))
  df_s$SubjectID  <- as.factor(df_s$SubjectID)
  df_s$Hemisphere <- as.factor(df_s$Hemisphere)
  df_s$GaitPhase  <- as.factor(df_s$GaitPhase)
}

tidy_fixef <- function(fit, ...) {
  co <- as.data.frame(summary(fit)$coefficients)
  ci <- confint(fit, parm = "beta_", method = "Wald")
  out <- data.frame(term = rownames(co), estimate = co$Estimate, SE = co$`Std. Error`,
                    df = co$df, t = co$`t value`, p = co$`Pr(>|t|)`,
                    lower = ci[rownames(co), 1], upper = ci[rownames(co), 2],
                    stringsAsFactors = FALSE)
  labels <- list(...)
  for (nm in names(labels)) out[[nm]] <- labels[[nm]]
  out
}

ctr_rows <- list(); fix_rows <- list(); vc_rows <- list()
add <- function(lst, x) { lst[[length(lst) + 1]] <- x; lst }

for (band in if (alignment == "anatomical") BANDS else character(0)) {
  y <- band_col(band)
  fml <- as.formula(paste(y, "~ Cohort * Area + Cohort * GaitPhase + (1 | SubjectID) + (1 | GaitCycle:SubjectID)"))
  res <- fit_lmer(fml, "df_c", REML = FALSE, control_name = "LMM_CONTROL_SUBMITTED")
  m <- res$fit
  a <- anova(m, type = 3, ddf = "Satterthwaite")
  lab <- list(model = "submitted", region = "cortical", datatype = datatype, band = band)
  c1 <- do.call(tidy_contrasts, c(list(m, ~ Cohort | Area, "Cohort:Area"), lab))
  c1$level <- as.character(c1$Area); c1$Area <- NULL
  c1$p_interaction <- a["Cohort:Area", "Pr(>F)"]
  c2 <- do.call(tidy_contrasts, c(list(m, ~ Cohort | GaitPhase, "Cohort:GaitPhase"), lab))
  c2$level <- as.character(c2$GaitPhase); c2$GaitPhase <- NULL
  c2$p_interaction <- a["Cohort:GaitPhase", "Pr(>F)"]
  ctr_rows <- add(ctr_rows, rbind(c1, c2))
  fix_rows <- add(fix_rows, do.call(tidy_fixef, c(list(m), lab)))
  vc_rows  <- add(vc_rows,  do.call(tidy_varcorr, c(list(m), lab)))
  cat("cortical", band, " singular:", isSingular(m),
      if (length(res$warnings)) paste(" warnings:", paste(res$warnings, collapse = " | ")) else "", "\n")
  rm(m); invisible(gc())
}

for (band in BANDS) {
  y <- band_col(band)
  fml <- as.formula(paste(y, "~ Hemisphere * GaitPhase + (1 | SubjectID)"))
  res <- fit_lmer(fml, "df_s", REML = TRUE, control_name = "LMM_CONTROL_SUBMITTED")
  m <- res$fit
  a <- anova(m, type = 3, ddf = "Satterthwaite")
  lab <- list(model = "submitted", region = "STN", datatype = datatype, band = band)
  c3 <- do.call(tidy_contrasts, c(list(m, ~ Hemisphere | GaitPhase, "Hemisphere:GaitPhase", reverse = FALSE), lab))
  c3$level <- as.character(c3$GaitPhase); c3$GaitPhase <- NULL
  c3$p_interaction <- a["Hemisphere:GaitPhase", "Pr(>F)"]
  ctr_rows <- add(ctr_rows, c3)
  fix_rows <- add(fix_rows, do.call(tidy_fixef, c(list(m), lab)))
  vc_rows  <- add(vc_rows,  do.call(tidy_varcorr, c(list(m), lab)))
  cat("STN", band, " singular:", isSingular(m),
      if (length(res$warnings)) paste(" warnings:", paste(res$warnings, collapse = " | ")) else "", "\n")
  rm(m); invisible(gc())
}

ctr <- do.call(rbind, ctr_rows)
ctr$p_fdr_within_family <- NULL   # one comparison per `by` level: not a correction
ctr <- ctr %>%
  group_by(region, datatype, family) %>%
  mutate(p_fdr_family = p.adjust(p_raw, method = "BH"), n_tests_in_family = n()) %>%
  ungroup() %>% as.data.frame()
keep <- c("region", "datatype", "family", "band", "level", "contrast", "estimate", "SE", "df",
          "lower.CL", "upper.CL", "t.ratio", "p_raw", "p_fdr_family", "n_tests_in_family", "p_interaction")
ctr <- ctr[, keep]

fwrite(ctr, outfile("lmm_submitted_posthoc", ".csv"))
fwrite(do.call(rbind, fix_rows), outfile("lmm_submitted_fixef", ".csv"))
fwrite(do.call(rbind, vc_rows),  outfile("lmm_submitted_varcorr", ".csv"))

cat("\n=== Post-hoc contrasts, BH across the family (bands x levels) ===\n")
op <- options(width = 200)
print(ctr %>% mutate(across(c(estimate, SE, df, lower.CL, upper.CL, t.ratio), ~ round(.x, 3)),
                     across(c(p_raw, p_fdr_family, p_interaction), ~ signif(.x, 3))),
      row.names = FALSE)
options(op)
close_log(con)
