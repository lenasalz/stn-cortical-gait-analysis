# ===================================================================
# STN LMM MODEL SELECTION (AIC / BIC) — explorative comparison
# -------------------------------------------------------------------
# Adapted for this project from:
#   ngb_study/notebooks/analysis/R_statistics/explorative_stats_STN_lmm_complex.R
#   (the original referenced an undefined f_stn_2_b / mod_stn_2 and would
#    error; fixed here, and UPDRS-only models gated on score availability.)
#
# Purpose: re-run the STN AIC/BIC model-structure comparison on the CURRENT
#   (post-fix) STN gERD/ERS data, to confirm the structure used in
#   stats_lmm_STN_complex_final.R / stats_LMM_STN_ipsicontra_final.R
#   (Hemisphere*GaitPhase + (1|SubjectID)) is still preferred.
#
# Inputs (relative to code/analysis/R/):
#   - ../../../data/derivatives/gerd/master_dbs_<norm>.csv            (anatomical L/R = "standard")
#   - ../../../data/derivatives/gerd/master_dbs_ipsicontra_<norm>.csv (ipsi/contra-aligned)
#   - Data/UPDRS_scores_USZ_finalscoreonly.xlsx  (OPTIONAL; stub=NA -> UPDRS skipped)
# Output (stdout; redirect to outputs/stats/lmm_modelsel_STN_<date>.txt):
#   - per dataset x norm x band: AIC/BIC/logLik table + which structure wins.
#
# NB: each PD subject contributes ONE hemisphere, so Hemisphere is a
#     between-subjects factor; (1|SubjectID) supplies its between-subjects
#     error term. Models fit with REML=FALSE (ML) for comparable AIC/BIC.
# Candidates:
#   add    : <band> ~ Hemisphere + GaitPhase + (1|SubjectID)
#   FINAL  : <band> ~ Hemisphere * GaitPhase + (1|SubjectID)
#   int_gc : <band> ~ Hemisphere * GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)
#   *_updrs (only if real UPDRS available)
# ===================================================================

set.seed(42)
cat("Random seed: 42\n")

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(data.table)
  library(dplyr)
  library(readxl)
})

GERD <- "../../../data/derivatives/gerd"
datasets <- list(standard   = "master_dbs_%s.csv",
                 ipsicontra = "master_dbs_ipsicontra_%s.csv")
datatypes  <- c("basenorm", "avggait")
freq_bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
make_band_colname <- function(b) paste0("ERD_ERS_", b)
FINAL_NAME <- "FINAL"

load_updrs <- function() {
  f <- "Data/UPDRS_scores_USZ_finalscoreonly.xlsx"
  if (!file.exists(f)) return(NULL)
  u <- tryCatch(as.data.frame(read_excel(f)), error = function(e) NULL)
  if (is.null(u) || !("UPDRS" %in% names(u)) || all(is.na(u$UPDRS))) return(NULL)
  u
}
updrs <- load_updrs()
cat(if (is.null(updrs)) "UPDRS: not available -> UPDRS models SKIPPED.\n"
    else "UPDRS: real scores loaded -> UPDRS models included.\n")

fit_ml <- function(fstr, data) {
  tryCatch(
    suppressMessages(suppressWarnings(
      lmer(as.formula(fstr), data = data, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    )),
    error = function(e) { cat("   [fit failed]", fstr, "::", conditionMessage(e), "\n"); NULL }
  )
}

compare <- function(models) {
  rows <- lapply(names(models), function(nm) {
    m <- models[[nm]]; if (is.null(m)) return(NULL)
    data.frame(model = nm, npar = attr(logLik(m), "df"),
               AIC = AIC(m), BIC = BIC(m), logLik = as.numeric(logLik(m)),
               singular = isSingular(m), is_final = (nm == FINAL_NAME),
               stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, rows)
  tab <- tab[order(tab$BIC), ]
  tab$dAIC <- round(tab$AIC - min(tab$AIC), 2)
  tab$dBIC <- round(tab$BIC - min(tab$BIC), 2)
  tab$AIC <- round(tab$AIC, 1); tab$BIC <- round(tab$BIC, 1); tab$logLik <- round(tab$logLik, 1)
  rownames(tab) <- NULL
  tab
}

for (ds in names(datasets)) {
  for (datatype in datatypes) {
    fp <- file.path(GERD, sprintf(datasets[[ds]], datatype))
    cat("\n\n################################################################\n")
    cat("# STN model selection | dataset:", ds, "| datatype:", datatype, "\n")
    cat("# data:", fp, "\n")
    cat("################################################################\n")
    if (!file.exists(fp)) { cat("MISSING:", fp, "- skipping.\n"); next }

    df <- fread(fp)
    df <- df[df$Hemisphere %in% c("Left", "Right"), ]
    if (!is.null(updrs)) {
      df <- merge(df, updrs, by = "SubjectID", all.x = TRUE)
      df$UPDRS[is.na(df$UPDRS)] <- mean(df$UPDRS, na.rm = TRUE)
    }
    df$SubjectID <- as.factor(df$SubjectID)
    df$Hemisphere <- as.factor(df$Hemisphere)
    df$GaitPhase  <- as.factor(df$GaitPhase)
    df$GaitCycle  <- as.factor(df$Epoch)
    cat("rows:", nrow(df), "| subjects:", nlevels(df$SubjectID),
        "| hemispheres:", paste(levels(df$Hemisphere), collapse="/"),
        "| phases:", paste(levels(df$GaitPhase), collapse="/"), "\n")

    for (band in freq_bands) {
      bc <- make_band_colname(band)
      if (!(bc %in% names(df))) { cat("  (no column", bc, "- skip)\n"); next }
      cat("\n================ Band:", band, "(", ds, "/", datatype, ") ================\n")

      fdefs <- list(
        add    = paste(bc, "~ Hemisphere + GaitPhase + (1|SubjectID)"),
        FINAL  = paste(bc, "~ Hemisphere * GaitPhase + (1|SubjectID)"),
        int_gc = paste(bc, "~ Hemisphere * GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)")
      )
      if (!is.null(updrs)) {
        fdefs[["FINAL_updrs"]] <- paste(bc, "~ Hemisphere * GaitPhase + UPDRS + (1|SubjectID)")
        fdefs[["updrs_gp"]]    <- paste(bc, "~ Hemisphere * GaitPhase + UPDRS*GaitPhase + (1|SubjectID)")
      }

      models <- lapply(fdefs, fit_ml, data = df)
      tab <- compare(models)
      print(tab)
      best_aic <- tab$model[which.min(tab$AIC)]
      best_bic <- tab$model[which.min(tab$BIC)]
      cat(sprintf("  -> best AIC: %s | best BIC: %s | FINAL is best by AIC: %s, by BIC: %s\n",
                  best_aic, best_bic,
                  ifelse(best_aic == FINAL_NAME, "YES", "no"),
                  ifelse(best_bic == FINAL_NAME, "YES", "no")))
    }
  }
}
cat("\n--- STN model selection complete ---\n")
