# ===================================================================
# CORTICAL LMM MODEL SELECTION (AIC / BIC) — explorative comparison
# -------------------------------------------------------------------
# Adapted for this project from:
#   ngb_study/notebooks/analysis/R_statistics/explorative_stats_lmm_complex.R
#
# Purpose: re-run Aline's AIC/BIC model-structure comparison on the
#   CURRENT (post-preprocessing-fix) cortical gERD/ERS data, to confirm
#   the model used in stats_lmm_cortical_complex_final.R is still the
#   preferred structure.
#
# Inputs  (read directly; relative to this script's dir code/analysis/R/):
#   - ../../../data/derivatives/gerd/master_eeg_basenorm.csv
#   - ../../../data/derivatives/gerd/master_eeg_avggait.csv
#   - Data/UPDRS_scores_USZ_finalscoreonly.xlsx        (OPTIONAL; server-only,
#     a stub with NA scores in this repo -> UPDRS models are skipped)
# Output (stdout; redirect to outputs/stats/lmm_modelsel_cortical_<date>.txt):
#   - per band x normalisation: AIC/BIC/logLik table + which structure wins.
#
# Candidate fixed/random structures compared (per band, REML=FALSE = ML, so
# AIC/BIC are comparable across differing fixed AND random terms):
#   group     : <band> ~ Group + (1|SubjectID)
#   int2      : <band> ~ Group*Area + Group*GaitPhase + (1|SubjectID)
#   int2_3way : <band> ~ Group*Area + Group*GaitPhase + Group:GaitPhase:Area + (1|SubjectID)
#   full3     : <band> ~ Group*Area*GaitPhase + (1|SubjectID)
#   FINAL     : <band> ~ Group*Area + Group*GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)
#   FINAL_updrs (only if real UPDRS available): FINAL + UPDRS
# ===================================================================

set.seed(42)  # reproducibility (project rule); lmer is deterministic but logged
cat("Random seed: 42\n")

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(data.table)
  library(dplyr)
  library(readxl)
})

GERD     <- "../../../data/derivatives/gerd"
datatypes <- c("basenorm", "avggait")
freq_bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
make_band_colname <- function(b) paste0("ERD_ERS_", b)
FINAL_NAME <- "FINAL"

# --- optional real UPDRS (skipped if absent or all-NA, e.g. the repo stub) ---
load_updrs <- function() {
  f <- "Data/UPDRS_scores_USZ_finalscoreonly.xlsx"
  if (!file.exists(f)) return(NULL)
  u <- tryCatch(as.data.frame(read_excel(f)), error = function(e) NULL)
  if (is.null(u) || !("UPDRS" %in% names(u)) || all(is.na(u$UPDRS))) return(NULL)
  u
}
updrs <- load_updrs()
cat(if (is.null(updrs)) "UPDRS: not available -> UPDRS models SKIPPED.\n"
    else "UPDRS: real scores loaded -> UPDRS model included.\n")

# --- fit one model (ML), return NULL on failure --------------------
fit_ml <- function(fstr, data) {
  tryCatch(
    suppressMessages(suppressWarnings(
      lmer(as.formula(fstr), data = data, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa",
                                 optCtrl = list(maxfun = 2e5)))
    )),
    error = function(e) { cat("   [fit failed]", fstr, "::", conditionMessage(e), "\n"); NULL }
  )
}

# --- comparison table ----------------------------------------------
compare <- function(models) {
  rows <- lapply(names(models), function(nm) {
    m <- models[[nm]]
    if (is.null(m)) return(NULL)
    data.frame(model = nm,
               npar = attr(logLik(m), "df"),
               AIC = AIC(m), BIC = BIC(m), logLik = as.numeric(logLik(m)),
               singular = isSingular(m),
               is_final = (nm == FINAL_NAME),
               stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, rows)
  tab <- tab[order(tab$BIC), ]
  tab$dAIC <- round(tab$AIC - min(tab$AIC), 2)
  tab$dBIC <- round(tab$BIC - min(tab$BIC), 2)
  tab$AIC <- round(tab$AIC, 1); tab$BIC <- round(tab$BIC, 1)
  tab$logLik <- round(tab$logLik, 1)
  rownames(tab) <- NULL
  tab
}

for (datatype in datatypes) {
  fp <- file.path(GERD, paste0("master_eeg_", datatype, ".csv"))
  cat("\n\n################################################################\n")
  cat("# CORTICAL model selection | datatype:", datatype, "\n")
  cat("# data:", fp, "\n")
  cat("################################################################\n")
  if (!file.exists(fp)) { cat("MISSING:", fp, "- skipping.\n"); next }

  df <- fread(fp)
  df <- df[df$Hemisphere == "Both", ]          # cortical ROI-averaged rows
  if (!is.null(updrs)) {
    df <- merge(df, updrs, by = "SubjectID", all.x = TRUE)
    df$UPDRS[df$Group == "HC"] <- 0
  }
  df$SubjectID <- as.factor(df$SubjectID)
  df$Group     <- as.factor(df$Group)
  df$Area      <- as.factor(df$Area)
  df$GaitPhase <- as.factor(df$GaitPhase)
  df$GaitCycle <- as.factor(df$Epoch)
  cat("rows:", nrow(df), "| subjects:", nlevels(df$SubjectID),
      "| areas:", paste(levels(df$Area), collapse="/"),
      "| phases:", paste(levels(df$GaitPhase), collapse="/"), "\n")

  for (band in freq_bands) {
    bc <- make_band_colname(band)
    if (!(bc %in% names(df))) { cat("  (no column", bc, "- skip)\n"); next }
    cat("\n================ Band:", band, "(", datatype, ") ================\n")

    fdefs <- list(
      group     = paste(bc, "~ Group + (1|SubjectID)"),
      int2      = paste(bc, "~ Group*Area + Group*GaitPhase + (1|SubjectID)"),
      int2_3way = paste(bc, "~ Group*Area + Group*GaitPhase + Group:GaitPhase:Area + (1|SubjectID)"),
      full3     = paste(bc, "~ Group*Area*GaitPhase + (1|SubjectID)"),
      FINAL     = paste(bc, "~ Group*Area + Group*GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)")
    )
    if (!is.null(updrs))
      fdefs[["FINAL_updrs"]] <- paste(bc,
        "~ Group*Area + Group*GaitPhase + UPDRS + (1|SubjectID) + (1|GaitCycle:SubjectID)")

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
cat("\n--- Cortical model selection complete ---\n")
