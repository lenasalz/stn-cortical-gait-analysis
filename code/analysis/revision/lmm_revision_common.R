# ===================================================================
# lmm_revision_common.R
# -------------------------------------------------------------------
# Purpose: Helper functions shared by the revision mixed-model scripts
#          (lmm_random_slopes.R, lmm_aggregated.R, lmm_refit_tables.R).
#          Data loading mirrors the submitted scripts
#          code/analysis/R/stats_lmm_cortical_complex_final.R and
#          code/analysis/R/stats_lmm_STN_complex_final.R; fitting with
#          a stepwise random-effects fallback ladder; tidy extraction of
#          Type III ANOVA tables (lmerTest, Satterthwaite), partial eta
#          squared, random-effect variances and emmeans contrasts.
# Inputs:  none directly (sourced by the scripts above)
# Outputs: none directly
# Author:  Lena
# Date:    2026-10-13
# Seed:    set.seed(42) is called by every script that sources this file
# ===================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

# Output date tag; override with the environment variable REVISION_DATE for a rerun.
REVISION_DATE <- Sys.getenv("REVISION_DATE", "2026-10-13")
PROJECT_ROOT  <- normalizePath(".")
DATA_DIR      <- file.path(PROJECT_ROOT, "code/analysis/R/Data")
OUT_DIR       <- file.path(PROJECT_ROOT, "outputs/stats/revision_2026-10")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

BANDS <- c("Theta", "Alpha", "Low Beta", "High Beta", "Gamma")
band_col <- function(band) paste0("ERD_ERS_", gsub(" ", "_", band))

# Satterthwaite df for emmeans (Kenward-Roger is too memory-hungry for
# random-slope models on ~15,000 rows); pbkrtest limit kept as submitted,
# lmerTest limit raised so that emmeans computes Satterthwaite df instead
# of falling back to asymptotic z tests.
emm_options(lmer.df = "satterthwaite", pbkrtest.limit = 17000, lmerTest.limit = 20000)

# Optimiser used for all new fits (the submitted scripts used the lme4
# default, nloptwrap; bobyqa with a large maxfun is more robust for the
# larger random-effect structures). Kept in the global environment so
# that emmeans can re-evaluate the model call.
LMM_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
LMM_CONTROL_SUBMITTED <- lmerControl()

# ------------------------------------------------------------------
# Logging: stdout is split to console and file; messages/warnings go
# to the file only.
# ------------------------------------------------------------------
open_log <- function(path) {
  con <- file(path, open = "wt")
  sink(con, split = TRUE)
  sink(con, type = "message")
  invisible(con)
}
close_log <- function(con) {
  sink(type = "message")
  sink()
  close(con)
}

# ------------------------------------------------------------------
# Data loading (identical handling to the submitted scripts). Numeric
# 0/1 indicator columns are added for the uncorrelated random-slope
# structures: lme4's `||` does not split a factor into independent
# dummy slopes (it builds a correlated `0 + factor` block next to a
# separate intercept, which is over-parameterised and always singular),
# so the dummies are written out explicitly (treatment coding, reference
# levels frontal and FDS as in the fixed-effect part).
# ------------------------------------------------------------------
add_indicators <- function(df) {
  if ("Area" %in% names(df)) df$A_motor <- as.numeric(df$Area == "motor")
  df$G_IDS <- as.numeric(df$GaitPhase == "IDS")
  df$G_LLS <- as.numeric(df$GaitPhase == "LLS")
  df$G_RLS <- as.numeric(df$GaitPhase == "RLS")
  if ("Area" %in% names(df)) {
    df$AG_IDS <- df$A_motor * df$G_IDS
    df$AG_LLS <- df$A_motor * df$G_LLS
    df$AG_RLS <- df$A_motor * df$G_RLS
  }
  df
}

load_cortical <- function(datatype = "basenorm") {
  f <- file.path(DATA_DIR,
                 paste0("for_R_EEG_epochs_PD-vs-HC_phase_median_", datatype, ".csv"))
  stopifnot(file.exists(f))
  df <- fread(f)
  df <- df[df$Hemisphere == "Both", ]
  df$SubjectID <- as.factor(df$SubjectID)
  df$Cohort    <- relevel(as.factor(df$Group), ref = "HC")
  df$Area      <- as.factor(df$Area)
  df$GaitPhase <- as.factor(df$GaitPhase)
  df$GaitCycle <- as.factor(df$Epoch)
  add_indicators(as.data.frame(df))
}

load_stn <- function(datatype = "basenorm") {
  f <- file.path(DATA_DIR,
                 paste0("STN_Staandard_PD_subj_epochs_phase_median_", datatype, ".csv"))
  stopifnot(file.exists(f))
  df <- fread(f)
  df$SubjectID  <- as.factor(df$SubjectID)
  df$Hemisphere <- as.factor(df$Hemisphere)
  df$GaitPhase  <- as.factor(df$GaitPhase)
  df$GaitCycle  <- as.factor(df$Epoch)
  add_indicators(as.data.frame(df))
}

# ------------------------------------------------------------------
# Fitting with warning capture. The call is built with the data and
# control referenced by global names and evaluated in the global
# environment so that emmeans / lmerTest can re-evaluate it.
# ------------------------------------------------------------------
fit_lmer <- function(formula, data_name, REML = FALSE, control_name = "LMM_CONTROL") {
  cl <- bquote(lmer(.(formula), data = .(as.name(data_name)), REML = .(REML),
                    control = .(as.name(control_name))))
  warns <- character(0)
  fit <- withCallingHandlers(
    tryCatch(eval(cl, envir = globalenv()),
             error = function(e) { warns <<- c(warns, paste("ERROR:", conditionMessage(e))); NULL }),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { warns <<- c(warns, conditionMessage(m)); invokeRestart("muffleMessage") })
  conv_msgs <- if (!is.null(fit)) unlist(fit@optinfo$conv$lme4$messages) else character(0)
  list(fit = fit, warnings = unique(trimws(c(warns, conv_msgs))))
}

# Fit a ladder of random-effects structures; accept the first that
# converges without a singular fit. Returns the accepted fit plus the
# record of every step tried, with the Type III ANOVA and variance
# components of every step that fitted (so the fallback can be reported
# and the effect of each simplification inspected).
fit_ladder <- function(lhs, fixed, ladder, data_name, REML = FALSE, labels = list()) {
  steps <- list(); step_anova <- list(); step_vc <- list()
  accepted <- NULL; last_fit <- NULL
  for (i in seq_along(ladder)) {
    fml <- as.formula(paste(lhs, "~", fixed, "+", ladder[[i]]$re))
    t0 <- Sys.time()
    res <- fit_lmer(fml, data_name, REML = REML)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    singular <- if (is.null(res$fit)) NA else isSingular(res$fit)
    converged <- !is.null(res$fit) &&
      !any(grepl("failed to converge|unable to evaluate", res$warnings, ignore.case = TRUE))
    steps[[i]] <- data.frame(step = i, label = ladder[[i]]$label, re = ladder[[i]]$re,
                             fitted = !is.null(res$fit), singular = singular,
                             converged = converged, seconds = round(secs, 1),
                             warnings = paste(res$warnings, collapse = " | "),
                             stringsAsFactors = FALSE)
    cat(sprintf("  step %d  %-52s fitted=%s singular=%s converged=%s (%.0f s)\n",
                i, ladder[[i]]$label, !is.null(res$fit), singular, converged, secs))
    if (length(res$warnings)) cat("    warnings:", paste(res$warnings, collapse = " | "), "\n")
    if (!is.null(res$fit)) {
      cat("    variance components:\n")
      print(VarCorr(res$fit), comp = c("Variance", "Std.Dev."))
      step_anova[[i]] <- do.call(tidy_anova, c(list(res$fit), labels,
        list(structure = ladder[[i]]$re, ladder_step = i, singular = singular,
             converged = converged, warnings = paste(res$warnings, collapse = " | "))))
      step_vc[[i]] <- do.call(tidy_varcorr, c(list(res$fit), labels,
        list(structure = ladder[[i]]$re, ladder_step = i, singular = singular)))
      last_fit <- list(fit = res$fit, step = i, label = ladder[[i]]$label,
                       re = ladder[[i]]$re, warnings = res$warnings)
    }
    if (!is.null(res$fit) && converged && isTRUE(!singular)) {
      accepted <- last_fit
      break
    }
  }
  if (is.null(accepted)) {
    cat("  No step met the criteria; keeping the last fitted structure (step",
        last_fit$step, ") and flagging it.\n")
    accepted <- last_fit
    accepted$fallback_exhausted <- TRUE
  } else accepted$fallback_exhausted <- FALSE
  accepted$steps <- do.call(rbind, steps)
  accepted$step_anova <- do.call(rbind, step_anova)
  accepted$step_vc <- do.call(rbind, step_vc)
  accepted
}

# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------
# Partial eta squared from F and df. Approximate for mixed models: it
# treats the Satterthwaite denominator df as if it were the error df of
# a fixed-effects ANOVA (Lakens-type conversion), ignoring the
# variance partition among random effects.
eta2p <- function(F, df1, df2) F * df1 / (F * df1 + df2)

tidy_anova <- function(fit, ...) {
  a <- anova(fit, type = 3, ddf = "Satterthwaite")
  out <- data.frame(term = rownames(a), NumDF = a$NumDF, DenDF = a$DenDF,
                    F = a$`F value`, p = a$`Pr(>F)`, stringsAsFactors = FALSE)
  out$eta2p <- eta2p(out$F, out$NumDF, out$DenDF)
  labels <- list(...)
  for (nm in names(labels)) out[[nm]] <- labels[[nm]]
  out
}

tidy_varcorr <- function(fit, ...) {
  v <- as.data.frame(VarCorr(fit))
  out <- data.frame(grp = v$grp, var1 = v$var1, var2 = v$var2,
                    vcov = v$vcov, sdcor = v$sdcor, stringsAsFactors = FALSE)
  labels <- list(...)
  for (nm in names(labels)) out[[nm]] <- labels[[nm]]
  out
}

# emmeans contrasts: `spec` such as ~ Cohort | GaitPhase. `reverse = TRUE`
# yields PD - HC (second level minus first) for Cohort (ref = HC).
tidy_contrasts <- function(fit, spec, family, reverse = TRUE, ...) {
  emm <- emmeans(fit, spec)
  ctr <- pairs(emm, reverse = reverse)
  raw <- as.data.frame(summary(ctr, infer = c(TRUE, TRUE), adjust = "none"))
  adj <- as.data.frame(summary(ctr, infer = c(FALSE, TRUE), adjust = "fdr"))
  raw$p_fdr_within_family <- adj$p.value
  names(raw)[names(raw) == "p.value"] <- "p_raw"
  raw$family <- family
  labels <- list(...)
  for (nm in names(labels)) raw[[nm]] <- labels[[nm]]
  raw
}

print_contrasts <- function(fit, spec, reverse = TRUE) {
  emm <- emmeans(fit, spec)
  print(summary(pairs(emm, reverse = reverse), infer = c(TRUE, TRUE), adjust = "fdr"))
}

# Benjamini-Hochberg across the five bands within model x region x
# datatype x term.
add_fdr_bands <- function(anova_tbl) {
  anova_tbl %>%
    group_by(model, region, datatype, term) %>%
    mutate(p_fdr_bands = p.adjust(p, method = "BH"), n_bands_in_family = n()) %>%
    ungroup() %>% as.data.frame()
}

session_header <- function(script) {
  cat("script:", script, "\n")
  cat("date:", REVISION_DATE, "  run at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  cat(R.version.string, "| lme4", as.character(packageVersion("lme4")),
      "| lmerTest", as.character(packageVersion("lmerTest")),
      "| emmeans", as.character(packageVersion("emmeans")), "\n")
  cat("seed: 42\n\n")
}
