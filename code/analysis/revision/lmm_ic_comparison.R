# Information criteria: submitted random-intercept model vs random-slope model
#
# Purpose:  Compare AIC/BIC/log-likelihood of the submitted cortical model
#           (random intercepts for Subject and GaitCycle:Subject) with the
#           random-slope structure used in the refit (uncorrelated by-subject
#           slopes for Area, GaitPhase and Area x GaitPhase), both with the
#           full fixed-effect structure, ML fit, per band (basenorm).
# Inputs:   code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_basenorm.csv
# Outputs:  outputs/stats/revision_2026-10/lmm_ic_comparison_2026-10-14.csv / .txt
# Author:   Lena
# Date:     2026-10-14
# Seed:     42
suppressPackageStartupMessages({ library(data.table); library(lme4) })
set.seed(42)
df <- fread("code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_basenorm.csv")[Hemisphere == "Both"]
df[, `:=`(SubjectID = factor(SubjectID), Cohort = factor(Cohort), Area = factor(Area), GaitPhase = factor(GaitPhase), GaitCycle = factor(Epoch))]
# indicator columns for uncorrelated slopes (same construction as lmm_random_slopes.R)
df[, A_motor := as.numeric(Area == "motor")]
for (g in c("IDS", "LLS", "RLS")) { df[, (paste0("G_", g)) := as.numeric(GaitPhase == g)]; df[, (paste0("AG_", g)) := A_motor * as.numeric(GaitPhase == g)] }
bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma"); out <- list()
for (b in bands) {
  y <- paste0("ERD_ERS_", b)
  f0 <- as.formula(paste(y, "~ Cohort*Area + Cohort*GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)"))
  f1 <- as.formula(paste(y, "~ Cohort*Area*GaitPhase + (1|SubjectID) + (1|GaitCycle:SubjectID)"))
  f2 <- as.formula(paste(y, "~ Cohort*Area*GaitPhase + (1 + A_motor + G_IDS + G_LLS + G_RLS + AG_IDS + AG_LLS + AG_RLS || SubjectID) + (1|GaitCycle:SubjectID)"))
  f3 <- as.formula(paste(y, "~ Cohort*Area*GaitPhase + (1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID) + (1|GaitCycle:SubjectID)"))
  ms <- list(submitted_2way_int = f0, int_3way = f1, slopes_Area_Phase_AxP = f2, slopes_Area_Phase = f3)
  fits <- lapply(ms, function(f) lmer(f, data = df, REML = FALSE, control = lmerControl(optimizer = "bobyqa")))
  for (nm in names(fits)) {
    m <- fits[[nm]]
    out[[length(out) + 1]] <- data.frame(band = b, model = nm, npar = attr(logLik(m), "df"), logLik = as.numeric(logLik(m)),
                                         AIC = AIC(m), BIC = BIC(m), singular = isSingular(m))
  }
  lrt <- anova(fits$int_3way, fits$slopes_Area_Phase, fits$slopes_Area_Phase_AxP)
  cat("\n==", b, "== LRT intercepts-only -> +Area,Phase slopes -> +AxP slopes\n"); print(lrt)
}
tab <- rbindlist(out); tab[, `:=`(dAIC = AIC - min(AIC), dBIC = BIC - min(BIC)), by = band]
print(tab[, .(band, model, npar, logLik = round(logLik, 1), AIC = round(AIC, 1), BIC = round(BIC, 1), dAIC = round(dAIC, 1), dBIC = round(dBIC, 1), singular)])
fwrite(tab, "outputs/stats/revision_2026-10/lmm_ic_comparison_2026-10-14.csv")
