# Does cycle-to-cycle variability of the gait-phase response differ between cohorts?
#
# Purpose:  Formal test of the hypothesis that the within-subject,
#           cycle-to-cycle variation of gait-phase-locked power (not its mean
#           pattern) differs between PD and HC: the same mixed model with
#           either one cycle-level residual variance or a separate one per
#           cohort (nlme varIdent), compared by likelihood-ratio test, per
#           band and normalisation.
# Inputs:   code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_{basenorm,avggait}.csv
# Outputs:  outputs/stats/revision_2026-10/cycle_variance_by_cohort_2026-10-14.{csv,txt}
# Author:   Lena
# Date:     2026-10-14
# Seed:     42
suppressPackageStartupMessages({ library(data.table); library(nlme) })
set.seed(42)
bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma"); out <- list()
sink("outputs/stats/revision_2026-10/cycle_variance_by_cohort_2026-10-14.txt")
cat("Cohort-specific cycle-level residual variance (nlme::lme, varIdent(~1|Cohort)), ML fits\n")
cat("Model: ERD ~ Cohort*Area*GaitPhase, random = ~1 | SubjectID/GaitCycle\n\n")
for (norm in c("basenorm", "avggait")) {
  df <- fread(sprintf("code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_%s.csv", norm))[Hemisphere == "Both"]
  df[, `:=`(SubjectID = factor(SubjectID), Cohort = factor(Cohort, levels = c("AgeMatched_NFB", "Patient_NFB")),
            Area = factor(Area), GaitPhase = factor(GaitPhase), GaitCycle = factor(Epoch))]
  for (b in bands) {
    y <- paste0("ERD_ERS_", b); d <- copy(df); setnames(d, y, "y"); d <- d[!is.na(y)]
    m0 <- lme(y ~ Cohort * Area * GaitPhase, random = ~ 1 | SubjectID / GaitCycle, data = d, method = "ML",
              control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200))
    m1 <- update(m0, weights = varIdent(form = ~ 1 | Cohort))
    lrt <- anova(m0, m1)
    ratio <- coef(m1$modelStruct$varStruct, unconstrained = FALSE)   # SD multiplier for the non-reference level (PD)
    sd_hc <- m1$sigma; sd_pd <- m1$sigma * as.numeric(ratio)
    cat(sprintf("%-9s %-10s cycle-level residual SD: HC %.2f dB, PD %.2f dB (ratio %.3f)  LRT chi2(1) = %.2f, p = %.4f\n",
                norm, b, sd_hc, sd_pd, as.numeric(ratio), lrt$L.Ratio[2], lrt$`p-value`[2]))
    out[[length(out) + 1]] <- data.frame(norm = norm, band = b, sd_hc = sd_hc, sd_pd = sd_pd, ratio_pd_hc = as.numeric(ratio),
                                         lrt_chi2 = lrt$L.Ratio[2], lrt_p = lrt$`p-value`[2], AIC_equal = AIC(m0), AIC_cohort = AIC(m1))
  }
}
sink()
fwrite(rbindlist(out), "outputs/stats/revision_2026-10/cycle_variance_by_cohort_2026-10-14.csv")
cat(readLines("outputs/stats/revision_2026-10/cycle_variance_by_cohort_2026-10-14.txt"), sep = "\n")
