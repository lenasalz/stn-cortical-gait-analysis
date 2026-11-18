# Illustration: random-intercept vs random-slope fits, per participant
#
# Purpose:  Show what the two random-effect structures assume: fitted
#           gait-phase curves per participant from (a) the submitted
#           random-intercept model (parallel curves, one shape per cohort)
#           and (b) the random-slope model (each participant has their own
#           shape), next to the observed per-participant medians.
# Inputs:   code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_basenorm.csv
# Outputs:  outputs/figures/revision_2026-10/random_effects_illustration_<band>_<area>.png
# Author:   Lena
# Date:     2026-10-14
# Seed:     42
suppressPackageStartupMessages({ library(data.table); library(lme4) })
set.seed(42)
args <- commandArgs(trailingOnly = TRUE)
band <- if (length(args) >= 1) args[1] else "Gamma"; area_sel <- if (length(args) >= 2) args[2] else "motor"
df <- fread("code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_basenorm.csv")[Hemisphere == "Both" & Area == area_sel]
df[, `:=`(SubjectID = factor(SubjectID), Cohort = factor(Cohort, levels = c("AgeMatched_NFB", "Patient_NFB")),
          GaitPhase = factor(GaitPhase, levels = c("IDS", "LLS", "FDS", "RLS")), GaitCycle = factor(Epoch))]
y <- paste0("ERD_ERS_", band); setnames(df, y, "y")
for (g in c("LLS", "FDS", "RLS")) df[, (paste0("G_", g)) := as.numeric(GaitPhase == g)]
m_int <- lmer(y ~ Cohort * GaitPhase + (1 | SubjectID) + (1 | GaitCycle:SubjectID), data = df, REML = FALSE)
m_slp <- lmer(y ~ Cohort * GaitPhase + (1 + G_LLS + G_FDS + G_RLS || SubjectID) + (1 | GaitCycle:SubjectID), data = df, REML = FALSE,
              control = lmerControl(optimizer = "bobyqa"))
grid <- unique(df[, .(SubjectID, Cohort, GaitPhase, G_LLS, G_FDS, G_RLS)])
grid[, fit_int := predict(m_int, newdata = grid, re.form = ~ (1 | SubjectID))]
grid[, fit_slp := predict(m_slp, newdata = grid, re.form = ~ (1 + G_LLS + G_FDS + G_RLS || SubjectID))]
obs <- df[, .(obs = median(y)), by = .(SubjectID, Cohort, GaitPhase)]
grid <- merge(grid, obs, by = c("SubjectID", "Cohort", "GaitPhase"))
cols <- c(AgeMatched_NFB = "#1f77b4", Patient_NFB = "#d62728")
png(sprintf("outputs/figures/revision_2026-10/random_effects_illustration_%s_%s.png", band, area_sel), width = 1500, height = 520, res = 120)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
panel <- function(col, title) {
  plot(NA, xlim = c(0.8, 4.2), ylim = range(grid[[col]]) + c(-0.5, 0.5), xaxt = "n", xlab = "Gait phase", ylab = "gERD/ERS (dB vs rest)", main = title)
  axis(1, at = 1:4, labels = levels(grid$GaitPhase))
  for (s in levels(grid$SubjectID)) { d <- grid[SubjectID == s][order(GaitPhase)]; lines(1:4, d[[col]], col = adjustcolor(cols[as.character(d$Cohort[1])], 0.45), lwd = 1.5) }
  for (co in names(cols)) { d <- grid[Cohort == co, .(v = mean(get(col))), by = GaitPhase][order(GaitPhase)]; lines(1:4, d$v, col = cols[co], lwd = 4) }
  legend("bottomleft", c("HC", "PD"), col = cols, lwd = 3, bty = "n")
}
panel("obs", sprintf("Observed: per-participant medians (%s, %s)", band, area_sel))
panel("fit_int", "Random intercept only (submitted):\nsame shape per cohort, shifted per participant")
panel("fit_slp", "Random intercept + gait-phase slopes:\nown shape per participant")
dev.off()
vc <- as.data.frame(VarCorr(m_slp)); cat("Random-slope model variances (dB^2):\n"); print(vc[, c("grp", "var1", "vcov")])
cat("Cohort x GaitPhase: intercept-only p =", anova(m_int)[["F value"]][3] |> round(2), "F;  see lmerTest output for p\n")
