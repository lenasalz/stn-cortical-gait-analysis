# Cohort x Condition mixed model on aperiodic-adjusted band power
#
# Purpose:  Replace the separate within- and between-cohort tests on
#           zero-imputed FOOOF peak power by one mixed model per band and
#           region: Cohort (PD, HC; between) x Condition (rest, walking;
#           within) with a subject random intercept, on peak-free band
#           power above the aperiodic fit. STN (PD only): Condition only.
#           Also run on relative band power and on spectrum variant B as
#           robustness checks.
# Inputs:   outputs/stats/revision_2026-10/band_power_unwarped_2026-10-13.csv
#               (band_power_above_aperiodic.py)
# Outputs:  outputs/stats/revision_2026-10/psd_mixed_anova_2026-10-13.txt
#           outputs/stats/revision_2026-10/psd_mixed_anova_2026-10-13.csv
#               (one row per analysis x region x band x term)
#           outputs/stats/revision_2026-10/psd_mixed_anova_contrasts_2026-10-13.csv
#               (emmeans contrasts: PD vs HC within condition, walking vs
#               rest within cohort; FDR within family)
# Author:   Lena
# Date:     2026-10-13
# Seed:     42 (set.seed; models are deterministic, seed for completeness)
#
# Run from the project root:
#   Rscript code/analysis/revision/psd_mixed_anova.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
set.seed(42)

# Date tag; override with the environment variable REVISION_DATE for a rerun.
rev_date <- Sys.getenv("REVISION_DATE", "2026-10-13")
in_csv  <- sprintf("outputs/stats/revision_2026-10/band_power_unwarped_%s.csv", rev_date)
out_txt <- sprintf("outputs/stats/revision_2026-10/psd_mixed_anova_%s.txt", rev_date)
out_csv <- sprintf("outputs/stats/revision_2026-10/psd_mixed_anova_%s.csv", rev_date)
out_con <- sprintf("outputs/stats/revision_2026-10/psd_mixed_anova_contrasts_%s.csv", rev_date)

dat <- fread(in_csv)
dat[, Cohort := factor(ifelse(cohort == "Patient_NFB", "PD", "HC"), levels = c("HC", "PD"))]
dat[, Condition := factor(ifelse(block == "baseline", "rest", "walking"), levels = c("rest", "walking"))]
dat[, subject := factor(subject)]
bands <- c("Theta", "Alpha", "Low Beta", "High Beta", "Gamma")

# Analyses: primary = variant A, min_peak_height 0.15, adjusted power.
analyses <- list(
  list(name = "primary_A_adj",   variant = "A", mph = 0.15, measure = "adj_power"),
  list(name = "robust_B_adj",    variant = "B", mph = 0.15, measure = "adj_power"),
  list(name = "robust_A_rel",    variant = "A", mph = 0.15, measure = "rel_power")
)

sink(out_txt)
cat("Cohort x Condition mixed models on band power, run", format(Sys.time()), "\n")
cat("R", R.version.string, "; lme4", as.character(packageVersion("lme4")),
    "; lmerTest", as.character(packageVersion("lmerTest")), "\n\n")

anova_rows <- list(); contrast_rows <- list()

for (an in analyses) {
  d_an <- dat[spectrum_variant == an$variant & abs(min_peak_height - an$mph) < 1e-9 & measure == an$measure]
  cat("\n################ ", an$name, " ################\n")

  # ---- Cortical: Cohort x Condition + (1|subject), per region x band ----
  for (reg in c("frontal", "motor")) {
    for (b in bands) {
      d <- d_an[modality == "eeg" & region == reg & band == b]
      if (nrow(d) != 40) { cat("Skipping", reg, b, ": n =", nrow(d), "\n"); next }
      cat("\n--- ", an$name, "| EEG", reg, "|", b, "(n =", nrow(d), ") ---\n")
      m <- lmer(value ~ Cohort * Condition + (1 | subject), data = d, REML = TRUE)
      a <- anova(m, type = 3)
      print(a)
      vc <- as.data.frame(VarCorr(m))
      cat("Singular:", isSingular(m), " subject var:", signif(vc$vcov[1], 3),
          " residual var:", signif(vc$vcov[2], 3), "\n")
      a_df <- as.data.frame(a) %>% tibble::rownames_to_column("term")
      for (i in seq_len(nrow(a_df))) {
        anova_rows[[length(anova_rows) + 1]] <- data.frame(
          analysis = an$name, modality = "eeg", region = reg, band = b, term = a_df$term[i],
          F = a_df$`F value`[i], df1 = a_df$NumDF[i], df2 = a_df$DenDF[i], p = a_df$`Pr(>F)`[i],
          eta2p = a_df$`F value`[i] * a_df$NumDF[i] / (a_df$`F value`[i] * a_df$NumDF[i] + a_df$DenDF[i]),
          singular = isSingular(m), stringsAsFactors = FALSE)
      }
      # Contrasts (always computed; interpret only where the interaction is significant)
      e1 <- summary(pairs(emmeans(m, ~ Cohort | Condition), adjust = "none"), infer = TRUE)
      e2 <- summary(pairs(emmeans(m, ~ Condition | Cohort), adjust = "none"), infer = TRUE)
      cat("PD vs HC within condition:\n"); print(e1)
      cat("rest vs walking within cohort:\n"); print(e2)
      for (e in list(cbind(family = "PD_vs_HC_within_condition", as.data.frame(e1)),
                     cbind(family = "rest_vs_walking_within_cohort", as.data.frame(e2)))) {
        names(e)[names(e) %in% c("Condition", "Cohort")] <- "by"
        contrast_rows[[length(contrast_rows) + 1]] <- data.frame(
          analysis = an$name, modality = "eeg", region = reg, band = b, family = e$family,
          contrast = e$contrast, by = e$by, estimate = e$estimate, SE = e$SE, df = e$df,
          lower = e$lower.CL, upper = e$upper.CL, t = e$t.ratio, p = e$p.value,
          stringsAsFactors = FALSE)
      }
    }
  }

  # ---- STN (PD only): Condition + (1|subject), per band ----
  for (b in bands) {
    d <- d_an[modality == "dbs" & band == b]
    if (nrow(d) != 20) { cat("Skipping STN", b, ": n =", nrow(d), "\n"); next }
    cat("\n--- ", an$name, "| STN |", b, "(n =", nrow(d), ") ---\n")
    m <- lmer(value ~ Condition + (1 | subject), data = d, REML = TRUE)
    a <- anova(m, type = 3); print(a)
    a_df <- as.data.frame(a) %>% tibble::rownames_to_column("term")
    anova_rows[[length(anova_rows) + 1]] <- data.frame(
      analysis = an$name, modality = "dbs", region = "STN", band = b, term = a_df$term[1],
      F = a_df$`F value`[1], df1 = a_df$NumDF[1], df2 = a_df$DenDF[1], p = a_df$`Pr(>F)`[1],
      eta2p = a_df$`F value`[1] * a_df$NumDF[1] / (a_df$`F value`[1] * a_df$NumDF[1] + a_df$DenDF[1]),
      singular = isSingular(m), stringsAsFactors = FALSE)
    e2 <- summary(pairs(emmeans(m, ~ Condition), adjust = "none"), infer = TRUE)
    print(e2)
    contrast_rows[[length(contrast_rows) + 1]] <- data.frame(
      analysis = an$name, modality = "dbs", region = "STN", band = b,
      family = "rest_vs_walking", contrast = e2$contrast, by = "PD",
      estimate = e2$estimate, SE = e2$SE, df = e2$df, lower = e2$lower.CL,
      upper = e2$upper.CL, t = e2$t.ratio, p = e2$p.value, stringsAsFactors = FALSE)
  }
}

anova_tab <- bind_rows(anova_rows)
# FDR across the five bands, within analysis x region x term
anova_tab <- anova_tab %>% group_by(analysis, region, term) %>%
  mutate(p_fdr_bands = p.adjust(p, method = "BH")) %>% ungroup()
contrast_tab <- bind_rows(contrast_rows) %>%
  group_by(analysis, region, family) %>%          # family = 5 bands x 2 levels of "by"
  mutate(p_fdr = p.adjust(p, method = "BH")) %>% ungroup()

cat("\n\n================ Summary (primary analysis, FDR across bands) ================\n")
print(as.data.frame(anova_tab %>% filter(analysis == "primary_A_adj") %>%
        select(region, band, term, F, df1, df2, p, p_fdr_bands, eta2p) %>%
        mutate(across(c(F, p, p_fdr_bands, eta2p), ~ signif(.x, 3)), df2 = round(df2, 1))))
sink()

fwrite(anova_tab, out_csv)
fwrite(contrast_tab, out_con)
cat("wrote", out_csv, "and", out_con, "\n")
