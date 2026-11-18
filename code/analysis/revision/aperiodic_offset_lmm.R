# Cohort x Condition mixed models on the aperiodic and periodic spectral components
#
# Purpose:  Test, per cortical region, whether the walking-vs-rest change in
#           the aperiodic component (FOOOF offset and exponent, and the
#           band-mean aperiodic fit in high beta and gamma) differs between
#           PD and HC, side by side with the same model on the periodic
#           (aperiodic-adjusted) band power and on total band power.
#           Model per region x measure: value ~ Cohort * Condition +
#           (1 | subject), REML, Satterthwaite df, type-III F tests, eta2p
#           from F and df; emmeans contrasts walking - rest within cohort and
#           the interaction contrast (PD change - HC change). Sensitivity:
#           without P4-2007 (near-flat rest spectrum) and without P4-2003
#           (lowest FOOOF R2).
# Inputs:   outputs/stats/revision_2026-10/aperiodic_offset_long_2026-10-14.csv
#               (aperiodic_offset_decomposition.py; subject x region x
#               condition x band x measure, variant A, min_peak_height 0.15)
# Outputs:  outputs/stats/revision_2026-10/aperiodic_offset_lmm_2026-10-14.txt
#           outputs/stats/revision_2026-10/aperiodic_offset_lmm_2026-10-14.csv
#               (one row per subset x region x band x measure x term)
#           outputs/stats/revision_2026-10/aperiodic_offset_lmm_contrasts_2026-10-14.csv
# Author:   Lena
# Date:     2026-10-14
# Seed:     42 (set.seed; models are deterministic, seed for completeness)
#
# Run from the project root:
#   Rscript code/analysis/revision/aperiodic_offset_lmm.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
set.seed(42)

in_csv  <- "outputs/stats/revision_2026-10/aperiodic_offset_long_2026-10-14.csv"
out_txt <- "outputs/stats/revision_2026-10/aperiodic_offset_lmm_2026-10-14.txt"
out_csv <- "outputs/stats/revision_2026-10/aperiodic_offset_lmm_2026-10-14.csv"
out_con <- "outputs/stats/revision_2026-10/aperiodic_offset_lmm_contrasts_2026-10-14.csv"

dat <- fread(in_csv)
dat[, Cohort := factor(cohort, levels = c("HC", "PD"))]
dat[, Condition := factor(condition, levels = c("rest", "walking"))]
dat[, subject := factor(subject)]

# measure x band cells to model (band "all" for the FOOOF parameters)
cells <- rbind(
  data.frame(measure = "aperiodic_offset",   band = "all"),
  data.frame(measure = "aperiodic_exponent", band = "all"),
  data.frame(measure = "aperiodic_band_db",  band = c("High Beta", "Gamma")),
  data.frame(measure = "periodic_band_db",   band = c("High Beta", "Gamma")),
  data.frame(measure = "total_band_db",      band = c("High Beta", "Gamma")))
subsets <- list(all = character(0), `without_P4-2007` = "P4-2007", `without_P4-2003` = "P4-2003")

sink(out_txt)
cat("Cohort x Condition mixed models on aperiodic and periodic components, run", format(Sys.time()), "\n")
cat("R", R.version.string, "; lme4", as.character(packageVersion("lme4")),
    "; lmerTest", as.character(packageVersion("lmerTest")),
    "; emmeans", as.character(packageVersion("emmeans")), "\n")
cat("Model: value ~ Cohort * Condition + (1 | subject), REML, Satterthwaite df, type-III F.\n")
cat("Units: offset and exponent in FOOOF log10 units; band measures in dB (10 x log10).\n")
cat("Contrast 'walking - rest' per cohort; 'interaction' = (PD walking - PD rest) - (HC walking - HC rest).\n\n")

anova_rows <- list(); contrast_rows <- list()
for (sub_name in names(subsets)) {
  d_sub <- dat[!(subject %in% subsets[[sub_name]])]
  cat("\n################ subset:", sub_name, "(n subjects =", length(unique(d_sub$subject)), ") ################\n")
  for (reg in c("frontal", "motor")) {
    for (i in seq_len(nrow(cells))) {
      ms <- cells$measure[i]; b <- cells$band[i]
      d <- d_sub[region == reg & measure == ms & band == b]
      if (nrow(d) == 0) { cat("Skipping", reg, ms, b, "\n"); next }
      m <- lmer(value ~ Cohort * Condition + (1 | subject), data = d, REML = TRUE)
      a <- anova(m, type = 3)
      a_df <- as.data.frame(a) %>% tibble::rownames_to_column("term")
      vc <- as.data.frame(VarCorr(m))
      for (k in seq_len(nrow(a_df))) {
        anova_rows[[length(anova_rows) + 1]] <- data.frame(
          subset = sub_name, region = reg, band = b, measure = ms, term = a_df$term[k],
          F = a_df$`F value`[k], df1 = a_df$NumDF[k], df2 = a_df$DenDF[k], p = a_df$`Pr(>F)`[k],
          eta2p = a_df$`F value`[k] * a_df$NumDF[k] / (a_df$`F value`[k] * a_df$NumDF[k] + a_df$DenDF[k]),
          subject_var = vc$vcov[1], residual_var = vc$vcov[2], singular = isSingular(m),
          n_obs = nrow(d), stringsAsFactors = FALSE)
      }
      em <- emmeans(m, ~ Condition | Cohort)
      e_within <- summary(pairs(em, reverse = TRUE, adjust = "none"), infer = TRUE)   # walking - rest per cohort
      e_inter  <- summary(contrast(emmeans(m, ~ Condition * Cohort), interaction = c("revpairwise", "revpairwise")),
                          infer = TRUE)                                                  # (walk-rest)_PD - (walk-rest)_HC
      e_cohort <- summary(pairs(emmeans(m, ~ Cohort | Condition), reverse = TRUE, adjust = "none"), infer = TRUE) # PD - HC per condition
      if (sub_name == "all") {
        cat("\n--- ", reg, "|", ms, "|", b, "(n obs =", nrow(d), ") ---\n")
        print(a)
        cat("Singular:", isSingular(m), " subject var:", signif(vc$vcov[1], 3), " residual var:", signif(vc$vcov[2], 3), "\n")
        cat("walking - rest within cohort:\n"); print(e_within)
        cat("interaction (PD change - HC change):\n"); print(e_inter)
        cat("PD - HC within condition:\n"); print(e_cohort)
      }
      add_con <- function(e, family, by_col) {
        e <- as.data.frame(e)
        by <- if (is.null(by_col)) rep("", nrow(e)) else as.character(e[[by_col]])
        contrast_rows[[length(contrast_rows) + 1]] <<- data.frame(
          subset = sub_name, region = reg, band = b, measure = ms, family = family,
          contrast = as.character(e[[1]]), by = by, estimate = e$estimate, SE = e$SE, df = e$df,
          lower = e$lower.CL, upper = e$upper.CL, t = e$t.ratio, p = e$p.value, stringsAsFactors = FALSE)
      }
      add_con(e_within, "walking_minus_rest_within_cohort", "Cohort")
      add_con(e_inter,  "interaction_PD_change_minus_HC_change", NULL)
      add_con(e_cohort, "PD_minus_HC_within_condition", "Condition")
    }
  }
}

anova_tab <- bind_rows(anova_rows)
contrast_tab <- bind_rows(contrast_rows)

cat("\n\n================ Summary table (subset all): F, p, eta2p per term ================\n")
s <- anova_tab %>% filter(subset == "all") %>%
  mutate(cell = paste(region, band, measure), stat = sprintf("F(%d,%.1f)=%.2f p=%.3f eta2p=%.2f", df1, df2, F, p, eta2p)) %>%
  select(cell, term, stat) %>% tidyr::pivot_wider(names_from = term, values_from = stat)
print(as.data.frame(s), right = FALSE)

cat("\n================ Interaction contrast (PD change - HC change), all subsets ================\n")
ci <- contrast_tab %>% filter(family == "interaction_PD_change_minus_HC_change") %>%
  mutate(est = sprintf("%+.2f [%+.2f, %+.2f] p=%.3f", estimate, lower, upper, p)) %>%
  select(subset, region, band, measure, est) %>% tidyr::pivot_wider(names_from = subset, values_from = est)
print(as.data.frame(ci), right = FALSE)

cat("\n================ walking - rest within cohort (subset all) ================\n")
cw <- contrast_tab %>% filter(subset == "all", family == "walking_minus_rest_within_cohort") %>%
  mutate(est = sprintf("%+.2f [%+.2f, %+.2f] p=%.3f", estimate, lower, upper, p)) %>%
  select(region, band, measure, by, est) %>% tidyr::pivot_wider(names_from = by, values_from = est)
print(as.data.frame(cw), right = FALSE)
sink()

fwrite(anova_tab, out_csv)
fwrite(contrast_tab, out_con)
cat("wrote", out_txt, out_csv, "and", out_con, "\n")
