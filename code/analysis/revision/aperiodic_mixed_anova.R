# Cohort x Condition mixed model on the aperiodic parameters of the unwarped spectra
#
# Purpose:  Reviewer comments 10 and 11 (npj Parkinson's Disease, decision
#           letter 2026-10-12). Test the aperiodic exponent and offset with
#           the same joint model as band power (psd_mixed_anova.R) instead of
#           separate within- and between-cohort tests on spectra from
#           time-warped epochs: Cohort (PD, HC) x Condition (rest, walking)
#           with a subject random intercept per cortical region; STN (PD
#           only): Condition with a subject random intercept.
# Inputs:   outputs/stats/revision_2026-10/fooof_unwarped_<date>.csv
#               (fooof_refit.py; spectrum variant A, min_peak_height 0.15)
# Outputs:  outputs/stats/revision_2026-10/aperiodic_mixed_anova_<date>.csv
#               (one row per region x parameter x term; p_fdr across the
#               two parameters within region and term)
#           outputs/stats/revision_2026-10/aperiodic_mixed_anova_desc_<date>.csv
#               (mean, SD, n per region x cohort x condition x parameter)
#           outputs/stats/revision_2026-10/aperiodic_mixed_anova_<date>.txt
# Author:   Lena
# Date:     2026-11-17
# Seed:     42 (set.seed; models are deterministic, seed for completeness)
#
# Run from the project root:
#   REVISION_DATE=2026-11-16 Rscript code/analysis/revision/aperiodic_mixed_anova.R

suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(lme4); library(lmerTest)
})
set.seed(42)

rev_date <- Sys.getenv("REVISION_DATE", "2026-11-16")
out_dir  <- "outputs/stats/revision_2026-10"
in_csv   <- file.path(out_dir, sprintf("fooof_unwarped_%s.csv", rev_date))
out_csv  <- file.path(out_dir, sprintf("aperiodic_mixed_anova_%s.csv", rev_date))
out_desc <- file.path(out_dir, sprintf("aperiodic_mixed_anova_desc_%s.csv", rev_date))
out_txt  <- file.path(out_dir, sprintf("aperiodic_mixed_anova_%s.txt", rev_date))

dat <- fread(in_csv)
dat <- dat[spectrum_variant == "A" & abs(min_peak_height - 0.15) < 1e-9]
dat[, Cohort := factor(ifelse(cohort == "Patient_NFB", "PD", "HC"), levels = c("HC", "PD"))]
dat[, Condition := factor(ifelse(block == "baseline", "rest", "walking"), levels = c("rest", "walking"))]
dat[, subject := factor(subject)]
params <- c(exponent = "aperiodic_exponent", offset = "aperiodic_offset")

sink(out_txt)
cat("Cohort x Condition mixed models on aperiodic parameters, run", format(Sys.time()), "\n")
cat(R.version.string, "; lme4", as.character(packageVersion("lme4")),
    "; lmerTest", as.character(packageVersion("lmerTest")), "; seed 42\n\n")

rows <- list(); desc <- list()
tidy <- function(a, region, param, singular) {
  a_df <- as.data.frame(a) %>% tibble::rownames_to_column("term")
  data.frame(region = region, parameter = param, term = a_df$term, F = a_df$`F value`,
             df1 = a_df$NumDF, df2 = a_df$DenDF, p = a_df$`Pr(>F)`, singular = singular,
             stringsAsFactors = FALSE)
}
for (reg in c("frontal", "motor", "STN")) {
  d_reg <- if (reg == "STN") dat[modality == "dbs"] else dat[modality == "eeg" & region == reg]
  for (pn in names(params)) {
    d <- as.data.frame(d_reg); d$value <- d[[params[[pn]]]]
    expected <- if (reg == "STN") 20 else 40
    if (nrow(d) != expected) { cat("Skipping", reg, pn, ": n =", nrow(d), "\n"); next }
    cat("\n---", reg, "|", pn, "(n =", nrow(d), ") ---\n")
    m <- if (reg == "STN") lmer(value ~ Condition + (1 | subject), data = d, REML = TRUE)
         else lmer(value ~ Cohort * Condition + (1 | subject), data = d, REML = TRUE)
    a <- anova(m, type = 3); print(a)
    cat("Singular:", isSingular(m), "\n")
    rows[[length(rows) + 1]] <- tidy(a, reg, pn, isSingular(m))
    desc[[length(desc) + 1]] <- d %>% group_by(Cohort, Condition) %>%
      summarise(mean = mean(value), sd = sd(value), n = n(), .groups = "drop") %>%
      mutate(region = reg, parameter = pn)
  }
}
tab <- bind_rows(rows) %>% group_by(region, term) %>%
  mutate(p_fdr = p.adjust(p, method = "BH"), n_tests_in_family = n()) %>% ungroup()
cat("\n=== Summary ===\n")
print(as.data.frame(tab %>% mutate(across(c(F, p, p_fdr), ~ signif(.x, 3)), df2 = round(df2, 1))), row.names = FALSE)
sink()
fwrite(tab, out_csv)
fwrite(bind_rows(desc), out_desc)
cat("wrote", out_csv, "\n")
