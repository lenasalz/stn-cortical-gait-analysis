# ===================================================================
# lmm_refit_tables.R
# -------------------------------------------------------------------
# Purpose: Build the Markdown comparison tables (submitted vs random-
#          slope vs aggregated models, per term and band; BH-corrected
#          p across bands; partial eta squared; random-effect variances;
#          emmeans contrasts) from the tidy CSVs written by
#          lmm_random_slopes.R and lmm_aggregated.R. The tables are
#          pasted into lmm_refit_report_<date>.md.
# Inputs:  outputs/stats/revision_2026-10/lmm_submitted_reproduction_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_varcorr_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_contrasts_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_random_slopes_ladder_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_aggregated_anova_2026-10-13[_<datatype>].csv
#          outputs/stats/revision_2026-10/lmm_aggregated_contrasts_2026-10-13[_<datatype>].csv
# Outputs: outputs/stats/revision_2026-10/lmm_refit_comparison_tables_2026-10-13[_<datatype>].md
#          outputs/stats/revision_2026-10/lmm_refit_anova_all_models_2026-10-13[_<datatype>].csv
# Author:  Lena
# Date:    2026-10-13
# Seed:    set.seed(42) (no random numbers are drawn)
# Usage:   Rscript code/analysis/revision/lmm_refit_tables.R [basenorm|avggait]
# ===================================================================

set.seed(42)
suppressPackageStartupMessages({ library(data.table); library(dplyr) })
# Output date tag; override with the environment variable REVISION_DATE for a rerun.
REVISION_DATE <- Sys.getenv("REVISION_DATE", "2026-10-13")
OUT_DIR <- "outputs/stats/revision_2026-10"
args     <- commandArgs(trailingOnly = TRUE)
datatype <- if (length(args) >= 1) args[1] else "basenorm"
suffix   <- if (datatype == "basenorm") "" else paste0("_", datatype)
infile   <- function(stem) file.path(OUT_DIR, paste0(stem, "_", REVISION_DATE, suffix, ".csv"))

sub <- fread(infile("lmm_submitted_reproduction_anova"))
rs  <- fread(infile("lmm_random_slopes_anova"))
agg <- fread(infile("lmm_aggregated_anova"))
vc  <- fread(infile("lmm_random_slopes_varcorr"))
ctr_rs  <- fread(infile("lmm_random_slopes_contrasts"))
ctr_agg <- fread(infile("lmm_aggregated_contrasts"))
ladder  <- fread(infile("lmm_random_slopes_ladder"))

all_models <- rbindlist(list(sub, rs, agg), fill = TRUE)
fwrite(all_models, infile("lmm_refit_anova_all_models"))

bands <- c("Theta", "Alpha", "Low Beta", "High Beta", "Gamma")
fmt_p <- function(p) ifelse(is.na(p), "-", ifelse(p < 0.001, formatC(p, format = "e", digits = 1), formatC(p, format = "f", digits = 3)))
fmt_F <- function(F, d1, d2) ifelse(is.na(F), "-", sprintf("%.2f (%d, %.0f)", F, d1, d2))
stars <- function(p) ifelse(is.na(p), "", ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", ""))))

cell <- function(tbl, mdl, reg, bnd, trm, with_fdr = TRUE) {
  r <- tbl[model == mdl & region == reg & band == bnd & term == trm]
  if (nrow(r) == 0) return("n/a")
  s <- sprintf("F=%s, p=%s%s", fmt_F(r$F, r$NumDF, r$DenDF), fmt_p(r$p), stars(r$p))
  if (with_fdr) s <- paste0(s, sprintf(", p_BH=%s, eta2p=%.3f", fmt_p(r$p_fdr_bands), r$eta2p))
  s
}

md <- c(sprintf("# LMM refit: comparison tables (%s, %s)", datatype, REVISION_DATE), "",
        "F (NumDF, DenDF), raw p, p_BH = Benjamini-Hochberg across the five bands within model and term, eta2p = approximate partial eta squared (F*df1/(F*df1+df2)). Stars: * p<0.05, ** p<0.01, *** p<0.001 (raw).", "")

for (reg in c("cortical", "STN")) {
  terms <- if (reg == "cortical") c("Cohort", "Cohort:Area", "Cohort:GaitPhase", "Area:GaitPhase", "Cohort:Area:GaitPhase") else c("Hemisphere", "GaitPhase", "Hemisphere:GaitPhase")
  md <- c(md, sprintf("## %s: submitted vs random-slope vs aggregated", reg), "",
          "| Band | Term | Submitted (intercepts only, cycle level) | Random slopes (cycle level) | Aggregated, median per cell | Aggregated, mean per cell |",
          "|---|---|---|---|---|---|")
  for (bnd in bands) for (trm in terms) {
    md <- c(md, sprintf("| %s | %s | %s | %s | %s | %s |", bnd, trm,
                        cell(sub, "submitted", reg, bnd, trm),
                        cell(rs, "random_slopes", reg, bnd, trm),
                        cell(agg, "aggregated_median", reg, bnd, trm),
                        cell(agg, "aggregated_mean", reg, bnd, trm, with_fdr = FALSE)))
  }
  md <- c(md, "")
  md <- c(md, sprintf("### %s: random-effects structure used per band", reg), "",
          "| Band | Ladder step | Structure | Singular | Warnings |", "|---|---|---|---|---|")
  for (bnd in bands) {
    r <- rs[region == reg & band == bnd & term == terms[1]]
    md <- c(md, sprintf("| %s | %d | `%s` | %s | %s |", bnd, r$ladder_step, r$structure, r$singular,
                        ifelse(is.na(r$warnings) | nchar(r$warnings) == 0, "none", r$warnings)))
  }
  md <- c(md, "", sprintf("### %s: ladder steps tried per band", reg), "",
          "| Band | Step | Structure | Fitted | Singular | Converged | Seconds |", "|---|---|---|---|---|---|---|")
  for (bnd in bands) {
    l <- ladder[region == reg & band == bnd]
    for (i in seq_len(nrow(l))) md <- c(md, sprintf("| %s | %d | %s | %s | %s | %s | %.0f |", bnd, l$step[i], l$label[i], l$fitted[i], l$singular[i], l$converged[i], l$seconds[i]))
  }
  md <- c(md, "", sprintf("### %s: random-effect variances of the random-slope model used", reg), "",
          "| Band | Group | Term | Variance | SD / corr |", "|---|---|---|---|---|")
  for (bnd in bands) {
    v <- vc[region == reg & band == bnd]
    for (i in seq_len(nrow(v))) md <- c(md, sprintf("| %s | %s | %s | %.4f | %.3f |", bnd, v$grp[i],
      ifelse(is.na(v$var2[i]) | v$var2[i] == "", ifelse(is.na(v$var1[i]) | v$var1[i] == "", "-", v$var1[i]), paste0("corr(", v$var1[i], ", ", v$var2[i], ")")), v$vcov[i], v$sdcor[i]))
  }
  md <- c(md, "")
}

# Contrast tables ------------------------------------------------------
ctr_tbl <- function(ct, mdl, reg, fam, label) {
  d <- ct[model == mdl & region == reg & family == fam]
  if (nrow(d) == 0) return(character(0))
  out <- c(sprintf("### %s: %s, %s (estimate in dB, 95%% CI unadjusted, p_FDR within family of %d contrasts per band)", reg, label, fam,
                   nrow(d[band == bands[1]])), "",
           "| Band | Area | GaitPhase | Contrast | Estimate | 95% CI | df | p raw | p FDR |", "|---|---|---|---|---|---|---|---|---|")
  for (bnd in bands) {
    r <- d[band == bnd]
    for (i in seq_len(nrow(r))) out <- c(out, sprintf("| %s | %s | %s | %s | %.2f | [%.2f, %.2f] | %.1f | %s | %s%s |", bnd,
      ifelse(is.na(r$Area[i]) | r$Area[i] == "", "-", r$Area[i]), ifelse(is.na(r$GaitPhase[i]) | r$GaitPhase[i] == "", "-", r$GaitPhase[i]),
      r$contrast[i], r$estimate[i], r$lower.CL[i], r$upper.CL[i], r$df[i], fmt_p(r$p_raw[i]), fmt_p(r$p_fdr_within_family[i]), stars(r$p_fdr_within_family[i])))
  }
  c(out, "")
}
md <- c(md, "## emmeans contrasts (PD - HC; STN: Left - Right)", "")
for (fam in c("Cohort:GaitPhase", "Cohort:Area", "Cohort:Area:GaitPhase")) {
  md <- c(md, ctr_tbl(ctr_rs, "random_slopes", "cortical", fam, "random-slope model"),
          ctr_tbl(ctr_agg, "aggregated_median", "cortical", fam, "aggregated model (median)"))
}
md <- c(md, ctr_tbl(ctr_rs, "random_slopes", "STN", "Hemisphere:GaitPhase", "random-slope model"),
        ctr_tbl(ctr_agg, "aggregated_median", "STN", "Hemisphere:GaitPhase", "aggregated model (median)"))

outmd <- file.path(OUT_DIR, paste0("lmm_refit_comparison_tables_", REVISION_DATE, suffix, ".md"))
writeLines(md, outmd)
cat("wrote", outmd, "\n")
