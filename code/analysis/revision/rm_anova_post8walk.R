# Classical mixed-design repeated-measures ANOVA on subject-level gERD/ERS, post8walk block
#
# Purpose:  Replicate rm_anova_classic.R (pre8walk, the manuscript block) on
#           the second walking block of the same session (post8walk). One
#           value per subject x area x gait phase (median across cycles, mean
#           as sensitivity), then a 2 (Cohort, between) x 2 (Area, within) x
#           4 (GaitPhase, within) RM-ANOVA with Greenhouse-Geisser correction
#           for the gait-phase terms, partial eta squared, and per-phase PD vs
#           HC Welch t-tests. STN (PD only): Hemisphere (between) x GaitPhase
#           (within). Model code is copied unchanged from rm_anova_classic.R;
#           only the input (per-cycle gERD tables of the post8walk block,
#           aggregated here) differs.
# Inputs:   data/derivatives/gerd/<subj>_post8walk-eeg-basenorm-gerd.csv  (Hemisphere == "Both")
#           data/derivatives/gerd/<subj>_post8walk-dbs-basenorm-gerd.csv
#               (compute_gerd_table.py, run 2026-10-14 for the post8walk block)
# Outputs:  outputs/stats/revision_2026-10/rm_anova_post8walk_cells_2026-10-14.csv
#           outputs/stats/revision_2026-10/rm_anova_post8walk_2026-10-14.txt
#           outputs/stats/revision_2026-10/rm_anova_post8walk_2026-10-14.csv
#           outputs/stats/revision_2026-10/rm_anova_post8walk_contrasts_2026-10-14.csv
# Author:   Lena
# Date:     2026-10-14
# Seed:     42 (set.seed; the analysis is deterministic)
#
# Caveats: post8walk derivatives come from the server proc-fif export (later
# preprocessing than the pickle-era pre8walk data); P4-2012 has no post8walk
# recording (n = 9 PD / 10 HC; STN n = 9); P4-1013 has 11 gait cycles.
# HC scaling: the HC post8walk proc fifs store microvolts as volts (1e6 x
# amplitude), the baseline block used for normalisation is in volts, so every
# HC basenorm value carries a constant 10*log10(1e12) = +120 dB (per-subject
# medians 116.6-124.0 dB vs PD -8.8 to +0.7 dB in the gait-warped maps).
# HC_CORRECTION_DB is subtracted from the HC basenorm values before
# aggregation. This is a constant per subject and leaves the within-subject
# terms (GaitPhase, Area and their interactions) unchanged; only the Cohort
# main effect and the PD - HC contrasts depend on it.
#
# Run from the project root:  Rscript code/analysis/revision/rm_anova_post8walk.R

suppressPackageStartupMessages({ library(data.table); library(dplyr) })
set.seed(42)
bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
ycols <- paste0("ERD_ERS_", bands)
out_cells <- "outputs/stats/revision_2026-10/rm_anova_post8walk_cells_2026-10-14.csv"
out_txt <- "outputs/stats/revision_2026-10/rm_anova_post8walk_2026-10-14.txt"
out_csv <- "outputs/stats/revision_2026-10/rm_anova_post8walk_2026-10-14.csv"
out_con <- "outputs/stats/revision_2026-10/rm_anova_post8walk_contrasts_2026-10-14.csv"

# --- Aggregate the per-cycle tables to one value per subject x cell -------
eeg_files <- Sys.glob("data/derivatives/gerd/P4-*_post8walk-eeg-basenorm-gerd.csv")
dbs_files <- Sys.glob("data/derivatives/gerd/P4-*_post8walk-dbs-basenorm-gerd.csv")
stopifnot(length(eeg_files) > 0, length(dbs_files) > 0)
df_c <- rbindlist(lapply(eeg_files, fread))[Hemisphere == "Both"]
df_s <- rbindlist(lapply(dbs_files, fread))
df_c[, Cohort := Group]
HC_CORRECTION_DB <- 120
df_c[Group == "HC", (ycols) := lapply(.SD, function(v) v - HC_CORRECTION_DB), .SDcols = ycols]
agg <- function(d, by, fun) d[, c(lapply(.SD, fun), list(n_cycles = .N)), by = by, .SDcols = ycols]
cells <- rbind(
  cbind(region = "cortical", summary = "median", as.data.frame(agg(df_c, c("SubjectID", "Cohort", "Area", "GaitPhase"), median)), Hemisphere = NA),
  cbind(region = "cortical", summary = "mean",   as.data.frame(agg(df_c, c("SubjectID", "Cohort", "Area", "GaitPhase"), mean)),   Hemisphere = NA),
  cbind(region = "STN", summary = "median", as.data.frame(agg(df_s, c("SubjectID", "Hemisphere", "GaitPhase"), median)), Cohort = "PD", Area = "STN"),
  cbind(region = "STN", summary = "mean",   as.data.frame(agg(df_s, c("SubjectID", "Hemisphere", "GaitPhase"), mean)),   Cohort = "PD", Area = "STN"),
  fill = TRUE)
cells <- as.data.table(cells)
fwrite(cells, out_cells)

gg_eps <- function(M) {
  k <- ncol(M); S <- cov(M)
  C <- diag(k) - matrix(1 / k, k, k)
  SC <- C %*% S %*% C
  ev <- Re(eigen(SC, only.values = TRUE)$values)[1:(k - 1)]
  sum(ev)^2 / ((k - 1) * sum(ev^2))
}

sink(out_txt)
cat("Classical mixed RM-ANOVA (aov with Error(SubjectID/(Area*GaitPhase))), subject medians across cycles — post8walk block\n")
cat("Greenhouse-Geisser epsilon applied to all terms involving GaitPhase.\n")
cat(sprintf("HC basenorm values corrected by -%g dB (microvolt-as-volt scaling of the HC post8walk export).\n", HC_CORRECTION_DB))
cat(sprintf("Input: %d EEG and %d STN per-subject post8walk gERD tables; cycles per cortical cell: min %d, median %g, max %d\n\n",
            length(eeg_files), length(dbs_files),
            min(cells[region == "cortical" & summary == "median"]$n_cycles),
            median(cells[region == "cortical" & summary == "median"]$n_cycles),
            max(cells[region == "cortical" & summary == "median"]$n_cycles)))
rows <- list(); con <- list()

for (summ in c("median", "mean")) {
  for (band in bands) {
    y <- paste0("ERD_ERS_", band)
    d <- cells[region == "cortical" & summary == summ]
    d[, `:=`(SubjectID = factor(SubjectID), Cohort = factor(Cohort, levels = c("HC", "PD")),
             Area = factor(Area), GaitPhase = factor(GaitPhase, levels = c("IDS", "LLS", "FDS", "RLS")))]
    d <- d[complete.cases(d[, ..y])]
    cat("\n======== cortical |", summ, "|", band, "| n =", uniqueN(d$SubjectID), "subjects,", nrow(d), "cells ========\n")
    f <- as.formula(paste(y, "~ Cohort * Area * GaitPhase + Error(SubjectID / (Area * GaitPhase))"))
    a <- aov(f, data = d)
    s <- summary(a)
    Mp <- dcast(d[, .(v = mean(get(y))), by = .(SubjectID, GaitPhase)], SubjectID ~ GaitPhase, value.var = "v")[, -1] |> as.matrix()
    Map <- dcast(d[, .(v = get(y)), by = .(SubjectID, Area, GaitPhase)], SubjectID ~ Area + GaitPhase, value.var = "v")[, -1] |> as.matrix()
    eps_p <- gg_eps(Mp); eps_ap <- gg_eps(Map)
    cat(sprintf("GG epsilon: GaitPhase %.3f, Area x GaitPhase %.3f\n", eps_p, eps_ap))
    for (stratum in names(s)) {
      tab <- s[[stratum]][[1]]
      if (is.null(tab) || !"F value" %in% colnames(tab)) next
      terms <- trimws(rownames(tab)); res_i <- which(terms == "Residuals")
      for (i in setdiff(seq_len(nrow(tab)), res_i)) {
        term <- terms[i]; Fv <- tab[i, "F value"]; df1 <- tab[i, "Df"]; df2 <- tab[res_i, "Df"]; p <- tab[i, "Pr(>F)"]
        eps <- if (grepl("GaitPhase", term) && grepl("Area", term)) eps_ap else if (grepl("GaitPhase", term)) eps_p else 1
        p_gg <- pf(Fv, df1 * eps, df2 * eps, lower.tail = FALSE)
        eta2p <- Fv * df1 / (Fv * df1 + df2)
        cat(sprintf("  %-24s F(%g, %g) = %6.2f  p = %.4f  GG eps = %.3f  p_GG = %.4f  eta2p = %.3f\n", term, df1, df2, Fv, p, eps, p_gg, eta2p))
        rows[[length(rows) + 1]] <- data.frame(block = "post8walk", region = "cortical", summary = summ, band = band, term = term, F = Fv, df1 = df1, df2 = df2,
                                               p = p, gg_eps = eps, p_gg = p_gg, eta2p = eta2p)
      }
    }
    for (ph in levels(d$GaitPhase)) {
      m <- d[GaitPhase == ph, .(v = mean(get(y))), by = .(SubjectID, Cohort)]
      tt <- t.test(v ~ Cohort, data = m)
      con[[length(con) + 1]] <- data.frame(block = "post8walk", region = "cortical", summary = summ, band = band, family = "PD_vs_HC_by_phase", level = ph,
                                           diff_PD_minus_HC = -diff(tt$estimate), ci_lo = -tt$conf.int[2], ci_hi = -tt$conf.int[1], t = -tt$statistic, df = tt$parameter, p = tt$p.value)
    }
    for (ar in levels(d$Area)) {
      m <- d[Area == ar, .(v = mean(get(y))), by = .(SubjectID, Cohort)]
      tt <- t.test(v ~ Cohort, data = m)
      con[[length(con) + 1]] <- data.frame(block = "post8walk", region = "cortical", summary = summ, band = band, family = "PD_vs_HC_by_area", level = ar,
                                           diff_PD_minus_HC = -diff(tt$estimate), ci_lo = -tt$conf.int[2], ci_hi = -tt$conf.int[1], t = -tt$statistic, df = tt$parameter, p = tt$p.value)
    }
  }
  for (band in bands) {
    y <- paste0("ERD_ERS_", band)
    d <- cells[region == "STN" & summary == summ]
    d[, `:=`(SubjectID = factor(SubjectID), Hemisphere = factor(Hemisphere), GaitPhase = factor(GaitPhase, levels = c("IDS", "LLS", "FDS", "RLS")))]
    d <- d[complete.cases(d[, ..y])]
    cat("\n======== STN |", summ, "|", band, "| n =", uniqueN(d$SubjectID), "subjects (", paste(table(unique(d[, .(SubjectID, Hemisphere)])$Hemisphere), collapse = "/"), ") ========\n")
    a <- aov(as.formula(paste(y, "~ Hemisphere * GaitPhase + Error(SubjectID / GaitPhase)")), data = d)
    s <- summary(a)
    Mp <- dcast(d[, .(v = get(y)), by = .(SubjectID, GaitPhase)], SubjectID ~ GaitPhase, value.var = "v")[, -1] |> as.matrix()
    eps_p <- gg_eps(Mp)
    for (stratum in names(s)) {
      tab <- s[[stratum]][[1]]; if (is.null(tab) || !"F value" %in% colnames(tab)) next
      terms <- trimws(rownames(tab)); res_i <- which(terms == "Residuals")
      for (i in setdiff(seq_len(nrow(tab)), res_i)) {
        term <- terms[i]; Fv <- tab[i, "F value"]; df1 <- tab[i, "Df"]; df2 <- tab[res_i, "Df"]; p <- tab[i, "Pr(>F)"]
        eps <- if (grepl("GaitPhase", term)) eps_p else 1
        p_gg <- pf(Fv, df1 * eps, df2 * eps, lower.tail = FALSE); eta2p <- Fv * df1 / (Fv * df1 + df2)
        cat(sprintf("  %-24s F(%g, %g) = %6.2f  p = %.4f  GG eps = %.3f  p_GG = %.4f  eta2p = %.3f\n", term, df1, df2, Fv, p, eps, p_gg, eta2p))
        rows[[length(rows) + 1]] <- data.frame(block = "post8walk", region = "STN", summary = summ, band = band, term = term, F = Fv, df1 = df1, df2 = df2,
                                               p = p, gg_eps = eps, p_gg = p_gg, eta2p = eta2p)
      }
    }
  }
}
tab <- bind_rows(rows) %>% group_by(region, summary, term) %>% mutate(p_gg_bh_bands = p.adjust(p_gg, "BH")) %>% ungroup()
ctab <- bind_rows(con) %>% group_by(region, summary, family, band) %>% mutate(p_bh = p.adjust(p, "BH")) %>% ungroup()
cat("\n\n================ Summary: cortical, subject medians, GG-corrected p (BH across bands in brackets) ================\n")
print(as.data.frame(tab %>% filter(region == "cortical", summary == "median") %>%
        transmute(band, term, F = round(F, 2), df1, df2, p_gg = signif(p_gg, 3), p_bh = signif(p_gg_bh_bands, 3), eta2p = round(eta2p, 3))))
cat("\nPD minus HC per gait phase (dB, Welch t on subject means; BH within band):\n")
print(as.data.frame(ctab %>% filter(region == "cortical", summary == "median", family == "PD_vs_HC_by_phase") %>%
        transmute(band, level, diff = round(diff_PD_minus_HC, 2), ci_lo = round(ci_lo, 2), ci_hi = round(ci_hi, 2), p = signif(p, 3), p_bh = signif(p_bh, 3))))
sink()
fwrite(tab, out_csv); fwrite(ctab, out_con); cat("wrote", out_csv, "\n")
