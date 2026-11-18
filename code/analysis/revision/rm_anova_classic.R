# Classical mixed-design repeated-measures ANOVA on subject-level gERD/ERS
#
# Purpose:  The conventional version of the aggregated analysis the reviewer
#           asked for (R1.4): one value per subject x region x gait phase
#           (median across cycles), then a 2 (Cohort, between) x 2 (Region,
#           within) x 4 (GaitPhase, within) repeated-measures ANOVA with
#           Greenhouse-Geisser correction for the gait-phase terms, partial
#           eta squared, and per-phase PD vs HC follow-up t-tests. STN
#           (PD only): Hemisphere (between) x GaitPhase (within).
# Inputs:   outputs/stats/revision_2026-10/lmm_aggregated_cells_2026-10-13.csv
#               (lmm_aggregated.R; subject-level medians and means)
# Outputs:  outputs/stats/revision_2026-10/rm_anova_classic_2026-10-14.txt
#           outputs/stats/revision_2026-10/rm_anova_classic_2026-10-14.csv
#           outputs/stats/revision_2026-10/rm_anova_classic_contrasts_2026-10-14.csv
# Author:   Lena
# Date:     2026-10-14
# Seed:     42 (set.seed; the analysis is deterministic)
#
# Run from the project root:  Rscript code/analysis/revision/rm_anova_classic.R

suppressPackageStartupMessages({ library(data.table); library(dplyr) })
set.seed(42)
# Date tags; override with the environment variable REVISION_DATE for a rerun.
in_date  <- Sys.getenv("REVISION_DATE", "2026-10-13")
out_date <- Sys.getenv("REVISION_DATE", "2026-10-14")
cells <- fread(sprintf("outputs/stats/revision_2026-10/lmm_aggregated_cells_%s.csv", in_date))
bands <- c("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
out_txt <- sprintf("outputs/stats/revision_2026-10/rm_anova_classic_%s.txt", out_date)
out_csv <- sprintf("outputs/stats/revision_2026-10/rm_anova_classic_%s.csv", out_date)
out_con <- sprintf("outputs/stats/revision_2026-10/rm_anova_classic_contrasts_%s.csv", out_date)

# Greenhouse-Geisser epsilon for a within-subject factor from the subject x level matrix
gg_eps <- function(M) {            # M: subjects x k levels (cell means per subject)
  k <- ncol(M); S <- cov(M)
  C <- diag(k) - matrix(1 / k, k, k)          # centring contrast (k-1 dims)
  SC <- C %*% S %*% C
  ev <- Re(eigen(SC, only.values = TRUE)$values)[1:(k - 1)]
  sum(ev)^2 / ((k - 1) * sum(ev^2))
}

sink(out_txt)
cat("Classical mixed RM-ANOVA (aov with Error(SubjectID/(Area*GaitPhase))), subject medians across cycles\n")
cat("Greenhouse-Geisser epsilon applied to all terms involving GaitPhase.\n\n")
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
    # GG epsilon for GaitPhase (subject x phase means, averaged over Area) and Area x GaitPhase (8 cells)
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
        rows[[length(rows) + 1]] <- data.frame(region = "cortical", summary = summ, band = band, term = term, F = Fv, df1 = df1, df2 = df2,
                                               p = p, gg_eps = eps, p_gg = p_gg, eta2p = eta2p)
      }
    }
    # follow-up: PD vs HC per gait phase (averaged over Area) and per Area (averaged over phase), Welch t
    for (ph in levels(d$GaitPhase)) {
      m <- d[GaitPhase == ph, .(v = mean(get(y))), by = .(SubjectID, Cohort)]
      tt <- t.test(v ~ Cohort, data = m)
      con[[length(con) + 1]] <- data.frame(region = "cortical", summary = summ, band = band, family = "PD_vs_HC_by_phase", level = ph,
                                           diff_PD_minus_HC = -diff(tt$estimate), ci_lo = -tt$conf.int[2], ci_hi = -tt$conf.int[1], t = -tt$statistic, df = tt$parameter, p = tt$p.value)
    }
    for (ar in levels(d$Area)) {
      m <- d[Area == ar, .(v = mean(get(y))), by = .(SubjectID, Cohort)]
      tt <- t.test(v ~ Cohort, data = m)
      con[[length(con) + 1]] <- data.frame(region = "cortical", summary = summ, band = band, family = "PD_vs_HC_by_area", level = ar,
                                           diff_PD_minus_HC = -diff(tt$estimate), ci_lo = -tt$conf.int[2], ci_hi = -tt$conf.int[1], t = -tt$statistic, df = tt$parameter, p = tt$p.value)
    }
  }
  # ---- STN: Hemisphere (between) x GaitPhase (within), PD only ----
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
        rows[[length(rows) + 1]] <- data.frame(region = "STN", summary = summ, band = band, term = term, F = Fv, df1 = df1, df2 = df2,
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
