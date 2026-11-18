# ===================================================================
# null_strength.R
# -------------------------------------------------------------------
# Purpose: Quantify how strong the null result for phase-specific
#          cohort differences in cortical gERD/ERS is (Reviewer 1,
#          comment 4 follow-up). After the random-slope / aggregated
#          refits, Cohort x GaitPhase is n.s. in every band; this script
#          asks how large an effect the data can still hide.
#          1. Equivalence tests (TOST, Welch) on subject-level cells:
#             per subject the phase pattern is centred on the subject
#             mean over phases (removes the cohort offset); the PD - HC
#             difference of the centred value per phase is the
#             interaction contrast. Bounds +/- 1.0 dB and +/- 0.5 dB
#             (pre-specified: about 40 % and 20 % of the observed gamma
#             offset of -2.4 to -2.8 dB). Same for the Area x Cohort
#             offset difference (motor - frontal, PD vs HC) and, as a
#             supplement, the three-way pattern (motor - frontal
#             centred pattern per phase, PD vs HC).
#          2. Bayes factors for Cohort:GaitPhase, the three-way term and
#             the Cohort main effect from the BIC approximation
#             (Wagenmakers 2007, BF01 = exp((BIC_full - BIC_reduced)/2))
#             on the aggregated model y ~ Cohort*Area*GaitPhase +
#             (1|SubjectID), ML. The approximation corresponds to a
#             unit-information prior on the dropped coefficients; BIC
#             uses lme4's nobs (160 cells), not the 20 subjects, so the
#             implied prior is the lme4/BIC default and the values are
#             indicative, not a full Bayesian analysis.
#          3. Simulation-based power / minimal detectable effect for
#             Cohort x GaitPhase in the cycle-level random-slope model
#             ERD ~ Cohort*Area*GaitPhase +
#             (1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID) +
#             (1 | GaitCycle:SubjectID), ML (same structure and indicator
#             coding as lmm_random_slopes.R, ladder step 4). The model is
#             fitted to the observed data per band; all Cohort:GaitPhase
#             and Cohort:Area:GaitPhase coefficients are set to zero
#             (null), then a PD - HC difference of delta dB is added in
#             the RLS phase only (coefficient CohortPD:GaitPhaseRLS, both
#             areas), delta in {0, 0.5, 1, 1.5, 2, 3}. Responses are
#             simulated from that parameter set with the observed design
#             (n = 10 per group, observed cycles), the same model is
#             refitted and the Type III Satterthwaite F test for
#             Cohort:GaitPhase recorded. Power = proportion p < 0.05;
#             delta = 0 gives the empirical type I error.
# Inputs:  outputs/stats/revision_2026-10/lmm_aggregated_cells_2026-10-13.csv
#          code/analysis/revision/null_strength_wording.txt (draft wording pasted into the report)
#          outputs/stats/revision_2026-10/lmm_random_slopes_varcorr_2026-10-13.csv
#              (read for the log only; the simulation re-estimates the
#              components from the cycle-level data)
#          code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_basenorm.csv
#          code/analysis/revision/lmm_revision_common.R (loader, control)
# Outputs: outputs/stats/revision_2026-10/null_strength_2026-10-14.txt   (log)
#          outputs/stats/revision_2026-10/null_strength_2026-10-14.csv   (all tables, long)
#          outputs/stats/revision_2026-10/null_strength_power_sims_2026-10-14.csv
#              (one row per simulation)
#          outputs/stats/revision_2026-10/null_strength_report_2026-10-14.md
# Author:  Lena
# Date:    2026-10-14
# Seed:    set.seed(42); simulate() calls use seed 42 + band index +
#          100 * effect index; refits run in forked workers but all
#          randomness is drawn in the master process.
# Usage:   Rscript code/analysis/revision/null_strength.R [nsim] [bands]
#          nsim = 0 runs the equivalence tests and Bayes factors only
#          (no simulation; a few seconds).
#          nsim default 500; bands default "all" or a comma list such as
#          "Gamma,Theta" (run in that order). Run from the project root.
#          Set OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 so that the
#          forked workers do not contend for BLAS threads.
# ===================================================================

set.seed(42)
suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(lme4); library(lmerTest); library(parallel)
})
source("code/analysis/revision/lmm_revision_common.R")   # load_cortical, band_col, LMM_CONTROL, OUT_DIR

args    <- commandArgs(trailingOnly = TRUE)
NSIM    <- if (length(args) >= 1) as.integer(args[1]) else 500L
BANDSEL <- if (length(args) >= 2 && args[2] != "all") strsplit(args[2], ",")[[1]] else BANDS
# Date tags; override both with the environment variable REVISION_DATE for a rerun.
DATE    <- Sys.getenv("REVISION_DATE", "2026-10-14")
IN_DATE <- Sys.getenv("REVISION_DATE", "2026-10-13")
out_txt <- file.path(OUT_DIR, paste0("null_strength_", DATE, ".txt"))
out_csv <- file.path(OUT_DIR, paste0("null_strength_", DATE, ".csv"))
out_sim <- file.path(OUT_DIR, paste0("null_strength_power_sims_", DATE, ".csv"))
out_md  <- file.path(OUT_DIR, paste0("null_strength_report_", DATE, ".md"))

SESOI   <- c(1.0, 0.5)
DELTAS  <- c(0, 0.5, 1.0, 1.5, 2.0, 3.0)
PHASES  <- c("FDS", "IDS", "LLS", "RLS")
N_CORES <- max(1L, min(8L, detectCores() - 2L))

log_con <- open_log(out_txt)
session_header("code/analysis/revision/null_strength.R")
cat("nsim:", NSIM, " bands:", paste(BANDSEL, collapse = ", "), " workers:", N_CORES, "\n\n")

rows <- list()
add_rows <- function(df) rows[[length(rows) + 1]] <<- as.data.frame(df)

# ===================================================================
# 1. Equivalence tests on subject-level cells
# ===================================================================
cat("###################################################################\n")
cat("# 1. TOST (Welch) on subject-level cells, cortical\n")
cat("###################################################################\n")
cells <- fread(file.path(OUT_DIR, paste0("lmm_aggregated_cells_", IN_DATE, ".csv")))
cells <- cells[region == "cortical"]
cells[, Cohort := factor(Cohort, levels = c("HC", "PD"))]
cat(sprintf("cells: %d rows; subjects per cohort: %s\n\n",
            nrow(cells), paste(names(table(unique(cells[, .(SubjectID, Cohort)])$Cohort)),
                               table(unique(cells[, .(SubjectID, Cohort)])$Cohort), collapse = ", ")))

# Welch t with 90 % CI and TOST p for bounds +/- b. x = PD values, y = HC values.
welch_tost <- function(x, y, bounds) {
  nx <- length(x); ny <- length(y)
  se <- sqrt(var(x) / nx + var(y) / ny)
  df <- se^4 / ((var(x) / nx)^2 / (nx - 1) + (var(y) / ny)^2 / (ny - 1))
  est <- mean(x) - mean(y)
  ci90 <- est + c(-1, 1) * qt(0.95, df) * se
  ci95 <- est + c(-1, 1) * qt(0.975, df) * se
  p_nhst <- 2 * pt(-abs(est / se), df)
  out <- data.frame(estimate = est, se = se, df = df, ci90_lo = ci90[1], ci90_hi = ci90[2],
                    ci95_lo = ci95[1], ci95_hi = ci95[2], p_welch = p_nhst)
  for (b in bounds) {
    p_lo <- pt((est + b) / se, df, lower.tail = FALSE)   # H0: est <= -b
    p_hi <- pt((est - b) / se, df, lower.tail = TRUE)    # H0: est >= +b
    p_tost <- max(p_lo, p_hi)
    tag <- sub("\\.", "p", sprintf("%.1f", b))
    out[[paste0("tost_p_", tag)]] <- p_tost
    out[[paste0("equiv_", tag)]] <- (ci90[1] > -b) & (ci90[2] < b)
  }
  out
}

tost_block <- function(summ) {
  d <- cells[summary == summ]
  res <- list()
  for (band in BANDS) {
    y <- band_col(band)
    # per subject x phase, averaged over Area
    sp <- d[, .(v = mean(get(y))), by = .(SubjectID, Cohort, GaitPhase)]
    sp[, v_c := v - mean(v), by = SubjectID]                      # centred on subject mean over phases
    for (ph in PHASES) {
      x <- sp[GaitPhase == ph & Cohort == "PD"]$v_c; yy <- sp[GaitPhase == ph & Cohort == "HC"]$v_c
      res[[length(res) + 1]] <- cbind(block = "tost_Cohort_x_GaitPhase", summary = summ, band = band,
                                      term = "Cohort:GaitPhase", level = ph, welch_tost(x, yy, SESOI))
    }
    # Cohort offset per phase (not centred) for reference
    for (ph in PHASES) {
      x <- sp[GaitPhase == ph & Cohort == "PD"]$v; yy <- sp[GaitPhase == ph & Cohort == "HC"]$v
      res[[length(res) + 1]] <- cbind(block = "welch_Cohort_offset_per_phase", summary = summ, band = band,
                                      term = "Cohort|GaitPhase", level = ph, welch_tost(x, yy, SESOI))
    }
    # Area x Cohort: motor - frontal per subject (mean over phases)
    sa <- d[, .(v = mean(get(y))), by = .(SubjectID, Cohort, Area)]
    sa <- dcast(sa, SubjectID + Cohort ~ Area, value.var = "v")
    sa[, diff := motor - frontal]
    res[[length(res) + 1]] <- cbind(block = "tost_Cohort_x_Area", summary = summ, band = band,
                                    term = "Cohort:Area", level = "motor-frontal",
                                    welch_tost(sa[Cohort == "PD"]$diff, sa[Cohort == "HC"]$diff, SESOI))
    # three-way: centred pattern per area, motor - frontal per phase
    sap <- d[, .(v = get(y)), by = .(SubjectID, Cohort, Area, GaitPhase)]
    sap[, v_c := v - mean(v), by = .(SubjectID, Area)]
    sap <- dcast(sap, SubjectID + Cohort + GaitPhase ~ Area, value.var = "v_c")
    sap[, diff := motor - frontal]
    for (ph in PHASES) {
      res[[length(res) + 1]] <- cbind(block = "tost_Cohort_x_Area_x_GaitPhase", summary = summ, band = band,
                                      term = "Cohort:Area:GaitPhase", level = ph,
                                      welch_tost(sap[GaitPhase == ph & Cohort == "PD"]$diff,
                                                 sap[GaitPhase == ph & Cohort == "HC"]$diff, SESOI))
    }
  }
  do.call(rbind, res)
}

for (summ in c("median", "mean")) {
  tb <- tost_block(summ)
  add_rows(tb)
  cat("\n--- summary =", summ, if (summ == "median") "(primary)" else "(sensitivity)", "---\n")
  show <- tb[tb$block %in% c("tost_Cohort_x_GaitPhase", "tost_Cohort_x_Area"), ]
  show <- show[order(match(show$band, BANDS)), ]
  cat(sprintf("%-10s %-22s %-13s %7s [%7s, %7s] %6s | TOST p (1.0) %6s eq %-5s | TOST p (0.5) %6s eq %-5s\n",
              "band", "term", "level", "PD-HC", "90lo", "90hi", "p", "", "", "", ""))
  for (i in seq_len(nrow(show))) with(show[i, ], cat(sprintf(
    "%-10s %-22s %-13s %7.2f [%7.2f, %7.2f] %6.3f | %18.3f    %-5s | %18.3f    %-5s\n",
    band, term, level, estimate, ci90_lo, ci90_hi, p_welch, tost_p_1p0, equiv_1p0, tost_p_0p5, equiv_0p5)))
}

# ===================================================================
# 2. BIC Bayes factors on the aggregated model
# ===================================================================
cat("\n###################################################################\n")
cat("# 2. BIC-approximated Bayes factors, aggregated model (median), ML\n")
cat("#    BF01 = exp((BIC_full - BIC_reduced)/2) (Wagenmakers 2007)\n")
cat("#    nobs used by BIC = number of cells (lme4 default)\n")
cat("###################################################################\n")
agg <- as.data.frame(cells[summary == "median"])
agg$SubjectID <- factor(agg$SubjectID); agg$Area <- factor(agg$Area); agg$GaitPhase <- factor(agg$GaitPhase)
agg$Cohort <- relevel(factor(agg$Cohort), ref = "HC")

bf_fit <- function(rhs, y) {
  m <- lmer(as.formula(paste(y, "~", rhs, "+ (1 | SubjectID)")), data = agg, REML = FALSE, control = LMM_CONTROL)
  list(bic = BIC(m), ll = as.numeric(logLik(m)), singular = isSingular(m), k = attr(logLik(m), "df"))
}
bf_specs <- list(
  list(term = "Cohort:Area:GaitPhase",
       full = "Cohort * Area * GaitPhase",
       red  = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase"),
  list(term = "Cohort:GaitPhase (given no three-way)",
       full = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase",
       red  = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase - Cohort:GaitPhase"),
  list(term = "Cohort:GaitPhase + Cohort:Area:GaitPhase (joint)",
       full = "Cohort * Area * GaitPhase",
       red  = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase - Cohort:GaitPhase"),
  list(term = "Cohort:Area (given no three-way)",
       full = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase",
       red  = "Cohort * Area * GaitPhase - Cohort:Area:GaitPhase - Cohort:Area"),
  list(term = "Cohort main effect (no Cohort interactions)",
       full = "Cohort + Area * GaitPhase",
       red  = "Area * GaitPhase")
)
cat(sprintf("%-10s %-50s %9s %9s %8s %8s  %s\n", "band", "term", "BIC_full", "BIC_red", "BF01", "BF10", "LRT | BF01 with n = subjects | singular(full/red)"))
for (band in BANDS) {
  y <- band_col(band)
  for (s in bf_specs) {
    f <- bf_fit(s$full, y); r <- bf_fit(s$red, y)
    bf01 <- exp((f$bic - r$bic) / 2)
    lr <- 2 * (f$ll - r$ll); dfk <- f$k - r$k; p_lrt <- pchisq(lr, dfk, lower.tail = FALSE)
    # BIC with the number of subjects (20) as effective sample size: between-subject
    # terms are informed by 20 independent units, not 160 cells; more conservative.
    n_subj <- nlevels(agg$SubjectID)
    bic_f20 <- -2 * f$ll + f$k * log(n_subj); bic_r20 <- -2 * r$ll + r$k * log(n_subj)
    bf01_20 <- exp((bic_f20 - bic_r20) / 2)
    add_rows(data.frame(block = "bic_bayes_factor", summary = "median", band = band, term = s$term, level = "",
                        full = s$full, reduced = s$red, k_full = f$k, k_red = r$k,
                        BIC_full = f$bic, BIC_red = r$bic, logLik_full = f$ll, logLik_red = r$ll,
                        LR_chisq = lr, LR_df = dfk, p_LRT = p_lrt,
                        BF01 = bf01, BF10 = 1 / bf01, BF01_nsubj = bf01_20, BF10_nsubj = 1 / bf01_20,
                        singular_full = f$singular, singular_red = r$singular))
    cat(sprintf("%-10s %-50s %9.2f %9.2f %8.2f %8.3f  LR %6.2f (df %d, p %.3f)  BF01[n=20] %7.2f  %s/%s\n",
                band, s$term, f$bic, r$bic, bf01, 1 / bf01, lr, dfk, p_lrt, bf01_20, f$singular, r$singular))
  }
}

# ===================================================================
# 3. Simulation-based power for Cohort x GaitPhase (random-slope model)
# ===================================================================
cat("\n###################################################################\n")
cat("# 3. Simulation power, cycle-level random-slope model, ML\n")
cat("###################################################################\n")
df_c <- load_cortical("basenorm")
cat(sprintf("cycle-level data: %d rows, %d subjects, %d gait cycles (max per subject)\n",
            nrow(df_c), nlevels(df_c$SubjectID), nlevels(df_c$GaitCycle)))
cyc <- df_c %>% distinct(SubjectID, Cohort, GaitCycle) %>% count(SubjectID, Cohort)
cat("cycles per subject: "); cat(sprintf("%s=%d", cyc$SubjectID, cyc$n), sep = " "); cat("\n")
cat(sprintf("cycles total: HC %d, PD %d\n\n", sum(cyc$n[cyc$Cohort == "HC"]), sum(cyc$n[cyc$Cohort == "PD"])))

vc_ref <- fread(file.path(OUT_DIR, paste0("lmm_random_slopes_varcorr_", IN_DATE, ".csv")))
cat("Variance components of the accepted random-slope models (lmm_random_slopes.R, for reference):\n")
print(vc_ref[region == "cortical" & (is.na(var2) | var2 == ""), .(band, grp, var1, vcov = round(vcov, 3))])

RE_STR <- "(1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID) + (1 | GaitCycle:SubjectID)"
sim_rows <- list(); power_rows <- list()
RNGkind("Mersenne-Twister")

one_refit <- function(m0, yvec) {
  w <- character(0)
  fit <- withCallingHandlers(
    tryCatch(refit(m0, yvec), error = function(e) NULL),
    warning = function(cw) { w <<- c(w, conditionMessage(cw)); invokeRestart("muffleWarning") },
    message = function(cm) invokeRestart("muffleMessage"))
  if (is.null(fit)) return(data.frame(F = NA, df2 = NA, p = NA, singular = NA, warn = "fit failed"))
  a <- tryCatch(anova(lmerTest::as_lmerModLmerTest(fit), type = 3, ddf = "Satterthwaite"), error = function(e) NULL)
  if (is.null(a)) return(data.frame(F = NA, df2 = NA, p = NA, singular = isSingular(fit), warn = "anova failed"))
  data.frame(F = a["Cohort:GaitPhase", "F value"], df2 = a["Cohort:GaitPhase", "DenDF"],
             p = a["Cohort:GaitPhase", "Pr(>F)"], singular = isSingular(fit),
             warn = paste(unique(w), collapse = " | "))
}

# nsim = 0 skips the simulation (sections 1 and 2 only).
if (NSIM <= 0) cat("nsim = 0: simulation skipped\n")
for (band in if (NSIM > 0) BANDSEL else character(0)) {            # order given on the command line (priority order)
  bi <- match(band, BANDS)
  y <- band_col(band)
  cat("\n======================================================\n")
  cat("Band:", band, "\n")
  fml <- as.formula(paste(y, "~ Cohort * Area * GaitPhase +", RE_STR))
  t0 <- Sys.time()
  m0 <- lmer(fml, data = df_c, REML = FALSE, control = LMM_CONTROL)
  cat(sprintf("observed fit: %.1f s, singular = %s\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), isSingular(m0)))
  print(VarCorr(m0), comp = c("Variance", "Std.Dev."))
  a0 <- anova(m0, type = 3, ddf = "Satterthwaite")
  cat(sprintf("observed Cohort:GaitPhase: F(%d, %.1f) = %.3f, p = %.3f\n", a0["Cohort:GaitPhase", "NumDF"],
              a0["Cohort:GaitPhase", "DenDF"], a0["Cohort:GaitPhase", "F value"], a0["Cohort:GaitPhase", "Pr(>F)"]))
  vc0 <- as.data.frame(VarCorr(m0))
  for (i in seq_len(nrow(vc0))) add_rows(data.frame(block = "power_sim_varcorr", summary = "cycle", band = band,
                                                     term = vc0$grp[i], level = ifelse(is.na(vc0$var1[i]), "", vc0$var1[i]),
                                                     vcov = vc0$vcov[i], sdcor = vc0$sdcor[i]))
  beta0 <- fixef(m0)
  zero_terms <- grep("^CohortPD:.*GaitPhase", names(beta0), value = TRUE)   # Cohort:GaitPhase and three-way
  beta_null <- beta0; beta_null[zero_terms] <- 0
  cat("coefficients set to 0 under the null:", paste(zero_terms, collapse = ", "), "\n")
  cat("effect added to: CohortPD:GaitPhaseRLS\n")

  for (di in seq_along(DELTAS)) {
    delta <- DELTAS[di]
    beta_d <- beta_null; beta_d["CohortPD:GaitPhaseRLS"] <- delta
    seed_i <- 42L + bi + 100L * di
    ysim <- simulate(m0, nsim = NSIM, seed = seed_i, newparams = list(beta = beta_d))
    t0 <- Sys.time()
    res <- mclapply(seq_len(NSIM), function(k) one_refit(m0, ysim[[k]]), mc.cores = N_CORES)
    res <- do.call(rbind, res)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    res$sim <- seq_len(NSIM); res$band <- band; res$delta <- delta; res$seed <- seed_i
    sim_rows[[length(sim_rows) + 1]] <- res
    ok <- !is.na(res$p)
    pw <- mean(res$p[ok] < 0.05)
    pw_ci <- binom.test(sum(res$p[ok] < 0.05), sum(ok))$conf.int
    power_rows[[length(power_rows) + 1]] <- data.frame(
      block = "power_sim", summary = "cycle", band = band, term = "Cohort:GaitPhase", level = "RLS",
      delta_dB = delta, nsim = NSIM, n_ok = sum(ok), power = pw, power_ci_lo = pw_ci[1], power_ci_hi = pw_ci[2],
      n_singular = sum(res$singular, na.rm = TRUE), n_warn = sum(nzchar(res$warn) & ok),
      median_df2 = median(res$df2, na.rm = TRUE), seed = seed_i, seconds = round(secs))
    cat(sprintf("delta = %.1f dB: power = %.3f [%.3f, %.3f]  (n_ok %d/%d, singular %d, warnings %d, median DenDF %.1f, %.0f s)\n",
                delta, pw, pw_ci[1], pw_ci[2], sum(ok), NSIM, sum(res$singular, na.rm = TRUE),
                sum(nzchar(res$warn) & ok), median(res$df2, na.rm = TRUE), secs))
    fwrite(do.call(rbind, sim_rows), out_sim)   # incremental save
  }
  pr <- do.call(rbind, power_rows); pr <- pr[pr$band == band, ]
  mde_grid <- if (any(pr$power >= 0.8)) min(pr$delta_dB[pr$power >= 0.8]) else NA
  # linear interpolation of the 80 % crossing between grid points (delta > 0 only)
  prp <- pr[pr$delta_dB > 0, ]
  mde_int <- NA
  if (any(prp$power >= 0.8) && any(prp$power < 0.8)) {
    j <- min(which(prp$power >= 0.8))
    if (j > 1) mde_int <- prp$delta_dB[j - 1] + (0.8 - prp$power[j - 1]) * (prp$delta_dB[j] - prp$delta_dB[j - 1]) / (prp$power[j] - prp$power[j - 1])
    else mde_int <- prp$delta_dB[j]
  }
  add_rows(data.frame(block = "power_mde", summary = "cycle", band = band, term = "Cohort:GaitPhase", level = "RLS",
                      mde_grid_dB = mde_grid, mde_interp_dB = mde_int,
                      type1_at_0 = pr$power[pr$delta_dB == 0], nsim = NSIM))
  cat(sprintf("MDE (80 %% power): smallest grid value %s dB; interpolated %s dB; type I error at delta 0: %.3f\n",
              format(mde_grid), ifelse(is.na(mde_int), "NA", sprintf("%.2f", mde_int)), pr$power[pr$delta_dB == 0]))
  rm(m0); invisible(gc())
}
for (p in power_rows) add_rows(p)

# ===================================================================
# Write combined CSV and the Markdown report
# ===================================================================
all_rows <- rbindlist(rows, fill = TRUE)
fwrite(all_rows, out_csv)
cat("\nwritten:", out_csv, "\n")

tost <- all_rows[block == "tost_Cohort_x_GaitPhase" & summary == "median"]
tarea <- all_rows[block == "tost_Cohort_x_Area" & summary == "median"]
bf <- all_rows[block == "bic_bayes_factor"]
pw <- all_rows[block == "power_sim"]
mde <- all_rows[block == "power_mde"]

fmt <- function(x, d = 2) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = d))
md <- c(
  paste0("# How strong is the null for phase-specific cohort differences? (", DATE, ")"),
  "",
  paste0("Script: `code/analysis/revision/null_strength.R`. Tables: `null_strength_", DATE, ".csv` (long format, column `block`), per-simulation results in `null_strength_power_sims_", DATE, ".csv`, console log `null_strength_", DATE, ".txt`. Cortical, basenorm, subject-level medians (sensitivity with means in the CSV)."),
  "",
  "## Methods in brief",
  "",
  "* **Equivalence (TOST).** Per subject and phase (averaged over the two areas) the value was centred on the subject's mean over the four phases, which removes any cohort offset; the PD minus HC difference of the centred value per phase is the Cohort x GaitPhase contrast. Welch two one-sided tests against pre-specified smallest effects of interest of 1.0 dB (about 40 % of the observed gamma offset of -2.4 to -2.8 dB) and 0.5 dB (about 20 %); equivalence is declared when the 90 % CI lies inside the bounds (equivalently TOST p < 0.05). The same procedure was applied to the Area x Cohort offset difference (motor minus frontal per subject, PD vs HC) and, as a supplement in the CSV, to the three-way pattern.",
  "* **Bayes factors.** BIC approximation (Wagenmakers 2007): BF01 = exp((BIC_full - BIC_reduced)/2), aggregated model `y ~ Cohort*Area*GaitPhase + (1|SubjectID)`, ML, 160 cells. Cohort:GaitPhase is tested in the model without the three-way term; the three-way term against the full model; the Cohort main effect as `Cohort + Area*GaitPhase` vs `Area*GaitPhase`. The approximation implies a unit-information prior on the dropped coefficients and uses lme4's nobs (160 cells, not 20 subjects) in the BIC penalty; the values are an approximation of a default-prior Bayes factor, not a full Bayesian analysis.",
  paste0("* **Power / minimal detectable effect.** Cycle-level random-slope model `ERD ~ Cohort*Area*GaitPhase + (1 + A_motor + G_IDS + G_LLS + G_RLS || SubjectID) + (1 | GaitCycle:SubjectID)`, ML, fitted to the observed data per band; all Cohort:GaitPhase and Cohort:Area:GaitPhase coefficients set to 0, then a PD minus HC difference of delta dB added in RLS only (both areas). Responses simulated with the observed design (10 vs 10 subjects, observed cycles), the same model refitted, Type III Satterthwaite F for Cohort:GaitPhase; power = share of p < 0.05 over ", NSIM, " simulations per delta (delta = 0 gives the empirical type I error). The MDE is the smallest delta with power >= 0.80 (grid 0.5, 1, 1.5, 2, 3 dB; linear interpolation between grid points also given)."),
  "",
  "## Results per band (median aggregation)",
  "",
  "### Cohort x GaitPhase contrasts (PD - HC of the phase-centred value, dB) and TOST",
  "",
  "| Band | Phase | PD - HC | 90 % CI | Welch p | TOST p (1.0 dB) | equiv 1.0 | TOST p (0.5 dB) | equiv 0.5 |",
  "|---|---|---|---|---|---|---|---|---|")
for (i in seq_len(nrow(tost))) md <- c(md, with(tost[i], sprintf("| %s | %s | %s | [%s, %s] | %s | %s | %s | %s | %s |",
  band, level, fmt(estimate), fmt(ci90_lo), fmt(ci90_hi), fmt(p_welch, 3), fmt(tost_p_1p0, 3), equiv_1p0, fmt(tost_p_0p5, 3), equiv_0p5)))
md <- c(md, "", "### Cohort x Area offset difference (motor - frontal, PD - HC, dB) and TOST", "",
        "| Band | PD - HC | 90 % CI | Welch p | TOST p (1.0 dB) | equiv 1.0 | TOST p (0.5 dB) | equiv 0.5 |", "|---|---|---|---|---|---|---|---|")
for (i in seq_len(nrow(tarea))) md <- c(md, with(tarea[i], sprintf("| %s | %s | [%s, %s] | %s | %s | %s | %s | %s |",
  band, fmt(estimate), fmt(ci90_lo), fmt(ci90_hi), fmt(p_welch, 3), fmt(tost_p_1p0, 3), equiv_1p0, fmt(tost_p_0p5, 3), equiv_0p5)))
md <- c(md, "", "### BIC Bayes factors (BF01 = evidence for the reduced model)", "",
        "| Band | Term | LR chi2 (df) | LRT p | BF01 (n = 160 cells) | BF01 (n = 20 subjects) |", "|---|---|---|---|---|---|")
for (i in seq_len(nrow(bf))) md <- c(md, with(bf[i], sprintf("| %s | %s | %s (%d) | %s | %s | %s |",
  band, term, fmt(LR_chisq, 2), LR_df, fmt(p_LRT, 3), fmt(BF01, 1), fmt(BF01_nsubj, 1))))
md <- c(md, "", "BF01 > 1 favours the model without the term. The n = 160 version is lme4's BIC; the n = 20 version uses the number of subjects as the effective sample size in the BIC penalty (k log n) and is the conservative reading for between-subject terms. With a likelihood ratio near zero the BF01 is set by the penalty alone (3 df: 160^1.5 = 2024, 20^1.5 = 89), so these values mostly express that the extra terms buy no fit.")
if (nrow(pw)) {
  md <- c(md, "", "### Simulation power for Cohort x GaitPhase (random-slope model, effect in RLS)", "",
          paste0("| Band | ", paste(sprintf("delta %.1f", DELTAS), collapse = " | "), " | MDE grid | MDE interp. |"),
          paste0("|---|", paste(rep("---|", length(DELTAS) + 2), collapse = "")))
  for (bnd in unique(pw$band)) {
    pb <- pw[pw$band == bnd]
    cellsp <- sapply(DELTAS, function(d) { r <- pb[delta_dB == d]; if (nrow(r)) sprintf("%.2f [%.2f, %.2f]", r$power, r$power_ci_lo, r$power_ci_hi) else "" })
    mb <- mde[mde$band == bnd]
    md <- c(md, sprintf("| %s | %s | %s | %s |", bnd, paste(cellsp, collapse = " | "),
                        ifelse(is.na(mb$mde_grid_dB), "> 3", fmt(mb$mde_grid_dB, 1)), fmt(mb$mde_interp_dB, 2)))
  }
  md <- c(md, "", "Column delta 0.0 is the empirical type I error of the random-slope test. Values in brackets are 95 % binomial CIs over the simulations.")
}

# summary table per band
md <- c(md, "", "### Summary per band", "",
        "| Band | largest abs. PD - HC phase contrast (dB) | widest 90 % CI bound (dB) | equiv. at 1.0 dB (phases) | equiv. at 0.5 dB (phases) | BF01 Cohort:GaitPhase (n = 160 / n = 20) | BF01 three-way (n = 160 / n = 20) | MDE 80 % (dB) |",
        "|---|---|---|---|---|---|---|---|---|")
for (bnd in BANDS) {
  tb <- tost[tost$band == bnd]
  bfi <- bf[bf$band == bnd & bf$term == "Cohort:GaitPhase (given no three-way)"]
  bf3 <- bf[bf$band == bnd & bf$term == "Cohort:Area:GaitPhase"]
  mb <- mde[mde$band == bnd]
  md <- c(md, sprintf("| %s | %s | %s | %d/4 | %d/4 | %s | %s | %s |", bnd,
                      fmt(max(abs(tb$estimate))), fmt(max(abs(c(tb$ci90_lo, tb$ci90_hi)))),
                      sum(tb$equiv_1p0), sum(tb$equiv_0p5), paste(fmt(bfi$BF01, 0), "/", fmt(bfi$BF01_nsubj, 0)), paste(fmt(bf3$BF01, 0), "/", fmt(bf3$BF01_nsubj, 0)),
                      if (nrow(mb)) ifelse(is.na(mb$mde_grid_dB), "> 3", fmt(mb$mde_grid_dB, 1)) else "not run"))
}
md <- c(md, "", "## Suggested wording", "",
        "Draft (numbers from the 2026-10-14 run with 200 simulations; re-check against the tables above after a rerun):", "",
        readLines("code/analysis/revision/null_strength_wording.txt", warn = FALSE))
writeLines(md, out_md)
cat("written:", out_md, "\n")
close_log(log_con)
