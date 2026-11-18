# Exploratory LMM model comparison (A.B.'s original candidate set) with random-slope candidates added
#
# Purpose:  Re-run the original exploratory model comparison on the current
#           data, unchanged except for two things: (i) the UPDRS covariate
#           model (mod7) is skipped because the UPDRS file in the repository
#           is an all-NA stub; (ii) two random-slope candidates are added to
#           the same anova() comparison, mod8 = (1 + GaitPhase | SubjectID)
#           (the formula that was commented out in the original) and
#           mod9 = (1 + Area + GaitPhase | SubjectID), both with the
#           gait-cycle intercept. Everything else (data filtering, factors,
#           candidate formulas, comparison call) is the original code.
# Inputs:   code/analysis/R/Data/for_R_EEG_epochs_PD-vs-HC_phase_median_{basenorm,avggait}.csv
#           code/analysis/R/Data/UPDRS_scores_USZ_finalscoreonly.xlsx (stub; loaded, not modelled)
# Outputs:  console only; captured in
#           outputs/stats/revision_2026-10/explorative_lmm_complex_AB_rerun_2026-10-14.txt
# Author:   A.B. (original), patched by Lena
# Date:     2026-10-14
# Seed:     not applicable (deterministic fits)
#
# Run from code/analysis/R (relative Data/ paths as in the original):
#   cd code/analysis/R && Rscript ../revision/explorative_stats_lmm_complex_patched.R
  # Install these if not already installed
  #install.packages(c("lme4", "lmerTest", "emmeans", "data.table"))
  
library(lme4)
library(lmerTest) # Satterthwaite degrees of freedom, as in JASP
library(emmeans)
library(data.table)
library(dplyr)
library(readxl)

datatypes <- list("basenorm", "avggait")
alpha_level <- 0.05

for (datatype in datatypes){
  print(paste("--- Processing data for:", datatype, "---"))
  # Load the dataset
  file <- paste("Data/for_R_EEG_epochs_PD-vs-HC_phase_median_" , datatype, sep="")
  filepath <- paste(file, ".csv", sep="")
  df <- fread(filepath)  
  
  #updrs scores
  # 1. Load the PD-specific values from an Excel file
  updrs <- read_excel("Data/UPDRS_scores_USZ_finalscoreonly.xlsx")
  
  # 2. Merge the scores into your main data frame
  df <- merge(df, updrs, by = "SubjectID", all.x = TRUE)
  
  # 3. Set the score for the HC group to 0
  # The merge will result in NA for HC subjects, so we replace NA with 0.
  df$UPDRS[df$Group == "HC"] <- 0
  
  df <- df[df$Hemisphere == "Both"] # | df$Hemisphere == "Right", ] #print(df)
  #df <- df[df$Hemisphere == "Left" | df$Hemisphere == "Right", ] #print(df)
  

  
  
  # If needed, make sure relevant columns are factors
  df$SubjectID   <- as.factor(df$SubjectID)
  df$Group       <- as.factor(df$Group)
  df$Area        <- as.factor(df$Area)
  df$GaitPhase   <- as.factor(df$GaitPhase)
  df$Block       <- as.factor(df$Block)
  df$GaitCycle   <- as.factor(df$Epoch)
  #df$GaitCycle   <- df$Epoch
  
  
  # Define frequency bands for analysis
  freq_bands <- list(
    #Delta     = c(1, 4),
    Theta     = c(4, 8),
    Alpha     = c(8, 12),
    `Low Beta`  = c(12, 20),
    `High Beta` = c(20, 30),
    Gamma     = c(30, 45)
  )
  
  # Check that all columns have the same length
  sapply(df, length)
  nrow(df)
  
  # This function fixes spaces and capitalization to match your column names
  make_band_colname <- function(band) {
    # Replace spaces with underscores and capitalize as in your headers
    # (Assumes original band names in freq_bands are already correctly capitalized)
    paste0("ERD_ERS_", gsub(" ", "_", band))
  }
  
  # Example usage:
  for (band in names(freq_bands)) {
    band_col <- make_band_colname(band)
    cat("Band:", band, "| Column name:", band_col, "\n")
    # Example: df[[band_col]]
  }
  
  # Initialize a list to store results
  results <- list()
  
  # Loop through each frequency band to perform analysis
  for (band in names(freq_bands)) {
    band_col <- make_band_colname(band)
    if (!(band_col %in% names(df))) {
      warning(paste("Column", band_col, "not found in data!"))
      next
    }
    print(band_col)
    
    # Define model formulas without Hemisphere
    formula_str0 <- paste0(
      band_col, "~ Group"
    )
    formula_str1 <- paste0(
      band_col, " ~ Group + (1 | SubjectID)"
    )
    formula_str2 <- paste0(
      band_col, " ~ Group + Area + GaitPhase + (1 | SubjectID)"
    )
    formula_str2_1 <- paste0(
      band_col, " ~ Group * Area + Group * GaitPhase + (1 | SubjectID)"
    )
    formula_str2_2 <- paste0(
      band_col, " ~ Group * Area + Area * GaitPhase + (1 | SubjectID)"
    )
    formula_str2_3 <- paste0(
      band_col, "~ Group * GaitPhase + GaitPhase * Area + (1| SubjectID)"
    )
    formula_str4 <- paste0(
      band_col,   " ~ Group * Area + Group * GaitPhase + Group:GaitPhase:Area + (1 | SubjectID)"
    )
    formula_str5 <- paste0(
      band_col, " ~ Group * Area * GaitPhase + (1 | SubjectID)"
    )
    # formula_str6 <- paste0(
    #   band_col, " ~ Group * Area + Group * GaitPhase + (1 | SubjectID) + (1 | GaitPhase:SubjectID)"
    # )
    formula_str6_2 <- paste0(
      band_col, " ~ Group * Area + Group * GaitPhase + (1 | SubjectID) + (1 | GaitCycle:SubjectID)"
    )
    formula_str7 <- paste0(
      band_col, "~ Group * Area + Group * GaitPhase + UPDRS + (1 | SubjectID) + (1 | GaitCycle:SubjectID)"
    )

    
    # Convert string formulas to formula objects
    fmla0 <- as.formula(formula_str0)
    fmla1 <- as.formula(formula_str1)
    fmla2 <- as.formula(formula_str2)
    fmla2_1 <- as.formula(formula_str2_1)
    fmla2_2 <- as.formula(formula_str2_2)
    fmla2_3 <- as.formula(formula_str2_3)
    fmla4 <- as.formula(formula_str4)
    fmla5 <- as.formula(formula_str5)
    #fmla6 <- as.formula(formula_str6)
    fmla6_2 <- as.formula(formula_str6_2)
    fmla7 <- as.formula(formula_str7)
    
    # Fit linear mixed-effects models
    mod0 <- lmer(fmla1, data = df)
    mod1 <- mod0
    mod1 <- lmer(fmla1, data = df)
    mod2 <- lmer(fmla2, data = df)
    mod2_1 <- lmer(fmla2_1, data = df)
    mod2_2 <- lmer(fmla2_2, data = df)
    mod2_3 <- lmer(fmla2_3, data = df)
    mod4 <- lmer(fmla4, data = df)
    mod5 <- lmer(fmla5, data = df)
    #mod6 <- lmer(fmla6, data = df)
    mod6_2 <- lmer(fmla6_2, data = df)
    # mod7 (UPDRS covariate) skipped: UPDRS file in the repository is an all-NA stub
    fmla8 <- as.formula(paste0(band_col, " ~ Group * Area + Group * GaitPhase + (1 + GaitPhase | SubjectID) + (1 | GaitCycle:SubjectID)"))
    fmla9 <- as.formula(paste0(band_col, " ~ Group * Area + Group * GaitPhase + (1 + Area + GaitPhase | SubjectID) + (1 | GaitCycle:SubjectID)"))
    mod8 <- lmer(fmla8, data = df, control = lmerControl(optimizer = "bobyqa"))
    mod9 <- lmer(fmla9, data = df, control = lmerControl(optimizer = "bobyqa"))
    cat("singular: mod6_2", isSingular(mod6_2), " mod8", isSingular(mod8), " mod9", isSingular(mod9), "\n")
    
    # Compare the models
    model_comparison <- anova(mod0, mod1, mod2_1, mod4, mod5, mod6_2, mod8, mod9)
    print(anova(mod0))
    print(band_col)
    print(model_comparison)
    
    # Optional: Post-hoc analysis (uncomment to run)
    #   # Estimated marginal means for Group within each GaitPhase
    #   emm <- emmeans(mod, ~ Group | GaitPhase)
    #   contrasts <- pairs(emm, adjust = "fdr")
    #   
    #   results[[band]] <- list(model = mod, emm = emm, contrasts = contrasts)
    # 
    #   print(summary(contrasts))
    #   
    #   emm2 <- emmeans(mod, ~ Group | GaitPhase * Area)
    #   contrasts2 <- pairs(emm2, adjust = "fdr")
    #   results[[band]]$emm_GaitPhaseArea <- emm2
    #   results[[band]]$contrasts_GaitPhaseArea <- contrasts2
    #   
    #   # Print or export:
    #   print(summary(contrasts2))
  }
}
  #plot(results[["ERD_ERS_Low_Beta"]]$emm)
  
  # # Install these if not already installed
  # #install.packages(c("lme4", "lmerTest", "emmeans", "data.table"))
  # 
  # library(lme4)
  # library(lmerTest) # Satterthwaite degrees of freedom, as in JASP
  # library(emmeans)
  # library(data.table)
  # 
  # 
  # df <- fread("Data/for_R_EEG_epochs_PD-vs-HC_phase_median_avggait.csv") # use read.csv("your_data.csv") if you prefer
  # # If needed, make sure relevant columns are factors
  # df$SubjectID   <- as.factor(df$SubjectID)
  # df$Group       <- as.factor(df$Group)
  # df$Area        <- as.factor(df$Area)
  # df$Hemisphere  <- as.factor(df$Hemisphere)
  # df$GaitPhase   <- as.factor(df$GaitPhase)
  # df$Block       <- as.factor(df$Block)
  # 
  # freq_bands <- list(
  #   #Delta     = c(1, 4),
  #   Theta     = c(4, 8),
  #   Alpha     = c(8, 12),
  #   `Low Beta`  = c(12, 20),
  #   `High Beta` = c(20, 30),
  #   Gamma     = c(30, 45)
  # )
  # 
  # # Check that all columns have the same length
  # sapply(df, length)
  # nrow(df)
  # 
  # # This function fixes spaces and capitalization to match your column names
  # make_band_colname <- function(band) {
  #   # Replace spaces with underscores and capitalize as in your headers
  #   # (Assumes original band names in freq_bands are already correctly capitalized)
  #   paste0("ERD_ERS_", gsub(" ", "_", band))
  # }
  # 
  # # Example usage:
  # for (band in names(freq_bands)) {
  #   band_col <- make_band_colname(band)
  #   cat("Band:", band, "| Column name:", band_col, "\n")
  #   # Example: df[[band_col]]
  # }
  # 
  # 
  # results <- list()
  # for (band in names(freq_bands)) {
  #   band_col <- make_band_colname(band)
  #   if (!(band_col %in% names(df))) {
  #     warning(paste("Column", band_col, "not found in data!"))
  #     next
  #   }
  #   print(band_col)
  #   formula_str1 <- paste0(
  #     band_col, " ~ Group + (1 | SubjectID)"
  #   )
  #   formula_str2 <- paste0(
  #     band_col, " ~ Group + Area + Hemisphere + GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_1 <- paste0(
  #     band_col, " ~ Group * Area + Hemisphere + Group * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_2 <- paste0(
  #     band_col, " ~ Group * Area + Group * Hemisphere + GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_3 <- paste0(
  #     band_col, " ~ Area + Group * Hemisphere + Group * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_4 <- paste0(
  #     band_col, " ~ Group + Area * Hemisphere + Hemisphere * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_5 <- paste0(
  #     band_col, " ~ Group + Area * Hemisphere + Area * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_6 <- paste0(
  #     band_col, " ~ Group + GaitPhase * Hemisphere + Area * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_7 <- paste0(
  #     band_col, " ~ Area + GaitPhase * Hemisphere + Group * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_8 <- paste0(
  #     band_col, " ~ Area + Group * Hemisphere + Hemisphere * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_9 <- paste0(
  #     band_col, " ~ GaitPhase * Area + Hemisphere + Group * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_10 <- paste0(
  #     band_col, " ~ Group * Area + Hemisphere + Area * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str2_11 <- paste0(
  #     band_col, " ~ Group * Area + GaitPhase + Area * Hemisphere + (1 | SubjectID)"
  #   )
  #   formula_str2_12 <- paste0(
  #     band_col, " ~ Group * Hemisphere + GaitPhase + Area * Hemisphere + (1 | SubjectID)"
  #   )
  #       formula_str3 <- paste0(
  #     band_col, " ~ Group * Area + Group * Hemisphere + Group * GaitPhase + (1 | SubjectID)"
  #   )
  #   
  #   ## GaitPhase-Area and Gait-Phase Hemisphere ?
  #   
  #   formula_str4 <- paste0(
  #     band_col,   " ~ Group * Area + Group * Hemisphere + Group * GaitPhase + Group:GaitPhase:Area + (1 | SubjectID)"
  #   )
  #   formula_str5 <- paste0(
  #     band_col, " ~ Group * Area * Hemisphere + Group * Area * GaitPhase + (1 | SubjectID)"
  #   )
  #   formula_str6 <- paste0(
  #     band_col, " ~ Group * Area * Hemisphere + Group * Area * GaitPhase + (1 +GaitPhase | SubjectID)"
  #   )
  #   
  #   #band_col,   " ~ Group  +  Hemisphere + Area + GaitPhase + (1 | SubjectID)")
  #   #band_col, " ~ Group * Hemisphere + Group *Area + Group * GaitPhase + (+|SubjectID) ")
  #   
  #   #    band_col, " ~ GaitPhase * Group + Hemisphere * Group + Area * Group + ",
  #   #    "Area * GaitPhase + Hemisphere * GaitPhase + Hemisphere * Area+ ",
  #   #    "(1 | SubjectID)"
  #   #  )
  #   
  #   fmla1 <- as.formula(formula_str1)
  #   fmla2 <- as.formula(formula_str2)
  #   fmla2_1 <- as.formula(formula_str2_1)
  #   fmla2_2 <- as.formula(formula_str2_2)
  #   fmla2_3 <- as.formula(formula_str2_3)
  #   fmla2_4 <- as.formula(formula_str2_4)
  #   fmla2_5 <- as.formula(formula_str2_5)
  #   fmla2_6 <- as.formula(formula_str2_6)
  #   fmla2_7 <- as.formula(formula_str2_7)
  #   fmla2_8 <- as.formula(formula_str2_8)
  #   fmla2_9 <- as.formula(formula_str2_9)
  #   fmla2_10 <- as.formula(formula_str2_10)
  #   fmla2_11 <- as.formula(formula_str2_11)
  #   fmla2_12 <- as.formula(formula_str2_12)
  #   fmla3 <- as.formula(formula_str3)
  #   fmla4 <- as.formula(formula_str4)
  #   fmla5 <- as.formula(formula_str5)
  #   fmla6 <- as.formula(formula_str6)
  #   
  #   
  #   # Fit model
  #   mod1 <- lmer(fmla1, data = df)
  #   mod2 <- lmer(fmla2, data = df)
  #   mod2_1 <- lmer(fmla2_1, data = df)
  #   mod2_2 <- lmer(fmla2_2, data = df)
  #   mod2_3 <- lmer(fmla2_3, data = df)
  #   mod2_4 <- lmer(fmla2_4, data = df)
  #   mod2_5 <- lmer(fmla2_5, data = df)
  #   mod2_6 <- lmer(fmla2_6, data = df)
  #   mod2_7 <- lmer(fmla2_7, data = df)
  #   mod2_8 <- lmer(fmla2_8, data = df)
  #   mod2_9 <- lmer(fmla2_9, data = df)
  #   mod2_10 <- lmer(fmla2_10, data = df)
  #   mod2_11 <- lmer(fmla2_11, data = df)
  #   mod2_12 <- lmer(fmla2_12, data = df)
  #   mod3 <- lmer(fmla3, data = df)
  #   mod4 <- lmer(fmla4, data = df)
  #   mod5 <- lmer(fmla5, data = df)
  #   mod6 <- lmer(fmla6, data = df)
  #   
  #   model_comparison <- anova(mod1, mod2, mod2_1, mod2_2, mod2_3,  mod2_4, mod2_5, mod2_6,  mod2_7, mod2_8, mod2_9, mod2_10, mod2_11, mod2_12,   mod3, mod4, mod5, mod6)
  #   #model_comparison <- anova(mod1, mod2, mod2_2, mod3) #, mod4, mod5)
  #   # install.packages("nonnest2") # If you don't have it
  # 
  #   print(band_col)
  #   print(model_comparison)
  #   # 
  #   # anova_summary <- anova(mod, type = 3)
  #   # print(band_col)
  #   # print(anova_summary)
  #   # 
  #   # anova_summary2 <- anova(mod2, type = 3)
  #   # print(band_col)
  #   # print(anova_summary2)
  #   
  #   #   # Estimated marginal means for Group within each GaitPhase
  #   #   emm <- emmeans(mod, ~ Group | GaitPhase)
  #   #   contrasts <- pairs(emm, adjust = "fdr")
  #   #   
  #   #   results[[band]] <- list(model = mod, emm = emm, contrasts = contrasts)
  #   # 
  #   #   print(summary(contrasts))
  #   #   
  #   #   emm2 <- emmeans(mod, ~ Group | GaitPhase * Area)
  #   #   contrasts2 <- pairs(emm2, adjust = "fdr")
  #   #   results[[band]]$emm_GaitPhaseArea <- emm2
  #   #   results[[band]]$contrasts_GaitPhaseArea <- contrasts2
  #   #   
  #   #   # Print or export:
  #   #   print(summary(contrasts2))
  # }
  # 
  # 
  # #plot(results[["ERD_ERS_Low_Beta"]]$emm)
  # 
