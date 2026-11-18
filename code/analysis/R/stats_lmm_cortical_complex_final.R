# 1. Load necessary libraries
library(data.table) 
library(dplyr)      
library(tidyr)      
library(lme4)       
library(lmerTest) # Explicitly load for anova() F-tests
library(emmeans)
library(tibble)

# --- Your existing code ---
datatypes <- list("basenorm", "avggait")
alpha_level <- 0.05

# Prevent emmeans from running out of memory on large models
emm_options(pbkrtest.limit = 17000)

for (datatype in datatypes){
  file <- paste("Data/for_R_EEG_epochs_PD-vs-HC_phase_median_" , datatype, sep="")
  filepath <- paste(file, ".csv", sep="")
  df <- fread(filepath)
  
  df <- df[df$Hemisphere == "Both", ]
  
  # --- Data Preparation ---
  df$SubjectID   <- as.factor(df$SubjectID)
  df$Cohort      <- as.factor(df$Group)
  df$Area        <- as.factor(df$Area)
  df$GaitPhase   <- as.factor(df$GaitPhase)
  df$Block       <- as.factor(df$Block)
  df$GaitCycle   <- as.factor(df$Epoch)
  
  # Set "HC" as the reference level for comparisons
  df$Cohort <- relevel(df$Cohort, ref = "HC")
  
  print(paste("--- Processing data for:", datatype, "---"))
  
  # --- Analysis Loop for Each Frequency Band ---
  freq_bands <- list(
    Theta     = c(4, 8),
    Alpha     = c(8, 12),
    `Low Beta`  = c(12, 20),
    `High Beta` = c(20, 30),
    Gamma     = c(30, 45)
  )
  
  make_band_colname <- function(band) {
    paste0("ERD_ERS_", gsub(" ", "_", band))
  }
  
  for (band in names(freq_bands)) {
    band_col <- make_band_colname(band)
    
    if (!(band_col %in% names(df))) {
      warning(paste("Column", band_col, "not found in data! Skipping."))
      next
    }
    
    cat("\n======================================================\n")
    cat("--- Analyzing Frequency Band:", band, "for", datatype, "data ---\n")
    cat("======================================================\n")
    
    # --- Model Fitting ---
    formula_str <- paste(band_col, "~ Cohort * Area + Cohort * GaitPhase + (1 | SubjectID) + (1 | GaitCycle:SubjectID)")
    model_slopes <- lmer(as.formula(formula_str), data = df, REML = FALSE)

    # --- Singular-fit detection (diagnostic flag only; does NOT change the fit) ---
    # A boundary (singular) fit means a random-effect variance collapsed to ~0.
    # That inflates the Satterthwaite denominator df of between-subjects terms
    # (here Cohort), making their p-values anticonservative / unreliable.
    if (isSingular(model_slopes)) {
      cat("\n[SINGULAR_FIT]", band, "/", datatype, "- boundary (singular) fit.\n")
      cat("  Random-effect variances (grp with vcov ~ 0 is the collapsed term):\n")
      print(as.data.frame(VarCorr(model_slopes))[, c("grp", "vcov", "sdcor")])
      cat("  NOTE: between-subjects (Cohort) df is inflated; its p-value is unreliable.\n")
    }

    # --- ANOVA Test ---
    # With lmerTest loaded, anova() correctly performs a Type III F-test. 

    anova_summary <- anova(model_slopes)
    print(anova_summary)
    # 
    anova_df <- as.data.frame(anova_summary) %>%
      rownames_to_column(var = "Term")
    
    # --- Post-Hoc Tests for Significant Interactions ---
    
    # Check for Cohort:Area interaction
    p_value_area <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:Area"]
    if (length(p_value_area) > 0 && !is.na(p_value_area) && p_value_area < alpha_level) {
      cat("\n** Significant Cohort:Area interaction found (p =", format(p_value_area, digits=3), "). Running post-hoc tests... **\n")
      emm_area <- emmeans(model_slopes, ~ Cohort | Area)
      print(summary(pairs(emm_area, adjust = "fdr")))
    }
    
    # Check for Cohort:GaitPhase interaction
    p_value_gaitphase <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:GaitPhase"]
    if (length(p_value_gaitphase) > 0 && !is.na(p_value_gaitphase) && p_value_gaitphase < alpha_level) {
      cat("\n** Significant Cohort:GaitPhase interaction found (p =", format(p_value_gaitphase, digits=3), "). Running post-hoc tests... **\n")
      emm_gaitphase <- emmeans(model_slopes, ~ Cohort | GaitPhase)
      print(summary(pairs(emm_gaitphase, adjust = "fdr")))
    }
  }
}

# # 1. Load necessary libraries
# library(data.table) # For fread()
# library(dplyr)      # For data manipulation verbs like rename(), mutate(), across()
# library(tidyr)      # For pivot_longer() to reshape the data
# library(lme4)       # For lmer()
# library(emmeans)
# library(tibble)
# 
# # --- Your existing code ---
# datatypes <- list("basenorm", "avggait")
# alpha_level <- 0.05
# 
# emm_options(pbkrtest.limit = 17000)
# 
# for (datatype in datatypes){
#   file <- paste("Data/for_R_EEG_epochs_PD-vs-HC_phase_median_" , datatype, sep="")
#   filepath <- paste(file, ".csv", sep="")
#   df <- fread(filepath)
#   
#   df <- df[df$Hemisphere == "Both", ]
#   # If needed, make sure relevant columns are factors
#   df$SubjectID   <- as.factor(df$SubjectID)
#   df$Cohort       <- as.factor(df$Group)
#   df$Area        <- as.factor(df$Area)
#   #df$Hemisphere  <- as.factor(df$Hemisphere)
#   df$GaitPhase   <- as.factor(df$GaitPhase)
#   df$Block       <- as.factor(df$Block)
#   print(df)
# 
#   # --- What to write afterwards ---
#   
#   # 2. Reshape the data from wide to long format
#   # We gather all the ERD_ERS columns into two new columns:
#   # 'FrequencyBand' will hold the name of the band (e.g., "Alpha")
#   # 'ERSD' will hold the corresponding value.
#   # df_long <- pivot_longer(df,
#   #                         cols = starts_with("ERD_ERS_"),
#   #                         names_to = "FrequencyBand",
#   #                         names_prefix = "ERD_ERS_", # This removes the "ERD_ERS_" part
#   #                         values_to = "ERSD")
#   
#   # 3. Prepare the data for modeling
#   # Rename columns for clarity and consistency with our discussion
#   # and convert character columns to factors.
# 
#   # df_long <- df_long %>%
#   #   rename(SubjectID = SubjectID,
#   #          Cohort = Group,
#   #          GaitCycle = Epoch) %>%
#   #   mutate(across(c(Subject, Cohort, Area, Hemisphere, GaitPhase, FrequencyBand), as.factor))
#   # 
#   # CRITICAL STEP: Set the reference level for the Cohort factor.
#   # We set "HC" as the reference so that the model coefficients for "PD"
#   # represent the *difference from the healthy controls*.
#   df$Cohort <- relevel(df$Cohort, ref = "HC")
#   
#   # You can check the structure and levels now
#   
#   print(paste("--- Processing data for:", datatype, "---"))
#   str(df)
#   print(levels(df$Cohort))
#   
#   
#   freq_bands <- list(
#     #Delta     = c(1, 4),
#     Theta     = c(4, 8),
#     Alpha     = c(8, 12),
#     `Low Beta`  = c(12, 20),
#     `High Beta` = c(20, 30),
#     Gamma     = c(30, 45)
#   )
#   
#   # Check that all columns have the same length
#   sapply(df, length)
#   nrow(df)
#   
#   # This function fixes spaces and capitalization to match your column names
#   make_band_colname <- function(band) {
#     # Replace spaces with underscores and capitalize as in your headers
#     # (Assumes original band names in freq_bands are already correctly capitalized)
#     paste0("ERD_ERS_", gsub(" ", "_", band))
#   }
#   
#   # Example usage:
#   for (band in names(freq_bands)) {
#     band_col <- make_band_colname(band)
#     cat("Band:", band, "| Column name:", band_col, "\n")
#     # Example: df[[band_col]]
#   }
#   
#   
#   results <- list()
#   for (band in names(freq_bands)) {
#     band_col <- make_band_colname(band)
#     if (!(band_col %in% names(df))) {
#       warning(paste("Column", band_col, "not found in data!"))
#       next
#     }
#     print(band_col)
#   
#   
#   # 4. Fit the initial Linear Mixed-Effects Model
#   # This is the "maximal" model with interaction fixed effects and nested random intercepts.
#   # We use REML = FALSE (Maximum Likelihood) because we will be comparing models
#   # with different fixed effects structures, for which ML is appropriate (Bates, p. 21).
#   
#   # Define the model formula
#   # model_formula <- ERSD ~ Cohort * FrequencyBand + Cohort * Area + Cohort * Hemisphere + Cohort * GaitPhase + (1 | Subject/GaitCycle)
#   # 
#   # # Fit the model
#   # control_params <- lmerControl(optimizer = "Nelder_Mead")
#   # model_intercepts <- lmer(model_formula, data = df_long, REML = FALSE,  control = control_params)
#   # 
#   # # Print the summary of the initial model
#   # print("--- Summary of Initial Model (Random Intercepts Only) ---")
#   # print(summary(model_intercepts))
#   
#   
#   # 5. Build and Test a More Complex Random Effects Model (Optional but recommended)
#   # Let's test if we need random slopes for FrequencyBand, as discussed.
#   # This model allows the effect of frequency band to vary across subjects.
#   # model_slopes <- lmer(ERSD ~ Cohort * FrequencyBand + Cohort * Area + Cohort * GaitPhase + 
#   #                        (1 | Subject) + (1 | Subject:GaitCycle), 
#   #                      data = df_long, REML = FALSE)
#     formula_str <- paste(band_col, "~ Cohort * Area + Cohort * GaitPhase + (1 | SubjectID) + (1 | GaitPhase:SubjectID)")
#     model_slopes <- lmer(as.formula(formula_str), data = df, REML = FALSE)
#   
#     # Compare the two models using a Likelihood Ratio Test
#     print("--- Model Comparison (Random Intercepts vs. Random Slopes) ---")
#     
#     anova_summary <- anova(model_slopes, type = 3)
#     print(anova_summary)
#     
#     anova_df <- as.data.frame(anova_summary) %>%
#     rownames_to_column(var = "Term") # Makes the term names a column
#     
#     # 3. Check for a significant Group:Area interaction
#     p_value_area <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:Area"]
#     if (length(p_value_area) > 0 && !is.na(p_value_area) && p_value_area < alpha_level) {
#       cat("\n** Interaction Cohort:Area significative trouvée (p =", format(p_value_area, digits=3), "). Lancement du test post-hoc... **\n")
#       emm_area <- emmeans(model_slopes, ~ Cohort | Area)
#       print(summary(pairs(emm_area, adjust = "fdr")))
#     }
#     
#     p_value_gaitphase <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:GaitPhase"]
#     if (length(p_value_gaitphase) > 0 && !is.na(p_value_gaitphase) && p_value_gaitphase < alpha_level) {
#       cat("\n** Interaction Cohort:GaitPhase significative trouvée (p =", format(p_value_gaitphase, digits=3), "). Lancement du test post-hoc... **\n")
#       emm_gaitphase <- emmeans(model_slopes, ~ Cohort | GaitPhase)
#       print(summary(pairs(emm_gaitphase, adjust = "fdr")))
#     }
#     
#     # p_value_area <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:GaitPhase"]
#     # if (length(p_value_area) > 0 && !is.na(p_value_area) && p_value_area < alpha_level) {
#     #   cat("\n** Interaction Cohort:Area significative trouvée (p =", format(p_value_area, digits=3), "). Lancement du test post-hoc... **\n")
#     #   emm_gaiphase <- emmeans(model_slopes, ~ Cohort | GaitPhase)
#     #   print(summary(pairs(emm_gaiphase, adjust = "fdr")))
#     # }
#     # 
#     # p_value_area <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:GaitPhase"]
#     # if (length(p_value_area) > 0 && !is.na(p_value_area) && p_value_area < alpha_level) {
#     #   cat("\n** Interaction Cohort:Area significative trouvée (p =", format(p_value_area, digits=3), "). Lancement du test post-hoc... **\n")
#     #   emm_gaiphase <- emmeans(model_slopes, ~ Cohort | GaitPhase)
#     #   print(summary(pairs(emm_gaiphase, adjust = "fdr")))
#     # }
#     
#     # p_value_area <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:Area"]
#     # if (length(p_value_area) > 0 && !is.na(p_value_area) && p_value_area < alpha_level) {
#     #   cat("\n** Interaction Cohort:Area significative trouvée (p =", format(p_value_area, digits=3), "). Lancement du test post-hoc... **\n")
#     #   emm_area <- emmeans(model_slopes, ~ Cohort | Area)
#     #   print(summary(pairs(emm_area, adjust = "fdr")))
#     # }
#     
#     # 4. Check for a significant Group:Hemisphere interaction
#     # p_value_hemi <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:FrequencyBand"]
#     # if (length(p_value_hemi) > 0 && !is.na(p_value_hemi) && p_value_hemi < alpha_level) {
#     #   cat("\n** Interaction Cohort:Hemisphere significative trouvée (p =", format(p_value_hemi, digits=3), "). Lancement du test post-hoc... **\n")
#     #   emm_hemi <- emmeans(model_slopes, ~ Cohort | FrequencyBand)
#     #   print(summary(pairs(emm_hemi, adjust = "fdr")))
#     # }
#     
#     # 5. Check for a significant Group:GaitPhase interaction
#     # p_value_gait <- anova_df$`Pr(>F)`[anova_df$Term == "Cohort:GaitPhase"]
#     # if (length(p_value_gait) > 0 && !is.na(p_value_gait) && p_value_gait < alpha_level) {
#     #   cat("\n** Interaction Cohort:GaitPhase significative trouvée (p =", format(p_value_gait, digits=3), "). Lancement du test post-hoc... **\n")
#     #   emm_gait <- emmeans(model_slopes, ~ Cohort | GaitPhase)
#     #   print(summary(pairs(emm_gait, adjust = "fdr")))
#     # }
#     
#     # Convert the anova table to a data frame for easy access to p-values
#     # anova_df <- as.data.frame(anova_summary) %>%
#     #   rownames_to_column(var = "Term") # Makes the term names a column
#   }
# } # End of the loop
# 
