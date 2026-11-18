# ===================================================================
# R SCRIPT FOR LINEAR MIXED-MODEL ANALYSIS OF STN DATA
# -------------------------------------------------------------------
# This script analyzes STN data, focusing on the effects of 
# Hemisphere and Gait Phase on neural activity in PD patients.
#
# - LOOPS: Iterates through normalization types ('basenorm', 'avggait') 
#   and all specified frequency bands.
# - MODEL: ERD/ERS ~ Hemisphere * GaitPhase 
# - RANDOM EFFECTS: Includes random intercepts for Subject and for 
#   GaitPhase nested within Subject.
# - POST-HOC: Automatically tests for a significant interaction and
#   compares Left vs. Right Hemisphere within each Gait Phase if found.
# ===================================================================

# 1. LOAD LIBRARIES
# -------------------------------------------------------------------
# Make sure these are installed: install.packages(c("lme4", "lmerTest", ...))
library(lme4)
library(lmerTest) # Essential for p-values in anova()
library(emmeans)  # For post-hoc analysis
library(data.table) # For fast data loading with fread()
library(dplyr)      # For data manipulation verbs
library(tibble)     # For rownames_to_column()
library(readxl)


# 2. ANALYSIS SETUP
# -------------------------------------------------------------------
# List of normalization types to process
datatypes<- list("basenorm", "avggait")
alpha_level <- 0.1

emm_options(pbkrtest.limit = 17000)


# Define the frequency bands for the analysis loop
freq_bands <- list(
  Theta     = c(4, 8),
  Alpha     = c(8, 12),
  `Low Beta`  = c(12, 20),
  `High Beta` = c(20, 30),
  Gamma     = c(30, 45)
)

# Helper function to create column names (e.g., "Low Beta" -> "ERD_ERS_Low_Beta")
make_band_colname <- function(band) {
  paste0("ERD_ERS_", gsub(" ", "_", band))
}

# 3. MAIN ANALYSIS LOOP
# -------------------------------------------------------------------

for (datatype in datatypes){
  
  # --- DATA LOADING ---
  # Construct the file path for the STN data based on the current datatype
  filepath <- paste0("Data/STN_IpsiContra_PD_subj_epochs_phase_median_", datatype, ".csv") 
  
  # Check if the file exists before trying to load it
  if (!file.exists(filepath)) {
    warning(paste("File not found:", filepath, "- Skipping this datatype."))
    next
  }
  
  print(paste("--- Processing STN data for:", datatype, "---"))
  df_stn <- fread(filepath)
  
  updrs <- read_excel("Data/UPDRS_scores_USZ_finalscoreonly.xlsx")
  
  # <-- FIX: Merge the scores into the correct data frame 'df_stn'
  df_stn <- merge(df_stn, updrs, by = "SubjectID", all.x = TRUE)
  
  # <-- FIX: Handle the single missing PD score using mean imputation
  # (The median is also a great choice if your scores are skewed)
  mean_pd_updrs <- mean(df_stn$UPDRS, na.rm = TRUE)
  df_stn$UPDRS[is.na(df_stn$UPDRS)] <- mean_pd_updrs
  
  # --- DATA PREPARATION ---
  
  # CRITICAL: Filter to only include Left and Right hemisphere data
  #df_stn <- df_stn[df_stn$Hemisphere == "Left" | df_stn$Hemisphere == "Right", ]
  
  # Convert relevant columns to factors for the model
  df_stn$SubjectID   <- as.factor(df_stn$SubjectID)
  df_stn$Hemisphere  <- as.factor(df_stn$Hemisphere)
  df_stn$GaitPhase   <- as.factor(df_stn$GaitPhase)
  df_stn$GaitCycle   <- as.factor(df_stn$Epoch)
  # --- ANALYSIS FOR EACH FREQUENCY BAND ---
  for (band in names(freq_bands)) {
    band_col <- make_band_colname(band)
    
    if (!(band_col %in% names(df_stn))) {
      warning(paste("Column", band_col, "not found in data for", datatype, "! Skipping."))
      next
    }
    
    cat("\n======================================================\n")
    cat("--- Analyzing Frequency Band:", band, "| Datatype:", datatype, "---\n")
    cat("======================================================\n")
    
    # --- MODEL FITTING ---
    # Define the model formula dynamically for each band
    # if (band == "Gamma" && datatype == "basenorm"){
    #   formula_str <- paste(band_col, "~ Hemisphere * GaitPhase + UPDRS + (1 | SubjectID)")
    # }
    # else{
    formula_str <- paste(band_col, "~ Hemisphere * GaitPhase + (1 | SubjectID)")
    #}
    
    # Fit the linear mixed-effects model
    model <- lmer(as.formula(formula_str), data = df_stn)

    # --- Singular-fit detection (diagnostic flag only; does NOT change the fit) ---
    # A boundary (singular) fit means a random-effect variance collapsed to ~0,
    # inflating the denominator df of the between-subjects fixed effects (here the
    # ipsi/contra-aligned Hemisphere factor).
    if (isSingular(model)) {
      cat("\n[SINGULAR_FIT]", band, "/", datatype, "- boundary (singular) fit.\n")
      cat("  Random-effect variances (grp with vcov ~ 0 is the collapsed term):\n")
      print(as.data.frame(VarCorr(model))[, c("grp", "vcov", "sdcor")])
      cat("  NOTE: a collapsed variance inflates the denominator df; affected between-subjects p-values are unreliable.\n")
    }

    # --- ANOVA TEST ---
    # Get the ANOVA table (Type III with Satterthwaite's method via lmerTest)
    anova_summary <- anova(model, type = 3)
    
    cat("\n--- ANOVA Table (Type III) ---\n")
    print(anova_summary)
    
    # Convert ANOVA summary to a data frame for easier p-value access
    anova_df <- as.data.frame(anova_summary) %>%
      rownames_to_column(var = "Term")
    
    # --- AUTOMATED POST-HOC TEST FOR INTERACTION ---
    
    # Check if the Hemisphere:GaitPhase interaction is significant
    interaction_term <- "Hemisphere:GaitPhase"
    p_value_interaction <- anova_df$`Pr(>F)`[anova_df$Term == interaction_term]
    
    if (length(p_value_interaction) > 0 && !is.na(p_value_interaction) && p_value_interaction < alpha_level) {
      cat(paste("\n** Significant", interaction_term, "interaction found (p =", format(p_value_interaction, digits=3), "). Running post-hoc tests... **\n"))
      
      # Calculate estimated marginal means for the interaction
      # This will compare Left vs. Right Hemisphere within each Gait Phase level.
      emm_interaction <- emmeans(model, ~ Hemisphere | GaitPhase)
      
      
      
      # Get the pairwise comparisons and print the summary with FDR correction
      print(summary(pairs(emm_interaction, adjust = "fdr")))
      
      #emm_interaction2 <- emmeans(model, ~ GaitPhase | Hemisphere)
      #print(summary(pairs(emm_interaction2, adjust = "fdr")))
      
      
    } else {
      cat("\nNo significant Hemisphere:GaitPhase interaction found. No post-hoc tests performed.\n")
    }
  }
}

print("--- STN Analysis Complete ---")

