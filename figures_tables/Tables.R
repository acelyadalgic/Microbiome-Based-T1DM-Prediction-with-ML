library(dplyr)

within_features <- read_excel("within_high_frequency_features.xlsx")
pooled_features <- read_excel("pooled_features.xlsx")
loso_features   <- read_excel("loso_bidirectional_features.xlsx")
metadata   <- read_excel("metadata.xlsx")
linda_disease <- read.csv("linda_significant_disease.csv")
linda_country <- read.csv("linda_significant_country.csv")

# -----------------------------------
# Table 1 Demographics
# -----------------------------------

library(dplyr)

table1 <- metadata %>%
  group_by(country, disease) %>%
  summarise(
    N = n(),
    age_mean = mean(age, na.rm = TRUE),
    age_sd = sd(age, na.rm = TRUE),
    male_n = sum(sex == "male", na.rm = TRUE),
    female_n = sum(sex == "female", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Age = paste0(round(age_mean,1), " (", round(age_sd,1), ")"),
    Male = paste0(male_n, " (", round(100*male_n/N,1), "%)"),
    Female = paste0(female_n, " (", round(100*female_n/N,1), "%)")
  ) %>%
  select(
    Cohort = country,
    Disease = disease,
    N,
    Age,
    Male,
    Female
  )

write_xlsx(table1,"Table 1. Demographic and clinical characteristics of the study cohorts.xlsx")

# -----------------------------------
# Table 2 Within Model Performance
# -----------------------------------
all_metrics_within <- read_excel("all_metrics_within.xlsx")
all_features_within <- read_excel("all_features_within.xlsx")
metric_summary <- all_metrics_within %>%
  group_by(Model, Country, Data) %>%
  summarise(
    AUC_mean = mean(AUC, na.rm = TRUE),
    AUC_sd   = sd(AUC, na.rm = TRUE),
    
    Accuracy_mean = mean(Accuracy, na.rm = TRUE),
    Accuracy_sd   = sd(Accuracy, na.rm = TRUE),
    
    Precision_mean = mean(Precision, na.rm = TRUE),
    Precision_sd   = sd(Precision, na.rm = TRUE),
    
    Recall_mean = mean(Recall, na.rm = TRUE),
    Recall_sd   = sd(Recall, na.rm = TRUE),
    
    F1_mean = mean(F1, na.rm = TRUE),
    F1_sd   = sd(F1, na.rm = TRUE),
    
    MCC_mean = mean(MCC, na.rm = TRUE),
    MCC_sd   = sd(MCC, na.rm = TRUE)
  )
format_metric <- function(mean, sd) {
  sprintf("%.3f (SD=%.3f)", mean, sd)
}

metric_summary <- metric_summary %>%
  mutate(
    AUC = format_metric(AUC_mean, AUC_sd),
    Accuracy = format_metric(Accuracy_mean, Accuracy_sd),
    Precision = format_metric(Precision_mean, Precision_sd),
    Recall = format_metric(Recall_mean, Recall_sd),
    F1 = format_metric(F1_mean, F1_sd),
    MCC = format_metric(MCC_mean, MCC_sd)
  ) %>%
  select(Model, Country, Data,
         AUC, Accuracy, Precision, Recall, F1, MCC)

stability_table <- all_features_within %>%
  filter(Model != "species") %>%
  
  
  group_by(Model, Country, Data, Repeat, Fold) %>%
  summarise(
    feature_list = list(Feature),
    .groups = "drop"
  ) %>%
  
  
  group_by(Model, Country, Data) %>%
  summarise(
    
    Stability_Index = calculate_jaccard(feature_list),
    
    
    Avg_Feature_Count = mean(map_int(feature_list, length)),
    
    
    Core_Feature_Count = length(reduce(feature_list, intersect)),
    
    .groups = "drop"
  )

table2 <- metric_summary %>%
  left_join(stability_table,
            by = c("Model", "Country", "Data"))

write_xlsx(table2,"Table 2. Performance of machine learning models under within-cohort repeated cross-validation.xlsx")


# -----------------------------------
# Table 3 LOSO Model Performance
# -----------------------------------

bayesian_df <- read_excel("bayesian_df.xlsx")
rf_df <- read_excel("rf_df.xlsx")
xgb_df <- read_excel("xgb_df.xlsx")


cols <- c("Model", "Data", "Train", "Test", "Features",
          "AUC", "Accuracy", "MCC", "F1 Score", "Precision", "Recall", "Inner_CV_AUC")

# --- 2. HER DATAFRAME'İ TEMİZLE + MODEL ADI EKLE ---
bayesian_df_clean <- bayesian_df %>%
  mutate(Model = "Bayesian Regression") %>%
  select(any_of(cols))

rf_df_clean <- rf_df %>%
  mutate(Model = "RF") %>%
  select(any_of(cols))

xgb_df_clean <- xgb_df %>%
  mutate(Model = "XGBOOST") %>%
  select(any_of(cols))

# --- 3. BİRLEŞTİR ---
final_table <- bind_rows(
  bayesian_df_clean,
  rf_df_clean,
  xgb_df_clean
)

# --- 4. FORMAT ---
final_table3 <- final_table %>%
  mutate(
    AUC = round(AUC, 3),
    Accuracy = round(Accuracy, 3),
    MCC = round(MCC, 3),
    `F1 Score` = round(`F1 Score`, 3),
    Precision = round(Precision, 3),
    Recall = round(Recall, 3),
    Inner_CV_AUC = round(Inner_CV_AUC, 3)
  ) %>%
  arrange(Model, Data, Train, Test)

write_xlsx(table3,"Table 3. Performance of machine learning models under leave-one-study-out (LOSO) validation.xlsx")


# -----------------------------------
# Table 4 POOLED Model Performance
# -----------------------------------

library(dplyr)
all_metrics_pooled <- read_excel("all_metrics_pooled.xlsx")
all_features_pooled <- read_excel("all_features_pooled.xlsx")
table4_pooled <- all_metrics_pooled %>%
  group_by(Model, Data) %>%
  summarise(
    AUC = sprintf("%.3f (SD=%.3f)", mean(AUC), sd(AUC)),
    Accuracy = sprintf("%.3f (SD=%.3f)", mean(Accuracy), sd(Accuracy)),
    Precision = sprintf("%.3f (SD=%.3f)", mean(Precision), sd(Precision)),
    Recall = sprintf("%.3f (SD=%.3f)", mean(Recall), sd(Recall)),
    F1 = sprintf("%.3f (SD=%.3f)", mean(F1), sd(F1)),
    MCC = sprintf("%.3f (SD=%.3f)", mean(MCC), sd(MCC)),
    .groups = "drop"
  )
stability_table <- all_features_pooled %>%
  filter(Model != "species") %>%
  group_by(Model, Data, Repeat, Fold) %>%
  summarise(feature_list = list(Feature), .groups = "drop") %>%
  
  group_by(Model, Data) %>%
  summarise(
    Stability = calculate_jaccard(feature_list),
    Avg_Feature_Count = mean(map_int(feature_list, length)),
    Core_Feature_Count = length(reduce(feature_list, intersect)),
    .groups = "drop"
  )
table4 <- table4_pooled %>%
  left_join(stability_table, by = c("Model", "Data")) %>%
  mutate(
    Stability = round(Stability, 3),
    Avg_Feature_Count = round(Avg_Feature_Count, 1)
  )

write_xlsx(table4,"Table 4. Performance of machine learning models trained on pooled datasets.xlsx")

# -----------------------------------
# Table 5 POOLED LinDA Features
# -----------------------------------

pooled_features <- pooled_features %>%
  mutate(model_rate = paste0(Model, ": ", Selection_Rate)) %>%
  group_by(Feature) %>%
  summarise(Model_Selection_Rate = paste(model_rate, collapse = " | "),Feature=Feature)

table5 <- pooled_features %>%
  left_join(linda_disease, by = "Feature") %>%
  left_join(linda_country, by = "Feature")%>%select(!c(diseasehealthy.pvalue,countryItaly.pvalue,Taxa_Level.x,Taxa_Level.y))%>%distinct()



write_xlsx(table5,"Table 5. Microbial features most frequently selected by machine learning models in the pooled analysis and their correspondence with differential abundance results.xlsx")



# -----------------------------------
# Supp Table 1 Read Depth
# -----------------------------------


asv_table <- read.csv("asv_table.csv", row.names = 1)

read_depth <- colSums(asv_table)

read_depth_df <- data.frame(
  SampleID = names(read_depth),
  ReadDepth = read_depth
)
write_xlsx(read_depth_df,"SuppTable1-ReadDepth.xlsx")



# -----------------------------------
# Supp Table 2 Permanova Results
# -----------------------------------

library(vegan)
library(dplyr)
library(purrr)

# metadata
meta <- metadata

# abundance tables
tables <- list(
  "Order" = "order_table.csv",
  "Family" = "family_table.csv",
  "Genus" = "genus_table.csv",
  "Full Hierarchical Taxonomy" = "path_table.csv"
)

results <- list()

for(level in names(tables)) {
  
  abund <- read.csv(tables[[level]], row.names = 1)
  abund <- abund[, meta$sample.id]
  # Bray-Curtis
  bc <- vegdist(t(abund), method = "bray")
  set.seed(42)
  # PERMANOVA
  perm <- adonis2(
    bc ~ disease + country + age + sex,
    data = meta,
    permutations = 9999,
    by="margin"
  )
  
  perm_df <- as.data.frame(perm)
  
  perm_df <- perm_df %>%
    mutate(
      Taxa_Level = level,
      Source = rownames(.)
    ) %>%
    select(Taxa_Level, Source, Df, R2, F, `Pr(>F)`)
  
  # Dispersion test (genelde disease için yapılır)
  disp <- betadisper(bc, meta$disease)
  disp_p <- anova(disp)$`Pr(>F)`[1]
  
  perm_df$Dispersion_P <- NA
  perm_df$Dispersion_P[1] <- disp_p
  
  results[[level]] <- perm_df
}

# tüm level'ları birleştir
permanova_table <- bind_rows(results)

# p-value formatting
permanova_table <- permanova_table %>%
  mutate(
    P_formatted = case_when(
      `Pr(>F)` < 0.001 ~ "<0.001",
      `Pr(>F)` < 0.01 ~ "<0.01",
      TRUE ~ sprintf("%.3f", `Pr(>F)`)
    ),
    Significance = case_when(
      `Pr(>F)` < 0.001 ~ "***",
      `Pr(>F)` < 0.01 ~ "**",
      `Pr(>F)` < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  select(Taxa_Level, Source, Df, R2, F, P_formatted, Significance, Dispersion_P)

write.csv(
  permanova_table,
  "permanova_results_with_dispersion.csv",
  row.names = FALSE
)



# -----------------------------------
# Supp Table 3 WITHIN LinDA Common
# -----------------------------------

within_linda <- within_features %>%
  left_join(linda_disease, by = "Feature") %>%
  left_join(linda_country, by = "Feature")%>%select(!c(diseasehealthy.pvalue,countryItaly.pvalue,Taxa_Level.x,Taxa_Level.y))

write_xlsx(within_linda,"Table3.Within_LINDA.xlsx")

# -----------------------------------
# Supp Table 4 LOSO LinDA Common
# -----------------------------------

loso_linda <- loso_features %>%
  left_join(linda_disease, by = "Feature") %>%
  left_join(linda_country, by = "Feature")%>%select(!c(diseasehealthy.pvalue,countryItaly.pvalue,Taxa_Level.x,Taxa_Level.y))%>%distinct()

write_xlsx(pooled_linda,"SuppTable4.LOSO_LINDA.xlsx")


# -----------------------------------
# Supp Table 5 Data Dimensions
# -----------------------------------
tables <- list(
  "Order" = "order_table.csv",
  "Family" = "family_table.csv",
  "Genus" = "genus_table.csv",
  "Full Taxonomic Hierarchy" = "path_table.csv"
)


supp_table_5 <- imap_dfr(tables, function(file_path, data_level) {
  
  df <- read_csv(file_path, show_col_types = FALSE)
  
  tibble(
    Data = data_level,
    Dimension = paste0(nrow(df), "x", ncol(df))
  )
})
write_xlsx(supp_table_5,"SupplementaryTable5.Dimensions.xlsx")


