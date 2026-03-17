library(tidyverse)
library(vegan)
library(ggplot2)
library(readxl)
# --------------------------
# Paths
# --------------------------
data_path  <- "data_summary/"
fig_path   <- "output/figures/"
tab_path   <- "output/tables/"

dir.create(fig_path, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_path, recursive = TRUE, showWarnings = FALSE)

# --------------------------
# Load data
# --------------------------
pcoa_df   <- read_csv(paste0(data_path, "Genus_PcoA_df.xlsx"))
loso   <- read_csv(paste0(data_path, "loso_results.csv"))
within <- read_csv(paste0(data_path, "within_results.csv"))
pooled <- read_csv(paste0(data_path, "pooled_results.csv"))

pooled_features <- read_excel(paste0(data_path, "pooled_high_freq_features.xlsx"))
# --------------------------
# Factor ordering
# --------------------------
order_levels <- c("Genus","Family","Order","Full Taxonomic Hierarchy")

loso$Data   <- factor(loso$Data, levels = order_levels)
within$Data <- factor(within$Data, levels = order_levels)
pooled$Data <- factor(pooled$Data, levels = order_levels)

# --------------------------
# FIGURE 1 — Genus PcoA
# --------------------------

figure1<-ggplot(pcoa_df,
       aes(x = PC1,
           y = PC2,
           color = Disease,
           shape = Country)) +
  
  geom_point(size = 5, alpha = 0.9) +
  
  stat_ellipse(
    aes(group=Disease,color = Disease),
    type = "norm",
    level = 0.95,
    linetype = "dashed",
    linewidth = 0.9
  ) +
  
  scale_color_manual(values = c(
    "Type 1 Diabetes" = "#D55E00",
    "Healthy" = "#009E73"
  )) +
  
  scale_shape_manual(values = c(
    "China" = 17,
    "Italy" = 15
  )) +
  
  labs(
    x = "PC1 (18%)",
    y = "PC2 (11%)"
  ) +
  
  theme_classic(base_size = 14) +
  
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 16),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )

ggsave(
  "figure1-genus_pcoa.pdf",
  plot = figure1,
  width = 12,
  height = 8,
  units = "in",
  dpi = 600
)
# --------------------------
# FIGURE 2 — LOSO dumbbell
# --------------------------
loso$Model <- factor(
  loso$Model,
  levels = c("Bayesian Regression","RF","XGBOOST")
)


figure2<-ggplot(loso, aes(y = Data)) +
  geom_segment(aes(x = AUC, xend = Inner_CV_AUC, yend = Data),
                                      color = "grey70",size=2) +
       geom_point(aes(x = Inner_CV_AUC),  shape = 21,          # Kenarlıklı nokta şekli
                  fill = "#1A73E8",    # İç dolgu (Mavi)
                  color = "black",     # Kenarlık rengi (Siyah)
                  size = 5, 
                  stroke = 1.2) +
       geom_point(aes(x = AUC), shape = 21,          # Kenarlıklı nokta şekli
                  fill = "#D93025",    # İç dolgu (Kırmızı)
                  color = "black",     # Kenarlık rengi (Siyah)
                  size = 5,            # Nokta büyüklüğü
                  stroke = 1.2) +
       facet_grid(Model ~ Comparison) +
       theme_classic()+
  theme(
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.2),
    panel.grid.minor.x = element_blank()
  ) +
       labs(x="AUC", y="Taxonomic Level")+
  theme(axis.text.x = element_text(size=16),axis.text.y = element_text(size=16),
                                                axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),strip.text.x=element_text(size=15),
                                                strip.text.y=element_text(size=16),
                                                plot.subtitle = element_text(size = 16, face = "plain", hjust = 0, margin = margin(b = 10)))+
  labs(subtitle = "Blue = Internal Cross−Validation   |   Red = External Test Cohort")

ggsave(
  "figure2-loso_dumbell.pdf",
  plot = figure2,
  width = 12,
  height = 8,
  units = "in",
  dpi = 600
)
# --------------------------
# FIGURE 3 — Pooled AUC Boxplot
# --------------------------

figure3<-ggplot(pooled, aes(x = Model, y = AUC, fill = as.factor(Fold))) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA,size=1) +
  geom_jitter(aes(color = as.factor(Fold)), width = 0.2, alpha = 0.5) +
  facet_wrap(~ Data, scales = "free_y") +
  theme_bw() +
  labs(
    x = "Model",
    y = "AUC Score",
    fill = "Fold No",
    color = "Fold No"
  ) +
  scale_fill_brewer(palette = "Reds") +
  scale_color_brewer(palette = "Reds") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,size = 16))+
  theme(axis.text.x = element_text(size=16,face = "bold"),axis.text.y = element_text(size=16,face = "bold"),
        axis.title.x = element_text(size=16,face = "bold"),axis.title.y = element_text(size=16,face = "bold"),strip.text.x=element_text(size=15,face = "bold"),
        strip.text.y=element_text(size=16,face = "bold"),legend.title =element_text(size=16,face = "bold"),legend.text = element_text(size=16,face = "bold"))
ggsave(
  "figure3-pooled_boxplot.pdf",
  plot = figure3,
  width = 12,
  height = 8,
  units = "in",
  dpi = 600
)
# --------------------------
# FIGURE 4 — Pooled Top Features
# --------------------------

pooled_features <- pooled_features2 %>%
  group_by(Feature) %>%
  mutate(Model_Count = sum(Selection_Rate >= 0.8)) %>%
  ungroup() %>%
  mutate(
    Feature = fct_reorder(Feature, Model_Count, .fun = max)
  )
figure4<-pooled_features%>%ggplot(  aes(x = Model,
                                                             y = Feature,
                                                             fill = Selection_Rate)) +
  
  geom_tile(color = "white", linewidth = 0.4) +
  
  
  
  scale_fill_gradient(
    low = "lightgray",
    high = "#7f0000",
    name = "Selection\nFrequency"
  ) +
  
  labs(
    x = "Machine Learning Model",
    y = "Microbial Taxa"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    axis.text.x = element_text(size=15),
    axis.text.y = element_text(size = 15,face="italic")
  )

ggsave(
  "figure4-pooled_feature_heatmap.pdf",
  plot = figure4,
  width = 20,
  height = 8,
  units = "in",
  dpi = 600
)
