

# 1) Importance
imp <- xgb.importance(feature_names = colnames(x_train), model = xgb_model)

imp_plot2 <- imp %>%
  arrange(desc(Gain)) %>%
  slice_head(n = 15)

# 2) Manuelles Mapping NUR für die Features, die du ändern willst
feat_map <- c(
  amount_tsh            = "available water ",
  gps_height            = "GPS height",
  mean_longitude        = "longitude",
  mean_latitude         = "latitude",
  region                = "Region",
  population            = "Population",
  construction_year     = "construction year",
  extraction_type       = "Extraction type 1",
  extraction_type_class = "Extraction type 2",
  management_group      = "Management group",
  payment               = "Payment",
  water_quality         = "Water quality",
  quantity              = "Quantity",
  source_type           = "Source type 2",
  source_class          = "Source class 1",
  waterpoint_type       = "Waterpoint type",
  installer             = "Installer"
)

imp_plot2 <- imp_plot2 %>%
  mutate(
    Feature_pretty = dplyr::recode(Feature, !!!feat_map, .default = Feature),
    Feature_pretty = factor(Feature_pretty, levels = rev(Feature_pretty))
  )

# 3) Plot mit theme_bw und Achsentiteln
ggplot(imp_plot2, aes(x = Gain, y = Feature_pretty)) +
  geom_col(fill = "#4C92C3") +
  labs(x = "Permutation Importance", y = "Feature") +
  theme_bw(base_size = 14) +
  theme(panel.grid.major.y = element_blank(),
        legend.position    = "none")











library(dplyr)
library(ggplot2)
library(purrr)

# 1) Importance roh
imp_raw <- xgb.importance(feature_names = colnames(x_train), model = xgb_model)


all_vars <- c(cat_vars, setdiff(colnames(train_imp), c(cat_vars, "status_group", "label")))

base_of <- function(feat){
  hit <- all_vars[startsWith(feat, all_vars)]
  if(length(hit)) hit[1] else feat
}

imp_collapsed <- imp_raw %>%
  mutate(Base = vapply(Feature, base_of, character(1))) %>%
  group_by(Base) %>%
  summarise(Gain = sum(Gain), .groups = "drop") %>%
  arrange(desc(Gain)) %>%
  slice_head(n = 15)

# 3) Umbennenung
feat_map <- c(
  amount_tsh        = "available water",
  gps_height        = "GPS height",
  mean_longitude    = "longitude",
  mean_latitude     = "latitude",
  construction_year = "construction year",
  population        = "population",
  quantity          = "quantity",
  source_type       = "source type",
  source_class      = "source class",
  extraction_type   = "extraction type 2",
  extraction_type_class = "extraction type 3",
  extraction_type_group = "extraction type 1",
  management_group  = "management group",
  waterpoint_type   = "waterpoint type",
  installer         = "installer",
  region            = "region",
  district_code     = "district code",
  payment_type      = "payment intervals"
)

imp_plot <- imp_collapsed %>%
  mutate(Base_pretty = dplyr::recode(Base, !!!feat_map, .default = Base),
         Base_pretty = factor(Base_pretty, levels = rev(Base_pretty)))

# 4) Plot
ggplot(imp_plot, aes(x = Gain, y = Base_pretty)) +
  geom_col(fill = "#4C92C3") +
  labs(x = "Importance", y = "Feature") +
  theme_bw(base_size = 14) +
  theme(panel.grid.major.y = element_blank(),
        legend.position    = "none")



