````
## XG BOOST Model
````


# 0) Pakete laden
install.packages("xgboost")
library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(Matrix)
library(xgboost)

# 1) Daten einlesen und zusammenführen
vars       <- read_csv("TrainingData_Variables.csv",
                       na = "NA", show_col_types = FALSE)
target     <- read_csv("TrainingData_TargetVariable.csv",
                       na = "NA", show_col_types = FALSE)
train_full <- inner_join(vars, target, by = "id")

test_full  <- read_csv("Testdaten.csv",
                       na = "NA", show_col_types = FALSE)

# 2) Ausgewählte Variablen definieren und auswählen
selected_vars <- c(
  "amount_tsh", "gps_height", "longitude", "latitude",
  "region", "population", "construction_year",
  "extraction_type", "extraction_type_class", "management_group",
  "payment", "water_quality", "quantity",
  "source_type", "source_class", "waterpoint_type",
  "installer"
)

train_sel <- train_full %>%
  select(id, all_of(selected_vars), status_group)

test_sel  <- test_full %>%
  select(id, all_of(selected_vars))

# 3) Datentypen setzen: faktorielle Prädiktoren & Zielvariable
cat_vars <- c(
  "extraction_type", "extraction_type_class", "management_group",
  "payment", "water_quality", "quantity",
  "source_type", "source_class", "waterpoint_type",
  "installer", "region"
)

train_sel <- train_sel %>%
  mutate(
    status_group = factor(status_group),
    across(all_of(cat_vars), as.character)
  ) %>%
  # NA in installer → "unknown"
  mutate(installer = ifelse(is.na(installer), "unknown", installer)) %>%
  mutate(across(all_of(cat_vars), as.factor))

test_sel <- test_sel %>%
  mutate(across(all_of(cat_vars), as.character)) %>%
  mutate(installer = ifelse(is.na(installer), "unknown", installer)) %>%
  mutate(across(all_of(cat_vars), as.factor))

# 4) Installer auf Top-25 beschränken, Rest = "other"
Top25_installers <- train_sel %>%
  count(installer, sort = TRUE) %>%
  slice_head(n = 25) %>%
  pull(installer)

train_sel <- train_sel %>%
  mutate(
    installer = as.character(installer),
    installer = ifelse(installer %in% Top25_installers, installer, "other"),
    installer = as.factor(installer)
  )

test_sel <- test_sel %>%
  mutate(
    installer = as.character(installer),
    installer = ifelse(installer %in% Top25_installers, installer, "other"),
    installer = as.factor(installer)
  )

# 5) IDs für später merken und nun aus den Data Frames entfernen
test_ids <- test_sel$id
train_df <- train_sel %>% select(-id)
test_df  <- test_sel  %>% select(-id)

# 6)  Faktor: NA → "missing"
train_imp <- train_df %>%
  mutate(
    across(where(is.numeric),
           ~ ifelse(is.na(.), median(.x, na.rm = TRUE), .x))
  ) %>%
  mutate(
    across(all_of(cat_vars),
           ~ replace_na(as.character(.x), "missing"))
  ) %>%
  mutate(across(all_of(cat_vars), as.factor)) %>%
  mutate(status_group = as.factor(status_group))

test_imp <- test_df %>%
  mutate(
    across(where(is.numeric),
           ~ ifelse(is.na(.),
                    median(train_imp[[cur_column()]], na.rm = TRUE),
                    .x))
  ) %>%
  mutate(
    across(all_of(cat_vars),
           ~ replace_na(as.character(.x), "missing"))
  ) %>%
  mutate(across(all_of(cat_vars), as.factor))

# 7) (Z-Transformation)
num_vars <- train_imp %>% select(where(is.numeric)) %>% names()
pp       <- preProcess(train_imp[num_vars], method = c("center","scale"))

train_imp[num_vars] <- predict(pp, train_imp[num_vars])
test_imp[num_vars]  <- predict(pp, test_imp[num_vars])

# 8) Faktor-Levels im Test an die im Training anpassen (Dummy-Konsistenz)
for (col in intersect(cat_vars, names(test_imp))) {
  test_imp[[col]] <- factor(
    test_imp[[col]],
    levels = levels(train_imp[[col]])
  )
}

# 9) Sparse Design-Matrizen für xgboost bauen
#    Zielvariable in numerische Codes umwandeln (0,1,2)
label_levels <- levels(train_imp$status_group)
train_imp$label <- as.integer(train_imp$status_group) - 1

x_train <- sparse.model.matrix(label ~ . - status_group - 1, data = train_imp)
y_train <- train_imp$label

x_test  <- sparse.model.matrix(~ . -1, data = test_imp)

# 10) Fehlende Dummy-Spalten in x_test ergänzen und Spaltenreihenfolge angleichen
train_cols   <- colnames(x_train)
test_cols    <- colnames(x_test)
missing_cols <- setdiff(train_cols, test_cols)

for (col in missing_cols) {
  x_test <- cbind(
    x_test,
    Matrix(0, nrow(x_test), 1, dimnames = list(NULL, col))
  )
}
x_test <- x_test[, train_cols]

# 11) optimale nrounds finden
dtrain_mat <- xgb.DMatrix(data = x_train, label = y_train)

num_class <- length(label_levels)
params <- list(
  objective        = "multi:softprob",
  eval_metric      = "mlogloss",
  num_class        = num_class,
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)

set.seed(42)
cvres <- xgb.cv(
  params              = params,
  data                = dtrain_mat,
  nrounds             = 1000,
  nfold               = 5,
  early_stopping_rounds = 20,
  verbose             = 0
)

best_nrounds <- cvres$best_iteration

# 12) Endgültiges Modell 
xgb_model <- xgb.train(
  params    = params,
  data      = dtrain_mat,
  nrounds   = best_nrounds,
  verbose   = 0
)

# 13) Vorhersage auf dem gesamten Testset (14 850 Zeilen)
dtest_mat   <- xgb.DMatrix(data = x_test)
pred_prob   <- predict(xgb_model, newdata = dtest_mat)
pred_matrix <- matrix(pred_prob, ncol = num_class, byrow = TRUE)
pred_label  <- max.col(pred_matrix) - 1

pred_factor <- factor(label_levels[pred_label + 1], levels = label_levels)

# 14) Prediction
output <- tibble(
  id           = test_ids,
  status_group = as.character(pred_factor)
)

write_csv(output, "XG_status_group.csv")

# 15) (Optional) Metriken auf dem Trainingssatz ausgeben:
#   Confusion-Matrix
train_pred_prob <- predict(xgb_model, newdata = dtrain_mat)
train_pred_mat  <- matrix(train_pred_prob, ncol = num_class, byrow = TRUE)
train_pred_lbl  <- max.col(train_pred_mat) - 1
train_pred_fac  <- factor(label_levels[train_pred_lbl + 1], levels = label_levels)
train_true_fac  <- train_imp$status_group

confusion_train <- confusionMatrix(train_pred_fac, train_true_fac)
print(confusion_train)
