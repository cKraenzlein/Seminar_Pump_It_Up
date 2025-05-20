# 0) Pakete laden
library(readr)
library(dplyr)
library(purrr)
library(caret)
library(Matrix)
library(glmnet)

# 1) Daten einlesen und zusammenführen
vars        <- read_csv("TrainingData_Variables.csv",    na = "NA", show_col_types = FALSE)
target      <- read_csv("TrainingData_TargetVariable.csv", na = "NA", show_col_types = FALSE)
train_raw   <- inner_join(vars, target, by = "id")

test_raw_full <- read_csv("Seminar/Testdaten.csv", na = "NA", show_col_types = FALSE)

# 2) IDs rauswerfen
test_ids  <- test_raw_full$id
train_raw <- train_raw   %>% select(-id)
test_raw  <- test_raw_full %>% select(-id)

# 3) Faktor markieren
train_raw <- train_raw %>% mutate(status_group = as.factor(status_group))

# 4) Konstante-Variablen entfernen
zv_metrics <- nearZeroVar(train_raw, saveMetrics = TRUE)
zero_vars  <- rownames(zv_metrics[zv_metrics$zeroVar, ])
train_kv   <- train_raw %>% select(-all_of(zero_vars))
test_kv    <- test_raw  %>% select(-all_of(zero_vars))

# 5) Level filtern
fac       <- train_kv %>% select(where(is.character)) %>% names()
lvl_cnts  <- map_int(train_kv[fac], ~ n_distinct(.x))
drop_hc   <- names(lvl_cnts[lvl_cnts > 100])
train_kv  <- train_kv %>% select(-all_of(drop_hc))
test_kv   <- test_kv  %>% select(-all_of(drop_hc))

# 6) Numerische Skalierung
num_vars <- train_kv  %>% select(where(is.numeric)) %>% names()
pp       <- preProcess(train_kv[num_vars], method = c("center", "scale"))
train_kv[num_vars] <- predict(pp, train_kv[num_vars])
test_kv[num_vars]  <- predict(pp, test_kv[num_vars])

# 7) Complete Case Analyse
train_cc <- train_kv %>% drop_na()
test_cc  <- test_kv  %>% drop_na()


test_ids_cc <- test_ids[complete.cases(test_kv)]


# 8) Faktor & Levels anpassen
pred_cols   <- setdiff(names(train_cc), "status_group")
factor_cols <- pred_cols[sapply(train_cc[pred_cols], is.factor)]

for(col in factor_cols) {
  test_cc[[col]] <- factor(
    test_cc[[col]], 
    levels = levels(train_cc[[col]])
  )
}

# 9) Sparse-Design-Matrizen aufbauen
library(Matrix)
x_train <- sparse.model.matrix(status_group ~ . -1, data = train_cc)
y_train <- train_cc$status_group

x_test  <- sparse.model.matrix(~ . -1, data = test_cc)

# 10) Ordnen
train_cols       <- colnames(x_train)
test_cols        <- colnames(x_test)
missing_in_test  <- setdiff(train_cols, test_cols)

# Null-Spalten an x_test anhängen
for(col in missing_in_test) {
  x_test <- cbind(
    x_test,
    Matrix(0, nrow(x_test), 1, dimnames = list(NULL, col))
  )
}

# Spaltenreihenfolge des Testsets an das Training anpassen
x_test <- x_test[, train_cols]

# 11) Penalisiertes multinomiales Logit-Modell mit CV fitten
set.seed(42)
cvfit <- cv.glmnet(
  x                = x_train,
  y                = y_train,
  family           = "multinomial",
  type.multinomial = "grouped"
)

# 12) Vorhersage auf dem angepassten Testset
pred_glmnet <- predict(
  cvfit, 
  newx = x_test, 
  s    = "lambda.min", 
  type = "class"
)

# 13) Ergebnis zusammenstellen & exportieren
output <- tibble(
  id           = test_ids_cc,
  status_group = as.character(pred_glmnet)
)

write_csv(output, "predicted_status_group.csv")