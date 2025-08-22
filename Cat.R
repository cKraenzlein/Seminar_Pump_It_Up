



------------------------ #8 stunden Tuning model
  suppressPackageStartupMessages({
    library(xgboost)
    library(dplyr)
    library(readr)
    library(purrr)
  })

set.seed(42)

# ========== 1) Parameter ==========
time_limit_sec <- 28800  # 8 Stunden
n_iter         <- 1000   # Max. Anzahl Kombinationen (zur Sicherheit, bricht vorher ab)
nfold          <- 5
early_stop     <- 50
eval_metric    <- "merror"
result_path    <- "xgb_tuning_results.csv"

# ========== 2) Daten vorbereiten ==========
# Verwendung nur der reduzierten Features (nach Feature Selection)
dtrain <- xgb.DMatrix(x_train[, best_set], label = y_train)

# ========== 3) Parameter-Räume definieren ==========
sample_param <- function() {
  list(
    eta              = runif(1, 0.01, 0.3),
    max_depth        = sample(3:12, 1),
    min_child_weight = sample(1:10, 1),
    subsample        = runif(1, 0.6, 1.0),
    colsample_bytree = runif(1, 0.6, 1.0),
    gamma            = runif(1, 0, 5),
    lambda           = runif(1, 0, 3),
    alpha            = runif(1, 0, 3)
  )
}

# ========== 4) Lauf starten ==========
results <- list()
t_start <- Sys.time()

for (i in 1:n_iter) {
  t_now <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  if (t_now > time_limit_sec) {
    message("⏳ Zeitlimit erreicht. Tuning abgebrochen.")
    break
  }
  
  param <- sample_param()
  param$objective   <- "multi:softprob"
  param$num_class   <- length(label_levels)
  param$eval_metric <- eval_metric
  
  cv <- xgb.cv(
    params                = param,
    data                  = dtrain,
    nrounds               = 5000,
    nfold                 = nfold,
    early_stopping_rounds = early_stop,
    verbose               = 0
  )
  
  best_iter <- cv$best_iteration
  acc       <- 1 - cv$evaluation_log$test_merror_mean[best_iter]
  
  result_row <- tibble(
    iter      = i,
    accuracy  = acc,
    nrounds   = best_iter,
    eta              = param$eta,
    max_depth        = param$max_depth,
    min_child_weight = param$min_child_weight,
    subsample        = param$subsample,
    colsample_bytree = param$colsample_bytree,
    gamma            = param$gamma,
    lambda           = param$lambda,
    alpha            = param$alpha,
    time_sec         = t_now
  )
  
  results[[i]] <- result_row
  
  # Zwischenspeichern
  write_csv(bind_rows(results), result_path)
  
  message(sprintf("Iter %03d | Acc = %.4f | Zeit: %.1f min", i, acc, t_now / 60))
}



# ========== 5) Beste Parameter-Kombi laden ==========
res_tbl  <- read_csv(result_path)
best_row <- res_tbl %>% slice_max(accuracy, n = 1)

best_param <- list(
  objective        = "multi:softprob",
  num_class        = length(label_levels),
  eval_metric      = eval_metric,
  eta              = best_row$eta,
  max_depth        = best_row$max_depth,
  min_child_weight = best_row$min_child_weight,
  subsample        = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  gamma            = best_row$gamma,
  lambda           = best_row$lambda,
  alpha            = best_row$alpha
)
best_nrounds <- best_row$nrounds

# ========== 6) Finales Modell trainieren ==========
final_model <- xgb.train(
  params  = best_param,
  data    = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# ========== 7) Vorhersage auf Testdaten ==========
dtest      <- xgb.DMatrix(x_test[, best_set])
prob       <- predict(final_model, dtest)
mat        <- matrix(prob, ncol = length(label_levels), byrow = TRUE)
lbl_idx    <- max.col(mat) - 1
pred_final <- factor(label_levels[lbl_idx + 1], levels = label_levels)

# ========== 8) Exportieren ==========
write_csv(tibble(id = FAS_ids, status_group = as.character(pred_final)),
          "XG_tuned_t3.csv")




library(readr)
library(dplyr)

# CSV-Datei einlesen
res_tbl <- read_csv("xgb_tuning_results.csv")

# Beste Zeile (nach Accuracy sortiert)
best_row <- res_tbl %>% slice_max(accuracy, n = 1)

# Parameter extrahieren
best_param <- list(
  objective        = "multi:softprob",
  num_class        = length(label_levels),
  eval_metric      = "merror",
  eta              = best_row$eta,
  max_depth        = best_row$max_depth,
  min_child_weight = best_row$min_child_weight,
  subsample        = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  gamma            = best_row$gamma,
  lambda           = best_row$lambda,
  alpha            = best_row$alpha
)

# Beste Rundenzahl (early stopped rounds)
best_nrounds <- best_row$nrounds

# Ausgabe der Werte:
print(best_param)
cat("\nBeste Anzahl an Boosting-Runden (nrounds):", best_nrounds, "\n")
