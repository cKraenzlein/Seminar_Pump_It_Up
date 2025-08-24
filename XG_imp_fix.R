# ============================
# CSV → mlr3-Pipeline → XGBoost → Batch-Tuning (mit per-Eval-Output) → Submission
# ============================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(data.table)
  library(mlr3); library(mlr3pipelines); library(mlr3learners)
  library(mlr3tuning); library(mlr3mbo); library(paradox); library(bbotk)
  library(future); library(lgr)
})

set.seed(24)
lgr::get_logger("bbotk")$set_threshold("info")
lgr::get_logger("mlr3")$set_threshold("info")

# -------- 1) Daten einlesen --------
vars       <- read_csv("TrainingData_Variables.csv",      na = "NA", show_col_types = FALSE)
target     <- read_csv("TrainingData_TargetVariable.csv", na = "NA", show_col_types = FALSE)
train_full <- dplyr::inner_join(vars, target, by = "id")

test_full  <- read_csv("Testdaten.csv", na = "NA", show_col_types = FALSE)

stopifnot("status_group fehlt im Training!" = "status_group" %in% names(train_full))
stopifnot("id fehlt!" = all(c("id") %in% names(train_full)) && "id" %in% names(test_full))

train_ids <- train_full$id
test_ids  <- test_full$id

# -------- 2) Typen vereinheitlichen & gemeinsame Featurebasis --------
train_full$status_group <- factor(train_full$status_group)

char_train <- setdiff(names(which(sapply(train_full, is.character))), c("id", "status_group"))
char_test  <- setdiff(names(which(sapply(test_full,  is.character))), c("id"))
if (length(char_train)) for (cc in char_train) train_full[[cc]] <- as.factor(train_full[[cc]])
if (length(char_test))  for (cc in char_test)  test_full[[cc]]  <- as.factor(test_full[[cc]])

feat_names <- setdiff(intersect(names(train_full), names(test_full)), c("id", "status_group"))
train_df <- as.data.table(train_full[, c(feat_names, "status_group")])
test_df  <- as.data.table(test_full[,  feat_names])
stopifnot(identical(names(test_df), setdiff(names(train_df), "status_group")))

# -------- 3) Task --------
task <- as_task_classif(train_df, target = "status_group")
task$col_roles$stratum <- "status_group"

# -------- 4) IMP Pipeline fix
thr <- max(1L, min(8L, parallel::detectCores(logical = FALSE)))

imp_num <- po("missind") %>>%
  po("imputelearner",
     learner = lrn("regr.ranger",
                   num.threads = thr, num.trees = 250,
                   min.node.size = 5, mtry = 6),
     affect_columns = selector_name(c("longitude", "latitude")),
     id = "impute_num")

imp_factor <- po("imputelearner",
                 learner = lrn("classif.ranger",
                               num.threads = thr, num.trees = 250,
                               min.node.size = 5, mtry = 6),
                 affect_columns = selector_type("factor"),
                 id = "impute_factor")

imp_bin <- po("imputelearner",
              learner = lrn("classif.ranger",
                            num.threads = thr, num.trees = 250,
                            min.node.size = 5, mtry = 6),
              affect_columns = selector_type("logical"),
              id = "impute_bin")

po_select <- po("select", selector = selector_invert(selector_name(
  c("missing_population_log", "missing_gps_height",
    "missing_longitude", "missing_construction_year", "missing_amount_tsh_log")
)))

enc     <- po("encode", method = "one-hot")
rmconst <- po("removeconstants", ratio = 1.0)

# -------- 4b) XGBoost-Learner (Tuningraum) --------
xgb <- lrn("classif.xgboost",
           booster          = "gbtree",
           tree_method      = "hist",
           predict_type     = "prob",
           objective        = "multi:softprob",
           eval_metric      = "merror",
           nthread          = thr,
           max_depth        = to_tune(p_int(3, 12)),      # praxisnah & schnell
           min_child_weight = to_tune(p_int(1, 8)),
           subsample        = to_tune(p_dbl(0.7, 1.0)),
           colsample_bytree = to_tune(p_dbl(0.7, 1.0)),
           eta              = to_tune(p_dbl(0.01, 0.3)),
           nrounds          = to_tune(p_int(500, 1500))
)

# -------- 4c) Graph --------
graph   <- imp_num %>>% imp_factor %>>% imp_bin %>>% po_select %>>% enc %>>% rmconst %>>% po(xgb)
learner <- as_learner(graph)

# -------- 5) Tuning-Setup: MBO mit per-Eval-Output je Batch --------
resampling <- rsmp("cv", folds = 3)
measure    <- msr("classif.acc")

batch_size <- 10L
n_batches  <- 100L

terminator <- trm("evals", n_evals = 0L)
instance   <- ti(task, learner, resampling, measure, terminator)

tuner <- tnr("mbo")
if ("init_design_size" %in% tuner$param_set$ids())
  tuner$param_set$values$init_design_size <- batch_size

future::plan("multisession")

for (batch_id in seq_len(n_batches)) {
  n_before <- nrow(instance$archive$data)
  instance$terminator$param_set$values$n_evals <- n_before + batch_size
  
  message(sprintf("🔁 Batch %03d | evals %d → %d",
                  batch_id, n_before, n_before + batch_size))
  
  tuner$optimize(instance)
  
  n_after <- nrow(instance$archive$data)
  if (n_after <= n_before) { message("  (keine neuen Evals)"); break }
  
  # ---- per-Eval-Output für alle NEUEN Konfigurationen dieses Batches ----
  new <- data.table::copy(instance$archive$data)[(n_before + 1):n_after]
  acc_col <- grep("acc", names(new), value = TRUE, ignore.case = TRUE)[1]
  for (i in seq_len(nrow(new))) {
    r <- new[i]
    message(sprintf(
      "  • eval %4d: acc=%6.4f | nrounds=%s | eta=%s | max_depth=%s | min_child_weight=%s | subsample=%s | colsample_bytree=%s",
      n_before + i, as.numeric(r[[acc_col]]),
      as.character(r$nrounds), as.character(r$eta), as.character(r$max_depth),
      as.character(r$min_child_weight), as.character(r$subsample),
      as.character(r$colsample_bytree)
    ))
  }
  
  # ---- Archiv exportieren & Instanz sichern ----
  arch <- data.table::copy(instance$archive$data)
  drop_cols <- intersect(names(arch), c("x_domain","learner_param_vals"))
  if (length(drop_cols)) arch[, (drop_cols) := NULL]
  data.table::fwrite(arch, sprintf("xgb_BO_results_batch_%03d.csv", batch_id))
  saveRDS(instance,        sprintf("xgb_BO_batch_%03d.rds",          batch_id))
  
  # ---- Bestes Ergebnis bisher ----
  best <- instance$archive$best()
  message(sprintf(
    "✅ Best so far: acc=%.4f | nrounds=%s | eta=%s | max_depth=%s | min_child_weight=%s | subsample=%s | colsample_bytree=%s",
    best$classif.acc, best$nrounds, best$eta, best$max_depth,
    best$min_child_weight, best$subsample, best$colsample_bytree
  ))
  
  if (instance$terminator$is_terminated(instance$archive)) {
    message("Terminator erreicht – Batches beendet."); break
  }
}

message("Tuning abgeschlossen")

# -------- 6) Final trainieren auf allen Daten --------
learner$param_set$values <- instance$result_learner_param_vals
learner$train(task)

# -------- 7) Vorhersage & Submission --------
pred <- learner$predict_newdata(test_df)
submission <- data.table(id = test_ids, status_group = as.character(pred$response))
data.table::fwrite(submission, "TunedXGB_fromCSV.csv")
message("📄 TunedXGB_fromCSV.csv geschrieben")
