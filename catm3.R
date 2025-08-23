# ============================
# CatBoost Tuning (mlr3): Random → Grid, 3×10 CV, ohne One-Hot
# ============================

remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-darwin-universal2-1.2.8.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))


suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(catboost); library(caret); library(readr)
})

suppressPackageStartupMessages({
  library(data.table)
  library(mlr3)
  library(mlr3pipelines)
  library(mlr3learners)        # ranger, kknn etc.
  library(mlr3tuning)
  library(bbotk)
})

# --- 0) Sicherstellen, dass CatBoost-Loader da ist ---
if (!requireNamespace("mlr3extralearners", quietly = TRUE)) install.packages("mlr3extralearners")
if (!requireNamespace("catboost", quietly = TRUE)) install.packages("catboost")
library(mlr3extralearners)
stopifnot("classif.catboost" %in% mlr_learners$keys())

set.seed(24)

# --- 1) Daten vorbereiten -----------------------------------------------------
# Variante A: vorhandenes train_df (Features + status_group als Faktor)
if (exists("train_df")) {
  train_dt <- as.data.table(train_df)
  if (!is.factor(train_dt$status_group)) train_dt[, status_group := factor(status_group)]
} else if (exists("x_train") && exists("y_train")) {
  # Variante B: x_train / y_train
  train_dt <- as.data.table(as.data.frame(x_train))
  train_dt[, status_group := factor(y_train)]
} else {
  stop("Bitte 'train_df' (mit status_group) oder (x_train, y_train) bereitstellen.")
}

task <- TaskClassif$new("catboost_task", backend = train_dt, target = "status_group")
task$col_roles$stratum <- "status_group"

# --- 2) Pipeline: Imputation (ohne One-Hot) + Konstanten entfernen -----------
thr <- max(1L, min(8L, parallel::detectCores(logical = FALSE)))

imp_num <- po("missind") %>>%
  po("imputelearner",
     learner = lrn("regr.ranger",
                   num.threads = thr, num.trees = 100, max.depth = 10,
                   min.node.size = 5, mtry.ratio = 0.5),
     affect_columns = selector_type("numeric"),
     id = "imp_num")

imp_fac <- po("imputemode", affect_columns = selector_type("factor"),  id = "imp_fac")
imp_log <- po("imputemode", affect_columns = selector_type("logical"), id = "imp_log")
rmconst <- po("removeconstants", ratio = 1.0)

# --- 3) CatBoost-Learner (Tuning-Space) --------------------------------------
cb <- lrn("classif.catboost",
          predict_type  = "prob",
          loss_function = "MultiClass",
          thread_count  = thr,
          # Tuning-Parameter (robust & schlank):
          iterations    = to_tune(p_int(400, 1500)),
          learning_rate = to_tune(p_dbl(0.02, 0.3)),
          depth         = to_tune(p_int(4, 10)),
          l2_leaf_reg   = to_tune(p_dbl(1, 10))
          # Optional zusätzlich:
          # subsample   = to_tune(p_dbl(0.6, 1.0)),
          # rsm         = to_tune(p_dbl(0.6, 1.0))   # column sampling
)

graph   <- imp_num %>>% imp_fac %>>% imp_log %>>% rmconst %>>% po(cb)
learner <- as_learner(graph)

# --- 4) CV & Metrik (einheitlich über alle Modelle) --------------------------
resampling <- rsmp("repeated_cv", folds = 3, repeats = 10)
measure    <- msr("classif.acc")

# --- 5) Stage 1 – Random Search (breit erkunden) -----------------------------
tuner_rs <- tnr("random_search")
term_rs  <- trm("evals", n_evals = 300L)   # Budget anpassen (z. B. 300–600)

inst_rs <- ti(
  task       = task,
  learner    = learner,          # nimmt die to_tune-Definitionen aus dem Learner
  resampling = resampling,
  measures   = measure,
  terminator = term_rs
)

if (requireNamespace("future", quietly = TRUE)) future::plan("sequential")

tuner_rs$optimize(inst_rs)

best_rs <- inst_rs$result_learner_param_vals
cat(sprintf("Stage 1 (Random): acc=%.4f | %s\n",
            inst_rs$result_y,
            paste(sprintf("%s=%s", names(best_rs), unlist(best_rs)), collapse = ", ")))

# --- 6) Stage 2 – Grid Search (fein um das beste Random-Ergebnis) ------------
clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)

b_it <- as.integer(best_rs$`classif.catboost.iterations`)
b_lr <- as.numeric(best_rs$`classif.catboost.learning_rate`)
b_dp <- as.integer(best_rs$`classif.catboost.depth`)
b_l2 <- as.numeric(best_rs$`classif.catboost.l2_leaf_reg`)

refined_space <- ps(
  classif.catboost.iterations    = p_int(clamp(round(b_it*0.7), 300, 3000),
                                         clamp(round(b_it*1.3), 300, 5000)),
  classif.catboost.learning_rate = p_dbl(clamp(b_lr*0.6, 0.01, 0.5),
                                         clamp(b_lr*1.4, 0.01, 0.5)),
  classif.catboost.depth         = p_int(max(3L, b_dp-1L), min(12L, b_dp+1L)),
  classif.catboost.l2_leaf_reg   = p_dbl(clamp(b_l2*0.5, 0.1, 20),
                                         clamp(b_l2*1.5, 0.1, 20))
)

tuner_gs <- tnr("grid_search", resolution = 3L)  # ~ bis zu 3^4 = 81 Punkte
term_gs  <- trm("evals", n_evals = 200L)         # Sicherheitskappe

inst_gs <- TuningInstanceBatchSingleCrit$new(
  task         = task,
  learner      = learner,
  resampling   = resampling,
  measure      = measure,
  search_space = refined_space,
  terminator   = term_gs
)

tuner_gs$optimize(inst_gs)

best_gs <- inst_gs$result_learner_param_vals
cat(sprintf("Stage 2 (Grid):   acc=%.4f | %s\n",
            inst_gs$result_y,
            paste(sprintf("%s=%s", names(best_gs), unlist(best_gs)), collapse = ", ")))

# --- 7) Final: beste Params setzen & auf allen Daten trainieren --------------
final_params <- inst_gs$result_learner_param_vals
if (is.null(final_params) || length(final_params) == 0L) {
  final_params <- inst_rs$result_learner_param_vals
}
learner$param_set$values <- final_params
learner$train(task)

saveRDS(learner, "catboost_final_model.rds")
cat("catboost_final_model.rds gespeichert\n")

# --- 8) (Optional) Vorhersagen / Submission ----------------------------------
# Falls du test_df/test_ids hast (gleiche Spalten wie train_df ohne status_group):
if (exists("test_df")) {
  pred <- learner$predict_newdata(as.data.table(test_df))
  print(table(pred$response))
  if (exists("test_ids")) {
    subm <- data.table(id = test_ids, status_group = as.character(pred$response))
    fwrite(subm, "CatBoost_Submission.csv")
    cat("CatBoost_Submission.csv geschrieben\n")
  }
}



