# Random Forest with mlr3
# Load required packages
library(mlr3)
library(mlr3learners) # Provides learners like 'classif.ranger'
#library(mlr3filters) # For feature selection
library(mlr3viz)     # For plotting
library(mlr3tuning)  # For hyperparameter tuning
library(mlr3fselect) # For feature selection
#library(ranger)      # The underlying Random Forest implementation
#library(iml)         # For model interpretation (including SHAP)
#library(data.table)  # mlr3 often works with data.table
library(mlr3pipelines) # For pipelines
library(paradox)

# Source R file for training data
here::here()
source("Training_Test_Data.r")
# Load the training data
Data <- TraingData_Formatted
#
set.seed(24)
future::plan("multisession", workers = 3)
task_NNET <- as_task_classif(Data, target = "status_group")

learner_NNET = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.nnet",
  size = to_tune(p_int(1,30)),
  decay = to_tune(p_dbl(1e-4, 0.2, logscale = TRUE)),
  maxit = to_tune(p_int(1, 2500)),
  MaxNWts = 10000))

instance_NNET <- ti(
  task = task_NNET,
  learner = learner_NNET,
  resampling = rsmp("cv", folds =3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 12 * 3600),
                                 trm("evals", n_evals = 100)), any = TRUE)
)
Tuner_NNET = tnr("random_search")

Tuner_NNET$optimize(instance_NNET)
# Results nach 4 Stunden Runtime, leider nur 7 Iterations
#decay              maxit         size        classif.acc
#-1.741247          1453          17         <  0.7710774
#--------------------------------------------------------
# Results nach 12 Stunden Runtime, leider nur 34 Iterations
#decay              maxit         size        classif.acc
#-5.677809          1544          26         <  0.7710774
autoplot(instance_NNET)

# XGBoost: 
task_XGBoost = as_task_classif(Data, target = "status_group")

numeric_feats = task_XGBoost$feature_types[type == "numeric", id]
ordinal_feats = c("payment_type")
nominal_feats = setdiff(task_XGBoost$feature_names, c(numeric_feats, ordinal_feats))

nominal_pipe = po("select", id = "select_nominal", selector = selector_name(nominal_feats)) %>>%
  po("encode", method = "one-hot", id = "encde_nominal")

ordinal_pipe = po("select", id = "select_ordinal", selector = selector_name(ordinal_feats)) %>>%
  po("encode", method = "treatment", id = "encode_ordinal")

numeric_pipe = po("select", id = "select_numeric", selector = selector_name(numeric_feats))

feature_union = gunion(list(nominal_pipe, ordinal_pipe, numeric_pipe)) %>>%
  po("featureunion", id = "combined_features")

learner_XGBoost = lrn("classif.xgboost", eval_metric = "error", nthread = 7)

graph = feature_union %>>% learner_XGBoost
graph_learner = GraphLearner$new(graph)

param_set = ps(
  classif.xgboost.eta = p_dbl(lower = 0.01, upper = 0.3),
  classif.xgboost.max_depth = p_int(lower = 10, upper = 100),
  classif.xgboost.nrounds = p_int(lower = 50, upper = 300),
  classif.xgboost.subsample = p_dbl(lower = 0.5, upper = 1),
  classif.xgboost.colsample_bytree = p_dbl(lower = 0.5, upper = 1)
)

tuner = tnr("random_search")
resampling = rsmp("cv", folds = 3)

at = AutoTuner$new(
  learner = graph_learner,
  resampling = resampling,
  measure = msr("classif.acc"),
  tuner = tuner,
  search_space = param_set,
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 12 * 3600),
                                 trm("evals", n_evals = 100)), any = TRUE)
)

at$train(task = task_XGBoost)

# Results after 42 evaluations (3h)
#eta            max_depth     nrounds           subsample         colsample_bytree      classif.acc
#0.1962411      10            134               0.8960142         0.9189042             0.8084343
autoplot(at$tuning_instance)
at$tuning_result
# Results adter 72 evaluations
#eta            max_depth     nrounds           subsample         colsample_bytree      classif.acc
#0.01908678     60            291               0.6791455         0.6432647             0.8098316


