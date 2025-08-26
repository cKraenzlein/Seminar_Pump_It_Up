# Load required packages
library(mlr3)
library(mlr3viz)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)
library(mlr3mbo)

# Set the working directory
here::here()

# Load the training data
source("./Data/CreateTrainingData.R")
Data <- Data_Training_Final

# Set seed for reproducibility
set.seed(24)

# Pipeline
encode_factors = po("encode", id = "encode_factors", method = "one-hot", affect_columns = selector_type("factor"))

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner
learner = lrn("classif.xgboost", id = "xgboost", nthread = 10)

# Combine graph and learner
graph_learner = as_learner(encode_factors %>>% learner)

# Hyperparameter space
hyper_param_space <- ps(
  xgboost.eta =               p_dbl(lower = 0.01, upper = 0.3),
  xgboost.max_depth =         p_int(lower = 3, upper = 10),
  xgboost.nrounds =           p_int(lower = 100, upper = 1000),
  xgboost.min_child_weight =  p_int(lower = 1, upper = 10),
  xgboost.subsample =         p_dbl(lower = 0.6, upper = 1),
  xgboost.colsample_bytree =  p_dbl(lower = 0.6, upper = 1),
  xgboost.gamma =             p_dbl(lower = 0, upper = 5)
)
# Resampling and Measure Strategy
resampling <- rsmp("cv", folds = 3)
measure    <- msr("classif.acc")

# Define Tuner and Termination
tuner_rs <- tnr("mbo")
term_rs  <- trm("combo", list(trm("clock_time", stop_time = Sys.time() + 4 * 3600),
                              trm("evals", n_evals = 100)), any = TRUE)

# Tuning instance
tuning_instance <- ti(
  task       = task,
  learner    = graph_learner,
  resampling = resampling,
  measures   = measure,
  search_space = hyper_param_space,
  terminator = term_rs
)

# Start tuning
tuner_rs$optimize(tuning_instance)
mlr3::autoplot(tuning_instance)
# Save tuning results
saveRDS(tuning_instance, "./HyperparameterTuning/Results/xgboost_tuning_data.rds")

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals

graph_learner$param_set$values <- final_params
graph_learner$train(task)

saveRDS(graph_learner, "./FinalModels/xgboost_final_model.rds")
cat("xgboost_final_model.rds gespeichert\n")

# Plot the feature importance
source("./HyperparameterTuning/VizHyper.r")
plot_importance(graph_learner$importance()%>%head(20), "XGBoost")
