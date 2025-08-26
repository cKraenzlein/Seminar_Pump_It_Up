# Load required packages
library(mlr3)
library(mlr3viz)
library(mlr3tuning) 
library(mlr3extralearners)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3mbo)

# Set the working directory
here::here()

# Load the training data
source("./Data/CreateTrainingData.R")
Data <- Data_Training_Final

# Set seed for reproducibility
set.seed(24)

logical_to_int = po("colapply", id = "encode_logical", applicator = as.numeric, affect_columns = selector_type("logical"))

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner
learner = lrn("classif.catboost", id = "catboost", thread_count = 10)

# Combine graph and learner
graph_learner = as_learner(logical_to_int %>>% learner)
graph_learner$param_set$data
# Hyperparameter space
hyper_param_space <- ps(
  catboost.iterations       = p_int(100, 1000),
  catboost.learning_rate    = p_dbl(0.001, 0.3),
  catboost.depth            = p_int(4, 10),
  catboost.l2_leaf_reg      = p_dbl(1, 10)
)

# Resampling and Measure Strategy
resampling <- rsmp("repeated_cv", folds = 3, repeats = 1)
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
  terminator = term_rs,
  search_space = hyper_param_space
)

# Start tuning
tuner_rs$optimize(tuning_instance)

# Save tuning results
saveRDS(tuning_instance, "./HyperparameterTuning/Results/catboost_tuning_data.rds")

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals

graph_learner$param_set$values <- final_params
graph_learner$train(task)

# Plot the feature importance
plot_importance(graph_learner$importance(), "CatBoost")

saveRDS(graph_learner, "./FinalModels/catboost_final_model.rds")
print("catboost_final_model.rds gespeichert")
