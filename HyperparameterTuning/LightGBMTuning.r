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

logical_to_int = po("colapply", id = "encode_logical", applicator = as.numeric, affect_columns = selector_type("logical"))

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner, number of threads can be choosen according to the available resources
learner = lrn("classif.lightgbm", id = "lightGBM", num_threads = 10)
# Combine graph and learner
graph_learner = as_learner(logical_to_int  %>>% learner)

# Define the hyperparameter space
hyper_param_space <- ps(
    lightGBM.learning_rate =               p_dbl(0, 0.5),
    lightGBM.num_leaves =                  p_int(2, 500),
    lightGBM.feature_fraction =            p_dbl(0.3, 1),
    lightGBM.bagging_fraction =            p_dbl(0.3, 1),
    lightGBM.min_sum_hessian_in_leaf =     p_dbl(0, 1),
    lightGBM.min_data_in_leaf =            p_int(1, 500),
    lightGBM.lambda_l1 =                   p_dbl(exp(-5), exp(2), logscale = TRUE),
    lightGBM.lambda_l2 =                   p_dbl(exp(-5), exp(2), logscale = TRUE)
)
# Resampling and Measure Strategy
resampling <- rsmp("repeated_cv", folds = 3, repeats = 3)
measure    <- msr("classif.acc")

# Define Tuner and Termination (time can be set)
tuner_rs <- tnr("mbo")
term_rs  <- trm("combo", list(trm("clock_time", stop_time = Sys.time() + 12 * 3600),
                              trm("evals", n_evals = 100)), any = TRUE)

# Tuning instance
tuning_instance <- ti(
  task       = task,
  learner    = graph_learner,
  resampling = resampling,
  measures   = measure,
  terminator = term_rs,
  search_space  = hyper_param_space
)

# Start tuning
tuner_rs$optimize(tuning_instance)

# Save tuning results
saveRDS(tuning_instance, "./HyperparameterTuning/Results/LightGBM_tuning_data.rds")

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals
graph_learner$param_set$values <- final_params

# Train the final model
graph_learner$train(task)
# Get the in Training Performance
performance_train = graph_learner$predict(task)$score(msr("classif.acc"))
print(performance_train)

# Plot the feature importance
source("./HyperparameterTuning/VizHyper.r")
plot_importance(graph_learner$importance(), "LightGBM")

saveRDS(graph_learner, "./FinalModels/lightGBM_final_model.rds")
print("lightGBM_final_model.rds gespeichert.")
