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

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner, number of threads can be choosen according to the available resources
learner = lrn("classif.ranger", id = "randomForest", num.threads = 12)

# Hyperparameter space
hyper_param_space <- ps(
  randomForest.max.depth      = p_int(10, 50),
  randomForest.num.trees      = p_int(100, 1000),
  randomForest.mtry.ratio     = p_dbl(0, 1),
  randomForest.min.node.size  = p_int(1, 10)
)

# Resampling and Measure Strategy
resampling <- rsmp("repeated_cv", folds = 3, repeats = 3)
measure    <- msr("classif.acc")

# Define Tuner and Termination
tuner_rs <- tnr("mbo")
term_rs  <- trm("combo", list(trm("clock_time", stop_time = Sys.time() + 12 * 3600),
                              trm("evals", n_evals = 3)), any = TRUE)

# Tuning instance
tuning_instance <- ti(
  task       = task,
  learner    = learner,
  resampling = resampling,
  measures   = measure,
  terminator = term_rs,
  search_space  = hyper_param_space
)

# Start tuning
tuner_rs$optimize(tuning_instance)

# Save visualization of the tuning process
source("./HyperparameterTuning/VizHyper.r")
Plot_Tuning_RF(tuning_instance)

# Save tuning results
saveRDS(tuning_instance, "./HyperparameterTuning/Results/randomForest_tuning_data.rds")

# Learner for the training
learner_train = lrn("classif.ranger", id = "randomForest", num.threads = 12)

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals

learner_train$param_set$values <- final_params

learner_train$param_set$values$importance = "impurity" # To get impurity-based feature importance
learner_train$train(task)

# Get the in Training Performance
performance_train = learner_train$predict(task)$score(msr("classif.acc"))
print(performance_train)

# Plot the feature importance
source("./HyperparameterTuning/VizHyper.r")
plot_importance(learner_train$importance(), "RandomForest")

saveRDS(learner_train, "./FinalModels/randomForest_final_TEST_model.rds")
print("randomForest_final_TEST_model.rds gespeichert.")