# Random Forest Model in R using mlr3 with hyperparameter tuning
# Features were seleceted using fselect from mlr3 and the feature importance

# Load necessary packages
library(mlr3)
library(mlr3learners) # Provides learners like 'classif.ranger'
library(mlr3filters) # For feature selection
library(mlr3viz)     # For plotting
library(mlr3tuning)  # For hyperparameter tuning
library(mlr3fselect) # For feature selection
library(ranger)      # The underlying Random Forest implementation
library(iml)         # For model interpretation (including SHAP)
library(data.table)  # mlr3 often works with data.table
library(xgboost)
library(mlr3pipelines)
library(mlr3extralearners)
library(mlr3torch)
library(doParallel)

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData_FormattedInstaller
# Set seed for reproducibility
set.seed(24)
# Create a classification task
Task <- as_task_classif(Data, target = "status_group")
# Create tuning/training split
train_data <- sample(Task$row_ids, 0.8 * Task$nrow)
tuning_data <- setdiff(Task$row_ids, train_data)
# 
Task_tuning <- Task$clone()$filter(tuning_data)

# ======================================================================================
# Tunung resampling and measur
resampling_tuning = rsmp("cv", folds = 5)
measure_tuning = msr("classif.acc")
# ======================================================================================
# Hyperparametertuning random forest:
learner_ranger = lrn("classif.ranger",
  num.trees  = to_tune(c(500, 1000, 1500)),
  mtry = to_tune(c(2, 4, 6, 8, 10, 12, 14)),
  min.node.size = to_tune(c(1,3,5,7,9)),
  splitrule = to_tune(c("gini", "extratrees")),
  num.threads = 6,
  predict_type = "response"
)

instance_ranger = ti(
  task = Task_tuning,
  learner = learner_ranger,
  resampling = resampling_tuning,
  measures = measure_tuning,
  terminator = trm("none")
)

tuner = tnr("grid_search")

# Tune the hyperparameters
tuner$optimize(instance_ranger)
# Visualize Tuning
autoplot(instance_ranger)
# Best hyperparameters
instance_ranger$result$learner_param_vals
hyper_ranger <- instance_ranger$result$learner_param_vals
# Best hyperparameters random forest
# min.node.size # mtry  # num.trees # splitrule
# 9             # 6      # 1000      # gini

# ========================================================================================
# Hyperparametertuning knn:
learner_kknn = lrn("classif.kknn", predict_type = "response")

search_space = ps(
  k = p_int(3, 50),
  distance = p_dbl(1, 3),
  kernel = p_fct(c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))
)

tuner_random = tnr("random_search", batch_size = 100)

instance_kknn = TuningInstanceBatchSingleCrit$new(
  task = Task_tuning,
  learner = learner_kknn,
  resampling = resampling_tuning,
  measure = measure_tuning,
  search_space = search_space,
  terminator = trm("evals", n_evals = 100)
)

# Tune the hyperparameters
tuner_random$optimize(instance_kknn)
# Visualize tuning results
autoplot(instance_kknn)
# Best hyperparameters
instance_kknn$result_learner_param_vals
hyper_knn <- instance_kknn$result$learner_param_vals
# Best hyperparameters kknn
# k         # distance          # kernel    
# 15        # 1.0674            # "inv" 

# ======================================================================================
# Stacking Model:

base_learners = list(
  lrn(
  "classif.ranger",
  predict_type = "prob",
  num.trees  = 1000,
  mtry = 4,
  min.node.size = 7,
  splitrule = "gini",
  num.threads = 6),

  lrn(
  "classif.kknn",
  predict_type = "prob",
  k = 24,
  distance = 1.188468,
  kernel = "inv")
)

super_learner = lrn("classif.ranger", id = "super.ranger", num.threads = 6)

stack = pipeline_stacking(base_learners, super_learner)
stacked_model_learner = as_learner(stack)
# Train the stacked model on the train Data
stacked_model_learner$train(Task, row_ids = train_data)

# -----------------------------------------------------------------------------------------------------------------------------
# Create the Submission DataFrame
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_FormattedInstaller

prediction = stacked_model_learner$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "submissionStackedModel_OldVariableSet(RangerKKNNSuperRanger).csv", row.names = FALSE)
# -----------------------------------------------------------------------------------------------------------------------------

# ====================================================================================
# Test our stacked model [ACC and BACC]

# Predict on the test set
pred_test = stacked_model_learner$predict(Task, row_ids = tuning_data)
# Calculate accuracy on the test set
measures = c(msrs('classif.acc'), msrs('classif.bacc'))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)
# =====================================================================================