# Load required packages
library(mlr3)
library(mlr3learners) # Provides learners like 'classif.ranger'
library(mlr3filters) # For feature selection
library(mlr3viz)     # For plotting
library(mlr3tuning)  # For hyperparameter tuning
library(mlr3fselect) # For feature selection
library(ranger)      # The underlying Random Forest implementation
library(iml)         # For model interpretation (including SHAP)
library(data.table)  # mlr3 often works with data.table
library(mlr3pipelines) # For pipelines

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData_Formatted
# Set seed for reproducibility
set.seed(24)
# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_Formatted
#---------------------------------------------------------------------------------------------------------------------------
# nnet with mlr3
task_nnet <- as_task_classif(Data, target = "status_group")
learner_nnet = as_learner(ppl("robustify") %>>% lrn("classif.nnet", maxit = 5000))
# Train the learner
learner_nnet$train(task_nnet)
# --------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_nnet$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "BaseLineNNET.csv", row.names = FALSE)
# --------------------------------------------------------------------------------------------------------------------------
# NNet with mlr3, Runtime approximately 2h 12min
# Task for nnet
task_nnet <- as_task_classif(Data, target = "status_group")
learner_nnet = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.nnet", maxit = 5000))
# Feature selection
instance_nnet = fselect(
  fselector = fs("sequential", strategy = "sbs"),
  task = task_nnet,
  learner = learner_nnet,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc")
)
# Best performing feature subset
instance_nnet$result
# --------------------------------------------------------------------------------------------------------------------------
# Train Model with selected features
task_nnet$select(instance_nnet$result_feature_set)
learner_nnet$train(task_nnet)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_nnet$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "FeatureSelNNet.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
# KKNN with mlr3, Runtime approximately #
# Create a task with the full dataset
future::plan("multisession", workers = 3)
task_NNET <- as_task_classif(Data, target = "status_group")
# Define the learner and the tuning spaces
learner_NNET = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.nnet",
  size = to_tune(p_int(1, 30)),
  decay = to_tune(p_dbl(1e-4, 0.1, logscale = TRUE)),
  maxit = to_tune(p_int(1, 2500)),
  MaxNWts = 10000
))
# Define the tuning instance
instance_NNET = ti(
  task = task_NNET,
  learner = learner_NNET,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 3 * 3600),
                                 trm("evals", n_evals = 100)), any = TRUE)
)
tuner_NNET = tnr("random_search")
# Tune the hyperparameters
tuner_NNET$optimize(instance_NNET)
# Result:
instance_NNET$result$learner_param_vals
# decay = 0.05878915  
# maxit = 1828
# size = 10

# Visualize the tuning results
autoplot(instance_NNET)
# -------------------------- -------------------------------------------------------------------------------------------------
# Create new learner with the best hyperparameters
learner_NNET_tuned = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.nnet",
  size = 10,
  decay = 0.05878915,
  maxit = 1828,
  MaxNWts = 10000
))
# Train the tuned learner
learner_NNET_tuned$train(task_NNET)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_NNET_tuned$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "TunedNNET.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
task_XGBoost <- as_task_classif(Data, target = "status_group")

lrn_xgb = lrn("classif.xgboost", nrounds = 100)

factor_pipeline =
    po("removeconstants") %>>%
    po("collapsefactors", no_collapse_above_prevalence = 0.01) %>>%
    po("encodeimpact", id = "high_card_enc") %>>%
    po("encode", method = "one-hot", id = "low_card_enc") %>>%
    po("encode", method = "treatment", id = "binary_enc")

glrn_xgb_impact = as_learner(factor_pipeline %>>% lrn_xgb)
glrn_xgb_impact$id = "XGB_enc_impact"


glrn_xgb_impact$train(task_XGBoost)
