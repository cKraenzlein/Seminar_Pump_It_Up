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
# Random Forest with mlr3
task_ranger <- as_task_classif(Data, target = "status_group")
learner_ranger = lrn("classif.ranger", num.threads = 6, importance = "impurity") # Using impurity for feature importance
# Train the learner
learner_ranger$train(task_ranger)
# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_Formatted
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_ranger$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "BaseLineRF.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
# kknn with mlr3
task_kknn <- as_task_classif(Data, target = "status_group")
learner_kknn = as_learner(ppl("robustify") %>>% lrn("classif.kknn"))
# Train the learner
learner_kknn$train(task_kknn)
# --------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_kknn$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "BaseLineKKNN.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
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