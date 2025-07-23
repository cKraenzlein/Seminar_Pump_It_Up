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
# --------------------------------------------------------------------------------------------------------------------------
# Load the training data
source("CreateTrainingData.R")
Data <- TrainingData_Formatted
# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_Formatted
#---------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(24)
# Create a classification task
task <- as_task_classif(Data, target = "status_group")
# --------------------------------------------------------------------------------------------------------------------------
# Random Forest with mlr3, Runtime approximately 2h 30min
learner_RF <- lrn("classif.ranger", num.threads = 6)
# Feature selection 
instance_RF = fselect(
  fselector = fs("sequential", strategy = "sbs"),
  task = task,
  learner = learner_RF,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc")
)
# Best performing feature subset
instance_RF$result
# --------------------------------------------------------------------------------------------------------------------------
#   amount_tsh  basin construction_year district_code extraction_type_class
#       <lgcl> <lgcl>            <lgcl>        <lgcl>                <lgcl>
#1:      FALSE   TRUE              TRUE          TRUE                  TRUE
#   extraction_type_group gps_height latitude longitude management
#                  <lgcl>     <lgcl>   <lgcl>    <lgcl>     <lgcl>
#1:                  TRUE       TRUE     TRUE      TRUE      FALSE
#   management_group payment_type permit population public_meeting quality_group
#             <lgcl>       <lgcl> <lgcl>     <lgcl>         <lgcl>        <lgcl>
#1:             TRUE         TRUE   TRUE       TRUE           TRUE         FALSE
#   quantity region source source_class source_type water_quality
#     <lgcl> <lgcl> <lgcl>       <lgcl>      <lgcl>        <lgcl>
#1:     TRUE  FALSE   TRUE         TRUE       FALSE          TRUE
#   waterpoint_type waterpoint_type_group year_recorded
#            <lgcl>                <lgcl>        <lgcl>
#1:            TRUE                 FALSE         FALSE
#                                                                                           features
#                                                                                             <list>
#1: basin,construction_year,district_code,extraction_type_class,extraction_type_group,gps_height,...
#   n_features classif.acc
#        <int>       <num>
#1:         18    0.811532
# --------------------------------------------------------------------------------------------------------------------------
# Train Model with selected features
#task$select(instance_RF$result_feature_set)
task$select(setdiff(task$feature_names, c("management", "installer", "source_type", "year_recorded")))
learner_RF$train(task)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_RF$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "FeatureSelRF.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
# KKNN with mlr3, Runtime approximately 2h 12min
# Task for kknn
task_kknn <- as_task_classif(Data, target = "status_group")
learner_kknn = as_learner(ppl("robustify") %>>% lrn("classif.kknn"))
# Feature selection 
instance_kknn = fselect(
  fselector = fs("sequential", strategy = "sbs"),
  task = task_kknn,
  learner = learner_kknn,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc")
)
# Best performing feature subset
instance_kknn$result
# n_features = 13, classif.acc = 0.7844444
# "amount_tsh","construction_year","district_code","extraction_type_class","mean_latitude","mean_longitude"
# "month_recorded","payment_type","quantity","region","source_type","waterpoint_type","year_recorded"
# --------------------------------------------------------------------------------------------------------------------------
# Train Model with selected features
# task_kknn$select(instance_kknn$result_feature_set)
task_kknn$select(c("amount_tsh","construction_year","district_code","extraction_type_class","mean_latitude","mean_longitude",
                   "month_recorded","payment_type","quantity","region","source_type","waterpoint_type","year_recorded"))
learner_kknn$train(task_kknn)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_kknn$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "FeatureSelKKNN.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------