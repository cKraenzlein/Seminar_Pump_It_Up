# Load required packages
library(mlr3)
library(mlr3learners)   # Provides learners like 'classif.ranger'
library(mlr3filters)    # For feature selection
library(mlr3viz)        # For plotting
library(mlr3tuning)     # For hyperparameter tuning
library(mlr3fselect)    # For feature selection
library(ranger)         # The underlying Random Forest implementation
library(iml)            # For model interpretation (including SHAP)
library(data.table)     # mlr3 often works with data.table
library(mlr3pipelines)  # For pipelines
library(ggplot2)        # For plotting
library(dplyr)          # For data manipulation
library(patchwork)      # Combining Plots

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
future::plan("multisession", workers = 3)
# --------------------------------------------------------------------------------------------------------------------------
# Task
task <- as_task_classif(Data, target = "status_group")

# Define learners with default params first
lrn_rf = lrn("classif.ranger", predict_type = "prob", id = "rf", num.trees  = 1000, mtry = 6, min.node.size = 3, splitrule = "gini", num.threads = 6)
lrn_knn = lrn("classif.kknn", predict_type = "prob", id = "knn", k = 25, distance = 1, kernel = "inv")
lrn_xgb = lrn("classif.xgboost", predict_type = "prob", id = "xgb", max_depth = 60, eta = 0.019, nrounds = 300, subsample = 0.6791455, nthread = 5)

# Clone preprocess for each learner to avoid ID conflicts
pre_rf  = po("imputeoor", id = "impute_rf") %>>% po("encodeimpact", id = "encode_rf") %>>% po("fixfactors", id = "fix_rf")
pre_knn = po("imputeoor", id = "impute_knn") %>>% po("encodeimpact", id = "encode_knn") %>>% po("fixfactors", id = "fix_knn")
pre_xgb = po("imputeoor", id = "impute_xgb") %>>% po("encodeimpact", id = "encode_xgb") %>>% po("fixfactors", id = "fix_xgb")

# Create learners with preprocessing applied
graph_rf = pre_rf %>>% lrn_rf
graph_knn = pre_knn %>>% lrn_knn
graph_xgb = pre_xgb %>>% lrn_xgb

# Stack them
stack = gunion(list(graph_rf, graph_knn, graph_xgb)) %>>%
        po("featureunion") %>>%
        po("classif.ranger", predict_type = "prob", num.threads = 5)  # master
# Wrap in GraphLearner
stack_learner = GraphLearner$new(stack)

# Tune meta-learner hyperparameters
at_master = AutoTuner$new(
  learner = stack_learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.ce"),
  search_space = ps(
    classif.ranger.mtry = p_int(1, 10),
    classif.ranger.min.node.size = p_int(1, 10)
  ),
  terminator = trm("evals", n_evals = 20),
  tuner = tnr("random_search")
)

at_master$train(task)