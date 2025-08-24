# Load required packages
library(mlr3)
library(mlr3viz)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)

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

# Load the optimized final learners
final_learner_RF <- readRDS("./FinalModels/randomForest_final_model.rds")

base_learner_RF = lrn("classif.ranger", model = final_learner_RF, predict_type = "prob")

super_learner_RF = lrn("classif.ranger", predict_type = "class")

Stacked_Model = pipeline_stacking(
  base_learners = list(base_learner_RF),
  super_learner = super_learner_RF,
  method = "cv",
  folds = 3,
  use_features = TRUE
)

Stacked_Model$train(task)