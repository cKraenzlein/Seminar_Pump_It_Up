# ------------------------------------------------------------------------------------------------------------------------------------ #
# Load necessary libraries
library(proxy)
library(mlr3)
library(rpart.plot)
library(mlr3learners)
library(data.table)
library(mlr3verse)
library(mlr3filters)
library(mlr3viz)
library(mlr3tuning)
library(ggplot2)
library(future)

# Set seed for reproducibility
set.seed(24)

#source the training data
here::here()
source("CreateTrainingData.R")
source("CreateTestDataSet.R")

# Load the training data
Data <- TrainingDataWrapper

# Parallelization: use 6 cores
plan("multisession", workers = 6)

# Define the task
task <- as_task_classif(Data, target = "status_group")

# Load the learner
learner <- lrn("classif.ranger", num.threads = 6)

# Feature selection 
instance = fselect(
  fselector = fs("sequential"),
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc")
)

# Best performing feature subset
instance$result

Data_NaRm <- na.omit(Data)
task_NaRm <- as_task_classif(Data_NaRm, target = "status_group")

# Feature Selection for diffrent learners
learners = list(
  lrn("classif.kknn", id = "kknn"),
  lrn("classif.ranger", id = "rf", importance = "permutation", num.trees = 200),
  lrn("classif.nnet", id = "nnet", maxit = 200)
)

efs = ensemble_fselect(
  fselector = fs("sequential", strategy = "sbs"),  # rfe does not work because kknn does not support importance
  task = task_NaRm,
  learners = learners,
  init_resampling = rsmp("subsampling", repeats = 5, ratio = 0.8),
  inner_resampling = rsmp("cv", folds = 3),
  inner_measure = msr("classif.ce"),
  measure = msr("classif.acc"),
  terminator = trm("run_time", secs = 10800)
)

print(efs)

efs$result

autoplot(efs, type = "performance", theme = theme_minimal(base_size = 14)) +
  scale_fill_brewer(palette = "Set1")

autoplot(efs, type = "n_features", theme = theme_minimal(base_size = 14)) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(breaks = seq(0, 60, 10))