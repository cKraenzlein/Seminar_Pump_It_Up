# Load required packages
library(mlr3)
library(mlr3viz)

# Set the working directory
here::here()

source("./Data/CreateTrainingData.R")

# Load Training Data 
Data <- Data_Training_Final

# Set seed for reproducibility
set.seed(24)

# Load the final model
final_learner_RF <-         readRDS("./FinalModels/randomForest_final_TEST_model.rds")
final_learner_XGBoost <-    readRDS("./FinalModels/xgboost_final_model.rds")
final_learner_lightGBM <-   readRDS("./FinalModels/lightGBM_final_model.rds")
final_learner_catboost <-   readRDS("./FinalModels/catboost_final_model.rds")
final_learner_Stacked <-   readRDS("./FinalModels/Stacked_Model_RF_CAT_final_model.rds")

final_learners <- list(final_learner_RF, final_learner_XGBoost, final_learner_lightGBM, final_learner_catboost, final_learner_Stacked)

# Define Task
task = as_task_classif(Data_Training_Final, id = "PumpItUp", target = "status_group")
# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))
# Define Benchmark Design
benchmark_design = mlr3::benchmark_grid(tasks = task,
                                        learners = final_learners, 
                                        resamplings = rsmp("cv", folds = 3))

# Run Benchmark
benchmark_result = mlr3::benchmark(benchmark_design)

# Aggregate Results
benchmark_result$aggregate(msr("classif.acc"))
InTrainingACC <- mlr3viz::autoplot(benchmark_result, measure = msr("classif.acc"))
# Visualize Benchmark Results
InTrainingACC <- mlr3viz::autoplot(benchmark_result, measure = msr("classif.acc")) +
  ggplot2::scale_x_discrete(labels = c("RF", "XGBoost", "LightGBM", "CATBoost", "Stacked Model")) +
  ggplot2::theme_bw() +
  ggplot2::labs(x = "Model", y = "Accuracy")

ggplot2::ggsave("./Plots/Model_Performance_Comparison.png", plot = InTrainingACC, dpi = 400)
