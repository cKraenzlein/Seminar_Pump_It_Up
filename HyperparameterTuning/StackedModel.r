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
final_learner_RF <- readRDS("./FinalModels/randomForest_final_TEST_model.rds")
final_learner_catboost <- readRDS("./FinalModels/catboost_final_model.rds")

# Create base learners with final parameters
base_learner_RF = lrn("classif.ranger", id = "base1", predict_type = "prob")
base_learner_RF$param_set$values = final_learner_RF$param_set$values

logical_to_int = po("colapply", id = "encode_logical", applicator = as.numeric, affect_columns = selector_type("logical"))

base_learner_catboost = lrn("classif.catboost", id = "base2", predict_type = "prob")

graph_learner = as_learner(logical_to_int %>>% base_learner_catboost)
# Set the parameters on the graph_learner using the 'base2' prefix
graph_learner$param_set$values$`base2.learning_rate`          = final_learner_catboost$param_set$values$learning_rate
graph_learner$param_set$values$`base2.iterations`             = final_learner_catboost$param_set$values$iterations
graph_learner$param_set$values$`base2.depth`                  = final_learner_catboost$param_set$values$depth
graph_learner$param_set$values$`base2.l2_leaf_reg`            = final_learner_catboost$param_set$values$l2_leaf_reg
graph_learner$param_set$values$`base2.thread_count`          = final_learner_catboost$param_set$values$thread_count

super_learner_RF = lrn("classif.ranger", num.threads = 10, id = "super_learner_RF", importance = "impurity", predict_type = "response")


graph_stack = pipeline_stacking(list(final_learner_RF, graph_learner), super_learner_RF)
graph_learner_stack = as_learner(graph_stack)
graph_learner_stack$train(task)

# Source the test data
source("./Data/CreateTestDataSet.R")

# Save the ID column and the test data
ID <- Id
Test_Data <- Data_Test_Final

# Predict on the test set
prediction = graph_learner_stack$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "Stacked_Model_RF_CAT.csv", row.names = FALSE)

saveRDS(graph_learner_stack, "./FinalModels/Stacked_Model_RF_CAT_final_model.rds")
