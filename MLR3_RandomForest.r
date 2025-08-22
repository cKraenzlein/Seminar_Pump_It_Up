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
Data <- TrainingData_VarSel
# Set seed for reproducibility
set.seed(24)
# Create a classification task
Task <- as_task_classif(Data, target = "status_group")
# Create tuning/training split
train_data <- sample(Task$row_ids, 0.8 * Task$nrow)
tuning_data <- setdiff(Task$row_ids, train_data)
# 
Task_tuning <- Task$clone()$filter(tuning_data)

# ==================================================================================
# Try mlr3Torch: 
lrn_torch <- lrn("classif.torch_model",
  neurons = c(50, 50),
  batch_size = 256,
  epochs = 30, # Number of training epochs
  device = "auto", # Uses GPU if available, otherwise CPU
  shuffle = TRUE, # because iris is sorted
  optimizer = t_opt("adamw"),
  loss = t_loss("cross_entropy")
)

lrn_torch$train(Task, row_ids = train_data)
lrn_torch$model$network

predictions <- lrn_torch$predict(Task, row_ids = tuning_data)
predictions$score(msr("classif.acc"))

# =====================================================================================

# Create a learner for classification using ranger and set hyperparameters to tune
learner = lrn("classif.ranger",
  num.trees  = to_tune(c(250, 500, 1000, 1500)),
  mtry = to_tune(c(2, 4, 6, 8, 10, 12, 14, 15)),
  min.node.size = 1,
  splitrule = "gini",
  num.threads = 6
)
# 
instance = ti(
  task = Task_tuning,
  learner = learner,
  resampling = rsmp("cv", folds = 5),
  measures = msr("classif.bacc"),
  terminator = trm("none")
)

tuner = tnr("grid_search")

# Tune the hyperparameters
tuner$optimize(instance)
# Best hyperparameters
instance$result$learner_param_vals
# min.node.size # mtry  # num.trees # splitrule
# 1             # 10     # 1500      # gini
# Visualize the tuning results
autoplot(instance)

# Train the Model with the best hyperparameters on the training data
learner_final = lrn("classif.ranger",
  num.trees  = 1000,
  mtry = 8,
  min.node.size = 1,
  splitrule = "gini",
  num.threads = 8
)

learner_final$train(Task, row_ids = train_data)
print(learner_final$model)

# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_VarSel
str(Test_Data)
prediction = learner_final$predict_newdata(Test_Data)
prediction

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "submissionRandomForest_mlr3_Tuning_fselect.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------


# XGBoost Model in R using mlr3 with hyperparameter tuning
# Define the XGBoost Learner
features_for_tabular_processing = Task_tuning$feature_types[
  type %in% c("numeric", "integer", "logical", "character", "factor", "ordered"),
  id
]

fencoder = po("encode", method = "treatment", affect_columns = selector_type("factor"))
ord_to_int = po("colapply", applicator = as.integer, affect_columns = selector_type("ordered"))
po_select_tabular = po("select", selector = selector_invert(selector_type("lazy_tensor")))

# We use predict_type = "prob" for classification to get probabilities
learner_xgb = lrn(
    "classif.xgboost", 
    predict_type = "response",
    nrounds = to_tune(c(50, 100, 150, 200, 500, 1000)), # Number of boosting rounds
    #eta = to_tune(c(0.01, 0.1, 0.2)), # Learning rate
    #max_depth = to_tune(c(2, 4, 6)), # Max tree depth
    #subsample = to_tune(c(0.6, 0.7, 0.8, 0.9, 1.0)), # Subsample ratio of instances
    nthread = 6#,
    #colsample_bytree = to_tune(c(0.6, 0.7, 0.8, 0.9, 1.0)) # Subsample ratio of columns
)
print(learner_xgb)

graph = po_select_tabular %>>% fencoder %>>% ord_to_int %>>% learner_xgb
print(graph)
graph_learner = as_learner(graph)

instance_xg =  tune(
  tuner = tnr("grid_search"),
  task = Task_tuning,
  learner = graph_learner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  terminator = trm("none")
)

# Best hyperparameters
autoplot(instance_xg)
instance_xg$result$learner_param_vals
# nrounds   # eta     # max_depth # subsample # colsample_bytree
# 50        # 0.2     # 6         # 0.8       # 0.9


# Train the Model with the best hyperparameters on the training data
learner_final_xgb = lrn(
  "classif.xgboost",
  nrounds = 50,
  eta = 0.2,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.9,
  predict_type = "response"
)
graph_final = fencoder %>>% learner_final_xgb
graph_learner_xgb = as_learner(graph_final)

graph_learner_xgb$train(Task, row_ids = train_data)
print(graph_learner_xgb$model)

# Testing the XGBoost model on the test data
prediction = graph_learner_xgb$predict_newdata(Test_Data)
prediction


learner_test = lrn(
  "classif.xgboost",
  predict_type = "response",
  nrounds = 1000,
  eta = 0.01,
  max_depth = 10)

learner_test <- as_learner(fencoder %>>% learner_test)
# Train the learner
learner_test$train(Task, row_ids = train_data)
# View the model
learner_test$model
# Predict on the test set
pred_test = learner_test$predict(Task, row_ids = tuning_data)
# Calculate accuracy on the test set
measures = msrs(c('classif.acc'))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "submissionRandomForest_mlr3_Tuning_fselect.csv", row.names = FALSE)





learner_lda = lrn("classif.lda", predict_type = "prob")





# Create a learner for classification using rpart
learner_lda = lrn("classif.lda", predict_type = "response") # Using impurity for feature importance
# Train the learner
learner_lda$train(Task, row_ids = train_data)
# View the model
learner_lda$model
# Predict on the test set
pred_test = learner_lda$predict(Task, row_ids = tuning_data)
# Calculate accuracy on the test set
measures = c(msrs("classif.acc"), msrs("classif.bacc"))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)


# Create a learner for classification using rpart

learner_kknn = lrn("classif.kknn", predict_type = "response")

search_space = ps(
  k = p_int(3, 50),
  distance = p_dbl(1, 3),
  kernel = p_fct(c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))
)

tuner_random = tnr("random_search", batch_size = 100)
resampling_tuning = rsmp("cv", folds = 10) # 5-fache Kreuzvalidierung
measure_tuning = msr("classif.bacc") # Metrik für die Optimierung

instance_tuning = TuningInstanceBatchSingleCrit$new(
  task = Task_tuning,
  learner = learner_kknn,
  resampling = resampling_tuning,
  measure = measure_tuning,
  search_space = search_space,
  terminator = trm("evals", n_evals = 100)
)

tuner_random$optimize(instance_tuning)

instance_tuning$result_learner_param_vals
# Best hyperparameters
# k         # distance          # kernel    
# 10        # 1.351592          # "inv" 


as.data.table(instance_tuning$archive)

autoplot(instance_tuning)


learner_kknn_tuned = lrn("classif.kknn",
  k = 10,
  distance = 1.351592,
  kernel = "inv",
  predict_type = "response"
)
# Train the learner
learner_kknn_tuned$train(Task, row_ids = train_data)
# View the model
learner_kknn_tuned$model
# Predict on the test set
pred_test = learner_kknn_tuned$predict(Task, row_ids = tuning_data)
# Calculate accuracy on the test set
measures = msrs(c('classif.acc'))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)
# 0.789 !!!!!



# Stacking 2. aptempt

library(mlr3)
library(mlr3learners)

base_learners = list(
  lrn(
  "classif.ranger",
  predict_type = "prob",
  num.trees  = 1500,
  mtry = 6,
  min.node.size = 1,
  splitrule = "gini",
  num.threads = 8),

  lrn(
  "classif.kknn",
  k = 10,
  distance = 1.351592,
  kernel = "inv",
  predict_type = "prob"),

  lrn("classif.lda", 
  predict_type = "prob")
)

super_learner = lrn("classif.rpart")

graph_stack = pipeline_stacking(base_learners, super_learner)
graph_learner = as_learner(graph_stack)

graph_learner$train(Task, row_ids = train_data)




# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_VarSel
str(Test_Data)

prediction = graph_learner$predict_newdata(Test_Data)
prediction

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "submissionStackedModel_ranger_kknn_lda_rpart.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------






# Predict on the test set
pred_test = graph_learner$predict(Task, row_ids = tuning_data)
# Calculate accuracy on the test set
measures = c(msrs('classif.acc'), msrs('classif.bacc'))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)


str(Data)

library(ggcorrplot)
model.matrix(~0+., data=Data) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag=FALSE, type="lower", lab=TRUE, lab_size=2)
